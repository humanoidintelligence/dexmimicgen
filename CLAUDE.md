# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This is a **reproduction** of DexMimicGen's bimanual data-generation pipeline, not the
original authors' release. The data-generation layer (`dexmimicgen/datagen/`,
`dexmimicgen/env_interfaces/`, `dexmimicgen/configs/`, and the `prepare_src_dataset.py` /
`generate_dataset.py` scripts) was reconstructed, debugged end-to-end on `TwoArmThreading`,
and extended to all **nine** tasks across three embodiments (Panda ×2, PandaDex ×2, GR1).
All **seven tasks that ship a source dataset** are validated end-to-end (the full
generate loop yields filtered successes): `two_arm_threading`, `two_arm_three_piece_assembly`,
`two_arm_lift_tray`, `two_arm_box_cleanup`, `two_arm_drawer_cleanup`, `two_arm_coffee`, and
`two_arm_pouring`. The remaining two, `two_arm_transport` and `two_arm_can_sort_random`, have
**no source hdf5** in this repo and so are static-only (env + interface instantiate, action
dims/slices correct, signals + `compose_action` compute) with their arm-role designs flagged
by `NOTE` comments for later confirmation via `scripts/analyze_src_demo.py` if source data
becomes available. The simulation `environments/` and `models/` are from the
original release. `REPRODUCTION.md` documents the reconstruction process, the two root-cause
fixes, the full task roster (§7), and the success-rate analysis — read it before modifying
the datagen pipeline.

## Setup

Requires robosuite installed from source first, then this package editable:

```bash
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite
pip install -e .                                    # this repo; Python >= 3.9
```

On headless servers, every rendering/datagen command needs:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

## Common Commands

```bash
# Smoke-test an environment (drop --render on headless machines)
python scripts/demo_random_action.py --env TwoArmThreading --render

# Full data-generation flow (see "Data-Generation Pipeline" below)
python scripts/prepare_src_dataset.py --dataset <src.hdf5> --task two_arm_threading --n 10
python scripts/generate_dataset.py --task two_arm_threading --source <src.hdf5> \
    --output <out.hdf5> --num_demos 5 --num_src 10

# Replay demos from any dataset
python scripts/playback_datasets.py --dataset <dataset.hdf5> --n 1

# Download published datasets (default dest ./datasets)
python scripts/download_hf_dataset.py --path /path/to/save/datasets
```

There is no test suite, linter, or build step. Validation is done empirically by running
`generate_dataset.py` and inspecting the per-attempt success rate (expected ~15-20% per
attempt for Threading; the pipeline oversamples and filters — see `REPRODUCTION.md`).

## Data-Generation Pipeline

The pipeline turns a few source demos into many by applying the SE(3) transform that maps
a source object pose to a newly sampled scene pose onto the recorded robot trajectory.
It runs in three stages, coordinated across the modules below:

1. **`prepare_src_dataset.py`** replays each source demo open-loop and annotates
   `data/<demo>/datagen_info` with per-arm `DatagenInfo` (eef poses, controller targets,
   gripper actions), object poses, and per-subtask termination signals.

2. **`generate_dataset.py` -> `datagen/data_generator.py::DataGenerator.generate()`** is
   the core loop. For each arm and subtask it selects a source demo, computes the object-
   frame SE(3) transform, warps the source segment, and hands per-arm `ArmSubtaskPlan`s to
   the executor. It oversamples on randomized scenes and keeps only successful attempts
   until `num_demos` successes are collected (cap `num_demos * max_num_attempts_per_demo`).

3. **`datagen/waypoint.py::MultiArmExecutor`** executes both arms' plans against the live
   robosuite env, interpolating between waypoints and stepping the simulation.

### Subtask types (the central abstraction)

Every subtask carries a `subtask_type` (`task_spec.py` / `waypoint.py`) that controls
cross-arm execution timing:

- **PARALLEL** — arms run their segments asynchronously and independently.
- **COORDINATION** — arms jointly manipulate one object; both arms MUST use a single
  shared SE(3) transform and the same source demo so their inter-arm relative pose is
  preserved. This is enforced in `DataGenerator.generate()` via the `coord_anchor` dict
  (the first arm to enter the subtask freezes the current/source object pose + source demo
  index; both arms reuse it through the `cur_object_pose_override` /
  `src_object_pose_override` args of `_plan_subtask`). Synchronized ending is achieved by
  plain async advancement over equal-length source segments — do NOT add hold-based sync
  in the executor (it was tried and reverted; see `REPRODUCTION.md` §4).
- **SEQUENTIAL** — an arm holds until the other arm's pre-subtask completes; this is the
  only case where `MultiArmExecutor` idles an arm.

### Adding or modifying a task

A task is defined in three places that must stay mutually consistent:

- **`configs/task_configs.py`** — a `DexMG_Config` subclass (registered by its `name`)
  that builds a `DexMG_TaskSpec`: per arm, an ordered list of subtasks each with an
  `object_ref` (which object's frame the transform uses), a `subtask_term_signal`, and a
  `subtask_type`. Generation knobs (`num_demos`, `guarantee`, `max_num_attempts_per_demo`,
  `select_src_per_subtask`, `transform_first_robot_pose`,
  `interpolate_from_last_target_pose`) default in `configs/config.py`.
- **`env_interfaces/robosuite.py`** — a `DexMG_EnvInterface` subclass mapping arm keys
  (`"left"`/`"right"`) to robosuite `robot_idx` + eef site, plus
  `get_subtask_term_signals()` returning the grasp/stage booleans named by the config.
- **`environments/two_arm_*.py`** — the robosuite env (object placement, success check).

The most error-prone part is keeping the arm-to-object mapping aligned across the config's
`object_ref`/`subtask_term_signal` and the interface's `robot_idx`. robosuite samples
objects at fixed +y/-y sides, and the source demos fix which arm handles which object; a
swap here silently introduces a large lever-arm offset and drives the success rate to zero
(this was the first root-cause bug — `REPRODUCTION.md` §4, Fix 1). Note the +y/-y → arm
convention is **mirrored** between embodiments: in the Panda/PandaDex tasks robot 0 ("right")
handles -y objects, while in the GR1 tasks the right hand handles +y objects.

### Two action conventions (OSC vs whole-body IK)

`compose_action` and the pose↔action helpers in the base `DexMG_EnvInterface` assume the
**OSC delta** convention: each arm emits `[dpos(3), daxisangle(3)]` (relative to current eef)
plus a gripper slice — Panda parallel-jaw `gripper_dim=1` (action dim 14), PandaDex hand
`gripper_dim=6` (action dim 24).

The **GR1** tasks (coffee, pouring, can_sort_random) are different and share a `GR1Interface`
base. Both arms are on a single robot driven by a `WholeBodyIK`/`WholeBodyMinkIK` composite
controller; the 24-dim action is `[right_eef(6), left_eef(6), right_hand(6), left_hand(6)]`
where each `*_eef(6)` is an **absolute** target pose `[pos(3), axis_angle(3)]` in the **world**
frame (controller `input_ref_frame="world"`), not a delta and not base-relative — so
`GR1Interface` packs/unpacks the pose directly with no base transform. (An earlier base-frame
version was wrong: decoding as base-frame was off by ~1.09 m, the base height.) `GR1Interface`
overrides `target_pose_to_action`, `action_to_target_pose`, `_arm_action_slices` and
`env_action_dim`. The real action split is the composite controller's
`_whole_body_controller_action_split_indexes`, NOT `robot._action_split_indexes`.

Dex-hand grasp caveat: for the GR1/PandaDex tasks the contact-based grasp subtask signal
cannot be recomputed by replaying a demo (the teleop finger contacts aren't repopulated by
`set_state`). The shipped source `datagen_info` is authoritative, so for `coffee`/`pouring`
run `generate_dataset.py` **directly on the shipped source** and skip `prepare_src_dataset.py`.
Subtask signal names must match the shipped canonical names (coffee: `grasp`/`insertion`/`rim`;
pouring: `cup_grasped`/`bowl_grasped`/`bowl_on_pad`). Only 7 of the 9 tasks ship a source
dataset — `transport` and `can_sort_random` have none and cannot be generated here.

## Datasets & Paths

- `./datasets` is a symlink to a large external store; `datasets_src/` and `datasets_gen/`
  hold multi-GB source/generated hdf5 and are gitignored (as are `report/`, `*.hdf5`, and
  vendored clones under `references/`).
- A generated demo contains `actions (T, A)` (A = 14 for Panda tasks, 24 for PandaDex/GR1
  tasks), `states (T, S)`, `datagen_info`, `src_demo_inds`, `src_demo_labels`, and a
  `model_file` for replay.

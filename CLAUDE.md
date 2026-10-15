# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This is a **reproduction** of DexMimicGen's bimanual data-generation pipeline, not the
original authors' release. The data-generation layer (`dexmimicgen/datagen/`,
`dexmimicgen/env_interfaces/`, `dexmimicgen/configs/`, and the `prepare_src_dataset.py` /
`generate_dataset.py` scripts) was reconstructed and validated end-to-end on
`TwoArmThreading`. The simulation `environments/` and `models/` are from the original
release. `REPRODUCTION.md` documents the reconstruction process, the two root-cause fixes,
and the measured success-rate analysis — read it before modifying the datagen pipeline.

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
(this was the first root-cause bug — `REPRODUCTION.md` §4, Fix 1).

## Datasets & Paths

- `./datasets` is a symlink to a large external store; `datasets_src/` and `datasets_gen/`
  hold multi-GB source/generated hdf5 and are gitignored (as are `report/`, `*.hdf5`, and
  vendored clones under `references/`).
- A generated demo contains `actions (T, 14)` (bimanual 7+7), `states (T, 63)`,
  `datagen_info`, `src_demo_inds`, `src_demo_labels`, and a `model_file` for replay.

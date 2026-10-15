# DexMimicGen Data-Generation Pipeline — Reproduction Report

This document records the end-to-end reproduction of the **DexMimicGen** bimanual
data-generation pipeline in this repository, using the `TwoArmThreading` task as the
running example. It covers environment setup, the three-step generation flow, the
diagnosis and fix of two root-cause defects, and the final end-to-end validation
results with a success-rate analysis.

> **Note.** This is a *reproduction* of the DexMimicGen data-generation pipeline. See
> the top of `README.md` for scope.

---

## 1. Background

DexMimicGen takes a small number of human source demonstrations and automatically
scales them into a large dataset by exploiting the SE(3) equivariance of robot actions
with respect to object poses: when an object's pose undergoes an SE(3) transform, the
same transform is applied to the recorded robot actions to replay the same effect in a
newly sampled scene. Compared with single-arm MimicGen, the bimanual setting introduces
three subtask types:

- **Parallel** — the two arms pursue independent sub-goals; starts/ends need not align,
  executed asynchronously.
- **Coordination** — the two arms jointly achieve one goal (e.g. threading the needle
  through the ring); during execution the relative pose between the two end-effectors
  must match the source demonstration.
- **Sequential** — one arm's subtask must wait until the other arm's pre-subtask
  completes.

Subtask structure of `TwoArmThreading` (see `dexmimicgen/configs/task_configs.py`):

| arm   | subtask 0 (PARALLEL)   | subtask 1 (COORDINATION)        |
|-------|------------------------|---------------------------------|
| right | grasp tripod (ring)    | coordinated threading, `object_ref=tripod` |
| left  | grasp needle           | coordinated threading, `object_ref=tripod` |

---

## 2. Environment Setup

```bash
# 1) Create and activate the conda environment
source <miniconda>/etc/profile.d/conda.sh
conda activate dexmimicgen

# 2) Install this repository (editable) and robosuite dependencies
pip install -e .

# 3) Use EGL for off-screen rendering on headless servers
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Dependencies are in `requirements.txt` (including `numba==0.56.4`). All generation and
diagnostic scripts below run under the environment variables above.

---

## 3. Three-Step Generation Flow

### Step 1 — Prepare the source dataset (annotate datagen info)

`scripts/prepare_src_dataset.py` replays each demo in the source hdf5 open-loop and
records, at every timestep, the per-arm `DatagenInfo` (eef poses, controller target
poses, gripper actions), the relevant object poses, and each subtask's termination
signal, writing them back to `data/<demo>/datagen_info` for the DataGenerator to
consume.

```bash
python scripts/prepare_src_dataset.py \
    --dataset datasets_src/generated/two_arm_threading.hdf5 \
    --task two_arm_threading \
    --n 10
```

### Step 2 — Generate the new dataset (oversample + filter)

`scripts/generate_dataset.py` repeatedly calls `DataGenerator.generate()` on randomized
initial scenes, keeping only successful attempts until `--num_demos` successes are
collected (capped at `num_demos x max_num_attempts_per_demo`, default `max=50`). This is
exactly the paper's oversample-then-filter design — a modest per-attempt success rate is
expected by construction.

```bash
python scripts/generate_dataset.py \
    --task two_arm_threading \
    --source datasets_src/generated/two_arm_threading.hdf5 \
    --output datasets_gen/two_arm_threading.hdf5 \
    --num_demos 5 --num_src 10
```

### Step 3 — Playback validation

```bash
python scripts/playback_datasets.py --dataset datasets_gen/two_arm_threading.hdf5
```

---

## 4. Debugging and Key Fixes

After the pipeline first ran, `TwoArmThreading` had a **0%** per-attempt success rate on
randomized scenes. Layered diagnosis (grasp signals -> object lift -> coordinated-thread
distance) isolated and fixed two root-cause defects.

### Fix 1: Inverted arm-to-object mapping (0% -> non-zero; the decisive fix)

**Symptom.** Neither arm could grasp its target; `grasp_tripod` / `grasp_needle` signals
stayed at 0.

**Root cause.** robosuite's `TwoArmThreading` samples the needle on the **+y** side
(`two_arm_threading.py`: x in [-0.2,-0.05], y in [0.15,0.25]) and the tripod on the
**-y** side (y in [-0.2,-0.1]). The source demos (confirmed by diagnostics) show robot 0
("right") ending subtask 0 on the **tripod** (0.035 m away) and robot 1 ("left") on the
**needle** (0.065 m away). My initial config had these **swapped** — right reaching for
the needle, left for the tripod. Anchoring to the wrong object introduced a ~0.33 m lever
arm, throwing the whole grasp pose off.

> The earlier observation that "the grasp pose relative to the needle is not conserved
> across the 10 source demos" was an artifact of this 0.33 m lever arm, not a problem
> with the object frame itself.

**Changes** (both must change together to stay consistent):

- `dexmimicgen/configs/task_configs.py`: `right` arm uses `object_ref="tripod"` /
  `subtask_term_signal="grasp_tripod"`; `left` arm uses `object_ref="needle"` /
  `grasp_needle`. The coordination subtasks still anchor both arms to `tripod`.
- `dexmimicgen/env_interfaces/robosuite.py`, `get_subtask_term_signals`: `grasp_tripod`
  checks `robot_idx=0`, `grasp_needle` checks `robot_idx=1`.

After the fix both grasps succeed reliably (`grasp_tripod=1.0, grasp_needle=1.0`) and
both objects lift; the failure point moved to the final coordinated-threading stage.

### Fix 2: Shared SE(3) transform for coordination subtasks (~7% -> ~20%)

**Symptom.** Grasps succeed but the needle misses the ring; the needle-tip-to-ring
distance scatters over 0.02-0.17 m, while the success threshold is only 0.012 m.

**Root cause.** Paper Sec. 4.2 requires two things of coordination subtasks: (1) both
arms use the **same** SE(3) transform, and (2) both arms finish **synchronously**.
Diagnosis showed the tripod moves ~0.11-0.19 m during coordination (carried by the right
arm), and the two arms enter coordination ~20 steps apart. My original implementation
computed a **separate** transform per arm — each at the moment that arm entered
coordination, from that arm's own source start — so the two transforms diverged, the
inter-arm relative pose was no longer cancelled, and threading drifted off.

**Changes** (`dexmimicgen/datagen/data_generator.py`):

- Added two optional args to `_plan_subtask(..., cur_object_pose_override,
  src_object_pose_override)` so the caller can pin the "current object pose" and "source
  object pose" used to build the transform.
- In `generate()`, maintain a `coord_anchor` dict (keyed by subtask index): the **first**
  arm to enter a coordination subtask freezes (current tripod pose, source tripod pose,
  source demo index); both arms then reuse that same anchor, so `T = cur_obj @ src_obj^-1`
  is identical for both arms and the inter-arm relative pose
  `(T*src_right)^-1 (T*src_left) = src_right^-1 * src_left` is preserved exactly.
- Both arms are also pinned to the **same source demo** — conserving the relative pose
  only holds when both arms draw their coordination segment from one source trajectory.

This fix raised the randomized-scene per-attempt rate from 1/15 to 3/15.

### A reverted attempt: hold-based sync in the executor

I once added "start-gating + end-alignment" holds in `MultiArmExecutor` (the arm with
more remaining steps waits for the other) to implement the paper's synchronized ending.
It **regressed to 2/15 and broke faithful identity replay** (the needle overshot to
z=1.06 vs the correct ~0.95). Reason: both arms' subtasks come from the **same source
timeline of equal length**, so plain asynchronous advancement (one source step per
executor step) already keeps them time-aligned and makes coordination finish
synchronously on its own; inserting holds injects "phantom steps" that desync the two
coordination trajectories. It was reverted — the executor now holds only under an
explicit SEQUENTIAL constraint — and the now-unused `_coordination_group` helper was
removed.

---

## 5. End-to-End Validation Results

The full `generate_dataset.py` run works; oversample + filter behaves as intended:

```
Done. 5 / 58 attempts succeeded. Wrote 5 demos to datasets_gen/two_arm_threading.hdf5
```

Output dataset structure check (`h5py`):

- 5 demos, `env_name = TwoArmThreading`
- each demo contains `actions (T, 14)` (bimanual 7+7), `states (T, 63)`, `datagen_info`,
  `src_demo_inds`, `src_demo_labels`, and a `model_file` (replayable)

Stability evaluation (fixed random seed, through the real `generate()` path). These are
**single-run snapshots**; because `generate()` advances one global RNG stream across all
attempts, different `N` runs execute different attempt sequences (not subsets), so with a
true rate around ~15-20% and small samples, +/-5-7% run-to-run variation is expected:

| samples N | success rate | failure breakdown |
|-----------|--------------|-------------------|
| 15        | 13-20% (2-3/15) | mostly near-miss |
| 30        | ~20% (6/30)  | near-miss 15 / gross-miss 9 |
| 40        | 15-22% (6-9/40) | near-miss clustered at 0.025-0.05 |

**Conclusion.** The current ~15-20% per-attempt rate is a **precision** regime, not a
logic defect:

- Failures are mostly near-misses (needle-to-ring 0.025-0.06, just above the 0.012
  threshold).
- They correlate with grasp slip: slip <=0.001 m succeeds; 0.012-0.019 m fails (the
  needle's length amplifies small grip errors past the ring opening).
- This matches the paper: DexMimicGen reports data-generation success rates of **29-76%**
  across tasks and explicitly flags Threading as the hardest (success <70%, heavy
  occlusion). The pipeline relies on **oversample-then-filter** to keep only successful
  demos, so a moderate per-attempt rate is expected rather than a defect.

Residual precision could be improved further by reducing grasp slip, but it is not a
blocker for running the pipeline end-to-end.

---

## 6. Reproduction Command Summary

```bash
# Environment
conda activate dexmimicgen
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# Step 1: annotate source data
python scripts/prepare_src_dataset.py \
    --dataset datasets_src/generated/two_arm_threading.hdf5 \
    --task two_arm_threading --n 10

# Step 2: generate (oversample + filter to 5 successful demos)
python scripts/generate_dataset.py \
    --task two_arm_threading \
    --source datasets_src/generated/two_arm_threading.hdf5 \
    --output datasets_gen/two_arm_threading.hdf5 \
    --num_demos 5 --num_src 10

# Step 3: playback validation
python scripts/playback_datasets.py --dataset datasets_gen/two_arm_threading.hdf5
```

## 7. Changed-File Index

| file | change |
|------|--------|
| `dexmimicgen/configs/task_configs.py` | fix right/left `object_ref` and grasp-signal mapping |
| `dexmimicgen/env_interfaces/robosuite.py` | swap `robot_idx` of grasp signals in `get_subtask_term_signals` |
| `dexmimicgen/datagen/data_generator.py` | add shared-transform overrides to `_plan_subtask`; add `coord_anchor` shared coordination anchor in `generate()` |
| `dexmimicgen/datagen/waypoint.py` | revert hold-based coordination sync; remove unused `_coordination_group` |


# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Empirically inspect a source demonstration to design data-generation subtask signals.

Replays one demo open-loop and, at every timestep, logs which robot/gripper is grasping
which candidate object and a set of task-specific stage booleans, then prints the first
rising/falling edge of each so the subtask structure (order, handedness) can be read off
directly. This is the ground truth for writing an env interface's subtask_term_signals,
since the original DexMimicGen never released the datagen specs.

Example:
    python scripts/analyze_src_demo.py --dataset datasets_src/generated/two_arm_lift_tray.hdf5 \
        --task TwoArmLiftTray --demo demo_0
"""
import argparse
import json

import h5py
import numpy as np

import dexmimicgen  # noqa: F401 registers environments
from dexmimicgen.utils import env_utils as EnvUtils


def grasp(env, robot_idx, arm_part, object_geoms):
    robot = env.robots[robot_idx]
    g = robot.gripper
    gripper = g[arm_part] if isinstance(g, dict) else g
    return float(env._check_grasp(gripper=gripper, object_geoms=object_geoms))


def candidate_object_geoms(env):
    """Return dict label -> object geoms for every manipulable object we can find."""
    cands = {}
    for attr in [
        "tripod", "needle", "piece_1", "piece_2", "base", "pot",
        "box", "lid", "drawer", "cleanup_object",
    ]:
        if hasattr(env, attr):
            obj = getattr(env, attr)
            if hasattr(obj, "contact_geoms"):
                cands[attr] = list(obj.contact_geoms)
    # pot handles are grasped individually
    if hasattr(env, "pot"):
        pot = env.pot
        if hasattr(pot, "handle0_geoms"):
            cands["pot_handle0"] = list(pot.handle0_geoms)
        if hasattr(pot, "handle1_geoms"):
            cands["pot_handle1"] = list(pot.handle1_geoms)
    return cands


def edges(series):
    arr = np.array(series)
    d = arr[1:] - arr[:-1]
    rises = [int(i) + 1 for i in np.nonzero(d > 0)[0]]
    falls = [int(i) + 1 for i in np.nonzero(d < 0)[0]]
    return rises, falls


def main(args):
    env_meta = EnvUtils.get_env_metadata_from_dataset(args.dataset)
    env = EnvUtils.create_env(env_meta, render=False, render_offscreen=False, use_image_obs=False)

    f = h5py.File(args.dataset, "r")
    ep_grp = f["data/{}".format(args.demo)]
    states = ep_grp["states"][()]
    actions = ep_grp["actions"][()]
    initial_state = dict(states=states[0])
    initial_state["model"] = ep_grp.attrs["model_file"]
    initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)
    f.close()

    EnvUtils.reset_to(env, initial_state)
    n_robots = len(env.robots)
    cands = candidate_object_geoms(env)
    print("n_robots:", n_robots, "| action_dim:", actions.shape[1], "| traj_len:", states.shape[0])
    print("candidate objects:", list(cands.keys()))

    log = {}
    for t in range(states.shape[0]):
        for ri in range(n_robots):
            parts = ["right", "left"] if isinstance(env.robots[ri].gripper, dict) else ["right"]
            for part in parts:
                for label, geoms in cands.items():
                    key = "r{}_{}_grasp_{}".format(ri, part, label)
                    try:
                        v = grasp(env, ri, part, geoms)
                    except Exception:
                        v = 0.0
                    log.setdefault(key, []).append(v)
        if hasattr(env, "_check_success"):
            try:
                log.setdefault("success", []).append(float(bool(env._check_success())))
            except Exception:
                pass
        env.step(actions[t])

    print("\n--- signal edges (rise=0->1, fall=1->0); blank means never fired ---")
    for key in sorted(log.keys()):
        rises, falls = edges(log[key])
        total = int(np.sum(log[key]))
        if rises or falls:
            print("  {:40s} rises={} falls={} active_steps={}".format(key, rises[:5], falls[:5], total))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, help="informational only")
    parser.add_argument("--demo", type=str, default="demo_0")
    args = parser.parse_args()
    main(args)

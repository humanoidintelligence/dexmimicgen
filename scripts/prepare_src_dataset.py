# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Prepare a source dataset for DexMimicGen data generation.

This script replays each demonstration in a source hdf5 (open-loop, using the stored
actions) and records, at every timestep, the per-arm DatagenInfo (eef poses, controller
target poses, gripper actions), the relevant object poses, and the per-subtask
termination signals produced by the task's environment interface. These are written
back into the dataset under `data/<demo>/datagen_info`, which the DataGenerator consumes.

Example:
    python scripts/prepare_src_dataset.py \
        --dataset datasets_src/generated/two_arm_threading.hdf5 \
        --task two_arm_threading \
        --n 10
"""
import argparse
import json

import h5py
import numpy as np
from tqdm import tqdm

import dexmimicgen  # noqa: F401 registers environments
from dexmimicgen.configs.config import config_factory
from dexmimicgen.env_interfaces.base import make_interface
from dexmimicgen.utils import env_utils as EnvUtils
from dexmimicgen.utils import file_utils as FileUtils


def extract_datagen_info_from_trajectory(env, env_interface, initial_state, states, actions):
    """Replay a trajectory open-loop and collect a DatagenInfo per timestep."""
    EnvUtils.reset_to(env, initial_state)
    traj_len = states.shape[0]
    all_datagen_infos = []
    for t in range(traj_len):
        # datagen info reflects the state BEFORE applying the action at this step
        di = env_interface.get_datagen_info(action=actions[t])
        all_datagen_infos.append(di)
        env.step(actions[t])
    return all_datagen_infos


def write_datagen_info_to_group(ep_grp, all_datagen_infos, env_interface_name):
    # remove any existing datagen_info
    if "datagen_info" in ep_grp:
        del ep_grp["datagen_info"]

    di_dicts = [di.to_dict() for di in all_datagen_infos]
    nested_keys = ["eef_pose", "target_pose", "gripper_action", "object_poses", "subtask_term_signals"]
    for k in nested_keys:
        subkeys = list(di_dicts[0][k].keys())
        for sk in subkeys:
            arr = np.array([d[k][sk] for d in di_dicts])
            ep_grp.create_dataset("datagen_info/{}/{}".format(k, sk), data=arr)

    ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name


def main(args):
    config = config_factory(args.task)

    env_meta = EnvUtils.get_env_metadata_from_dataset(args.dataset)
    env = EnvUtils.create_env(env_meta, render=False, render_offscreen=False, use_image_obs=False)
    env_interface = make_interface(config.env_interface, env=env)
    print("Prepared env interface:", env_interface)

    demo_keys = FileUtils.get_all_demos_from_dataset(
        dataset_path=args.dataset, filter_key=args.filter_key, n=args.n
    )
    print("Annotating {} demos".format(len(demo_keys)))

    f = h5py.File(args.dataset, "a")
    for ep in tqdm(demo_keys):
        ep_grp = f["data/{}".format(ep)]
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = ep_grp.attrs["model_file"]
        initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

        all_datagen_infos = extract_datagen_info_from_trajectory(
            env=env,
            env_interface=env_interface,
            initial_state=initial_state,
            states=states,
            actions=actions,
        )
        write_datagen_info_to_group(ep_grp, all_datagen_infos, config.env_interface)
    f.close()
    print("Done. Wrote datagen_info into", args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to source hdf5 dataset")
    parser.add_argument("--task", type=str, required=True, help="registered task config name")
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="limit number of demos to annotate")
    args = parser.parse_args()
    main(args)

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
File utilities for the DexMimicGen data generation pipeline. Adapted from MimicGen
(NVlabs/mimicgen) for the bimanual (per-arm) datagen_info layout and raw-robosuite
environments.
"""
import json
import os
import shutil
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm

from dexmimicgen.datagen.datagen_info import DatagenInfo


def write_json(json_dic, json_path):
    with open(json_path, "w") as f:
        json.dump(json_dic, f, sort_keys=False, indent=4)


def get_all_demos_from_dataset(dataset_path, filter_key=None, start=None, n=None):
    f = h5py.File(dataset_path, "r")
    if filter_key is not None:
        demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
    else:
        demo_keys = list(f["data"].keys())
    # sort in increasing order of demo number
    inds = np.argsort([int(elem[5:]) for elem in demo_keys])
    demo_keys = [demo_keys[i] for i in inds]
    if start is not None:
        demo_keys = demo_keys[start:]
    if n is not None:
        demo_keys = demo_keys[:n]
    f.close()
    return demo_keys


def get_env_interface_info_from_dataset(dataset_path, demo_keys):
    f = h5py.File(dataset_path, "r")
    names = []
    for ep in demo_keys:
        key = "data/{}/datagen_info".format(ep)
        assert key in f, (
            "Could not find DexMimicGen metadata in dataset {}. Ensure you have run "
            "prepare_src_dataset.py on this hdf5".format(dataset_path)
        )
        names.append(f[key].attrs["env_interface_name"])
    f.close()
    name = names[0]
    assert all(elem == name for elem in names)
    return name


# arm-index -> arm-name mapping for the released (stacked) datagen_info layout.
# robosuite "single-arm-parallel" orders robots as [robot0, robot1]; the DexMimicGen
# interfaces map robot0->"right", robot1->"left".
_STACKED_ARM_ORDER = ["right", "left"]


def _load_datagen_info_from_group(ep_datagen_info):
    """
    Reconstruct a DatagenInfo from an hdf5 'datagen_info' group. Supports two layouts:
    - per-arm named: eef_pose/<arm>, target_pose/<arm>, gripper_action/<arm>
    - released stacked: eef_pose (T, n_arms, 4, 4), gripper_action (T, n_arms, 1)
      where arm index follows @_STACKED_ARM_ORDER.
    """

    def is_stacked(key):
        return isinstance(ep_datagen_info[key], h5py.Dataset)

    def load_per_arm(key, trailing_ndim):
        node = ep_datagen_info[key]
        if is_stacked(key):
            arr = node[:]  # (T, n_arms, ...)
            n_arms = arr.shape[1]
            return {_STACKED_ARM_ORDER[i]: arr[:, i] for i in range(n_arms)}
        return {k: node[k][:] for k in node}

    eef_pose = load_per_arm("eef_pose", 2)
    target_pose = load_per_arm("target_pose", 2)
    gripper_action = load_per_arm("gripper_action", 1)
    object_poses = {
        k: ep_datagen_info["object_poses"][k][:] for k in ep_datagen_info["object_poses"]
    }
    subtask_term_signals = {
        k: ep_datagen_info["subtask_term_signals"][k][:]
        for k in ep_datagen_info["subtask_term_signals"]
    }
    return DatagenInfo(
        eef_pose=eef_pose,
        object_poses=object_poses,
        subtask_term_signals=subtask_term_signals,
        target_pose=target_pose,
        gripper_action=gripper_action,
    )


def parse_source_dataset(dataset_path, demo_keys, task_spec):
    """
    Parse a source dataset into per-demo DatagenInfo objects and per-arm subtask indices.

    Returns:
        datagen_infos (list): one DatagenInfo per demo (per-arm layout)
        subtask_indices (dict): maps arm key -> np.array of shape (N, S_arm, 2)
    """
    arms = task_spec.arms

    # per-arm sequences of subtask termination signals / offsets
    arm_term_signals = {}
    arm_offset_ranges = {}
    for arm in arms:
        sigs = [s["subtask_term_signal"] for s in task_spec[arm]]
        offs = [s["subtask_term_offset_range"] for s in task_spec[arm]]
        assert sigs[-1] is None, "end of final subtask (arm {}) does not need detection".format(arm)
        offs[-1] = (0, 0)
        arm_term_signals[arm] = sigs
        arm_offset_ranges[arm] = offs

    f = h5py.File(dataset_path, "r")
    datagen_infos = []
    subtask_indices = {arm: [] for arm in arms}

    for ind in tqdm(range(len(demo_keys))):
        ep = demo_keys[ind]
        ep_grp = f["data/{}".format(ep)]
        di = _load_datagen_info_from_group(ep_grp["datagen_info"])
        datagen_infos.append(di)

        traj_len = ep_grp["actions"].shape[0]

        for arm in arms:
            sigs = arm_term_signals[arm]
            offs = arm_offset_ranges[arm]
            ep_subtask_indices = []
            prev = 0
            for subtask_ind in range(len(sigs)):
                sig = sigs[subtask_ind]
                if sig is None:
                    end = traj_len
                else:
                    indicators = di.subtask_term_signals[sig]
                    diffs = indicators[1:] - indicators[:-1]
                    end_ind = next(
                        (int(nz) + 1 for nz in diffs.nonzero()[0] if int(nz) + 1 >= prev),
                        None,
                    )
                    assert end_ind is not None, (
                        "No valid subtask termination for {} (arm {}) in demo {}".format(sig, arm, ep)
                    )
                    end = end_ind + 1
                ep_subtask_indices.append([prev, end])
                prev = end

            # sanity check on offset ranges to avoid empty subtasks
            for i in range(1, len(ep_subtask_indices)):
                prev_max = offs[i - 1][1]
                assert (
                    ep_subtask_indices[i - 1][1] + prev_max
                    < ep_subtask_indices[i][1] + offs[i][0]
                ), "subtask sanity check violation in demo {} arm {} subtask {}".format(ep, arm, i)

            subtask_indices[arm].append(ep_subtask_indices)

    f.close()
    subtask_indices = {arm: np.array(subtask_indices[arm]) for arm in arms}
    return datagen_infos, subtask_indices


def _list_of_flat_dict_to_dict_of_list(list_of_dict):
    """Convert a list of (flat) dicts to a dict of lists, keyed by the dict keys."""
    assert len(list_of_dict) > 0
    keys = list(list_of_dict[0].keys())
    out = {k: [d[k] for d in list_of_dict] for k in keys}
    return out


def write_demo_to_hdf5(
    folder,
    env_args,
    initial_state,
    states,
    observations,
    datagen_info,
    actions,
    src_demo_inds=None,
    src_demo_labels=None,
):
    """
    Write a single demonstration to an hdf5 in @folder. Each file holds one demo under
    data/demo_0, matching the layout expected by merge_all_hdf5.
    """
    os.makedirs(folder, exist_ok=True)
    num_existing = len(glob(os.path.join(folder, "*.hdf5")))
    new_path = os.path.join(folder, "{}.hdf5".format(num_existing))
    writer = h5py.File(new_path, "w")
    data_grp = writer.create_group("data")
    ep_grp = data_grp.create_group("demo_0")

    # states
    ep_grp.create_dataset("states", data=np.array(states))

    # observations
    if observations is not None and len(observations) > 0 and observations[0] is not None:
        obs = _list_of_flat_dict_to_dict_of_list(observations)
        for k in obs:
            ep_grp.create_dataset("obs/{}".format(k), data=np.array(obs[k]), compression="gzip")

    # datagen info (per-arm nested dicts for eef/target/gripper, plus object_poses + signals)
    di = _list_of_flat_dict_to_dict_of_list([x.to_dict() for x in datagen_info])
    for k in di:
        if k in ["object_poses", "subtask_term_signals", "eef_pose", "target_pose", "gripper_action"]:
            nested = _list_of_flat_dict_to_dict_of_list(di[k])
            for k2 in nested:
                ep_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(nested[k2]))
        else:
            ep_grp.create_dataset("datagen_info/{}".format(k), data=np.array(di[k]))

    if src_demo_inds is not None:
        ep_grp.create_dataset("src_demo_inds", data=np.array(src_demo_inds))
    if src_demo_labels is not None:
        ep_grp.create_dataset("src_demo_labels", data=np.array(src_demo_labels))

    if ("model" in initial_state) and (initial_state["model"] is not None):
        ep_grp.attrs["model_file"] = initial_state["model"]
    if initial_state.get("ep_meta", None) is not None:
        ep_grp.attrs["ep_meta"] = initial_state["ep_meta"]
    ep_grp.attrs["num_samples"] = actions.shape[0]

    data_grp.attrs["total"] = actions.shape[0]
    data_grp.attrs["env_args"] = env_args
    # actions written last so num_samples/total are consistent
    ep_grp.create_dataset("actions", data=np.array(actions))
    writer.close()
    return new_path


def merge_all_hdf5(folder, new_hdf5_path, delete_folder=False, env_args=None):
    """
    Merge all per-episode hdf5 files in @folder into a single robomimic-style dataset.
    """
    source_hdf5s = sorted(glob(os.path.join(folder, "*.hdf5")), key=lambda p: int(os.path.basename(p).split(".")[0]))

    f_new = h5py.File(new_hdf5_path, "w")
    grp = f_new.create_group("data")
    total = 0
    env_meta_str = env_args
    for i, src in enumerate(source_hdf5s):
        with h5py.File(src, "r") as f:
            demo_str = "demo_{}".format(i)
            f.copy("data/demo_0", grp, name=demo_str)
            total += f["data"].attrs["total"]
            if env_meta_str is None:
                env_meta_str = f["data"].attrs["env_args"]
    grp.attrs["total"] = total
    grp.attrs["env_args"] = env_meta_str if env_meta_str is not None else ""
    f_new.close()

    if delete_folder:
        shutil.rmtree(folder)
    return len(source_hdf5s)

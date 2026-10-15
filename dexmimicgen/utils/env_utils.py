# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Environment helpers shared by the DexMimicGen data generation scripts: creating a
robosuite environment from stored env metadata, and resetting to a stored sim state.
These mirror the logic in scripts/playback_datasets.py so that generated datasets are
format-compatible with the released DexMimicGen datasets.
"""
import json

import numpy as np
import robosuite

import dexmimicgen  # noqa: F401  (registers environments)


def get_env_metadata_from_dataset(dataset_path):
    import h5py

    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta


def create_env(env_meta, render=False, render_offscreen=False, use_image_obs=False):
    """
    Create a robosuite environment from stored env metadata (the "env_args" attr).
    """
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = render
    env_kwargs["has_offscreen_renderer"] = render_offscreen
    env_kwargs["use_camera_obs"] = use_image_obs
    env_kwargs.pop("env_lang", None)
    if render:
        env_kwargs["renderer"] = "mjviewer"
    env = robosuite.make(**env_kwargs)
    return env


def reset_to(env, state):
    """
    Reset to a specific simulator state. @state may contain "model" (xml), "ep_meta",
    and/or "states" (flattened mujoco state). Ported from scripts/playback_datasets.py.
    """
    if "model" in state and state["model"] is not None:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state and state["states"] is not None:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return None


def get_env_args_string(env):
    """
    Build the robomimic-style "env_args" JSON string for a raw robosuite env, matching the
    format used by the released DexMimicGen datasets.
    """
    env_kwargs = dict(getattr(env, "_input_kwargs", {}) or {})
    # fall back to reconstructing from robosuite private attributes if needed
    env_name = type(env).__name__
    env_args = dict(
        env_name=env_name,
        env_version=robosuite.__version__,
        type=1,  # robosuite type in robomimic EnvType
        env_kwargs=env_kwargs,
    )
    return json.dumps(env_args, indent=4)

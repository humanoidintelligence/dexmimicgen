# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Generate a new bimanual dataset with DexMimicGen.

Given a source dataset that has been annotated with `prepare_src_dataset.py`, this
script repeatedly resets the environment to new initial states and uses the
DataGenerator to synthesize new demonstrations by transforming + replaying the
source subtask segments for each arm under the configured coordination strategy.

Example:
    python scripts/generate_dataset.py \
        --task two_arm_threading \
        --source datasets_src/generated/two_arm_threading.hdf5 \
        --output datasets_gen/two_arm_threading.hdf5 \
        --num_demos 10 --num_src 10
"""
import argparse
import json
import os
import tempfile

import h5py
import imageio
import numpy as np

import dexmimicgen  # noqa: F401 registers environments
from dexmimicgen.configs.config import config_factory
from dexmimicgen.datagen.data_generator import DataGenerator
from dexmimicgen.env_interfaces.base import make_interface
from dexmimicgen.utils import env_utils as EnvUtils
from dexmimicgen.utils import file_utils as FileUtils


def main(args):
    config = config_factory(args.task)
    np.random.seed(args.seed)

    # pull env_args verbatim from the source dataset so the generated dataset is
    # format-compatible (same env_name / env_kwargs / version)
    with h5py.File(args.source, "r") as f:
        env_args_str = f["data"].attrs["env_args"]
    env_meta = json.loads(env_args_str)

    env = EnvUtils.create_env(
        env_meta,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        use_image_obs=False,
    )
    env_interface = make_interface(config.env_interface, env=env)
    print("Env interface:", env_interface)

    demo_keys = FileUtils.get_all_demos_from_dataset(
        dataset_path=args.source, filter_key=args.filter_key, n=args.num_src
    )
    data_generator = DataGenerator(
        task_spec=config.task_spec(),
        dataset_path=args.source,
        demo_keys=demo_keys,
    )
    print(data_generator)

    video_writer = None
    if args.video_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)

    tmp_folder = tempfile.mkdtemp(prefix="dexmg_gen_")
    num_success = 0
    num_attempts = 0

    while num_success < args.num_demos:
        if num_attempts >= args.num_demos * config.max_num_attempts_per_demo:
            print("Reached attempt budget; stopping.")
            break
        num_attempts += 1

        results = data_generator.generate(
            env=env,
            env_interface=env_interface,
            select_src_per_subtask=config.select_src_per_subtask,
            transform_first_robot_pose=config.transform_first_robot_pose,
            interpolate_from_last_target_pose=config.interpolate_from_last_target_pose,
            render=args.render,
            video_writer=video_writer if args.video_all else None,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
        )

        success = results["success"]
        print("attempt {}: success={} (len={})".format(num_attempts, success, len(results["actions"])))

        if success or (not config.guarantee):
            if success and video_writer is not None and not args.video_all:
                _write_video_from_states(env, results, video_writer, args)
            FileUtils.write_demo_to_hdf5(
                folder=tmp_folder,
                env_args=env_args_str,
                initial_state=results["initial_state"],
                states=results["states"],
                observations=results["observations"],
                datagen_info=results["datagen_infos"],
                actions=results["actions"],
                src_demo_inds=results["src_demo_inds"],
                src_demo_labels=results["src_demo_labels"],
            )
            if success:
                num_success += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_written = FileUtils.merge_all_hdf5(
        folder=tmp_folder,
        new_hdf5_path=args.output,
        delete_folder=True,
        env_args=env_args_str,
    )
    if video_writer is not None:
        video_writer.close()

    print(
        "\nDone. {} / {} attempts succeeded. Wrote {} demos to {}".format(
            num_success, num_attempts, n_written, args.output
        )
    )
    env.close()


def _write_video_from_states(env, results, video_writer, args):
    """Render a successful demo to video by replaying its recorded states."""
    EnvUtils.reset_to(env, results["initial_state"])
    for t in range(len(results["states"])):
        EnvUtils.reset_to(env, {"states": results["states"][t]})
        if t % args.video_skip == 0:
            imgs = [env.sim.render(height=512, width=512, camera_name=c)[::-1] for c in args.camera_names]
            video_writer.append_data(np.concatenate(imgs, axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="registered task config name")
    parser.add_argument("--source", type=str, required=True, help="path to prepared source hdf5")
    parser.add_argument("--output", type=str, required=True, help="path to output hdf5")
    parser.add_argument("--num_demos", type=int, default=10, help="number of successful demos to generate")
    parser.add_argument("--num_src", type=int, default=None, help="number of source demos to use")
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_all", action="store_true", help="record every attempt, not just successes")
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--camera_names", type=str, nargs="+", default=["agentview"])
    args = parser.parse_args()
    main(args)

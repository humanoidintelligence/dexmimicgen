# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
DexMimicGen data generator. Loads a source dataset parsed into per-arm subtask
segments and generates new bimanual demonstrations by transforming each arm's
source segments to the new scene's object poses and executing them with the
MultiArmExecutor (which realizes the parallel / coordination / sequential
coordination strategies).

Adapted from MimicGen (NVlabs/mimicgen).
"""
import numpy as np

import dexmimicgen.utils.file_utils as DexMG_FileUtils
import dexmimicgen.utils.pose_utils as PoseUtils
from dexmimicgen.datagen.datagen_info import DatagenInfo
from dexmimicgen.datagen.selection_strategy import make_selection_strategy
from dexmimicgen.datagen.waypoint import (
    COORDINATION,
    ArmSubtaskPlan,
    MultiArmExecutor,
    WaypointSequence,
    WaypointTrajectory,
)


class DataGenerator(object):
    def __init__(self, task_spec, dataset_path, demo_keys):
        self.task_spec = task_spec
        self.task_spec.sanity_check()
        self.dataset_path = dataset_path
        self.demo_keys = demo_keys
        self.arms = task_spec.arms

        print("\nDataGenerator: loading dataset at path {}...".format(dataset_path))
        self.src_dataset_infos, self.src_subtask_indices = DexMG_FileUtils.parse_source_dataset(
            dataset_path=dataset_path,
            demo_keys=demo_keys,
            task_spec=task_spec,
        )
        print("DataGenerator: done loading\n")

    def __repr__(self):
        return "{}(\n  dataset_path={}\n  demo_keys={}\n  arms={}\n)".format(
            self.__class__.__name__, self.dataset_path, len(self.demo_keys), self.arms
        )

    def randomize_subtask_boundaries(self, arm):
        """
        Sample per-demo subtask boundaries for a given arm, applying random offsets to the
        subtask end indices as specified by the task spec.
        """
        src_subtask_indices = np.array(self.src_subtask_indices[arm])
        spec = self.task_spec[arm]
        for i in range(src_subtask_indices.shape[1] - 1):
            end_offsets = np.random.randint(
                low=spec[i]["subtask_term_offset_range"][0],
                high=spec[i]["subtask_term_offset_range"][1] + 1,
                size=src_subtask_indices.shape[0],
            )
            src_subtask_indices[:, i, 1] = src_subtask_indices[:, i, 1] + end_offsets
            src_subtask_indices[:, i + 1, 0] = src_subtask_indices[:, i, 1]

        assert np.all((src_subtask_indices[:, :, 1] - src_subtask_indices[:, :, 0]) > 0), "empty subtasks!"
        return src_subtask_indices

    def select_source_demo(
        self,
        arm,
        eef_pose,
        object_pose,
        subtask_ind,
        src_subtask_inds,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
    ):
        if subtask_object_name is None:
            assert selection_strategy_name == "random"

        src_subtask_datagen_infos = []
        for i in range(len(self.demo_keys)):
            src_ep = self.src_dataset_infos[i]
            s, e = src_subtask_inds[i][0], src_subtask_inds[i][1]
            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=({arm: src_ep.eef_pose[arm][s:e]}),
                    object_poses=(
                        {subtask_object_name: src_ep.object_poses[subtask_object_name][s:e]}
                        if subtask_object_name is not None
                        else None
                    ),
                    subtask_term_signals=None,
                    target_pose={arm: src_ep.target_pose[arm][s:e]},
                    gripper_action={arm: src_ep.gripper_action[arm][s:e]},
                )
            )

        selection_strategy = make_selection_strategy(selection_strategy_name)
        kwargs = selection_strategy_kwargs if selection_strategy_kwargs is not None else {}
        # the strategy expects a single-arm eef / target pose, pass through per-arm views
        selected = selection_strategy.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **kwargs,
        )
        return selected

    def _plan_subtask(
        self,
        arm,
        subtask_ind,
        env_interface,
        all_subtask_inds,
        selected_src_demo_ind,
        select_src_per_subtask,
        transform_first_robot_pose,
        interpolate_from_last_target_pose,
        prev_last_waypoint,
        cur_object_pose_override=None,
        src_object_pose_override=None,
    ):
        """
        Plan a SINGLE subtask (stage) for one arm, transforming the source segment to the
        object pose observed *right now* in the current scene. Planning must be interleaved
        with execution (one subtask at a time), because manipulated objects move during
        execution: planning a later subtask against the initial object pose would transform
        the segment to a stale frame. This mirrors MimicGen's per-subtask re-observation.

        For COORDINATION subtasks both arms must be transformed by the *same* SE(3) matrix so
        that the relative pose between the two end-effectors is preserved (DexMimicGen paper,
        Sec. 4.2). The caller enforces this by passing a shared ``cur_object_pose_override``
        (the reference-object pose observed when the first arm entered the coordination
        subtask) and ``src_object_pose_override`` (the matching source-object pose), so that
        T = cur_obj @ src_obj^{-1} is identical for both arms.

        Returns (ArmSubtaskPlan, selected_src_demo_ind).
        """
        spec = self.task_spec[arm]
        is_first_subtask = (subtask_ind == 0)
        subtask_object_name = spec[subtask_ind]["object_ref"]

        # (re)select source demo for this subtask using the CURRENT observation
        if is_first_subtask or select_src_per_subtask:
            selected_src_demo_ind = self.select_source_demo(
                arm=arm,
                eef_pose=env_interface.get_robot_eef_pose(arm),
                object_pose=(
                    env_interface.get_object_poses()[subtask_object_name]
                    if subtask_object_name is not None
                    else None
                ),
                subtask_ind=subtask_ind,
                src_subtask_inds=all_subtask_inds[:, subtask_ind],
                subtask_object_name=subtask_object_name,
                selection_strategy_name=spec[subtask_ind]["selection_strategy"],
                selection_strategy_kwargs=spec[subtask_ind]["selection_strategy_kwargs"],
            )

        selected_inds = all_subtask_inds[selected_src_demo_ind, subtask_ind]
        src_ep = self.src_dataset_infos[selected_src_demo_ind]
        s, e = selected_inds[0], selected_inds[1]
        src_eef_poses = src_ep.eef_pose[arm][s:e]
        src_target_poses = src_ep.target_pose[arm][s:e]
        src_gripper_actions = src_ep.gripper_action[arm][s:e]
        src_object_pose = (
            src_ep.object_poses[subtask_object_name][s] if subtask_object_name is not None else None
        )
        if src_object_pose_override is not None:
            src_object_pose = src_object_pose_override

        if is_first_subtask or transform_first_robot_pose:
            src_poses = np.concatenate([src_eef_poses[0:1], src_target_poses], axis=0)
            # account for the extra (prepended) eef timestep so poses and gripper
            # actions stay the same length and remain index-aligned (pose[i] <-> gripper[i]).
            src_gripper_actions = np.concatenate(
                [src_gripper_actions[0:1], src_gripper_actions], axis=0
            )
        else:
            # no extra pose timestep is prepended here, so the gripper actions must NOT be
            # prepended either -- otherwise gripper[i] lags pose[i] by one step, which
            # misfires the gripper exactly at a subtask (e.g. grasp) boundary.
            src_poses = np.array(src_target_poses)

        # transform source segment to the object frame observed right now
        if subtask_object_name is not None:
            if cur_object_pose_override is not None:
                cur_object_pose = cur_object_pose_override
            else:
                cur_object_pose = env_interface.get_object_poses()[subtask_object_name]
            transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                obj_pose=cur_object_pose,
                src_eef_poses=src_poses,
                src_obj_pose=src_object_pose,
            )
        else:
            transformed_eef_poses = src_poses

        traj_to_execute = WaypointTrajectory()
        if interpolate_from_last_target_pose and (not is_first_subtask) and (prev_last_waypoint is not None):
            init_sequence = WaypointSequence(sequence=[prev_last_waypoint])
        else:
            init_sequence = WaypointSequence.from_poses(
                poses=env_interface.get_robot_eef_pose(arm)[None],
                gripper_actions=src_gripper_actions[0:1],
                action_noise=spec[subtask_ind]["action_noise"],
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_gripper_actions,
            action_noise=spec[subtask_ind]["action_noise"],
        )
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)

        traj_to_execute.merge(
            transformed_traj,
            num_steps_interp=spec[subtask_ind]["num_interpolation_steps"],
            num_steps_fixed=spec[subtask_ind]["num_fixed_steps"],
            action_noise=(
                float(spec[subtask_ind]["apply_noise_during_interpolation"])
                * spec[subtask_ind]["action_noise"]
            ),
        )
        # discard the seed waypoint used to drive interpolation
        traj_to_execute.pop_first()

        plan = ArmSubtaskPlan(
            waypoints=traj_to_execute.to_waypoint_list(),
            subtask_type=spec[subtask_ind]["subtask_type"],
            subtask_term_signal=spec[subtask_ind]["subtask_term_signal"],
            wait_for=spec[subtask_ind]["wait_for"],
        )
        return plan, selected_src_demo_ind

    def generate(
        self,
        env,
        env_interface,
        select_src_per_subtask=False,
        transform_first_robot_pose=False,
        interpolate_from_last_target_pose=True,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
    ):
        """Attempt to generate a new bimanual demonstration.

        Planning and execution are interleaved one stage (subtask) at a time: at each stage
        both arms plan their subtask against the object poses observed in the current scene,
        the stage is executed with the MultiArmExecutor (realizing parallel / coordination
        sync between arms), and only then is the next stage planned -- using freshly observed
        object poses. This is required because manipulated objects move during execution.
        """
        env.reset()
        new_initial_state = env.get_state()

        # sample new subtask boundaries per arm
        all_subtask_inds = {arm: self.randomize_subtask_boundaries(arm) for arm in self.arms}

        # Lazy, interleaved planning + execution in a SINGLE continuous pass. Each arm plans
        # its next subtask on demand -- at the moment it actually reaches that subtask -- so
        # object-frame transforms always use freshly observed object poses (objects move a lot
        # during execution). Running one continuous executor pass (rather than one pass per
        # stage) also means an arm flows directly from one subtask into the next without idling
        # its grasp against the shared scene, which previously perturbed objects and broke the
        # downstream transforms.
        src_demo_inds = {arm: None for arm in self.arms}
        # Shared coordination anchors, keyed by subtask index. The FIRST arm to enter a given
        # coordination subtask records (cur_obj, src_obj) for the reference object; every arm
        # in that coordination subtask then transforms with the SAME T = cur_obj @ src_obj^-1,
        # which preserves the relative pose between the two end-effectors (paper Sec. 4.2).
        coord_anchor = {}

        def plan_fn(arm, subtask_ind, prev_last_waypoint):
            spec = self.task_spec[arm][subtask_ind]
            cur_override = None
            src_override = None
            if spec["subtask_type"] == COORDINATION and spec["object_ref"] is not None:
                obj_name = spec["object_ref"]
                if subtask_ind not in coord_anchor:
                    # first arm into this coordination subtask: fix the shared anchor. Both the
                    # live object pose (observed now) and the source object pose (this arm's
                    # coordination-segment start) are frozen so T is identical for both arms.
                    # The source demo is pinned too: preserving the inter-arm relative pose only
                    # makes sense when both arms draw their coordination segment from the SAME
                    # source demonstration.
                    sel = src_demo_inds[arm]
                    s0 = all_subtask_inds[arm][sel, subtask_ind][0]
                    coord_anchor[subtask_ind] = dict(
                        src_demo=sel,
                        cur_obj=env_interface.get_object_poses()[obj_name].copy(),
                        src_obj=self.src_dataset_infos[sel].object_poses[obj_name][s0].copy(),
                    )
                # pin every arm in this coordination subtask to the shared source demo
                src_demo_inds[arm] = coord_anchor[subtask_ind]["src_demo"]
                cur_override = coord_anchor[subtask_ind]["cur_obj"]
                src_override = coord_anchor[subtask_ind]["src_obj"]

            plan, sel = self._plan_subtask(
                arm=arm,
                subtask_ind=subtask_ind,
                env_interface=env_interface,
                all_subtask_inds=all_subtask_inds[arm],
                selected_src_demo_ind=src_demo_inds[arm],
                select_src_per_subtask=select_src_per_subtask,
                transform_first_robot_pose=transform_first_robot_pose,
                interpolate_from_last_target_pose=interpolate_from_last_target_pose,
                prev_last_waypoint=prev_last_waypoint,
                cur_object_pose_override=cur_override,
                src_object_pose_override=src_override,
            )
            src_demo_inds[arm] = sel
            return plan

        executor = MultiArmExecutor(
            env=env,
            env_interface=env_interface,
            plan_fn=plan_fn,
            n_subtasks={arm: len(self.task_spec[arm]) for arm in self.arms},
        )
        exec_results = executor.execute(
            render=render,
            video_writer=video_writer,
            video_skip=video_skip,
            camera_names=camera_names,
        )

        actions = np.asarray(exec_results["actions"])
        if actions.shape[0] == 0:
            actions = np.zeros((0, env_interface.env_action_dim))

        results = dict(
            initial_state=new_initial_state,
            states=exec_results["states"],
            observations=exec_results["observations"],
            datagen_infos=exec_results["datagen_infos"],
            actions=actions,
            success=bool(exec_results["success"]),
            src_demo_inds=[src_demo_inds[arm] for arm in self.arms],
            src_demo_labels=np.array(
                [[src_demo_inds[self.arms[0]]]] * actions.shape[0], dtype=int
            )
            if actions.shape[0] > 0
            else np.zeros((0, 1), dtype=int),
        )
        return results

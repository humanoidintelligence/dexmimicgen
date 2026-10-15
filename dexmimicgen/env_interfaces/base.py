# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base class for DexMimicGen environment interfaces. Adapted from MimicGen
(NVlabs/mimicgen) but generalized to the bimanual setting: eef poses, target
poses and gripper/hand actions are tracked per logical arm.

A "logical arm" is identified by a string key (e.g. "right", "left"). Each arm
maps to a specific robosuite robot index, a controller part name, and an eef
site. This indirection lets a single interface describe both the two-robot
layouts (Panda x2, PandaDex x2) and the single-robot humanoid layout (GR1).
"""
import abc

import numpy as np
import robosuite.utils.transform_utils as T

import dexmimicgen.utils.pose_utils as PoseUtils
from dexmimicgen.datagen.datagen_info import DatagenInfo

REGISTERED_ENV_INTERFACES = {}


def make_interface(name, *args, **kwargs):
    if name not in REGISTERED_ENV_INTERFACES:
        raise Exception(
            "make_interface: interface name {} not found. Registered: {}".format(
                name, ", ".join(REGISTERED_ENV_INTERFACES)
            )
        )
    return REGISTERED_ENV_INTERFACES[name](*args, **kwargs)


def register_env_interface(cls):
    ignore_classes = ["DexMG_EnvInterface", "RobosuiteInterface"]
    if cls.__name__ not in ignore_classes:
        REGISTERED_ENV_INTERFACES[cls.__name__] = cls


class DexMG_EnvInterfaceMeta(type):
    def __new__(meta, name, bases, class_dict):
        cls = super(DexMG_EnvInterfaceMeta, meta).__new__(meta, name, bases, class_dict)
        register_env_interface(cls)
        return cls


class ArmSpec(object):
    """
    Describes how one logical arm maps onto robosuite control structures.

    Args:
        robot_idx (int): index into env.robots
        part_name (str): robosuite controller part name, e.g. "right" or "left"
        eef_site (str): mujoco site name for this arm's end effector
        gripper_dim (int): dimension of the gripper/hand actuation for this arm
    """

    def __init__(self, robot_idx, part_name, eef_site, gripper_dim):
        self.robot_idx = robot_idx
        self.part_name = part_name
        self.eef_site = eef_site
        self.gripper_dim = gripper_dim


class DexMG_EnvInterface(object, metaclass=DexMG_EnvInterfaceMeta):
    """
    Environment interface API that DexMimicGen environment interfaces conform to.
    """

    def __init__(self, env):
        self.env = env
        # subclasses must populate self.arm_specs: dict(arm_key -> ArmSpec)
        self.arm_specs = self._get_arm_specs()
        self.arms = list(self.arm_specs.keys())

    def __repr__(self):
        return "{}(arms={})".format(self.__class__.__name__, self.arms)

    # ------------------------------------------------------------------
    # methods subclasses must implement
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _get_arm_specs(self):
        """Returns dict mapping arm key -> ArmSpec."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_object_poses(self):
        """Returns dict mapping object name -> 4x4 pose."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_subtask_term_signals(self):
        """Returns dict mapping subtask signal name -> {0, 1}."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # robosuite helpers
    # ------------------------------------------------------------------
    def get_object_pose(self, obj_name, obj_type):
        assert obj_type in ["body", "geom", "site"]
        if obj_type == "body":
            obj_id = self.env.sim.model.body_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.body_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.body_xmat[obj_id].reshape(3, 3))
        elif obj_type == "geom":
            obj_id = self.env.sim.model.geom_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.geom_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.geom_xmat[obj_id].reshape(3, 3))
        else:
            obj_id = self.env.sim.model.site_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.site_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.site_xmat[obj_id].reshape(3, 3))
        return PoseUtils.make_pose(obj_pos, obj_rot)

    def _controller(self, arm_key):
        spec = self.arm_specs[arm_key]
        robot = self.env.robots[spec.robot_idx]
        return robot.composite_controller.part_controllers[spec.part_name]

    def get_robot_eef_pose(self, arm_key):
        """Current eef pose (4x4) for a logical arm, in the OSC control site frame."""
        spec = self.arm_specs[arm_key]
        return self.get_object_pose(obj_name=spec.eef_site, obj_type="site")

    def get_all_eef_poses(self):
        return {arm: self.get_robot_eef_pose(arm) for arm in self.arms}

    # ------------------------------------------------------------------
    # pose <-> action conversion (OSC convention; GR1 overrides)
    # ------------------------------------------------------------------
    def target_pose_to_action(self, arm_key, target_pose, relative=True):
        """
        Convert a 4x4 target eef pose into the arm action (minus gripper) using the
        OSC delta-pose convention.
        """
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
        curr_pose = self.get_robot_eef_pose(arm_key)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        controller = self._controller(arm_key)
        max_dpos = controller.output_max[0]
        max_drot = controller.output_max[3]

        if relative:
            delta_position = target_pos - curr_pos
            delta_position = np.clip(delta_position / max_dpos, -1.0, 1.0)

            delta_rot_mat = target_rot.dot(curr_rot.T)
            delta_quat = T.mat2quat(delta_rot_mat)
            delta_rotation = T.quat2axisangle(delta_quat)
            delta_rotation = np.clip(delta_rotation / max_drot, -1.0, 1.0)
            return np.concatenate([delta_position, delta_rotation])

        target_quat = T.mat2quat(target_rot)
        abs_rotation = T.quat2axisangle(target_quat)
        return np.concatenate([target_pos, abs_rotation])

    def action_to_target_pose(self, arm_key, action, relative=True):
        """
        Inverse of target_pose_to_action for a single arm. @action is the arm (non-gripper)
        action slice for this arm.
        """
        if not relative:
            target_pos = action[:3]
            target_quat = T.axisangle2quat(action[3:6])
            target_rot = T.quat2mat(target_quat)
        else:
            controller = self._controller(arm_key)
            max_dpos = controller.output_max[0]
            max_drot = controller.output_max[3]

            delta_position = action[:3] * max_dpos
            delta_rotation = action[3:6] * max_drot

            curr_pose = self.get_robot_eef_pose(arm_key)
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            target_pos = curr_pos + delta_position
            delta_quat = T.axisangle2quat(delta_rotation)
            delta_rot_mat = T.quat2mat(delta_quat)
            target_rot = delta_rot_mat.dot(curr_rot)

        return PoseUtils.make_pose(target_pos, target_rot)

    # number of pose-action dims per arm (OSC: 6). GR1 overrides.
    ARM_ACTION_DIM = 6

    # ------------------------------------------------------------------
    # per-arm action slice bookkeeping within the full env action vector
    # ------------------------------------------------------------------
    def _arm_action_slices(self, arm_key):
        """
        Returns (arm_slice, gripper_slice) as (start, end) tuples into the FULL env action
        vector for this logical arm. Uses robosuite's per-robot action split indices,
        offset by the cumulative action dim of preceding robots.
        """
        spec = self.arm_specs[arm_key]
        offset = 0
        for i in range(spec.robot_idx):
            offset += self.env.robots[i].action_dim
        robot = self.env.robots[spec.robot_idx]
        split = robot._action_split_indexes
        arm_start, arm_end = split[spec.part_name]
        grip_key = "{}_gripper".format(spec.part_name)
        if grip_key in split:
            grip_start, grip_end = split[grip_key]
        else:
            grip_start, grip_end = arm_end, arm_end
        return (
            (offset + arm_start, offset + arm_end),
            (offset + grip_start, offset + grip_end),
        )

    @property
    def env_action_dim(self):
        return int(sum(r.action_dim for r in self.env.robots))

    def compose_action(self, step_waypoints):
        """
        Build the full env action vector from a dict mapping arm_key -> Waypoint (or None).

        For each arm, the waypoint pose is converted to an OSC delta-pose action and placed
        in the arm's slice; the waypoint gripper_action is placed in the gripper slice.
        """
        action = np.zeros(self.env_action_dim)
        for arm_key in self.arms:
            wp = step_waypoints.get(arm_key, None)
            if wp is None:
                continue
            arm_sl, grip_sl = self._arm_action_slices(arm_key)
            arm_action = self.target_pose_to_action(arm_key, wp.pose, relative=True)
            action[arm_sl[0] : arm_sl[1]] = arm_action
            if grip_sl[1] > grip_sl[0]:
                action[grip_sl[0] : grip_sl[1]] = wp.gripper_action
        return action

    def action_to_gripper_action(self, arm_key, action):
        """Extract the gripper slice for this arm from a full env action vector."""
        _, grip_sl = self._arm_action_slices(arm_key)
        return action[grip_sl[0] : grip_sl[1]]

    def action_arm_slice(self, arm_key, action):
        """Extract the arm (non-gripper) slice for this arm from a full env action vector."""
        arm_sl, _ = self._arm_action_slices(arm_key)
        return action[arm_sl[0] : arm_sl[1]]

    def get_datagen_info(self, action=None):
        """
        Build a DatagenInfo for the current timestep. If @action (a full env action vector)
        is provided, target poses and gripper actions are inferred from it.
        """
        eef_pose = self.get_all_eef_poses()
        object_poses = self.get_object_poses()
        subtask_term_signals = self.get_subtask_term_signals()

        target_pose = None
        gripper_action = None
        if action is not None:
            target_pose = {}
            gripper_action = {}
            for arm_key in self.arms:
                arm_a = self.action_arm_slice(arm_key, action)
                target_pose[arm_key] = self.action_to_target_pose(
                    arm_key, arm_a, relative=True
                )
                gripper_action[arm_key] = self.action_to_gripper_action(arm_key, action)

        return DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses,
            subtask_term_signals=subtask_term_signals,
            target_pose=target_pose,
            gripper_action=gripper_action,
        )

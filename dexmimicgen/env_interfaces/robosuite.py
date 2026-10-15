# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Robosuite environment interfaces for DexMimicGen tasks. Each interface provides the
per-arm end-effector mapping, the object poses relevant to data generation, and the
per-subtask termination signals used to segment source demonstrations.
"""
import numpy as np

from dexmimicgen.env_interfaces.base import ArmSpec, DexMG_EnvInterface


class RobosuiteInterface(DexMG_EnvInterface):
    """Shared helpers for robosuite-based interfaces."""

    def _grasp(self, robot_idx, arm, object_geoms):
        """Return 1.0 if the given robot arm's gripper is grasping @object_geoms, else 0.0."""
        robot = self.env.robots[robot_idx]
        gripper = robot.gripper[arm] if isinstance(robot.gripper, dict) else robot.gripper
        return float(self.env._check_grasp(gripper=gripper, object_geoms=object_geoms))


class TwoArmThreadingInterface(RobosuiteInterface):
    """
    Threading: the right arm (robot 0) manipulates the tripod; the left arm (robot 1)
    manipulates the needle. (In robosuite TwoArmThreading the needle is sampled at +y and
    the tripod at -y, and robot 0 holds the tripod while robot 1 threads the needle --
    verified against the source demonstrations.) Threading is a coordination task: the final
    insertion is executed with both arms synchronized.
    """

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=1),
            "left": ArmSpec(robot_idx=1, part_name="right", eef_site="gripper1_right_grip_site", gripper_dim=1),
        }

    def get_object_poses(self):
        return dict(
            needle=self.get_object_pose(obj_name=self.env.needle.root_body, obj_type="body"),
            tripod=self.get_object_pose(obj_name=self.env.tripod.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # right arm (robot 0) grasps the tripod; left arm (robot 1) grasps the needle
        signals["grasp_tripod"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self.env.tripod.contact_geoms
        )
        signals["grasp_needle"] = self._grasp(
            robot_idx=1, arm="right", object_geoms=self.env.needle.contact_geoms
        )
        return signals

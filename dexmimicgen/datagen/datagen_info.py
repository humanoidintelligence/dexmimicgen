# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Defines structure of information that is needed from an environment for data generation.
Ported from MimicGen (NVlabs/mimicgen). For DexMimicGen, eef_pose / target_pose /
gripper_action are per-arm dictionaries (keys e.g. "right", "left").
"""
from copy import deepcopy

import numpy as np


class DatagenInfo(object):
    """
    Structure of information needed from an environment for data generation. To allow for
    flexibility, not all information must be present.

    For the bimanual (DexMimicGen) setting:
        eef_pose (dict): maps arm name -> eef poses of shape [..., 4, 4]
        target_pose (dict): maps arm name -> target eef poses of shape [..., 4, 4]
        gripper_action (dict): maps arm name -> gripper/hand actions of shape [..., D]
        object_poses (dict): maps object name -> object poses of shape [..., 4, 4]
        subtask_term_signals (dict): maps subtask name -> binary indicator
    """

    def __init__(
        self,
        eef_pose=None,
        object_poses=None,
        subtask_term_signals=None,
        target_pose=None,
        gripper_action=None,
    ):
        self.eef_pose = None
        if eef_pose is not None:
            self.eef_pose = {k: np.array(eef_pose[k]) for k in eef_pose}

        self.object_poses = None
        if object_poses is not None:
            self.object_poses = {k: np.array(object_poses[k]) for k in object_poses}

        self.subtask_term_signals = None
        if subtask_term_signals is not None:
            self.subtask_term_signals = dict()
            for k in subtask_term_signals:
                if isinstance(subtask_term_signals[k], (float, int)):
                    self.subtask_term_signals[k] = subtask_term_signals[k]
                else:
                    self.subtask_term_signals[k] = np.array(subtask_term_signals[k])

        self.target_pose = None
        if target_pose is not None:
            self.target_pose = {k: np.array(target_pose[k]) for k in target_pose}

        self.gripper_action = None
        if gripper_action is not None:
            self.gripper_action = {k: np.array(gripper_action[k]) for k in gripper_action}

    def to_dict(self):
        """
        Convert this instance to a dictionary containing the same information.
        """
        ret = dict()
        if self.eef_pose is not None:
            ret["eef_pose"] = deepcopy(self.eef_pose)
        if self.object_poses is not None:
            ret["object_poses"] = deepcopy(self.object_poses)
        if self.subtask_term_signals is not None:
            ret["subtask_term_signals"] = deepcopy(self.subtask_term_signals)
        if self.target_pose is not None:
            ret["target_pose"] = deepcopy(self.target_pose)
        if self.gripper_action is not None:
            ret["gripper_action"] = deepcopy(self.gripper_action)
        return ret

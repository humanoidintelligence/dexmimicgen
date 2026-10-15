# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Selection strategies used to select source subtask segments from source human
demonstrations. Ported from MimicGen (NVlabs/mimicgen).
"""
import abc

import numpy as np
import six

import dexmimicgen.utils.pose_utils as PoseUtils

REGISTERED_SELECTION_STRATEGIES = {}


def make_selection_strategy(name, *args, **kwargs):
    assert_selection_strategy_exists(name)
    return REGISTERED_SELECTION_STRATEGIES[name](*args, **kwargs)


def register_selection_strategy(cls):
    ignore_classes = ["MG_SelectionStrategy"]
    if cls.__name__ not in ignore_classes:
        REGISTERED_SELECTION_STRATEGIES[cls.NAME] = cls


def assert_selection_strategy_exists(name):
    if name not in REGISTERED_SELECTION_STRATEGIES:
        raise Exception(
            "assert_selection_strategy_exists: name {} not found. Registered: {}".format(
                name, ", ".join(REGISTERED_SELECTION_STRATEGIES)
            )
        )


class MG_SelectionStrategyMeta(type):
    def __new__(meta, name, bases, class_dict):
        cls = super(MG_SelectionStrategyMeta, meta).__new__(meta, name, bases, class_dict)
        register_selection_strategy(cls)
        return cls


@six.add_metaclass(MG_SelectionStrategyMeta)
class MG_SelectionStrategy(object):
    """
    Defines methods and functions for selection strategies to implement.
    """

    def __init__(self):
        pass

    @property
    @classmethod
    def NAME(self):
        raise NotImplementedError

    @abc.abstractmethod
    def select_source_demo(self, eef_pose, object_pose, src_subtask_datagen_infos):
        raise NotImplementedError


class RandomStrategy(MG_SelectionStrategy):
    """Pick source demonstration randomly."""

    NAME = "random"

    def select_source_demo(self, eef_pose, object_pose, src_subtask_datagen_infos):
        n_src_demo = len(src_subtask_datagen_infos)
        return np.random.randint(0, n_src_demo)


class NearestNeighborObjectStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration to be the one with the closest object pose to the object
    in the current scene.
    """

    NAME = "nearest_neighbor_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.0,
        rot_weight=1.0,
        nn_k=3,
    ):
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_object_poses = np.array(src_object_poses)

        all_src_obj_pos, all_src_obj_rot = PoseUtils.unmake_pose(src_object_poses)
        obj_pos, obj_rot = PoseUtils.unmake_pose(object_pose)

        obj_pos = obj_pos.reshape(-1, 3)
        obj_rot_T = obj_rot.T.reshape(-1, 3, 3)

        pos_dists = np.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(axis=-1))

        delta_R = np.matmul(all_src_obj_rot, obj_rot_T)
        arc_cos_in = (np.trace(delta_R, axis1=-2, axis2=-1) - 1.0) / 2.0
        arc_cos_in = np.clip(arc_cos_in, -1.0, 1.0)
        rot_dists = np.arccos(arc_cos_in)

        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        nn_k = min(nn_k, len(dists_to_minimize))
        rand_k = np.random.randint(0, nn_k)
        top_k_neighbors_in_order = np.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]


class NearestNeighborRobotDistanceStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration to be the one that minimizes the distance between the current
    robot eef pose and the first (transformed) target eef pose in the source subtask segment.
    """

    NAME = "nearest_neighbor_robot_distance"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.0,
        rot_weight=1.0,
        nn_k=3,
    ):
        # collect first target eef pose of each source subtask segment (relative to its object frame)
        src_eef_poses = []
        src_obj_poses = []
        for di in src_subtask_datagen_infos:
            # single arm per datagen info passed to selection
            src_eef = list(di.target_pose.values())[0]
            src_eef_poses.append(src_eef[0])
            src_obj = list(di.object_poses.values())[0]
            src_obj_poses.append(src_obj[0])
        src_eef_poses = np.array(src_eef_poses)
        src_obj_poses = np.array(src_obj_poses)

        # express src eef poses relative to src object frame, then transform to the current object frame
        src_eef_poses_in_obj = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses,
            pose_A_in_B=PoseUtils.pose_inv(src_obj_poses),
        )
        transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses_in_obj,
            pose_A_in_B=object_pose,
        )

        all_transformed_eef_pos, all_transformed_eef_rot = PoseUtils.unmake_pose(transformed_eef_poses)
        eef_pos, eef_rot = PoseUtils.unmake_pose(eef_pose)

        eef_pos = eef_pos.reshape(-1, 3)
        eef_rot_T = eef_rot.T.reshape(-1, 3, 3)

        pos_dists = np.sqrt(((all_transformed_eef_pos - eef_pos) ** 2).sum(axis=-1))
        delta_R = np.matmul(all_transformed_eef_rot, eef_rot_T)
        arc_cos_in = (np.trace(delta_R, axis1=-2, axis2=-1) - 1.0) / 2.0
        arc_cos_in = np.clip(arc_cos_in, -1.0, 1.0)
        rot_dists = np.arccos(arc_cos_in)

        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        nn_k = min(nn_k, len(dists_to_minimize))
        rand_k = np.random.randint(0, nn_k)
        top_k_neighbors_in_order = np.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]

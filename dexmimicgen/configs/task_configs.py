# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Per-task DexMimicGen data generation configs. Importing this module registers all
configs into the registry in dexmimicgen.configs.config.
"""
from dexmimicgen.configs.config import DexMG_Config
from dexmimicgen.configs.task_spec import COORDINATION, PARALLEL


class TwoArmThreadingConfig(DexMG_Config):
    name = "two_arm_threading"
    env_name = "TwoArmThreading"
    robots = ["Panda", "Panda"]
    env_interface = "TwoArmThreadingInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # right arm (robot 0) grasps the tripod (parallel), then holds it steady while the
        # other arm threads (coordination). NOTE: in robosuite TwoArmThreading the needle is
        # sampled at +y and the tripod at -y; robot 0 ("right") manipulates the tripod and
        # robot 1 ("left") manipulates the needle -- verified against the source demos.
        ts.add_subtask(
            arm="right",
            object_ref="tripod",
            subtask_term_signal="grasp_tripod",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="tripod",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )

        # left arm (robot 1) grasps the needle (parallel), then threads it into the ring
        # on the tripod (coordination).
        ts.add_subtask(
            arm="left",
            object_ref="needle",
            subtask_term_signal="grasp_needle",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="tripod",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )
        return ts

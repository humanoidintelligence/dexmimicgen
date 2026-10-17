# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Per-task DexMimicGen data generation configs. Importing this module registers all
configs into the registry in dexmimicgen.configs.config.
"""
from dexmimicgen.configs.config import DexMG_Config
from dexmimicgen.configs.task_spec import COORDINATION, PARALLEL, SEQUENTIAL


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


class TwoArmLiftTrayConfig(DexMG_Config):
    name = "two_arm_lift_tray"
    env_name = "TwoArmLiftTray"
    robots = ["PandaDexRH", "PandaDexLH"]
    env_interface = "TwoArmLiftTrayInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # Both arms grasp their respective tray handle (PARALLEL), then jointly lift the tray
        # (COORDINATION, object_ref="pot" shared). robot 0 ("right", -y) -> handle0;
        # robot 1 ("left", +y) -> handle1.
        ts.add_subtask(
            arm="right",
            object_ref="pot",
            subtask_term_signal="grasp_handle0",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="pot",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )

        ts.add_subtask(
            arm="left",
            object_ref="pot",
            subtask_term_signal="grasp_handle1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="pot",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )
        return ts


class TwoArmThreePieceAssemblyConfig(DexMG_Config):
    name = "two_arm_three_piece_assembly"
    env_name = "TwoArmThreePieceAssembly"
    robots = ["Panda", "Panda"]
    env_interface = "TwoArmThreePieceAssemblyInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # SEQUENTIAL assembly. Right arm (robot 0) stacks piece_1 (sampled -y) onto the base;
        # the left arm (robot 1) then stacks piece_2 (sampled +y) onto piece_1. The left arm's
        # placement waits for the right arm's "place_piece_1" signal (ordering constraint).
        # right arm: grasp piece_1, place it on the base (emits "place_piece_1"), then hold.
        # "place_piece_1" must be a NON-final subtask signal so it is added to completed_signals
        # for the left arm's sequential gate (the final subtask of each arm must have signal=None).
        ts.add_subtask(
            arm="right",
            object_ref="piece_1",
            subtask_term_signal="grasp_piece_1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="base",
            subtask_term_signal="place_piece_1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="base",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )

        # left arm: grasp piece_2, then stack it on piece_1 -- gated on the right arm having
        # placed piece_1 (wait_for="place_piece_1").
        ts.add_subtask(
            arm="left",
            object_ref="piece_2",
            subtask_term_signal="grasp_piece_2",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="piece_1",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=SEQUENTIAL,
            wait_for="place_piece_1",
        )
        return ts


class TwoArmBoxCleanupConfig(DexMG_Config):
    name = "two_arm_box_cleanup"
    env_name = "TwoArmBoxCleanup"
    robots = ["PandaDexRH", "PandaDexLH"]
    env_interface = "TwoArmBoxCleanupInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # Both arms grasp the lid (PARALLEL), then jointly place it onto the box
        # (COORDINATION, object_ref="box" shared transform). box sampled -y, lid +y.
        ts.add_subtask(
            arm="right",
            object_ref="lid",
            subtask_term_signal="grasp_lid_right",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )

        ts.add_subtask(
            arm="left",
            object_ref="lid",
            subtask_term_signal="grasp_lid_left",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )
        return ts


class TwoArmDrawerCleanupConfig(DexMG_Config):
    name = "two_arm_drawer_cleanup"
    env_name = "TwoArmDrawerCleanup"
    robots = ["PandaDexRH", "PandaDexLH"]
    env_interface = "TwoArmDrawerCleanupInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # SEQUENTIAL: right arm opens the drawer (drawer at -y); left arm grasps the object
        # (at +y) and places it inside once the drawer is open; right arm then closes the
        # drawer once the object is inside.
        # right arm: open drawer (emit "drawer_open"), then close drawer (wait for the object
        # to be placed). "drawer_open" is non-final so it reaches completed_signals.
        ts.add_subtask(
            arm="right",
            object_ref="drawer",
            subtask_term_signal="drawer_open",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="drawer",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=SEQUENTIAL,
            wait_for="object_in_drawer",
        )

        # left arm: grasp object, then place into the drawer (emit "object_in_drawer") once the
        # drawer is open.
        ts.add_subtask(
            arm="left",
            object_ref="object",
            subtask_term_signal="grasp_object",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="drawer",
            subtask_term_signal="object_in_drawer",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=SEQUENTIAL,
            wait_for="drawer_open",
        )
        ts.add_subtask(
            arm="left",
            object_ref="drawer",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        return ts


class TwoArmTransportConfig(DexMG_Config):
    name = "two_arm_transport"
    env_name = "TwoArmTransport"
    robots = ["Panda", "Panda"]
    env_interface = "TwoArmTransportInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # Handover flow. right arm (robot 0): remove lid -> grasp payload -> hand payload over
        # (COORDINATION). left arm (robot 1): dispose trash -> receive payload (COORDINATION) and
        # drop into target bin. Both coordination subtasks share object_ref="payload".
        ts.add_subtask(
            arm="right",
            object_ref="lid",
            subtask_term_signal="grasp_lid",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="payload",
            subtask_term_signal="grasp_payload_r0",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="payload",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )

        ts.add_subtask(
            arm="left",
            object_ref="trash",
            subtask_term_signal="grasp_trash",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="trash_bin",
            subtask_term_signal="grasp_payload_r1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="payload",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )
        return ts


class TwoArmCoffeeConfig(DexMG_Config):
    name = "two_arm_coffee"
    env_name = "TwoArmCoffee"
    robots = ["GR1FixedLowerBody"]
    env_interface = "TwoArmCoffeeInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # SEQUENTIAL single-arm-dominant task. The right hand does the work: grasp the pod (+y),
        # insert it into the machine (-y) pod holder, then settle. The left hand is idle/stabilizing
        # so it is given a single trivial subtask relative to the machine. No cross-arm wait is
        # needed because the pod manipulation is entirely on the right arm.
        ts.add_subtask(
            arm="right",
            object_ref="coffee_pod",
            subtask_term_signal="grasp",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="coffee_machine",
            subtask_term_signal="insertion",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=SEQUENTIAL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="coffee_machine",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )

        # left hand: single subtask stabilizing the machine for the whole episode
        ts.add_subtask(
            arm="left",
            object_ref="coffee_machine",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        return ts


class TwoArmPouringConfig(DexMG_Config):
    name = "two_arm_pouring"
    env_name = "TwoArmPouring"
    robots = ["GR1FixedLowerBody"]
    env_interface = "TwoArmPouringInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # SEQUENTIAL: left hand places the bowl (-y) onto the pad (emit "bowl_on_pad"); right hand
        # grasps the cup (+y) and pours the ball into the bowl, gated on the bowl being placed.
        # left: grasp bowl -> place on pad (emit bowl_on_pad, non-final) -> hold.
        ts.add_subtask(
            arm="left",
            object_ref="bowl",
            subtask_term_signal="bowl_grasped",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="pad",
            subtask_term_signal="bowl_on_pad",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="pad",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )

        # right: grasp cup -> pour into bowl, waiting for the bowl to be on the pad.
        ts.add_subtask(
            arm="right",
            object_ref="cup",
            subtask_term_signal="cup_grasped",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="bowl",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=SEQUENTIAL,
            wait_for="bowl_on_pad",
        )
        return ts


class TwoArmCanSortRandomConfig(DexMG_Config):
    name = "two_arm_can_sort_random"
    env_name = "TwoArmCanSortRandom"
    robots = ["GR1ArmsOnly"]
    env_interface = "TwoArmCanSortRandomInterface"

    def task_spec(self):
        ts = self.new_task_spec()

        # COORDINATION / handover: the left hand grasps the can (sampled near center) and hands it
        # to the right hand (COORDINATION on the can), which drops it into the correct box (+y).
        # Both coordination subtasks share object_ref="can".
        ts.add_subtask(
            arm="left",
            object_ref="can",
            subtask_term_signal="grasp_can_left",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="left",
            object_ref="can",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )

        ts.add_subtask(
            arm="right",
            object_ref="can",
            subtask_term_signal="grasp_can_right",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            subtask_type=PARALLEL,
        )
        ts.add_subtask(
            arm="right",
            object_ref="can",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            subtask_type=COORDINATION,
        )
        return ts

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Robosuite environment interfaces for DexMimicGen tasks. Each interface provides the
per-arm end-effector mapping, the object poses relevant to data generation, and the
per-subtask termination signals used to segment source demonstrations.
"""
import numpy as np
import robosuite.utils.transform_utils as T

import dexmimicgen.utils.pose_utils as PoseUtils
from dexmimicgen.env_interfaces.base import ArmSpec, DexMG_EnvInterface


class RobosuiteInterface(DexMG_EnvInterface):
    """Shared helpers for robosuite-based interfaces."""

    def _grasp(self, robot_idx, arm, object_geoms):
        """Return 1.0 if the given robot arm's gripper is grasping @object_geoms, else 0.0."""
        robot = self.env.robots[robot_idx]
        gripper = robot.gripper[arm] if isinstance(robot.gripper, dict) else robot.gripper
        return float(self.env._check_grasp(gripper=gripper, object_geoms=object_geoms))

    def _near(self, eef_site, object_geoms, threshold):
        """
        Return 1.0 if the eef site is within @threshold (meters) of the centroid of
        @object_geoms, else 0.0.

        Proximity-based stand-in for a grasp signal. The Inspire dexterous hand grasps by
        cradling/wrapping rather than pinching, so robosuite's contact-based ``_check_grasp``
        frequently reports no grasp even while the hand is clearly holding the object. For the
        dex tasks the grip-site-to-object distance is the reliable segmentation signal.
        """
        eef = self.get_object_pose(obj_name=eef_site, obj_type="site")[:3, 3]
        pts = np.array(
            [self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(g)] for g in object_geoms]
        )
        centroid = pts.mean(axis=0)
        return float(np.linalg.norm(eef - centroid) < threshold)


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


class TwoArmLiftTrayInterface(RobosuiteInterface):
    """
    Lift Tray (PandaDex): both arms jointly lift a tray (pot with two handles) loaded with
    two cubes. robot 0 ("right", -y side) grasps handle0; robot 1 ("left", +y side) grasps
    handle1 (mapping taken from the env reward: g0=robots[0]->handle0_geoms,
    g1=robots[1]->handle1_geoms). After each arm grasps its handle (PARALLEL), the lift is a
    COORDINATION subtask -- both arms move the single tray together, so they must share one
    SE(3) transform and the same source demo to preserve the inter-arm relative pose.
    Hands are 6-DoF dexterous (gripper_dim=6).
    """

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=6),
            "left": ArmSpec(robot_idx=1, part_name="right", eef_site="gripper1_right_grip_site", gripper_dim=6),
        }

    def get_object_poses(self):
        return dict(
            pot=self.get_object_pose(obj_name=self.env.pot.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # The dex hand cradles the handle, so contact-based _check_grasp never fires here;
        # use grip-site proximity instead (crosses cleanly at ~0.13 m, before the lift begins).
        signals = dict()
        signals["grasp_handle0"] = self._near(
            eef_site="gripper0_right_grip_site",
            object_geoms=self.env.pot.handle0_geoms,
            threshold=0.13,
        )
        signals["grasp_handle1"] = self._near(
            eef_site="gripper1_right_grip_site",
            object_geoms=self.env.pot.handle1_geoms,
            threshold=0.13,
        )
        return signals


class TwoArmThreePieceAssemblyInterface(RobosuiteInterface):
    """
    Three-Piece Assembly (Panda x2): a SEQUENTIAL task. The base sits fixed at the center;
    piece_1 is sampled at -y (robot 0 "right"), piece_2 at +y (robot 1 "left"). The right arm
    first grasps and stacks piece_1 onto the base; only then does the left arm stack piece_2
    on top of piece_1 (the env success check requires first_piece_is_assembled before
    second_piece_is_assembled). The ordering constraint is realized by making the left arm's
    placement subtask wait for the right arm's piece_1 placement.
    Parallel-jaw grippers (gripper_dim=1).
    """

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=1),
            "left": ArmSpec(robot_idx=1, part_name="right", eef_site="gripper1_right_grip_site", gripper_dim=1),
        }

    def get_object_poses(self):
        return dict(
            piece_1=self.get_object_pose(obj_name=self.env.piece_1.root_body, obj_type="body"),
            piece_2=self.get_object_pose(obj_name=self.env.piece_2.root_body, obj_type="body"),
            base=self.get_object_pose(obj_name=self.env.base.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # right arm (robot 0) grasps piece_1; left arm (robot 1) grasps piece_2
        signals["grasp_piece_1"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self.env.piece_1.contact_geoms
        )
        signals["grasp_piece_2"] = self._grasp(
            robot_idx=1, arm="right", object_geoms=self.env.piece_2.contact_geoms
        )
        # stage signals: a piece is "placed" once it is assembled (env releases + aligns)
        signals["place_piece_1"] = float(self.env._check_first_piece_is_assembled())
        return signals


class TwoArmBoxCleanupInterface(RobosuiteInterface):
    """
    Box Cleanup / Box Packing (PandaDex): a COORDINATION task. The box (container) is sampled
    at -y and the lid at +y. Both arms grasp the lid and lift it onto the box together, so the
    placement is a coordination subtask sharing one SE(3) transform (object_ref="box"). The
    box itself is not relocated. robot 0 ("right", -y) and robot 1 ("left", +y) each grasp a
    side of the lid. Dexterous hands (gripper_dim=6).

    NOTE: handedness and whether both arms (vs one) grasp the lid are to be confirmed against
    the source demos via scripts/analyze_src_demo.py once the dataset is available.
    """

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=6),
            "left": ArmSpec(robot_idx=1, part_name="right", eef_site="gripper1_right_grip_site", gripper_dim=6),
        }

    def get_object_poses(self):
        return dict(
            box=self.get_object_pose(obj_name=self.env.box.root_body, obj_type="body"),
            lid=self.get_object_pose(obj_name=self.env.lid.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # both arms grasp the lid (coordination lift)
        signals["grasp_lid_right"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self.env.lid.contact_geoms
        )
        signals["grasp_lid_left"] = self._grasp(
            robot_idx=1, arm="right", object_geoms=self.env.lid.contact_geoms
        )
        return signals


class TwoArmDrawerCleanupInterface(RobosuiteInterface):
    """
    Drawer Cleanup (PandaDex): a SEQUENTIAL task. The drawer is sampled at -y (robot 0
    "right") and the cleanup object at +y (robot 1 "left"). The right arm opens the drawer;
    the left arm then picks the object and places it inside, after which the right arm closes
    the drawer. The object-placement/closing ordering is realized via sequential gating.
    Dexterous hands (gripper_dim=6).

    NOTE: exact stage order (open -> place -> close) and handedness to be confirmed against the
    source demos via scripts/analyze_src_demo.py.
    """

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=6),
            "left": ArmSpec(robot_idx=1, part_name="right", eef_site="gripper1_right_grip_site", gripper_dim=6),
        }

    def get_object_poses(self):
        return dict(
            drawer=self.get_object_pose(obj_name=self.env.drawer.root_body, obj_type="body"),
            object=self.get_object_pose(obj_name=self.env.cleanup_object.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # left arm grasps the cleanup object; right arm opens/closes the drawer
        signals["grasp_object"] = self._grasp(
            robot_idx=1, arm="right", object_geoms=self.env.cleanup_object.contact_geoms
        )
        # drawer opened enough for the object to be placed (qpos more negative than threshold)
        signals["drawer_open"] = float(
            self.env.sim.data.qpos[self.env.drawer_qpos_addr] < -0.05
        )
        # object resting inside the drawer
        signals["object_in_drawer"] = float(self.env._check_object())
        return signals


class TwoArmTransportInterface(RobosuiteInterface):
    """
    Transport (Panda x2): a long-horizon HANDOVER task on two tables. The start bin (with a
    lid over the payload) sits on table 0 near robot 0 ("right", -y side); the target bin
    (with trash in it) and the trash bin sit on table 1 near robot 1 ("left", +y side).

    Flow (handover): robot 0 removes the lid, grasps the payload, and hands it across to
    robot 1 (COORDINATION); robot 1 clears the trash from the target bin into the trash bin,
    then receives the payload and drops it into the target bin. Success = payload in target
    bin AND trash in trash bin.

    Parallel-jaw grippers (gripper_dim=1). NOTE: the precise arm roles, stage order and
    handedness here are an informed design; they must be confirmed/corrected against the
    source demos via scripts/analyze_src_demo.py before trusting generation.
    """

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=1),
            "left": ArmSpec(robot_idx=1, part_name="right", eef_site="gripper1_right_grip_site", gripper_dim=1),
        }

    def _tobj(self, name):
        return self.env.transport.objects[name]

    def get_object_poses(self):
        return dict(
            payload=self.get_object_pose(obj_name=self._tobj("payload").root_body, obj_type="body"),
            lid=self.get_object_pose(obj_name=self._tobj("lid").root_body, obj_type="body"),
            trash=self.get_object_pose(obj_name=self._tobj("trash").root_body, obj_type="body"),
            target_bin=self.get_object_pose(obj_name=self._tobj("target_bin").root_body, obj_type="body"),
            trash_bin=self.get_object_pose(obj_name=self._tobj("trash_bin").root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # robot 0: grasp lid, then payload; robot 1: grasp trash, then (after handover) payload
        signals["grasp_lid"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self._tobj("lid").contact_geoms
        )
        signals["grasp_payload_r0"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self._tobj("payload").contact_geoms
        )
        signals["grasp_trash"] = self._grasp(
            robot_idx=1, arm="right", object_geoms=self._tobj("trash").contact_geoms
        )
        signals["grasp_payload_r1"] = self._grasp(
            robot_idx=1, arm="right", object_geoms=self._tobj("payload").contact_geoms
        )
        return signals


class GR1Interface(RobosuiteInterface):
    """
    Shared base for the GR-1 humanoid tasks (Coffee, Pouring, Can Sort). Both arms live on a
    single robot (env.robots[0]) and are driven by a whole-body IK composite controller
    (WHOLE_BODY_MINK_IK). Unlike the OSC Panda tasks, the per-arm action is an ABSOLUTE target
    end-effector pose [pos(3), axis_angle(3)] expressed in the WORLD frame, not an OSC delta.
    The Inspire hand is 6-DoF (gripper_dim=6).

    The controller config records input_ref_frame="world" (verified against the source demos:
    decoding a recorded action as a world-frame pose reproduces the achieved right-eef pose to
    ~0.02 m / ~1.8 deg, whereas a base-frame interpretation is off by ~1.09 m, i.e. the base
    height). So the conversion below is a straight world-frame pack/unpack with no base
    transform. Subclasses only specify object poses and subtask signals.
    """

    BASE_BODY = "robot0_base"

    def _get_arm_specs(self):
        return {
            "right": ArmSpec(robot_idx=0, part_name="right", eef_site="gripper0_right_grip_site", gripper_dim=6),
            "left": ArmSpec(robot_idx=0, part_name="left", eef_site="gripper0_left_grip_site", gripper_dim=6),
        }

    def _base_pose(self):
        """World-frame pose (4x4) of the IK reference (robot base)."""
        return self.get_object_pose(obj_name=self.BASE_BODY, obj_type="body")

    def _arm_action_slices(self, arm_key):
        """
        GR1 overrides robosuite's per-part split: the whole-body controller expects
        [right_eef(6), left_eef(6), right_hand(6), left_hand(6)] regardless of the raw
        part-controller joint dims. Mirror that layout here.
        """
        order = {"right": 0, "left": 1}
        i = order[arm_key]
        arm_sl = (i * 6, i * 6 + 6)
        grip_start = 12 + i * 6
        grip_sl = (grip_start, grip_start + 6)
        return arm_sl, grip_sl

    @property
    def env_action_dim(self):
        return 24

    def target_pose_to_action(self, arm_key, target_pose, relative=True):
        """Encode a world-frame target eef pose as an absolute world-frame [pos, axisangle]."""
        pos, rot = PoseUtils.unmake_pose(target_pose)
        axisangle = T.quat2axisangle(T.mat2quat(rot))
        return np.concatenate([pos, axisangle])

    def action_to_target_pose(self, arm_key, action, relative=True):
        """Decode an absolute world-frame [pos, axisangle] action into a world-frame pose."""
        pos = action[:3]
        rot = T.quat2mat(T.axisangle2quat(action[3:6]))
        return PoseUtils.make_pose(pos, rot)


class TwoArmCoffeeInterface(GR1Interface):
    """
    Coffee (GR-1): SEQUENTIAL. The coffee machine (with a hinged lid) sits at -y; the coffee pod
    sits at +y. The right hand opens the lid, grasps the pod, inserts it into the pod holder, then
    closes the lid (env _check_pod_is_grasped uses robots[0].gripper["right"]). The left hand
    stabilizes the machine. Stage ordering: pod grasped -> pod on rim -> pod inserted -> lid closed.

    Signal names mirror the env's _get_partial_task_metrics() keys (grasp, insertion, rim, ...),
    which are the canonical names stored in the shipped source datagen_info. NOTE: the dex-hand
    ``grasp`` signal (contact-based _check_pod_is_grasped) does NOT reproduce under replay --
    set_state + forward does not repopulate the finger contacts that generated it. For dex/GR1
    tasks the shipped source datagen_info is authoritative, so run generate_dataset directly on
    the shipped source rather than re-running prepare_src_dataset (which would overwrite the
    grasp boundary with an all-zero recomputation). The non-grasp signals (insertion, rim) do
    reproduce exactly via per-step reset_to + _get_partial_task_metrics().
    """

    def get_object_poses(self):
        return dict(
            coffee_pod=self.get_object_pose(obj_name=self.env.coffee_pod.root_body, obj_type="body"),
            coffee_machine=self.get_object_pose(obj_name=self.env.coffee_machine.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # Canonical names from env._get_partial_task_metrics(); for dex grasp see class docstring.
        m = self.env._get_partial_task_metrics()
        return dict(
            grasp=float(m["grasp"]),
            insertion=float(m["insertion"]),
            rim=float(m["rim"]),
        )


class TwoArmPouringInterface(GR1Interface):
    """
    Pouring (GR-1): SEQUENTIAL. The cup (with a ball inside) sits at +y; the bowl sits at -y and
    must end up on the pad (further -y). The left hand picks the bowl and places it on the pad; the
    right hand grasps the cup and pours the ball into the bowl. Pouring is gated on the bowl being
    placed first (ordering constraint). Success = ball in bowl AND bowl upright on pad.

    Signal names match the shipped source datagen_info: cup_grasped, bowl_grasped, bowl_on_pad,
    ball_in_bowl (confirmed on demo_0: cup_grasped@101, bowl_grasped@271, bowl_on_pad@323,
    monotonic per-arm). As with coffee, the dex-hand *_grasped contact signals do NOT reproduce
    under replay, so run generate_dataset directly on the shipped source (authoritative
    datagen_info) rather than re-running prepare_src_dataset. See GR1Interface /
    TwoArmCoffeeInterface docstrings for the full rationale.
    """

    def get_object_poses(self):
        return dict(
            cup=self.get_object_pose(obj_name=self.env.cup.root_body, obj_type="body"),
            bowl=self.get_object_pose(obj_name=self.env.bowl.root_body, obj_type="body"),
            ball=self.get_object_pose(obj_name=self.env.ball.root_body, obj_type="body"),
            pad=self.get_object_pose(obj_name=self.env.pad.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # Canonical names from the shipped source; dex *_grasped signals are contact-based and
        # only authoritative in the shipped datagen_info (see class docstring).
        signals = dict()
        signals["cup_grasped"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self.env.cup.contact_geoms
        )
        signals["bowl_grasped"] = self._grasp(
            robot_idx=0, arm="left", object_geoms=self.env.bowl.contact_geoms
        )
        # bowl placed upright on the pad (gate for pouring)
        signals["bowl_on_pad"] = float(self.env.check_contact(self.env.pad, self.env.bowl))
        return signals


class TwoArmCanSortRandomInterface(GR1Interface):
    """
    Can Sort (GR-1, GR1ArmsOnly): a COORDINATION / handover task. A single can is sampled near the
    center; it must be sorted into the red box (+y, far) or blue box (+y, near) depending on
    env.is_red. Because the can starts between the arms and the target bins are on one side, the
    arms hand the can over: one arm grasps it and passes it to the other, which drops it in the
    correct box. The handover is a coordination subtask sharing the can's transform.

    NOTE: which arm initiates vs receives the handover is an informed design; confirm against the
    source demos via scripts/analyze_src_demo.py.
    """

    def get_object_poses(self):
        return dict(
            can=self.get_object_pose(obj_name=self.env.can.root_body, obj_type="body"),
            red_box=self.get_object_pose(obj_name=self.env.red_box.root_body, obj_type="body"),
            blue_box=self.get_object_pose(obj_name=self.env.blue_box.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # both hands may grasp the can across the handover
        signals["grasp_can_left"] = self._grasp(
            robot_idx=0, arm="left", object_geoms=self.env.can.contact_geoms
        )
        signals["grasp_can_right"] = self._grasp(
            robot_idx=0, arm="right", object_geoms=self.env.can.contact_geoms
        )
        return signals

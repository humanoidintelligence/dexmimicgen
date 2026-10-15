# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Classes used to represent waypoints and single-arm trajectories, together with a
multi-arm execution engine that implements the DexMimicGen asynchronous /
coordination / sequential execution strategies.

The single-arm Waypoint / WaypointSequence / WaypointTrajectory classes are ported
from MimicGen (NVlabs/mimicgen). The MultiArmExecutor is new for DexMimicGen.
"""
from copy import deepcopy

import numpy as np

import dexmimicgen.utils.pose_utils as PoseUtils


class Waypoint(object):
    """
    A single desired 6-DoF waypoint, along with corresponding gripper/hand actuation.
    """

    def __init__(self, pose, gripper_action, noise=None):
        self.pose = np.array(pose)
        self.gripper_action = np.array(gripper_action)
        self.noise = noise
        assert len(self.gripper_action.shape) == 1


class WaypointSequence(object):
    """A sequence of Waypoint objects."""

    def __init__(self, sequence=None):
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence given a sequence of poses, gripper actions, and action noise.
        """
        assert isinstance(action_noise, float) or isinstance(action_noise, np.ndarray)

        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * np.ones((num_timesteps, 1))
        action_noise = action_noise.reshape(-1, 1)

        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, ind):
        return self.sequence[ind]

    def __add__(self, other):
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    @property
    def last_waypoint(self):
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)


class WaypointTrajectory(object):
    """A sequence of WaypointSequence objects corresponding to a full 6-DoF trajectory."""

    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        return self.waypoint_sequences[-1].last_waypoint

    def add_waypoint_sequence(self, sequence):
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.0,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose, either via
        linear interpolation from the last waypoint (default) or a constant set of targets.
        """
        if len(self.waypoint_sequences) == 0:
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            assert num_steps is not None
            poses = np.array([pose for _ in range(num_steps)])
            gripper_actions = np.array([gripper_action for _ in range(num_steps)])
        else:
            last_waypoint = self.last_waypoint
            poses, num_steps_2 = PoseUtils.interpolate_poses(
                pose_1=last_waypoint.pose,
                pose_2=pose,
                num_steps=num_steps,
            )
            assert num_steps == num_steps_2
            gripper_actions = np.array([gripper_action for _ in range(num_steps + 2)])
            # skip the first element of the new path, which already exists on the current trajectory
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            self.waypoint_sequences[0] = rest
        return first

    def merge(self, other, num_steps_interp=None, num_steps_fixed=None, action_noise=0.0):
        """
        Merge this trajectory with another (@other), optionally adding an interpolation
        segment and/or a fixed segment that bridges the end of this trajectory with @other.
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = need_interp or need_fixed

        if use_interpolation_segment:
            other_first = other.pop_first()
            target_for_interpolation = other_first[0]

            if need_interp:
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                )

            if need_fixed:
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                )

            # preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        self.waypoint_sequences += other.waypoint_sequences

    def to_waypoint_list(self):
        """Flatten into a plain list of Waypoint objects."""
        out = []
        for seq in self.waypoint_sequences:
            out.extend(seq.sequence)
        return out


# ---------------------------------------------------------------------------
# Multi-arm execution engine (new for DexMimicGen)
# ---------------------------------------------------------------------------

# subtask type constants
PARALLEL = "parallel"
COORDINATION = "coordination"
SEQUENTIAL = "sequential"


class ArmSubtaskPlan(object):
    """
    Container for one arm's plan for a single subtask: the flattened list of Waypoint
    objects to execute, plus metadata describing how this subtask coordinates with the
    other arm.

    Args:
        waypoints (list of Waypoint): the (already transformed + interpolated) waypoints.
        subtask_type (str): one of PARALLEL, COORDINATION, SEQUENTIAL.
        subtask_term_signal (str or None): name of the subtask (used for sequential
            pre/post constraints).
        wait_for (str or None): for a SEQUENTIAL post-subtask, the name of the pre-subtask
            (on the OTHER arm) that must complete before this subtask may begin.
    """

    def __init__(self, waypoints, subtask_type=PARALLEL, subtask_term_signal=None, wait_for=None):
        self.waypoints = list(waypoints)
        self.subtask_type = subtask_type
        self.subtask_term_signal = subtask_term_signal
        self.wait_for = wait_for


class MultiArmExecutor(object):
    """
    Executes per-arm sequences of subtask plans in the shared environment, one simulation
    step at a time, implementing the three DexMimicGen subtask coordination strategies:

      * parallel     : each arm dequeues its own actions independently; when an arm's queue
                       is empty it loads its next subtask (asynchronous execution).
      * coordination : both arms must execute the coordination subtask synchronized so that
                       they finish at the same simulation step. We align by waiting (holding
                       the last pose) until both arms have the same number of remaining steps.
      * sequential   : an arm's post-subtask is held until the paired pre-subtask on the
                       other arm has completed.

    Args:
        env: the (raw robosuite) environment.
        env_interface (DexMG_EnvInterface): interface providing per-arm action composition.
        arm_plans (dict): maps arm name -> list of ArmSubtaskPlan, in execution order.
    """

    def __init__(self, env, env_interface, arm_plans=None, plan_fn=None, n_subtasks=None):
        """
        Args:
            arm_plans (dict): maps arm -> list of pre-built ArmSubtaskPlan (eager mode).
            plan_fn (callable): lazy mode. ``plan_fn(arm, subtask_ind, prev_last_waypoint)``
                returns the ArmSubtaskPlan for that arm's subtask, planned against the CURRENT
                scene observation at the moment it is called. Planning each subtask lazily (only
                when the arm actually reaches it) is required so that object-frame transforms use
                up-to-date object poses -- planning all subtasks up front would transform later
                subtasks to stale (pre-execution) object frames.
            n_subtasks (dict): maps arm -> number of subtasks (required in lazy mode).
        """
        self.env = env
        self.env_interface = env_interface
        if arm_plans is not None:
            # eager mode: wrap the pre-built lists behind the same lazy interface
            self.arms = list(arm_plans.keys())
            self._n_subtasks = {arm: len(arm_plans[arm]) for arm in self.arms}
            self._plan_fn = lambda arm, idx, prev: arm_plans[arm][idx]
        else:
            assert plan_fn is not None and n_subtasks is not None
            self.arms = list(n_subtasks.keys())
            self._n_subtasks = dict(n_subtasks)
            self._plan_fn = plan_fn

    def execute(
        self,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
    ):
        """
        Run the multi-arm execution loop. Returns a results dict matching the single-arm
        MimicGen execute() contract.
        """
        write_video = video_writer is not None
        video_count = 0

        states = []
        actions = []
        observations = []
        datagen_infos = []
        success = False

        # per-arm execution cursors
        subtask_idx = {arm: 0 for arm in self.arms}      # which subtask the arm is on
        wp_idx = {arm: 0 for arm in self.arms}           # waypoint cursor within current subtask
        completed_signals = set()                        # subtask_term_signals completed so far
        last_waypoint = {arm: None for arm in self.arms}  # for holding pose when waiting
        plan_cache = {arm: {} for arm in self.arms}      # subtask_ind -> ArmSubtaskPlan (lazy)

        def current_plan(arm):
            """
            Return the arm's current ArmSubtaskPlan, planning it lazily (against the current
            scene) the first time the arm reaches this subtask index. Returns None when the
            arm has exhausted all of its subtasks.
            """
            idx = subtask_idx[arm]
            if idx >= self._n_subtasks[arm]:
                return None
            if idx not in plan_cache[arm]:
                plan_cache[arm][idx] = self._plan_fn(arm, idx, last_waypoint[arm])
            return plan_cache[arm][idx]

        def arm_finished(arm):
            return current_plan(arm) is None

        def can_start(arm, plan):
            """Check sequential ordering constraint for the start of a subtask."""
            if plan.subtask_type == SEQUENTIAL and plan.wait_for is not None:
                return plan.wait_for in completed_signals
            return True

        def hold_target(arm):
            """
            Resolve the waypoint to hold when an arm must wait. Prefer the last applied
            waypoint; if the arm has not acted yet, hold its first upcoming target pose.
            """
            if last_waypoint[arm] is not None:
                return last_waypoint[arm]
            plan = current_plan(arm)
            if plan is not None and len(plan.waypoints) > 0:
                return plan.waypoints[0]
            return None

        max_steps = 100 + 4 * sum(self._n_subtasks[arm] for arm in self.arms) * 500

        step_count = 0
        while not all(arm_finished(arm) for arm in self.arms):
            step_count += 1
            if step_count > max_steps:
                break

            cur_plans = {arm: current_plan(arm) for arm in self.arms}

            # decide, per arm, which waypoint to apply this step. Arms flow through their own
            # subtask timelines asynchronously: each arm advances one waypoint per step and
            # transitions straight into its next subtask with no idle hold.
            #
            # Coordination end-synchronization is handled WITHOUT inserting holds. Both arms'
            # subtasks are segmented from the same source-demo timeline and have equal total
            # length up to the end of the coordination subtask, so advancing both arms one
            # source-step per executor-step keeps them aligned in absolute (source) time and
            # makes the coordination subtask finish on the same step automatically -- which is
            # exactly the source-demo property the paper's sync strategy reproduces (Sec. 4.2).
            # The paper's second coordination requirement -- a single shared SE(3) transform for
            # both arms -- is enforced upstream in DataGenerator.generate (coord_anchor). Idling
            # an arm here was found to inject phantom steps that desync the two coordination
            # trajectories and distort even a faithful (identity-transform) replay, so arms are
            # never held except for an explicit SEQUENTIAL ordering constraint.
            step_waypoints = {}
            advance = {arm: False for arm in self.arms}
            for arm in self.arms:
                plan = cur_plans[arm]
                if plan is None:
                    # arm done: hold last pose while the other arm finishes
                    step_waypoints[arm] = hold_target(arm)
                    continue

                # sequential gating: hold until the pre-subtask on the other arm completes
                if wp_idx[arm] == 0 and not can_start(arm, plan):
                    step_waypoints[arm] = hold_target(arm)
                    continue

                step_waypoints[arm] = plan.waypoints[wp_idx[arm]]
                advance[arm] = True

            # deadlock guard: if nothing can advance (every unfinished arm is blocked on a
            # sequential constraint that can no longer be satisfied), stop.
            if not any(advance.values()):
                any_unfinished = any(not arm_finished(a) for a in self.arms)
                if any_unfinished:
                    break

            # render
            if render:
                if getattr(self.env, "viewer", None) is None:
                    self.env.initialize_renderer()
                self.env.viewer.update()
            if write_video and (video_count % video_skip == 0):
                video_img = []
                for cam_name in camera_names:
                    video_img.append(
                        self.env.sim.render(height=512, width=512, camera_name=cam_name)[::-1]
                    )
                video_writer.append_data(np.concatenate(video_img, axis=1))
            if write_video:
                video_count += 1

            # record flattened sim state BEFORE stepping (fast; avoids XML regeneration)
            state = np.array(self.env.sim.get_state().flatten())
            obs = None

            # compose and step the combined action
            play_action = self.env_interface.compose_action(step_waypoints)
            datagen_info = self.env_interface.get_datagen_info(action=play_action)
            self.env.step(play_action)

            states.append(state)
            actions.append(play_action)
            observations.append(obs)
            datagen_infos.append(datagen_info)

            success = success or bool(self.env._check_success())

            # advance cursors and record last applied waypoint
            for arm in self.arms:
                if step_waypoints[arm] is not None:
                    last_waypoint[arm] = step_waypoints[arm]
                if advance[arm]:
                    wp_idx[arm] += 1
                    plan = cur_plans[arm]
                    if wp_idx[arm] >= len(plan.waypoints):
                        # subtask complete
                        if plan.subtask_term_signal is not None:
                            completed_signals.add(plan.subtask_term_signal)
                        subtask_idx[arm] += 1
                        wp_idx[arm] = 0

        results = dict(
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=np.array(actions),
            success=bool(success),
        )
        return results

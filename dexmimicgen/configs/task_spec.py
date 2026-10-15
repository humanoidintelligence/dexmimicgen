# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Task specification objects for DexMimicGen. Each task is described by a set of
per-arm subtask sequences. In addition to the standard MimicGen per-subtask
settings, each subtask carries a `subtask_type` that controls how it is executed
relative to the other arm:

    - "parallel"     : executed asynchronously w.r.t. the other arm (default)
    - "coordination" : executed synchronized with the other arm's coordination subtask
    - "sequential"   : execution is gated on the completion of a named subtask on
                       the other arm (via `wait_for`)
"""
import json

from dexmimicgen.datagen.selection_strategy import assert_selection_strategy_exists

PARALLEL = "parallel"
COORDINATION = "coordination"
SEQUENTIAL = "sequential"
VALID_SUBTASK_TYPES = [PARALLEL, COORDINATION, SEQUENTIAL]


class DexMG_SingleArmTaskSpec:
    """Stores the ordered list of subtasks for a single arm."""

    def __init__(self):
        self.spec = []

    def add_subtask(
        self,
        object_ref,
        subtask_term_signal,
        subtask_term_offset_range=None,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.0,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
        subtask_type=PARALLEL,
        wait_for=None,
    ):
        if subtask_term_offset_range is None:
            subtask_term_offset_range = (0, 0)
        assert isinstance(subtask_term_offset_range, tuple)
        assert len(subtask_term_offset_range) == 2
        assert subtask_term_offset_range[0] <= subtask_term_offset_range[1]
        assert subtask_type in VALID_SUBTASK_TYPES
        assert_selection_strategy_exists(selection_strategy)
        self.spec.append(
            dict(
                object_ref=object_ref,
                subtask_term_signal=subtask_term_signal,
                subtask_term_offset_range=subtask_term_offset_range,
                selection_strategy=selection_strategy,
                selection_strategy_kwargs=selection_strategy_kwargs,
                action_noise=action_noise,
                num_interpolation_steps=num_interpolation_steps,
                num_fixed_steps=num_fixed_steps,
                apply_noise_during_interpolation=apply_noise_during_interpolation,
                subtask_type=subtask_type,
                wait_for=wait_for,
            )
        )

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, ind):
        return self.spec[ind]

    def __iter__(self):
        return iter(self.spec)


class DexMG_TaskSpec:
    """
    Stores per-arm task specs. Behaves like a dict mapping arm key -> list of subtask dicts.
    """

    def __init__(self):
        self.arm_specs = {}

    def add_arm(self, arm_key):
        if arm_key not in self.arm_specs:
            self.arm_specs[arm_key] = DexMG_SingleArmTaskSpec()
        return self.arm_specs[arm_key]

    def add_subtask(self, arm, **kwargs):
        if arm not in self.arm_specs:
            self.add_arm(arm)
        self.arm_specs[arm].add_subtask(**kwargs)

    @property
    def arms(self):
        return list(self.arm_specs.keys())

    def __getitem__(self, arm_key):
        return self.arm_specs[arm_key]

    def sanity_check(self):
        """
        Final subtask of each arm must not have offset randomization (end is task end).
        """
        for arm in self.arms:
            spec = self.arm_specs[arm]
            assert len(spec) > 0, "arm {} has no subtasks".format(arm)
            assert spec[-1]["subtask_term_offset_range"][0] == 0
            assert spec[-1]["subtask_term_offset_range"][1] == 0

    @classmethod
    def from_json(cls, json_string=None, json_dict=None):
        if json_dict is None:
            json_dict = json.loads(json_string)
        task_spec = cls()
        for arm_key in json_dict:
            for subtask_name in json_dict[arm_key]:
                sub = dict(json_dict[arm_key][subtask_name])
                if sub.get("subtask_term_offset_range", None) is not None:
                    sub["subtask_term_offset_range"] = tuple(sub["subtask_term_offset_range"])
                task_spec.add_subtask(arm=arm_key, **sub)
        return task_spec

    def serialize(self):
        json_dict = dict()
        for arm_key in self.arms:
            json_dict[arm_key] = dict()
            for i, elem in enumerate(self.arm_specs[arm_key].spec):
                json_dict[arm_key]["subtask_{}".format(i + 1)] = elem
        return json.dumps(json_dict, indent=4)

    def __repr__(self):
        return self.serialize()

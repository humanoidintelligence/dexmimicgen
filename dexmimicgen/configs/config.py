# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Lightweight configuration objects + registry for DexMimicGen data generation.

A task config bundles together:
  * the robosuite environment name and robots
  * the environment interface class name
  * the per-arm task spec (subtasks + coordination strategy)
  * generation hyperparameters
"""
from dexmimicgen.configs.task_spec import DexMG_TaskSpec

REGISTERED_CONFIGS = {}


def register_config(cls):
    inst = cls()
    REGISTERED_CONFIGS[inst.name] = cls


def config_factory(name):
    if name not in REGISTERED_CONFIGS:
        raise Exception(
            "config_factory: config {} not found. Registered: {}".format(
                name, ", ".join(REGISTERED_CONFIGS)
            )
        )
    return REGISTERED_CONFIGS[name]()


class DexMG_ConfigMeta(type):
    def __new__(meta, cls_name, bases, class_dict):
        cls = super(DexMG_ConfigMeta, meta).__new__(meta, cls_name, bases, class_dict)
        if cls_name != "DexMG_Config":
            register_config(cls)
        return cls


class DexMG_Config(object, metaclass=DexMG_ConfigMeta):
    """
    Base class for a DexMimicGen task config. Subclasses set `name`, `env_name`,
    `robots`, `env_interface`, and implement `task_spec()`.
    """

    # unique registry name
    name = None
    # robosuite environment name
    env_name = None
    # list of robots to pass to robosuite.make
    robots = None
    # name of the DexMG_EnvInterface subclass
    env_interface = None

    # generation defaults
    num_demos = 10
    guarantee = True
    max_num_attempts_per_demo = 50
    seed = 1
    select_src_per_subtask = False
    transform_first_robot_pose = False
    interpolate_from_last_target_pose = True

    def task_spec(self):
        """Return a populated DexMG_TaskSpec for this task."""
        raise NotImplementedError

    def new_task_spec(self):
        return DexMG_TaskSpec()

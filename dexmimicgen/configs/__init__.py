# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from dexmimicgen.configs.config import DexMG_Config, config_factory, REGISTERED_CONFIGS
from dexmimicgen.configs.task_spec import (
    COORDINATION,
    PARALLEL,
    SEQUENTIAL,
    DexMG_TaskSpec,
)

# importing registers all task configs
import dexmimicgen.configs.task_configs  # noqa: F401,E402

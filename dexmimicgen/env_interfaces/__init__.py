# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from dexmimicgen.env_interfaces.base import (
    DexMG_EnvInterface,
    REGISTERED_ENV_INTERFACES,
    make_interface,
)

# importing registers all robosuite interfaces
import dexmimicgen.env_interfaces.robosuite  # noqa: F401,E402

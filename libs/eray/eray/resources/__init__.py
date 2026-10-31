# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware types, Ray resource specs, and accelerator configurations."""

from .configs import (
    AcceleratorConfigType,
    ComputeResourceConfig,
    CpuAcceleratorConfig,
    GpuAcceleratorConfig,
    TpuAcceleratorConfig,
)
from .hardware import HardwareType
from .ray_resources import RayResources, available_cpu_cores
from .topology import (
    DEFAULT_HOST_CHIP_GRIDS,
    DEFAULT_SPLIT_BASE_PORT,
    HostPartitionPlan,
    HostSplitPlan,
    plan_host_partition,
    plan_host_split,
)

__all__ = (
    "DEFAULT_HOST_CHIP_GRIDS",
    "DEFAULT_SPLIT_BASE_PORT",
    "AcceleratorConfigType",
    "ComputeResourceConfig",
    "CpuAcceleratorConfig",
    "GpuAcceleratorConfig",
    "HardwareType",
    "HostPartitionPlan",
    "HostSplitPlan",
    "RayResources",
    "TpuAcceleratorConfig",
    "available_cpu_cores",
    "plan_host_partition",
    "plan_host_split",
)

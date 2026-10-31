# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Executor module for distributed computing with Ray and SLURM.

This module provides utilities for managing distributed execution of workloads
across various compute environments, including:

- Ray-based distributed execution with support for TPUs, GPUs, and CPUs
- SLURM cluster integration for HPC environments
- Automatic cluster discovery and initialization
- Resource management and allocation

Key Components:
    - **DistributedConfig**: Configuration for distributed JAX execution
    - **RayClusterConfig**: Configuration for Ray cluster initialization
    - **auto_ray_cluster**: Automatic Ray cluster setup and discovery
    - **eSlurmCluster**: Extended SLURM cluster implementation
    - **ray**: Submodule containing Ray-specific executor utilities

Example:
    Basic distributed setup:

    >>> from eformer.executor import DistributedConfig, auto_ray_cluster
    >>>
    >>> # Initialize distributed JAX
    >>> config = DistributedConfig()
    >>> config.initialize()
    >>>
    >>> # Set up Ray cluster
    >>> auto_ray_cluster()
"""

from . import ray
from .cluster_util import DistributedConfig, RayClusterConfig, auto_ray_cluster, eSlurmCluster
from .ray import (
    ComputeResourceConfig,
    CpuAcceleratorConfig,
    GpuAcceleratorConfig,
    TpuAcceleratorConfig,
    autoscale_execute,
    autoscale_execute_resumable,
    execute,
    execute_resumable,
)

__all__ = (
    "ComputeResourceConfig",
    "CpuAcceleratorConfig",
    "DistributedConfig",
    "GpuAcceleratorConfig",
    "RayClusterConfig",
    "TpuAcceleratorConfig",
    "auto_ray_cluster",
    "autoscale_execute",
    "autoscale_execute_resumable",
    "eSlurmCluster",
    "execute",
    "execute_resumable",
    "ray",
)

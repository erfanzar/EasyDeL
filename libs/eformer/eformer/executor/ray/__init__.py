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


"""Ray-based distributed execution framework for EasyDeL/eFormer.

This module provides a comprehensive framework for distributed execution of
machine learning workloads using Ray. It supports various hardware accelerators
(CPU, GPU, TPU), Docker containerization, resource management, and fault-tolerant
execution patterns.

Key Components:
    - **Execution**: Functions for running distributed jobs with automatic
      retries and resumption capabilities
    - **Docker Support**: Tools for building, pushing, and running Docker
      containers on distributed nodes
    - **Resource Management**: Flexible resource allocation and pooling for
      efficient hardware utilization
    - **Type System**: Rich type definitions for job status, errors, and
      execution metadata

Example:
    Basic distributed execution:

    >>> from eformer.executor.ray import execute, ComputeResourceConfig
    >>>
    >>> config = ComputeResourceConfig(
    ...     cpu_cores=4,
    ...     memory_gb=16,
    ...     accelerator=GpuAcceleratorConfig(count=1, type="v100")
    ... )
    >>>
    >>> result = execute(
    ...     my_function,
    ...     args=(data,),
    ...     resources=config
    ... )

    Multi-slice TPU execution with Docker:

    >>> from eformer.executor.ray import run_docker_multislice, DockerConfig
    >>>
    >>> docker_config = DockerConfig(
    ...     image="my-ml-image:latest",
    ...     registry="gcr.io/my-project"
    ... )
    >>>
    >>> run_docker_multislice(
    ...     command="python train.py",
    ...     docker_config=docker_config,
    ...     num_slices=4,
    ...     tpu_type="v4-32"
    ... )
"""

from .docker_executor import (
    DockerConfig,
    build_and_push_docker_image,
    make_docker_run_command,
    run_docker_async,
    run_docker_multislice,
    run_docker_on_pod,
)
from .executor import (
    RayExecutor,
    autoscale_execute,
    autoscale_execute_resumable,
    device_remote,
    execute,
    execute_resumable,
)
from .pool_manager import (
    ActorPoolMember,
    DeviceHostActor,
    ResourcePoolManager,
    SlicePoolManager,
)
from .resource_manager import (
    AcceleratorConfigType,
    ComputeResourceConfig,
    CpuAcceleratorConfig,
    GpuAcceleratorConfig,
    HardwareType,
    RayResources,
    TpuAcceleratorConfig,
    available_cpu_cores,
)
from .types import (
    DONE,
    DoneSentinel,
    ExceptionInfo,
    JobError,
    JobFailed,
    JobInfo,
    JobPreempted,
    JobStatus,
    JobSucceeded,
    MultisliceInfo,
    RefBox,
    SliceInfo,
    SnitchRecipient,
    StopwatchActor,
    current_actor_handle,
    handle_ray_error,
    log_failures_to,
    print_remote_raise,
)

__all__ = (
    "DONE",
    "AcceleratorConfigType",
    "ActorPoolMember",
    "ComputeResourceConfig",
    "CpuAcceleratorConfig",
    "DeviceHostActor",
    "DockerConfig",
    "DoneSentinel",
    "ExceptionInfo",
    "GpuAcceleratorConfig",
    "HardwareType",
    "JobError",
    "JobFailed",
    "JobInfo",
    "JobPreempted",
    "JobStatus",
    "JobSucceeded",
    "MultisliceInfo",
    "RayExecutor",
    "RayResources",
    "RefBox",
    "ResourcePoolManager",
    "SliceInfo",
    "SlicePoolManager",
    "SnitchRecipient",
    "StopwatchActor",
    "TpuAcceleratorConfig",
    "autoscale_execute",
    "autoscale_execute_resumable",
    "available_cpu_cores",
    "build_and_push_docker_image",
    "current_actor_handle",
    "device_remote",
    "execute",
    "execute_resumable",
    "handle_ray_error",
    "log_failures_to",
    "make_docker_run_command",
    "print_remote_raise",
    "run_docker_async",
    "run_docker_multislice",
    "run_docker_on_pod",
)

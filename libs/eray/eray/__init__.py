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


"""eray — Ray-based distributed execution and scaling framework.

A standalone extraction of the Ray executor stack from eFormer/EasyDeL.
Provides distributed execution of machine learning workloads using Ray,
with support for CPUs, GPUs, TPUs, Docker containerization, resource
pooling, multi-slice coordination, and fault-tolerant execution.

Subpackages:
    core:       Job status types, exceptions, sentinels, monitoring, cluster info
    resources:  Hardware constants, Ray resource specs, accelerator configs
    pool:       Actor pool management, device hosts, multi-slice coordination
    execution:  RayExecutor engine, decorators, device-remote machinery
    docker:     Docker container config and execution

Example:
    Basic distributed execution:

    >>> from eray import execute, GpuAcceleratorConfig
    >>>
    >>> @execute(GpuAcceleratorConfig(count=1, gpu_model="A100"))
    ... def train_step(batch):
    ...     ...
    ...     return loss

    Multi-slice TPU execution with Docker:

    >>> from eray import run_docker_multislice, DockerConfig, TpuAcceleratorConfig
    >>>
    >>> docker_config = DockerConfig(
    ...     image="my-ml-image:latest",
    ...     command="python train.py",
    ... )
    >>> run_docker_multislice(
    ...     docker_config,
    ...     TpuAcceleratorConfig(tpu_version="v4-32", pod_count=4),
    ... )
"""

from ray import init
from ray.runtime_env import RuntimeEnv

from .core import (
    DONE,
    DoneSentinel,
    ExceptionInfo,
    HostInfo,
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
    start_raylet_log_guard,
    sweep_raylet_logs,
)
from .docker import (
    DockerConfig,
    build_and_push_docker_image,
    make_docker_run_command,
    run_docker_async,
    run_docker_multislice,
    run_docker_on_pod,
)
from .execution import (
    ENV_CALL_INDEX,
    ENV_CALL_SLICE,
    MEGASCALE_DEFAULT_PORT,
    RayExecutor,
    autoscale_execute,
    autoscale_execute_resumable,
    device_remote,
    execute,
    execute_resumable,
    resolve_maybe_refs,
)
from .pool import (
    ActorPoolMember,
    DeviceHostActor,
    InsufficientSlicesError,
    ResourcePoolManager,
    SliceActor,
    SlicePoolManager,
)
from .provision import ensure_tpu, watch_and_reconnect
from .resources import (
    AcceleratorConfigType,
    ComputeResourceConfig,
    CpuAcceleratorConfig,
    GpuAcceleratorConfig,
    HardwareType,
    HostPartitionPlan,
    HostSplitPlan,
    RayResources,
    TpuAcceleratorConfig,
    available_cpu_cores,
    plan_host_partition,
    plan_host_split,
)
from .swarm import GpuSwarmConfig, SwarmConfig, SwarmRun, shutdown_swarm, swarm_execute, swarmed

__version__ = "0.1.0"
__all__ = (
    "DONE",
    "ENV_CALL_INDEX",
    "ENV_CALL_SLICE",
    "MEGASCALE_DEFAULT_PORT",
    "AcceleratorConfigType",
    "ActorPoolMember",
    "ComputeResourceConfig",
    "CpuAcceleratorConfig",
    "DeviceHostActor",
    "DockerConfig",
    "DoneSentinel",
    "ExceptionInfo",
    "GpuAcceleratorConfig",
    "GpuSwarmConfig",
    "HardwareType",
    "HostInfo",
    "HostPartitionPlan",
    "HostSplitPlan",
    "InsufficientSlicesError",
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
    "RuntimeEnv",
    "SliceActor",
    "SliceInfo",
    "SlicePoolManager",
    "SnitchRecipient",
    "StopwatchActor",
    "SwarmConfig",
    "SwarmRun",
    "TpuAcceleratorConfig",
    "autoscale_execute",
    "autoscale_execute_resumable",
    "available_cpu_cores",
    "build_and_push_docker_image",
    "current_actor_handle",
    "device_remote",
    "ensure_tpu",
    "execute",
    "execute_resumable",
    "handle_ray_error",
    "init",
    "log_failures_to",
    "make_docker_run_command",
    "plan_host_partition",
    "plan_host_split",
    "print_remote_raise",
    "resolve_maybe_refs",
    "run_docker_async",
    "run_docker_multislice",
    "run_docker_on_pod",
    "shutdown_swarm",
    "start_raylet_log_guard",
    "swarm_execute",
    "swarmed",
    "sweep_raylet_logs",
    "watch_and_reconnect",
)

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


"""Accelerator configurations for CPU, GPU, and TPU."""

import typing as tp
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import ray
from ray._private.accelerators import NvidiaGPUAcceleratorManager, TPUAcceleratorManager
from ray.remote_function import RemoteFunction
from ray.runtime_env import RuntimeEnv

from .ray_resources import RayResources, available_cpu_cores


def _safe_tpu_chips_per_host() -> int:
    """Best-effort TPU chip count lookup with a safe fallback for non-TPU nodes.

    Queries the TPUAcceleratorManager to determine the number of TPU chips
    available on the current node. Designed to work safely in mixed environments
    where some nodes may not have TPUs.

    Returns:
        int: Number of TPU chips on the current host, or 0 if detection fails
            or the node does not have TPUs.
    """
    try:
        count = TPUAcceleratorManager.get_current_node_num_accelerators()
    except Exception:
        return 0
    return int(count or 0)


def _safe_tpu_worker_count() -> int:
    """Best-effort TPU worker count with a safe fallback for non-TPU drivers.

    Queries the Ray TPU utilities to determine the number of workers in
    the current TPU pod. Returns a safe default for non-TPU environments.

    Returns:
        int: Number of TPU workers in the current pod, or 1 if detection fails
            or the node is not part of a TPU pod.
    """
    try:
        count = ray.util.accelerators.tpu.get_current_pod_worker_count()
    except Exception:
        return 1
    return int(count) if count else 1


def _safe_tpu_runtime_name() -> str:
    """Best-effort TPU pod name with a stable fallback.

    Queries the Ray TPU utilities to get the current TPU pod name. If the
    name cannot be determined (e.g., not running on a TPU pod), generates
    a unique UUID as a fallback.

    Returns:
        str: The TPU pod name if available, otherwise a randomly generated UUID.
    """
    try:
        name = ray.util.accelerators.tpu.get_current_pod_name()
    except Exception:
        name = None
    return name or str(uuid.uuid4())

class ComputeResourceConfig(Protocol):
    """Protocol defining the interface for hardware resource configurations.

    This protocol establishes a standardized contract that all resource configuration
    classes must implement. It ensures consistency across different accelerator types
    (CPU, GPU, TPU) and provides the necessary methods for Ray task and actor
    deployment. The implementations are primarily used for distributed training
    and inference workloads.

    The protocol defines both required attributes and methods that enable:
    - Resource specification conversion to Ray formats
    - Runtime environment management
    - Hardware-specific configuration handling
    - Remote function decoration with appropriate resources

    Attributes:
        execution_env (RuntimeEnv): Ray runtime environment configuration.
        head_name (str | None): Optional identifier for the head node.
        head_workers (int): Number of workers on the head node.

    Example:
        >>> def deploy_model(config: ComputeResourceConfig):
        ...     remote_options = config.get_remote_options()
        ...     @ray.remote(**remote_options)
        ...     class ModelServer:
        ...         def predict(self, data): pass
        ...     return ModelServer
    """

    execution_env: RuntimeEnv
    head_name: str | None = None
    head_workers: int = 1

    def hardware_identifier(self) -> str | None:
        """Get the identifier for the hardware accelerator being used.

        This method returns a string identifier that specifies the exact hardware
        accelerator type or model required for the computation. The identifier
        typically corresponds to values from the HardwareType class.

        Returns:
            str | None: String identifier for the hardware accelerator (e.g., "A100",
                "TPU-V4") or None if no specific accelerator is required.

        Example:
            >>> config = GpuAcceleratorConfig(gpu_model="A100")
            >>> print(config.hardware_identifier())
        """
        return None

    def get_remote_options(self) -> dict[str, Any]:
        """Get keyword arguments for ray.remote() based on this resource configuration.

        This method converts the resource configuration into a dictionary format
        that can be directly passed to Ray's remote decorator. It handles the
        translation between high-level resource specifications and Ray's internal
        resource format.

        Returns:
            dict[str, Any]: Dictionary of arguments suitable for passing to ray.remote().
                Common keys include num_cpus, num_gpus, resources, runtime_env,
                and accelerator_type.

        Example:
            >>> config = GpuAcceleratorConfig(device_count=2, cpu_count=4)
            >>> options = config.get_remote_options()
            >>> @ray.remote(**options)
            ... def gpu_task():
            ...     return "Task complete"
        """
        return self.to_ray_resources().to_kwargs()

    def to_ray_resources(self) -> RayResources:
        """Convert this configuration to a RayResources object.

        This method transforms the configuration into a RayResources instance,
        which provides a standardized representation of resource requirements
        that can be used across different parts of the system.

        Returns:
            RayResources: RayResources instance representing the hardware resources.
                Contains all necessary information for Ray task/actor deployment.

        Example:
            >>> config = CpuAcceleratorConfig(core_count=4)
            >>> resources = config.to_ray_resources()
            >>> print(resources.num_cpus)
        """
        ...

    def create_remote_decorator(self) -> Callable[[Any], Any]:
        """Create a ray.remote decorator with this resource configuration.

        This convenience method creates a pre-configured Ray remote decorator
        that includes all the resource specifications from this configuration.
        The decorator can then be applied to functions or classes to make them
        Ray remote operations.

        Returns:
            Callable[[Any], Any]: A ray.remote decorator that can be applied to
                functions or classes. The decorator includes all resource requirements
                from this configuration.

        Example:
            >>> config = GpuAcceleratorConfig(device_count=1)
            >>> remote_decorator = config.create_remote_decorator()
            >>> @remote_decorator
            ... def train_model():
            ...     return "Training complete"
            >>> result = ray.get(train_model.remote())
        """
        return ray.remote(**self.get_remote_options())

    def with_environment_variables(
        self,
        env_vars: dict[str, str] | None = None,
        /,
        **kwargs,
    ) -> "ComputeResourceConfig":
        """Create a new resource configuration with additional environment variables.

        This method allows for adding or overriding environment variables without
        modifying other aspects of the resource configuration. It creates a new
        configuration instance with updated environment variables, preserving
        immutability of the original configuration.

        Args:
            env_vars (dict[str, str] | None): Dictionary of environment variables
                to add or override. If None, only kwargs are used.
            **kwargs: Additional environment variables as keyword arguments.
                These are merged with env_vars if provided.

        Returns:
            ComputeResourceConfig: A new ComputeResourceConfig instance with the
                combined environment variables from existing config, env_vars, and kwargs.

        Example:
            >>> config = CpuAcceleratorConfig()
            >>> new_config = config.with_environment_variables(
            ...     {'CUDA_VISIBLE_DEVICES': '0,1'},
            ...     OMP_NUM_THREADS='4'
            ... )
        """
        current_env_vars = self.execution_env.get("env_vars", {})
        new_env_vars = {**current_env_vars, **(env_vars or {}), **kwargs}
        updated_env = RuntimeEnv(**{**self.execution_env, "env_vars": new_env_vars})
        return replace(self, execution_env=updated_env)

@dataclass(frozen=True)
class CpuAcceleratorConfig(ComputeResourceConfig):
    """Resource configuration for CPU-only workloads.

    This configuration is designed for computational tasks that rely solely on
    CPU processing power. It's suitable for local development, batch processing,
    data preprocessing, or any tasks that don't require specialized hardware
    acceleration like GPUs or TPUs.

    The configuration automatically detects available CPU cores and provides
    sensible defaults for CPU-bound workloads. It can be used for distributed
    CPU computing across multiple nodes in a Ray cluster.

    Attributes:
        core_count (int): Number of CPU cores to allocate. Defaults to all available cores.
        execution_env (RuntimeEnv): Ray runtime environment for dependencies and setup.
        resource_name (str): Name identifier for the resource type (default: "CPU").
        runtime_name (str): Unique runtime identifier for this configuration.
        worker_count (int): Number of worker processes to spawn.

    Example:
        >>> config = CpuAcceleratorConfig(core_count=4, worker_count=2)
        >>> @config.create_remote_decorator()
        ... def cpu_intensive_task(data):
        ...     return process_data_on_cpu(data)
        >>> result = ray.get(cpu_intensive_task.remote(my_data))
    """

    core_count: int = field(default_factory=available_cpu_cores)
    execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
    resource_name: str = field(default="CPU")
    runtime_name: str = field(default_factory=uuid.uuid4)
    worker_count: int = 1

    def hardware_identifier(self) -> str | None:
        """Get the hardware identifier (none for CPU-only configuration).

        For CPU-only configurations, there is no specialized hardware accelerator,
        so this method always returns None to indicate that any available CPU
        resources can be used.

        Returns:
            None: Always None since no specialized hardware accelerator is used.

        Example:
            >>> config = CpuAcceleratorConfig()
            >>> print(config.hardware_identifier())
        """
        return None

    def get_remote_options(self) -> dict[str, Any]:
        """Get Ray remote options for CPU-only execution.

        This method returns the Ray remote options specifically configured for
        CPU-only tasks, including the CPU core count and runtime environment.
        GPU allocation is explicitly set to 0.

        Returns:
            dict[str, Any]: Dictionary of options for ray.remote() containing:
                - num_cpus: Number of CPU cores to allocate
                - runtime_env: Runtime environment configuration

        Example:
            >>> config = CpuAcceleratorConfig(core_count=2)
            >>> options = config.get_remote_options()
            >>> print(options)
            {'num_cpus': 2, 'runtime_env': {}}
        """
        return {"num_cpus": self.core_count, "runtime_env": self.execution_env}

    def to_ray_resources(self) -> RayResources:
        """Convert to Ray resource specifications for CPU-only allocation.

        This method creates a RayResources object that represents CPU-only
        resource allocation with no GPU or specialized accelerator requirements.

        Returns:
            RayResources: RayResources object representing CPU-only allocation
                with the specified core count, zero GPUs, and runtime environment.

        Example:
            >>> config = CpuAcceleratorConfig(core_count=8)
            >>> resources = config.to_ray_resources()
            >>> print(f"CPUs: {resources.num_cpus}, GPUs: {resources.num_gpus}")
            CPUs: 8, GPUs: 0
        """
        return RayResources(
            num_cpus=self.core_count,
            num_gpus=0,
            runtime_env=self.execution_env,
        )

    def redecorate_remote_fn_for_call(
        self,
        remote_fn: RemoteFunction | tp.Callable,
        **extra_envs,
    ):
        """Prepare a remote function for CPU execution with merged runtime environment.

        Wraps a remote function with CPU-specific resource requirements and
        runtime environment configuration. The function is forkified to run
        in a separate process for isolation.

        Args:
            remote_fn: The remote function or callable to configure.
            **extra_envs: Additional environment variables to merge into
                the runtime environment.

        Returns:
            RemoteFunction: Configured remote function with CPU resources and
                merged runtime environment.
        """
        remote_fn = RayResources.forkify_remote_fn(remote_fn)
        if not isinstance(remote_fn, RemoteFunction):
            remote_fn = ray.remote(remote_fn)

        runtime_env = RayResources.update_fn_resource_env(
            remote_fn=remote_fn,
            runtime_env=self.execution_env,
            **extra_envs,
        )

        return remote_fn.options(num_cpus=self.core_count, runtime_env=runtime_env)

@dataclass(frozen=True)
class GpuAcceleratorConfig(ComputeResourceConfig):
    """Resource configuration for GPU-accelerated workloads.

    This configuration specifies GPU requirements for computationally intensive
    tasks such as neural network training, inference, and other parallel computing
    workloads that benefit from GPU acceleration. It supports both generic GPU
    allocation and specific GPU model requirements.

    The configuration automatically detects available GPU resources on the current
    node and provides flexible options for multi-GPU setups. It's designed to work
    with NVIDIA GPUs through Ray's GPU resource management system.

    Attributes:
        device_count (int): Number of GPU devices to allocate per task/actor.
        execution_env (RuntimeEnv): Ray runtime environment for CUDA/GPU dependencies.
        gpu_model (str | None): Specific GPU model identifier (e.g., "A100", "V100").
        cpu_count (int): Number of CPU cores to allocate alongside GPUs.
        chips_per_host (int): Number of GPU devices available per host node.
        runtime_name (str): Unique runtime identifier for this configuration.
        worker_count (int): Number of worker processes to spawn.
        resource_name (str): Name identifier for the resource type.

    Example:
        >>> config = GpuAcceleratorConfig(
        ...     device_count=2,
        ...     gpu_model=HardwareType.NVIDIA_A100,
        ...     cpu_count=8
        ... )
        >>> @config.create_remote_decorator()
        ... def train_model(data):
        ...     return gpu_training_function(data)
        >>> result = ray.get(train_model.remote(training_data))
    """

    device_count: int = 1
    execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
    gpu_model: str | None = None
    cpu_count: int = 1
    chips_per_host: int = field(default_factory=NvidiaGPUAcceleratorManager.get_current_node_num_accelerators)
    runtime_name: str = field(default_factory=uuid.uuid4)
    worker_count: int = 1
    resource_name: str = field(default="GPU")

    def hardware_identifier(self) -> str | None:
        """Get the hardware identifier for the GPU model.

        This method returns the specific GPU model identifier if one was specified
        during configuration. This is used by Ray's accelerator management system
        to ensure tasks are scheduled on nodes with the required GPU hardware.

        Returns:
            str | None: String identifier for the GPU model (e.g., "A100", "V100")
                or None if any available GPU is acceptable.

        Example:
            >>> config = GpuAcceleratorConfig(gpu_model="A100")
            >>> print(config.hardware_identifier())
            >>> generic_config = GpuAcceleratorConfig()
            >>> print(generic_config.hardware_identifier())
        """
        return self.gpu_model

    def get_remote_options(self) -> dict[str, Any]:
        """Get Ray remote options for GPU-accelerated execution.

        This method constructs the Ray remote options dictionary for GPU-accelerated
        tasks, including CPU and GPU allocation, runtime environment, and optionally
        specific accelerator type requirements.

        Returns:
            dict[str, Any]: Dictionary of options for ray.remote() containing:
                - num_cpus: Number of CPU cores to allocate
                - num_gpus: Number of GPU devices to allocate
                - runtime_env: Runtime environment configuration
                - accelerator_type: Specific GPU model (if specified)

        Example:
            >>> config = GpuAcceleratorConfig(device_count=1, cpu_count=4, gpu_model="A100")
            >>> options = config.get_remote_options()
            >>> print(options)
            {'num_cpus': 4, 'num_gpus': 1, 'runtime_env': {}, 'accelerator_type': 'A100'}
        """
        remote_options = {
            "num_cpus": self.cpu_count,
            "num_gpus": self.device_count,
            "runtime_env": self.execution_env,
        }

        if self.gpu_model is not None:
            remote_options["accelerator_type"] = self.gpu_model

        return remote_options

    def to_ray_resources(self) -> RayResources:
        """Convert to Ray resource specifications for GPU allocation.

        This method creates a RayResources object that represents GPU resource
        allocation with the specified number of GPUs, CPUs, accelerator type,
        and runtime environment configuration.

        Returns:
            RayResources: RayResources object representing GPU resource allocation
                with specified device count, CPU count, and optional accelerator type.

        Example:
            >>> config = GpuAcceleratorConfig(device_count=2, cpu_count=8)
            >>> resources = config.to_ray_resources()
            >>> print(f"CPUs: {resources.num_cpus}, GPUs: {resources.num_gpus}")
            CPUs: 8, GPUs: 2
        """
        return RayResources(
            num_cpus=self.cpu_count,
            num_gpus=self.device_count,
            accelerator_type=self.gpu_model,
            runtime_env=self.execution_env,
        )

    def redecorate_remote_fn_for_call(
        self,
        remote_fn: RemoteFunction | tp.Callable,
        **extra_envs,
    ):
        """Prepare a remote function for GPU execution with merged runtime environment.

        Wraps a remote function with GPU-specific resource requirements and
        runtime environment configuration. The function is forkified to run
        in a separate process for isolation.

        Args:
            remote_fn: The remote function or callable to configure.
            **extra_envs: Additional environment variables to merge into
                the runtime environment.

        Returns:
            RemoteFunction: Configured remote function with GPU resources,
                optional accelerator type, and merged runtime environment.
        """
        remote_fn = RayResources.forkify_remote_fn(remote_fn)
        if not isinstance(remote_fn, RemoteFunction):
            remote_fn = ray.remote(remote_fn)

        runtime_env = RayResources.update_fn_resource_env(
            remote_fn=remote_fn,
            runtime_env=self.execution_env,
            **extra_envs,
        )

        remote_options = {
            "num_cpus": self.cpu_count,
            "num_gpus": self.device_count,
            "runtime_env": runtime_env,
        }
        if self.gpu_model is not None:
            remote_options["accelerator_type"] = self.gpu_model

        return remote_fn.options(**remote_options)

@dataclass(frozen=True)
class TpuAcceleratorConfig(ComputeResourceConfig):
    """Resource configuration for TPU-accelerated workloads.

    This configuration is designed for large-scale machine learning tasks using
    Google's Tensor Processing Units (TPUs). TPUs are particularly well-suited
    for transformer models, large neural networks, and other matrix-heavy
    computations that benefit from TPU's specialized architecture.

    The configuration handles TPU pod management, resource allocation, and
    integration with Ray's distributed computing framework. It supports various
    TPU versions and pod configurations for different computational requirements.

    Attributes:
        tpu_version (str): TPU version identifier (e.g., "TPU-V4", "TPU-V5P").
        pod_count (int): Number of TPU pods to allocate for the task.
        execution_env (RuntimeEnv): Ray runtime environment for TPU dependencies.
        cpu_count (int): Number of CPU cores to allocate alongside TPUs.
        chips_per_host (int): Number of TPU chips available per host.
        worker_count (int): Number of worker processes for distributed TPU training.
        runtime_name (str): TPU pod name identifier.
        resource_name (str): Resource type identifier for TPU resources.

    Example:
        >>> config = TpuAcceleratorConfig(
        ...     tpu_version=HardwareType.GOOGLE_TPU_V4,
        ...     pod_count=1,
        ...     cpu_count=4
        ... )
        >>> @config.create_remote_decorator()
        ... def train_transformer(model_config):
        ...     return tpu_training_loop(model_config)
        >>> result = ray.get(train_transformer.remote(config))
    """

    tpu_version: str
    pod_count: int = 1
    execution_env: RuntimeEnv = field(default_factory=RuntimeEnv)
    cpu_count: int = 2
    chips_per_host: int = field(default_factory=_safe_tpu_chips_per_host)
    worker_count: int = field(default_factory=_safe_tpu_worker_count)
    runtime_name: str = field(default_factory=_safe_tpu_runtime_name)
    resource_name: str = field(default="TPU")

    def hardware_identifier(self) -> str:
        """Get the hardware identifier for the TPU configuration.

        This method returns the TPU version identifier that specifies the exact
        TPU hardware generation and capabilities required for the computation.
        This ensures tasks are scheduled on appropriate TPU resources.

        Returns:
            str: String identifier for the TPU version (e.g., "TPU-V4", "TPU-V5P").
                This corresponds to the tpu_version attribute.

        Example:
            >>> config = TpuAcceleratorConfig(tpu_version="TPU-V4")
            >>> print(config.hardware_identifier())
        """
        return self.tpu_version

    def get_remote_options(self) -> dict[str, Any]:
        """Get Ray remote options for TPU-accelerated execution.

        This method constructs the Ray remote options dictionary for TPU-accelerated
        tasks. TPU resources are specified using Ray's custom resource system,
        with the TPU version as the resource name and pod count as the quantity.

        Returns:
            dict[str, Any]: Dictionary of options for ray.remote() containing:
                - num_cpus: Number of CPU cores to allocate
                - resources: Custom resource specification for TPU (version: count)
                - runtime_env: Runtime environment configuration

        Example:
            >>> config = TpuAcceleratorConfig(tpu_version="TPU-V4", pod_count=1, cpu_count=2)
            >>> options = config.get_remote_options()
            >>> print(options)
            {'num_cpus': 2, 'resources': {'TPU-V4': 1}, 'runtime_env': {}}
        """
        return {
            "num_cpus": self.cpu_count,
            "resources": {self.tpu_version: self.pod_count},
            "runtime_env": self.execution_env,
        }

    def to_ray_resources(self) -> RayResources:
        """Convert to Ray resource specifications for TPU resources.

        This method creates a RayResources object that represents TPU resource
        allocation using Ray's custom resource system. TPU resources are specified
        as custom resources with the TPU version as the key.

        Returns:
            RayResources: RayResources object representing TPU resource allocation
                with specified CPU count, TPU version and pod count as custom resources,
                and runtime environment.

        Example:
            >>> config = TpuAcceleratorConfig(tpu_version="TPU-V4", pod_count=2)
            >>> resources = config.to_ray_resources()
            >>> print(resources.resources)
            {'TPU-V4': 2.0}
        """
        return RayResources(
            num_cpus=self.cpu_count,
            resources={self.tpu_version: float(self.pod_count)},
            runtime_env=self.execution_env,
        )

    def redecorate_remote_fn_for_call(
        self,
        remote_fn: RemoteFunction | tp.Callable,
        **extra_envs,
    ):
        """Redecorate a remote function with TPU-specific resource requirements.

        This method applies TPU-specific configuration to a remote function,
        including process isolation (forkification), TPU pod resource allocation,
        and runtime environment updates. It's specifically designed for TPU
        workloads that require special handling.

        Args:
            remote_fn (RemoteFunction | tp.Callable): The remote function or callable
                to be configured for TPU execution.
            **extra_envs: Additional environment variables to merge into the
                runtime environment.

        Returns:
            RemoteFunction: A reconfigured remote function with TPU resource
                requirements and updated runtime environment.

        Example:
            >>> config = TpuAcceleratorConfig(tpu_version="TPU-V4")
            >>> def my_tpu_function():
            ...     return "TPU computation"
            >>> tpu_fn = config.redecorate_remote_fn_for_call(
            ...     my_tpu_function,
            ...     JAX_PLATFORMS='tpu'
            ... )
            >>> result = ray.get(tpu_fn.remote())
        """
        remote_fn = RayResources.forkify_remote_fn(remote_fn)
        if not isinstance(remote_fn, RemoteFunction):
            remote_fn = ray.remote(remote_fn)

        tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
        runtime_env = RayResources.update_fn_resource_env(
            remote_fn=remote_fn,
            runtime_env=self.execution_env,
            **extra_envs,
        )
        remote_fn = remote_fn.options(
            runtime_env=runtime_env,
            resources={tpu_name: 1, self.resource_name: self.chips_per_host},
        )
        return remote_fn

AcceleratorConfigType: tp.TypeAlias = TpuAcceleratorConfig | GpuAcceleratorConfig | CpuAcceleratorConfig

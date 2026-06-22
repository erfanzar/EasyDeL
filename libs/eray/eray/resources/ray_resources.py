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


"""Ray resource specification and process isolation."""

import functools
import logging
import multiprocessing
import os
import typing as tp
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from typing import Any

import mergedeep
import ray
from ray.remote_function import RemoteFunction
from ray.runtime_env import RuntimeEnv

from ..core.exceptions import ExceptionInfo

logger = logging.getLogger("ray")

@dataclass
class RayResources:
    """A representation of resource requirements for Ray tasks and actors.

    This dataclass encapsulates all resource specifications needed when creating
    Ray tasks or actors, allowing for easy conversion between different resource
    representation formats used by Ray. It provides methods for converting between
    Ray's internal resource representation and user-friendly specifications.

    Attributes:
        num_cpus: Number of CPU cores to allocate for the task/actor.
        num_gpus: Number of GPU devices to allocate for the task/actor.
        resources: Custom resource requirements as name-value pairs.
        runtime_env: Ray runtime environment configuration for dependencies.
        accelerator_type: Specific accelerator type identifier (e.g., "A100").

    Example:
        >>> resources = RayResources(
        ...     num_cpus=4,
        ...     num_gpus=2,
        ...     accelerator_type="A100"
        ... )
        >>> kwargs = resources.to_kwargs()
        >>> @ray.remote(**kwargs)
        ... def my_task():
        ...     return "Hello from Ray!"
    """

    num_cpus: int = 1
    num_gpus: int = 0
    resources: dict[str, float] = field(default_factory=dict)
    runtime_env: RuntimeEnv = field(default_factory=RuntimeEnv)
    accelerator_type: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        """Convert resource specifications to kwargs for ray.remote() decorator.

        This method transforms the resource specifications into a format directly
        compatible with Ray's remote decorator, handling all the necessary parameter
        mapping and filtering.

        Returns:
            dict[str, Any]: Dictionary of keyword arguments compatible with ray.remote().
                Includes num_cpus, num_gpus, resources, runtime_env, and optionally
                accelerator_type if specified.

        Example:
            >>> resources = RayResources(num_cpus=2, num_gpus=1)
            >>> kwargs = resources.to_kwargs()
            >>> print(kwargs)
            {'num_cpus': 2, 'num_gpus': 1, 'resources': {}, 'runtime_env': {}}
        """
        remote_kwargs = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "resources": self.resources,
            "runtime_env": self.runtime_env,
        }

        if self.accelerator_type is not None:
            remote_kwargs["accelerator_type"] = self.accelerator_type

        return remote_kwargs

    def to_resource_dict(self) -> dict[str, float]:
        """Convert resource specifications to a dictionary format for resource reporting.

        This method creates a flattened view of all resource requirements, suitable
        for monitoring, logging, and resource visualization tools. It standardizes
        resource names and handles accelerator type encoding.

        Note:
            This is primarily for resource visualization and reporting, not for
            direct use with ray.remote(). For ray.remote(), use to_kwargs() instead.

        Returns:
            dict[str, float]: Dictionary mapping resource names to quantities.
                Standard keys include "CPU", "GPU", and any custom resources.
                Accelerator types are encoded as "accelerator_type:<type>".

        Example:
            >>> resources = RayResources(num_cpus=4, num_gpus=2, accelerator_type="A100")
            >>> resource_dict = resources.to_resource_dict()
            >>> print(resource_dict)
            {'CPU': 4, 'GPU': 2, 'accelerator_type:A100': 0.001}
        """
        resource_dict = {"CPU": self.num_cpus, "GPU": self.num_gpus}
        resource_dict.update(self.resources)

        if self.accelerator_type is not None:
            resource_dict[f"accelerator_type:{self.accelerator_type}"] = 0.001

        return resource_dict

    @staticmethod
    def from_resource_dict(resource_spec: dict[str, float]) -> "RayResources":
        """Create a RayResources instance from a resource dictionary.

        This factory method reconstructs a RayResources object from a flattened
        resource specification dictionary, handling the reverse transformation
        of to_resource_dict().

        Args:
            resource_spec (dict[str, float]): Dictionary mapping resource names to quantities.
                Expected keys include "CPU", "GPU", custom resources, and optionally
                "accelerator_type:<type>" for accelerator specifications.

        Returns:
            RayResources: A new RayResources instance representing the specified resources.

        Example:
            >>> resource_dict = {'CPU': 4, 'GPU': 2, 'accelerator_type:A100': 0.001}
            >>> resources = RayResources.from_resource_dict(resource_dict)
            >>> print(f"CPUs: {resources.num_cpus}, GPUs: {resources.num_gpus}")
            CPUs: 4, GPUs: 2
        """
        resources = dict(resource_spec)
        num_cpus = resources.pop("CPU", 0)
        num_gpus = resources.pop("GPU", 0)

        accelerator_type = None
        accelerator_keys = [k for k in resources.keys() if k.startswith("accelerator_type:")]
        if accelerator_keys:
            accelerator_type = accelerator_keys[0].split(":", 1)[1]
            for key in accelerator_keys:
                resources.pop(key)

        return RayResources(
            num_cpus=int(num_cpus),
            num_gpus=int(num_gpus),
            resources=resources,
            accelerator_type=accelerator_type,
        )

    @staticmethod
    def forkify_remote_fn(remote_fn: RemoteFunction | Callable):
        """Wrap a remote function to execute in a separate process.

        This method transforms a Ray remote function or callable to execute in
        an isolated subprocess, providing additional process isolation and
        error handling capabilities. Useful for functions that may cause
        memory leaks or require process-level isolation.

        Args:
            remote_fn (RemoteFunction | Callable): The remote function or callable
                to be wrapped with process isolation.

        Returns:
            RemoteFunction | functools.partial: The wrapped function that will
                execute in a separate process.

        Example:
            >>> @ray.remote
            ... def my_function(x):
            ...     return x * 2
            >>> forked_fn = RayResources.forkify_remote_fn(my_function)
            >>> result = ray.get(forked_fn.remote(5))
        """
        if isinstance(remote_fn, RemoteFunction):
            fn = remote_fn._function

            @functools.wraps(fn)
            def wrapped_fn(*args, **kwargs):
                return RayResources.separate_process_fn(fn, args, kwargs)

            remote_fn = RemoteFunction(
                language=remote_fn._language,
                function=wrapped_fn,
                function_descriptor=remote_fn._function_descriptor,
                task_options=remote_fn._default_options,
            )
            return remote_fn
        else:
            return functools.partial(RayResources.separate_process_fn, remote_fn)

    @staticmethod
    def separate_process_fn(underlying_function, args, kwargs):
        """Execute a function in a separate subprocess with error handling.

        This method runs the specified function in an isolated subprocess,
        capturing results or exceptions and handling process lifecycle management.
        It provides robust error handling and timeout protection.

        Args:
            underlying_function (Callable): The function to execute in subprocess.
            args (tuple): Positional arguments to pass to the function.
            kwargs (dict): Keyword arguments to pass to the function.

        Returns:
            Any: The return value from the function execution.

        Raises:
            RuntimeError: If the subprocess times out.
            ValueError: If the subprocess execution fails with an exception.

        Example:
            >>> def add(x, y):
            ...     return x + y
            >>> result = RayResources.separate_process_fn(add, (2, 3), {})
            >>> print(result)
        """

        def target_fn(queue, args, kwargs):
            try:
                result = underlying_function(*args, **kwargs)
                queue.put((True, result))
            except Exception as e:
                info = ExceptionInfo.ser_exc_info(e)
                queue.put((False, info))

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
        timeout_s = float(os.getenv("EFORMER_SUBPROCESS_TIMEOUT_S", "1000000"))
        process.start()
        process.join(timeout=timeout_s)
        if process.is_alive():
            logger.error("Process timed out")
            process.terminate()
            process.join(timeout=10)
            raise RuntimeError("Process timed out")

        logger.info("Process finished")
        try:
            success, value = queue.get(timeout=5)
        except QueueEmpty as e:
            logger.error("Process timed out")
            process.terminate()
            raise RuntimeError("Process timed out") from e

        if success:
            return value
        else:
            raise ValueError(value)

    @staticmethod
    def update_fn_resource_env(
        remote_fn: RemoteFunction | tp.Callable,
        runtime_env: dict[str, str] | dict[str, dict[str, str]],
        **extra_env,
    ):
        """Merge runtime environment configurations for a remote function.

        This method combines multiple sources of runtime environment configuration,
        including the function's existing environment, provided runtime_env, and
        additional environment variables. Uses deep merging to handle nested
        configurations properly.

        Args:
            remote_fn (RemoteFunction | tp.Callable): The remote function whose
                runtime environment will be updated.
            runtime_env (dict[str, str] | dict[str, dict[str, str]]): Runtime
                environment configuration to merge.
            **extra_env: Additional environment variables as keyword arguments.

        Returns:
            dict: Merged runtime environment configuration.

        Example:
            >>> @ray.remote
            ... def my_fn():
            ...     return os.getenv('MY_VAR')
            >>> new_env = RayResources.update_fn_resource_env(
            ...     my_fn,
            ...     {'env_vars': {'MY_VAR': 'value1'}},
            ...     MY_OTHER_VAR='value2'
            ... )
        """
        sources = [e for e in [remote_fn._runtime_env, runtime_env, extra_env] if e is not None]
        return mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)

    @staticmethod
    def cancel_all_futures(futures):
        """Cancel all Ray futures in the provided collection.

        This utility method attempts to cancel all Ray futures/ObjectRefs in the
        given iterable, providing error handling for individual cancellation failures.
        Useful for cleanup operations when a batch of tasks needs to be terminated.

        Args:
            futures (Iterable[ray.ObjectRef]): Collection of Ray futures to cancel.

        Note:
            Individual cancellation failures are logged but do not stop the
            cancellation of remaining futures.

        Example:
            >>> futures = [my_remote_fn.remote(i) for i in range(10)]
            >>>
            >>> RayResources.cancel_all_futures(futures)
        """
        for future in futures:
            try:
                ray.cancel(future)
            except Exception:
                logger.exception("Failed to kill job after primary failure")

def available_cpu_cores() -> int:
    """Determine the number of logical CPU cores available on the current system.

    This function checks for SLURM environment variables first (common in HPC
    clusters), then falls back to the system's reported CPU count. It provides
    a reliable way to determine available compute capacity across different
    deployment environments.

    Returns:
        int: Number of available logical CPU cores. Returns 1 as fallback
            if the system doesn't support CPU count detection.

    Example:
        >>> cores = available_cpu_cores()
        >>> print(f"Available CPU cores: {cores}")
        Available CPU cores: 8
    """
    num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
    if num_cpus is not None:
        return int(num_cpus)

    try:
        return os.cpu_count()
    except NotImplementedError:
        return 1

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


"""Convenience decorator wrappers for RayExecutor."""

import functools

from ray.remote_function import RemoteFunction

from ..resources.configs import AcceleratorConfigType
from .executor import RayExecutor


def execute_resumable(accelerator_config: AcceleratorConfigType):
    """Decorator for fault-tolerant single-pod execution.

    Wraps a Ray remote function to automatically use RayExecutor.execute_resumable
    with the specified accelerator configuration. The decorated function will
    automatically retry on preemption or failure according to the default retry
    policies (1,000,000 retries for preemption, 10 for failures).

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources to use for execution. Should have pod_count=1 for
            single-pod execution.

    Returns:
        Callable: Decorator function that wraps the remote function and adds
            automatic retry logic.

    Note:
        To customize retry behavior, use RayExecutor.execute_resumable directly
        with max_retries_preemption and max_retries_failure parameters.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
        >>>
        >>> @execute_resumable(tpu_config)
        >>> @ray.remote
        >>> def my_task(data):
        ...     return process(data)
        >>>
        >>> result = my_task(input_data)
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute_resumable(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def execute(accelerator_config: AcceleratorConfigType):
    """Decorator for single-pod execution without retry.

    Wraps a Ray remote function to automatically use RayExecutor.execute
    with the specified accelerator configuration. Results are automatically
    retrieved with ray.get(). This decorator is suitable for tasks that
    don't require fault tolerance or where failures should be handled
    by the caller.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources to use for execution. Should have pod_count=1 for
            single-pod execution.

    Returns:
        Callable: Decorator function that wraps the remote function and
            automatically retrieves results.

    Note:
        Unlike execute_resumable, this decorator does not retry on failure.
        Use this for quick tasks or when you want to handle failures yourself.

    Example:
        >>> gpu_config = GpuAcceleratorConfig(count=2, type="a100")
        >>>
        >>> @execute(gpu_config)
        >>> @ray.remote
        >>> def gpu_task(tensor):
        ...     return tensor.cuda() * 2
        >>>
        >>> result = gpu_task(my_tensor)
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def autoscale_execute(accelerator_config: AcceleratorConfigType):
    """Decorator for multi-slice execution without retry.

    Wraps a Ray remote function to automatically use RayExecutor.autoscale_execute
    with the specified accelerator configuration. Results from all slices are
    automatically retrieved with ray.get(). The function will be executed
    across multiple TPU slices in parallel, with MegaScale coordination
    automatically configured.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources with multi-slice support. Must have pod_count > 1.

    Returns:
        Callable: Decorator function that wraps the remote function and returns
            a list of results, one from each slice.

    Note:
        The decorator handles slice actor creation, placement group setup,
        and MegaScale environment configuration automatically.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
        >>>
        >>> @autoscale_execute(tpu_config)
        >>> @ray.remote
        >>> def parallel_compute(data_shard):
        ...     return compute_result(data_shard)
        >>>
        >>> results = parallel_compute(sharded_data)
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.autoscale_execute(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def autoscale_execute_resumable(accelerator_config: AcceleratorConfigType):
    """Decorator for fault-tolerant multi-slice execution.

    Wraps a Ray remote function to automatically use RayExecutor.autoscale_execute_resumable
    with the specified accelerator configuration. Provides automatic retry on
    preemption or failure of any slice. Uses an all-or-nothing retry policy:
    if any slice fails, the entire multi-slice execution is retried.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources with multi-slice support. Must have pod_count > 1.

    Returns:
        Callable: Decorator function that wraps the remote function and adds
            automatic retry logic for all slices.

    Note:
        Default retry limits are 1,000,000 for preemptions and 10 for failures.
        To customize these limits, use RayExecutor.autoscale_execute_resumable
        directly with max_retries_preemption and max_retries_failure parameters.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4, preemptible=True)
        >>>
        >>> @autoscale_execute_resumable(tpu_config)
        >>> @ray.remote
        >>> def resilient_training(data_batch):
        ...
        ...     return train_model(data_batch)
        >>>
        >>> results = resilient_training(training_data)
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.autoscale_execute_resumable(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator

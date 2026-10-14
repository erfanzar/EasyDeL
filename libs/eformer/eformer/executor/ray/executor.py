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


"""Ray-based executor for distributed machine learning workloads.

This module provides the core execution framework for running distributed
workloads on accelerators (TPUs, GPUs) using Ray. It supports single-pod,
multi-slice, and fault-tolerant execution patterns with automatic retry
mechanisms.

Key Features:
    - Single-pod and multi-slice execution on TPUs/GPUs
    - Automatic retry mechanisms for preemption and failures
    - Resource management and allocation via Ray
    - Support for both synchronous and asynchronous execution
    - Decorator-based API for easy integration
    - MegaScale coordination for multi-slice TPU workloads
    - Flexible result flattening for multi-host scenarios

Environment Variables:
    - EXECUTOR_CALL_INDEX: Set to worker index within a pod (0-based)
    - EXECUTOR_CALL_SLICE: Set to slice ID for multi-slice execution
    - COORD_PORT: Coordinator port for MegaScale (default: 8192)
    - TPU_NAME, TPU_VERSION, TPU_ZONE: TPU configuration passed to workers
    - MEGASCALE_* variables: Auto-configured for multi-slice coordination

Example:
    Basic single-pod execution:

    >>> import ray
    >>> from eformer.executor.ray import RayExecutor, TpuAcceleratorConfig
    >>>
    >>> @ray.remote
    >>> def train_model(data):
    ...
    ...     return trained_model
    >>>
    >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
    >>> result = RayExecutor.execute_resumable(
    ...     train_model,
    ...     tpu_config,
    ...     max_retries_preemption=10,
    ...     max_retries_failure=3
    ... )

    Multi-slice execution with decorator:

    >>> from eformer.executor.ray import autoscale_execute_resumable
    >>>
    >>> @autoscale_execute_resumable(tpu_config)
    >>> @ray.remote
    >>> def distributed_train(slice_data):
    ...
    ...     return slice_results
    >>>
    >>> results = distributed_train(training_data)

Classes:
    RayExecutor: Core executor with static methods for various execution patterns

Functions:
    execute: Decorator for single-pod execution without retry
    execute_resumable: Decorator for single-pod execution with retry
    autoscale_execute: Decorator for multi-slice execution without retry
    autoscale_execute_resumable: Decorator for multi-slice execution with retry
"""

import functools
import logging
import os
import time
from collections.abc import Callable

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from .pool_manager import InsufficientSlicesError, SlicePoolManager
from .resource_manager import AcceleratorConfigType, RayResources, TpuAcceleratorConfig
from .types import (
    JobError,
    JobFailed,
    JobInfo,
    JobPreempted,
    JobStatus,
    JobSucceeded,
    SliceInfo,
    handle_ray_error,
)

ENV_CALL_INDEX = "EXECUTOR_CALL_INDEX"
ENV_CALL_SLICE = "EXECUTOR_CALL_SLICE"
MEGASCALE_DEFAULT_PORT = 8081

logger = logging.getLogger("ray")


def resolve_maybe_refs(items):
    """Resolve Ray ObjectRefs to their values if present.

    Checks if all items in the provided list are Ray ObjectRefs and
    resolves them using ray.get(). If the items are not ObjectRefs
    or the list is empty, returns them unchanged.

    Args:
        items: List of items that may be Ray ObjectRefs or regular values.

    Returns:
        List of resolved values if input contained ObjectRefs,
        otherwise the original items unchanged.

    Example:
        >>> refs = [task.remote() for task in tasks]
        >>> results = resolve_maybe_refs(refs)
        >>>
        >>> values = [1, 2, 3]
        >>> results = resolve_maybe_refs(values)
    """
    if not items:
        return items
    import ray

    if all(isinstance(x, ray.ObjectRef) for x in items):
        return ray.get(items)
    return items


class RayExecutor:
    """Core executor for Ray-based distributed workloads.

    Provides static methods to execute Ray remote functions on various
    accelerators (TPUs, GPUs) with support for single-pod, multi-slice,
    and fault-tolerant execution patterns.

    This class serves as the main interface for running distributed ML
    workloads with automatic resource allocation, retry mechanisms, and
    failure handling.

    Methods:
        execute: Single-pod execution without retry
        autoscale_execute: Multi-slice execution without retry
        execute_resumable: Single-pod execution with automatic retry
        autoscale_execute_resumable: Multi-slice execution with automatic retry

    All methods return JobStatus objects that encapsulate:
        - JobSucceeded: Successful completion with results
        - JobFailed: Failure due to exceptions
        - JobPreempted: Preemption on preemptible resources
        - JobError: Unexpected errors

    Note:
        All methods are static and can be called directly on the class.
        The class does not maintain state between executions.
    """

    @staticmethod
    def execute(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        **kwargs,
    ):
        """Execute a Ray remote function on a single pod or slice.

        Runs a Ray remote function on a single accelerator pod (TPU/GPU)
        with the specified resource configuration. For multi-slice TPU
        workloads, use autoscale_execute instead.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute.
                Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources (TPU, GPU, or CPU).
            **kwargs: Additional keyword arguments passed to the remote function.

        Returns:
            ray.JobStatus: actual result.

        Raises:
            ValueError: If pod_count in accelerator_config is not 1,
                indicating that autoscale_execute should be used instead.

        Example:
            >>> @ray.remote
            >>> def compute(x):
            ...     return x * 2
            >>>
            >>> config = GpuAcceleratorConfig(count=1, type="v100")
            >>> result = RayExecutor.execute(compute, config, x=10)
        """
        if getattr(accelerator_config, "pod_count", 1) != 1:
            raise ValueError("Multi-slice workloads on TPUs should use 'autoscale_execute'.")

        def do_run(
            remote_fn,
            accelerator_config: AcceleratorConfigType,
            kwargs,
        ) -> JobStatus:
            """Internal function to run the remote function with proper resource allocation.

            This function handles the actual execution of the remote function,
            managing multiple workers if specified in the configuration and
            capturing any errors that occur during execution.

            Args:
                remote_fn: The remote function to execute.
                accelerator_config: Accelerator configuration specifying resources.
                kwargs: Keyword arguments to pass to the remote function.

            Returns:
                JobStatus: Status object indicating:
                    - JobSucceeded: Execution completed successfully with results
                    - JobFailed: Execution failed with an error (non-preemption)
                    - JobPreempted: Execution was preempted (for preemptible instances)

            Note:
                Creates one Ray future per worker as specified in accelerator_config.worker_count.
                Each worker receives an ENV_CALL_INDEX environment variable set to its index.
            """
            info = JobInfo(accelerator_config.runtime_name, "running", accelerator_config.resource_name)
            futures = []
            for idx in range(accelerator_config.worker_count):
                _call = accelerator_config.redecorate_remote_fn_for_call(
                    remote_fn=remote_fn,
                    env_vars={ENV_CALL_INDEX: str(idx)},
                )
                futures.append(_call.remote(**kwargs))
            try:
                out = ray.get(futures)
                return JobSucceeded(info, out)
            except RayError as e:
                RayResources.cancel_all_futures(futures)
                return handle_ray_error(info, e)
            except Exception as e:
                RayResources.cancel_all_futures(futures)
                return JobFailed(info, e)

        if accelerator_config.head_name is None and not isinstance(accelerator_config, TpuAcceleratorConfig):
            do_run = ray.remote(do_run)
        else:
            default_name = f"TPU-{accelerator_config.tpu_version}-head"
            resources = {accelerator_config.head_name or default_name: accelerator_config.head_workers}
            do_run = ray.remote(resources=resources)(do_run)
        return ray.get(do_run.remote(remote_fn, accelerator_config, kwargs))

    @staticmethod
    def autoscale_execute(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        flatten: bool = True,
        **kwargs,
    ) -> JobStatus:
        """Execute a Ray remote function across multiple TPU slices.

        Distributes execution of a remote function across multiple TPU slices
        for large-scale parallel processing. This method sets up the necessary
        infrastructure including slice actors, placement groups, and MegaScale
        coordination environment variables.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute
                on each slice. Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources, must include multi-slice details
                (pod_count > 1).
            flatten (bool): If True (default), returns a flat list of results
                from all hosts across all slices. If False, returns nested
                lists where outer list represents slices and inner lists
                contain results from hosts within each slice.
            **kwargs: Additional keyword arguments passed to the remote
                function on each slice.

        Returns:
            JobStatus: A single JobStatus object containing results from all slices.
                - JobSucceeded: Contains results list (flat or nested based on flatten)
                - JobFailed: Contains the exception that caused the failure
                - JobPreempted: Contains preemption error details
                - JobError: Contains unexpected error information

        Raises:
            InsufficientSlicesError: If requested number of slices cannot be allocated.
            RayError: If slice actor creation fails, coordinator IP cannot
                be determined, or remote function calls fail.
            RuntimeError: If no SliceActors available after scaling or
                coordinator IP cannot be determined.

        Note:
            - The method automatically sets up MegaScale environment variables
              for multi-slice coordination including coordinator address, slice IDs,
              and port configuration.
            - Each slice gets its own SliceActor which manages multiple DeviceHostActors.
            - The pool manager is automatically drained after execution completes
              or if an error occurs.
            - Environment variables set include: MEGASCALE_COORDINATOR_ADDRESS,
              MEGASCALE_NUM_SLICES, MEGASCALE_PORT, MEGASCALE_SLICE_ID,
              TPU_SLICE_NAME, and more.

        Example:
            >>> @ray.remote
            >>> def train_on_slice(data, slice_id):
            ...
            ...     return model_weights
            >>>
            >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
            >>>
            >>>
            >>> job_status = RayExecutor.autoscale_execute(
            ...     train_on_slice,
            ...     tpu_config,
            ...     data=training_data
            ... )
            >>> if isinstance(job_status, JobSucceeded):
            ...     flat_results = job_status.result
            >>>
            >>>
            >>> job_status = RayExecutor.autoscale_execute(
            ...     train_on_slice,
            ...     tpu_config,
            ...     flatten=False,
            ...     data=training_data
            ... )
            >>> if isinstance(job_status, JobSucceeded):
            ...     results_by_slice = job_status.result
        """
        pool_manager = SlicePoolManager(tpu_type=accelerator_config.tpu_version)
        per_slice_futures = None

        info = JobInfo(accelerator_config.runtime_name, "running", accelerator_config.resource_name)

        try:
            pool_manager.scale_multislice(accelerator_config.pod_count)
            pool_manager.prepare_all_slices()

            members = pool_manager.get_all_pool_members()
            if not members:
                raise RuntimeError("No SliceActors available after scaling.")
            ray.get([m.actor.ensure_host_pool.remote() for m in members])

            slice_infos = ray.get([m.actor.get_info.remote() for m in members])
            coord_ip = slice_infos[0].ip_address
            if not coord_ip:
                raise RuntimeError("Could not determine coordinator IP.")
            port = int(os.getenv("COORD_PORT", str(MEGASCALE_DEFAULT_PORT)))
            base_env = dict(
                TPU_NAME=os.getenv("TPU_NAME", "EMPTY"),
                TPU_VERSION=accelerator_config.tpu_version,
                TPU_ZONE=os.getenv("TPU_ZONE", "EMPTY"),
                TPU_POD_COUNT=str(len(members)),
            )
            if accelerator_config.execution_env:
                base_env.update({str(k): str(v) for k, v in accelerator_config.execution_env.items() if v is not None})

            per_slice_futures = []
            for slice_id, member in enumerate(members):
                if len(members) > 1:
                    env_for_slice = dict(
                        **base_env,
                        MEGASCALE_COORDINATOR_ADDRESS=f"{coord_ip}:{port}",
                        MEGASCALE_NUM_SLICES=str(len(members)),
                        MEGASCALE_PORT=str(port),
                        MEGASCALE_SLICE_ID=str(slice_id),
                        EXECUTOR_CALL_SLICE=str(slice_id),
                        TPU_SLICE_NAME=slice_infos[slice_id].slice_name,
                    )
                else:
                    env_for_slice = base_env
                env_for_slice = {str(k): str(v) for k, v in env_for_slice.items()}

                host_handles = ray.get(member.actor.get_all_actors_in_pool.remote())
                host_futures = []
                for host_idx, handle in enumerate(host_handles):
                    env_for_host = dict(env_for_slice, EXECUTOR_CALL_INDEX=str(host_idx))
                    host_futures.append(
                        handle.run_remote_fn.remote(
                            remote_fn,
                            f_args=(),
                            f_kwargs=kwargs,
                            runtime_env=accelerator_config.execution_env,
                            env=env_for_host,
                        )
                    )
                per_slice_futures.append(host_futures)

            if flatten:
                outer_refs = [f for sub in per_slice_futures for f in sub]
                inner = ray.get(outer_refs)
                results = resolve_maybe_refs(inner)
            else:
                inner_by_slice = [ray.get(lst) for lst in per_slice_futures]
                results = [resolve_maybe_refs(lst) for lst in inner_by_slice]

            return JobSucceeded(info, results)

        except InsufficientSlicesError as e:
            raise e
        except (
            ray.exceptions.RayError,
            ray.exceptions.RayTaskError,
            ray.exceptions.TaskUnschedulableError,
        ) as e:
            if per_slice_futures:
                try:
                    for lst in per_slice_futures:
                        RayResources.cancel_all_futures(lst)
                except Exception:
                    logger.debug("Failed to cancel per-slice futures after Ray error.", exc_info=True)

            s = str(e).lower()
            if ("preempt" in s) or ("unhealthy or preempted" in s) or ("owner died" in s) or ("owner has exited" in s):
                return JobPreempted(info, e)

            return handle_ray_error(info, e)
        except Exception as e:
            if per_slice_futures:
                try:
                    for lst in per_slice_futures:
                        RayResources.cancel_all_futures(lst)
                except Exception:
                    logger.debug("Failed to cancel per-slice futures after failure.", exc_info=True)
            info2 = JobInfo(accelerator_config.runtime_name, "running", accelerator_config.resource_name)
            return JobFailed(info2, e)
        finally:
            try:
                pool_manager.drain_actor_pool()
            except Exception:
                logger.debug("Failed to drain actor pool.", exc_info=True)

    @classmethod
    def execute_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """Execute a remote function with automatic retry on failures.

        Provides fault-tolerant execution of Ray remote functions with
        configurable retry policies for both preemptions and failures.
        Particularly useful for long-running jobs on preemptible resources.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute.
                Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources.
            max_retries_preemption (int): Maximum number of retries on
                preemption. Defaults to 1,000,000 (effectively unlimited).
            max_retries_failure (int): Maximum number of retries on failure.
                Defaults to 10.
            **kwargs: Additional keyword arguments passed to the remote function.

        Returns:
            Any: The result from successful execution of the remote function.
                The actual return type depends on what the remote function returns.

        Raises:
            RuntimeError: If the job is preempted more than max_retries_preemption
                times or fails more than max_retries_failure times. The error
                message indicates whether it was due to preemptions or failures.
            ray.exceptions.RayTaskError: Re-raised if it occurs and is not
                preemption-related after max retries.
            Exception: The last encountered exception if all retries are exhausted.

        Note:
            - Preemptions and failures are tracked separately
            - Each attempt logs status information for debugging
            - The method distinguishes between preemption (often recoverable)
              and failures (may indicate code issues)
            - RayTaskErrors containing "preempted" are treated as preemptions

        Example:
            >>> @ray.remote
            >>> def long_running_task(data):
            ...
            ...     return process(data)
            >>>
            >>> config = TpuAcceleratorConfig(type="v4-8", preemptible=True)
            >>> result = RayExecutor.execute_resumable(
            ...     long_running_task,
            ...     config,
            ...     max_retries_preemption=100,
            ...     max_retries_failure=5,
            ...     data=my_data
            ... )
        """
        num_failures = 0
        num_preemptions = 0
        attempt = 0
        problem: Exception | None = None

        while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
            logger.info(f"Running on Attempt {attempt}")
            attempt += 1
            problem = None
            try:
                out = cls.execute(remote_fn=remote_fn, accelerator_config=accelerator_config, **kwargs)
            except ray.exceptions.RayTaskError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(f"Preempted {num_preemptions} times, {e}")
                else:
                    num_failures += 1
                    logger.warning(f"Failed {num_failures} times (RayTaskError)", exc_info=e)
                continue
            except Exception as e:
                problem = e
                num_failures += 1
                if num_failures >= max_retries_failure:
                    logger.exception("Failed too many times", exc_info=e)
                    raise e
                else:
                    logger.warning(f"Failed {num_failures} times", exc_info=e)
                    continue

            if isinstance(out, JobSucceeded):
                result = out.result
                logger.info("Success")
                return result
            elif isinstance(out, JobPreempted):
                problem = out.error
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
            elif isinstance(out, JobFailed):
                problem = out.error
                num_failures += 1
                logger.warning(
                    f"JobFailed reported. Incrementing failure count to {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
            elif isinstance(out, JobError):
                problem = out.error
                num_failures += 1
                logger.warning(f"Failed {num_failures} times", exc_info=problem)
            else:
                raise RuntimeError(f"Unexpected result: {out}")

        if num_preemptions >= max_retries_preemption:
            raise RuntimeError("Preempted too many times") from problem
        elif num_failures >= max_retries_failure:
            raise RuntimeError("Failed too many times") from problem

    @classmethod
    def autoscale_execute_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """Execute a multi-slice function with automatic retry on failures.

        Provides fault-tolerant execution of Ray remote functions across
        multiple TPU slices with coordinated retry mechanisms. All slices
        must succeed for the execution to be considered successful.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute
                on each slice. Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources with multi-slice support (pod_count > 1).
            max_retries_preemption (int): Maximum number of retries when
                any slice is preempted. Defaults to 1,000,000.
            max_retries_failure (int): Maximum number of retries when any
                slice fails. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the remote
                function on each slice. The 'flatten' parameter can be used
                to control result structure.

        Returns:
            list[Any]: List of results from successful execution on all slices.
                The structure depends on the flatten parameter passed in kwargs:
                - If flatten=True (default): Flat list of all results
                - If flatten=False: List of lists, one per slice

        Raises:
            RuntimeError: If any slice is preempted more than max_retries_preemption
                times, fails more than max_retries_failure times, or if
                autoscale_execute returns None or unexpected result type.
            RayError: If autoscale_execute fails during setup or coordination
                (slice actor creation, placement group setup, etc.).
            ray.exceptions.RayTaskError: Re-raised if it occurs and indicates
                preemption or failure after max retries.
            Exception: The last encountered exception if retries are exhausted.

        Note:
            - Implements an all-or-nothing retry policy: if any slice fails
              or is preempted, the entire multi-slice execution is retried
            - Different error types are handled with appropriate retry logic:
              * RayTaskError/RayError with "preempted" -> preemption counter
              * Other errors -> failure counter
            - Each retry attempt creates new slice actors and placement groups
            - Detailed logging tracks retry attempts and error types

        Example:
            >>> @ray.remote
            >>> def distributed_training(data_shard):
            ...
            ...     return trained_weights
            >>>
            >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
            >>>
            >>>
            >>> results = RayExecutor.autoscale_execute_resumable(
            ...     distributed_training,
            ...     tpu_config,
            ...     max_retries_preemption=50,
            ...     max_retries_failure=3,
            ...     data_shard=sharded_data
            ... )
            >>>
            >>>
            >>> results_by_slice = RayExecutor.autoscale_execute_resumable(
            ...     distributed_training,
            ...     tpu_config,
            ...     max_retries_preemption=50,
            ...     max_retries_failure=3,
            ...     data_shard=sharded_data,
            ...     flatten=False
            ... )
        """
        num_failures = 0
        num_preemptions = 0
        attempt = 0
        problem: Exception | None = None

        while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
            logger.info(f"Running multislice on Attempt {attempt}")
            attempt += 1
            problem = None
            job_status: JobStatus | None = None

            try:
                job_status = cls.autoscale_execute(remote_fn=remote_fn, accelerator_config=accelerator_config, **kwargs)

            except ray.exceptions.RayTaskError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(
                        f"A slice was preempted (RayTaskError). Preemption count: {num_preemptions}. Error: {e}"
                    )
                else:
                    num_failures += 1
                    logger.warning(f"A slice failed (RayTaskError). Failure count: {num_failures}.", exc_info=e)
                continue
            except RayError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(
                        f"Multislice operation preempted during setup/coordination (RayError). "
                        f"Preemption count: {num_preemptions}. Error: {e}"
                    )
                else:
                    num_failures += 1
                    logger.warning(
                        f"Multislice operation failed during setup/coordination (RayError)."
                        f" Failure count: {num_failures}.",
                        exc_info=e,
                    )
                continue
            except InsufficientSlicesError as e:
                problem = e
                num_preemptions += 1
                logger.warning(
                    f"Not enough TPU slices (likely preemption/capacity). "
                    f"Preemption count: {num_preemptions}. Error: {e}"
                )
                time.sleep(int(os.getenv("EFORMER_SCALE_RETRY_SLEEP_S", "60")))
                continue
            except Exception as e:
                problem = e
                num_failures += 1
                if num_failures >= max_retries_failure:
                    logger.exception(
                        "Multislice operation failed too many times (non-Ray/RayTaskError).",
                        exc_info=e,
                    )
                    raise e
                else:
                    logger.warning(
                        f"Multislice operation failed (non-Ray/RayTaskError). Failure count: {num_failures}.",
                        exc_info=e,
                    )
                    continue

            if not job_status:
                logger.warning("autoscale_execute returned None. Treating as failure.")
                num_failures += 1
                problem = problem or RuntimeError("No job status from autoscale_execute")
                continue

            if isinstance(job_status, JobSucceeded):
                logger.info("All slices succeeded in this attempt.")
                return job_status.result
            elif isinstance(job_status, JobPreempted):
                problem = job_status.error
                num_preemptions += 1
                logger.warning(
                    f"Multislice execution preempted. Preemption count: {num_preemptions}. Error: {problem}",
                    exc_info=problem,
                )
                continue
            elif isinstance(job_status, JobFailed):
                problem = job_status.error
                num_failures += 1
                logger.warning(
                    f"Multislice execution failed (JobFailed). Failure count: {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
                continue
            elif isinstance(job_status, JobError):
                problem = job_status.error
                num_failures += 1
                logger.warning(
                    f"Multislice execution reported JobError. Failure count: {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
                continue
            else:
                err_msg = f"Unexpected result type {type(job_status)} from autoscale_execute: {job_status}"
                problem = RuntimeError(err_msg)
                num_failures += 1
                logger.error(err_msg)
                continue

        if num_preemptions >= max_retries_preemption:
            logger.error(f"Multislice job preempted too many times ({num_preemptions} >= {max_retries_preemption}).")
            raise RuntimeError(f"Preempted too many times ({num_preemptions})") from problem
        elif num_failures >= max_retries_failure:
            logger.error(f"Multislice job failed too many times ({num_failures} >= {max_retries_failure}).")
            raise RuntimeError(f"Failed too many times ({num_failures})") from problem

        raise RuntimeError(
            "Exhausted retries for multislice execution without explicit success or reaching failure/preemption limits."
        ) from problem

    autoscale_execute = autoscale_execute
    autoscale_execute_resumable = autoscale_execute_resumable


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


def _build_multislice_envs(
    slice_infos: list[SliceInfo],
    base_env: dict[str, str],
    coord_port: int,
) -> list[dict[str, str]]:
    """Build environment variables for multi-slice TPU execution.

    Creates environment variable dictionaries for each TPU slice in a multi-slice
    setup, including MegaScale coordination variables required for inter-slice
    communication.

    Args:
        slice_infos: List of SliceInfo objects containing metadata about each
            TPU slice, including IP addresses and slice names.
        base_env: Base environment variables to include for all slices.
        coord_port: Port number for the MegaScale coordinator service.

    Returns:
        List of environment variable dictionaries, one per slice. Each dictionary
        contains:
        - All base environment variables
        - MegaScale coordination variables (if multi-slice):
            - MEGASCALE_COORDINATOR_ADDRESS: IP:port of coordinator
            - MEGASCALE_NUM_SLICES: Total number of slices
            - MEGASCALE_PORT: Coordinator port
            - MEGASCALE_SLICE_ID: This slice's ID (0-indexed)
            - EXECUTOR_CALL_SLICE: Same as MEGASCALE_SLICE_ID
            - TPU_SLICE_NAME: Name of this TPU slice

    Note:
        The first slice's IP address is used as the coordinator address.
        All values are converted to strings for environment compatibility.
        EXECUTOR_CALL_INDEX is assigned per host when scheduling tasks.
    """
    envs = []
    coord_ip = slice_infos[0].ip_address
    for i, si in enumerate(slice_infos):
        env = dict(base_env)
        if len(slice_infos) > 1:
            env.update(
                MEGASCALE_COORDINATOR_ADDRESS=f"{coord_ip}:{coord_port}",
                MEGASCALE_NUM_SLICES=str(len(slice_infos)),
                MEGASCALE_PORT=str(coord_port),
                MEGASCALE_SLICE_ID=str(i),
                EXECUTOR_CALL_SLICE=str(i),
                TPU_SLICE_NAME=si.slice_name,
            )
        envs.append({str(k): str(v) for k, v in env.items()})
    return envs


class TpuRemoteManager:
    """Session-based manager for TPU remote execution across multiple slices.

    Provides a high-level interface for managing TPU resources and executing
    functions or class methods across multiple TPU slices. Each execution runs
    in a short-lived worker process to avoid TPU handle leaks.

    Key features:
    - Automatic slice scaling and warming
    - Persistent DeviceHostActors for efficient execution
    - Broadcasting of function/method calls across all hosts
    - Ephemeral worker processes for each execution
    - MegaScale coordination for multi-slice workloads

    Attributes:
        tpu_version: TPU version string (e.g., 'v4', 'v5p')
        pod_count: Number of TPU pods/slices to use
        base_env: Base environment variables for all workers
        runtime_env: Ray runtime environment configuration
        coord_port: Port for MegaScale coordinator (default 8081)

    Example:
        >>> manager = TpuRemoteManager(
        ...     tpu_version='v4',
        ...     pod_count=2,
        ...     base_env={'MY_VAR': 'value'}
        ... )
        >>> manager.ensure()
        >>> results = manager.run_function(my_func, arg1, arg2)
        >>> manager.close()
    """

    def __init__(
        self,
        tpu_version: str,
        pod_count: int,
        *,
        base_env: dict[str, str] | None = None,
        runtime_env: dict | None = None,
        coord_port: int = 8081,
    ):
        self.tpu_version = tpu_version
        self.pod_count = int(pod_count)
        self.base_env = dict(base_env or {})
        self.runtime_env = dict(runtime_env or {})
        self.coord_port = int(coord_port)

        self._pool = SlicePoolManager(tpu_type=tpu_version)
        self._members = None
        self._slice_infos: list[SliceInfo] = []
        self._per_slice_envs: list[dict[str, str]] = []
        self._host_handles_by_slice: list[list[ray.actor.ActorHandle]] = []

    def ensure(self):
        """Initialize and prepare TPU resources if not already done.

        This method:
        1. Scales the slice pool to the requested pod_count
        2. Prepares all slices (creates placement groups, etc.)
        3. Ensures DeviceHostActors are created on each host
        4. Builds per-slice environment variables including MegaScale config
        5. Caches host actor handles for efficient execution

        Raises:
            RuntimeError: If no SliceActors are available after scaling.

        Note:
            This method is idempotent - calling it multiple times has no
            additional effect after the first successful call.
        """
        if self._members is not None:
            return

        self._pool.scale_multislice(self.pod_count)
        self._pool.prepare_all_slices()

        members = self._pool.get_all_pool_members()
        if not members:
            raise RuntimeError("No SliceActors available after scaling.")
        ray.get([m.actor.ensure_host_pool.remote() for m in members])

        slice_infos = ray.get([m.actor.get_info.remote() for m in members])

        base_env = dict(self.base_env)
        base_env.update(
            TPU_VERSION=self.tpu_version,
            TPU_POD_COUNT=str(len(members)),
        )
        self._per_slice_envs = _build_multislice_envs(slice_infos, base_env, self.coord_port)

        host_handles_by_slice = ray.get([m.actor.get_all_actors_in_pool.remote() for m in members])

        self._members = members
        self._slice_infos = slice_infos
        self._host_handles_by_slice = host_handles_by_slice

    def close(self):
        """Clean up all TPU resources and reset state.

        Performs cleanup in the following order:
        1. Cancels any current work on all host actors
        2. Drains the actor pool (terminates all actors)
        3. Resets internal state

        Note:
            This method is safe to call multiple times and will suppress
            any errors during cleanup to ensure resources are freed.
        """

        try:
            if self._members:
                for m in self._members:
                    try:
                        hosts = ray.get(m.actor.get_all_actors_in_pool.remote())
                        ray.get([h.cancel_current.remote() for h in hosts], timeout=10)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self._pool.drain_actor_pool()
        finally:
            self._members = None
            self._slice_infos = []
            self._per_slice_envs = []
            self._host_handles_by_slice = []

    def run_function(self, fn: Callable, *args, flatten: bool = True, **kwargs):
        """Execute a Python function on every TPU host across all slices.

        Runs the provided function in ephemeral worker processes on each host
        to avoid TPU handle leaks. The function is executed with the same
        arguments on every host.

        Args:
            fn: Python callable to execute on each host. Should be pickleable.
            *args: Positional arguments to pass to the function.
            flatten: If True, returns a flat list of results from all hosts.
                If False, returns nested lists where outer list represents
                slices and inner lists contain results from hosts within
                each slice.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            If flatten=True: List of results from all hosts (flattened).
            If flatten=False: List of lists, where each inner list contains
                results from hosts within a slice.

        Note:
            Each execution runs in a new process to avoid TPU handle leaks.
            The function must be pickleable for Ray serialization.
        """
        self.ensure()

        outer_refs_per_slice = []
        for sl, hosts in enumerate(self._host_handles_by_slice):
            env = self._per_slice_envs[sl]

            outer = [
                h.run_remote_fn.remote(
                    fn,
                    f_args=args,
                    f_kwargs=kwargs,
                    runtime_env=self.runtime_env,
                    env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                )
                for host_idx, h in enumerate(hosts)
            ]
            outer_refs_per_slice.append(outer)

        inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
        if flatten:
            flat_refs = [ref for refs in inner_per_slice for ref in refs]
            return ray.get(flat_refs)
        else:
            return [ray.get(refs) for refs in inner_per_slice]

    def run_class_method(
        self,
        cls_obj: type,
        init_args: tuple,
        init_kwargs: dict,
        method_name: str,
        *call_args,
        flatten: bool = True,
        **call_kwargs,
    ):
        """Execute a class method on every TPU host across all slices.

        Instantiates the class and calls the specified method in ephemeral
        worker processes on each host. The class is reconstructed inside
        each worker to ensure clean TPU state.

        Args:
            cls_obj: The class object (not an instance). Must be pickleable.
            init_args: Positional arguments for class initialization.
            init_kwargs: Keyword arguments for class initialization.
            method_name: Name of the method to call on the instantiated object.
            *call_args: Positional arguments for the method call.
            flatten: If True, returns a flat list of results from all hosts.
                If False, returns nested lists by slice.
            **call_kwargs: Keyword arguments for the method call.

        Returns:
            If flatten=True: List of results from all hosts (flattened).
            If flatten=False: List of lists, where each inner list contains
                results from hosts within a slice.

        Example:
            >>> results = manager.run_class_method(
            ...     MyModel,
            ...     init_args=(config,),
            ...     init_kwargs={'checkpoint': 'path/to/ckpt'},
            ...     method_name='train',
            ...     epochs=10,
            ...     batch_size=32
            ... )

        Note:
            The class is instantiated fresh in each worker process,
            avoiding TPU handle leaks from persistent objects.
        """
        self.ensure()

        def _class_method_entry(cls_obj, i_args, i_kwargs, mname, c_args, c_kwargs):
            obj = cls_obj(*i_args, **i_kwargs)
            return getattr(obj, mname)(*c_args, **c_kwargs)

        outer_refs_per_slice = []
        for sl, hosts in enumerate(self._host_handles_by_slice):
            env = self._per_slice_envs[sl]
            payload = (cls_obj, init_args, init_kwargs, method_name, call_args, call_kwargs)
            outer = [
                h.run_remote_fn.remote(
                    _class_method_entry,
                    f_args=payload,
                    runtime_env=self.runtime_env,
                    env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                )
                for host_idx, h in enumerate(hosts)
            ]
            outer_refs_per_slice.append(outer)

        inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
        if flatten:
            flat_refs = [ref for refs in inner_per_slice for ref in refs]
            return ray.get(flat_refs)
        else:
            return [ray.get(refs) for refs in inner_per_slice]


class _BoundRemoteClass:
    """Bound handle for remote class execution on TPU hosts.

    Internal class that provides a proxy interface for calling methods on
    a class that will be instantiated and executed on TPU hosts. Supports
    both synchronous execution (returning values) and asynchronous execution
    (returning Ray ObjectRefs).

    This class is created by the tpu_remote decorator when applied to classes.

    Attributes:
        _manager: TpuRemoteManager instance for resource management
        _cls: The class object to instantiate on remote hosts
        _init_args: Positional arguments for class initialization
        _init_kwargs: Keyword arguments for class initialization
        _flatten: Whether to flatten results from multiple hosts
    """

    def __init__(self, manager: TpuRemoteManager, cls: type, init_args: tuple, init_kwargs: dict, flatten: bool):
        self._manager = manager
        self._cls = cls
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._flatten = flatten

    def __getattr__(self, name: str):
        def _call_method(*args, **kwargs):
            return self._manager.run_class_method(
                self._cls,
                self._init_args,
                self._init_kwargs,
                name,
                *args,
                flatten=self._flatten,
                **kwargs,
            )

        _call_method.remote = self._remote(name)
        return _call_method

    def _remote(self, name: str):
        def _remote_impl(*args, **kwargs):
            def _class_method_entry(cls_obj, i_args, i_kwargs, mname, c_args, c_kwargs):
                obj = cls_obj(*i_args, **i_kwargs)
                return getattr(obj, mname)(*c_args, **c_kwargs)

            self._manager.ensure()
            outer_refs_per_slice = []
            for sl, hosts in enumerate(self._manager._host_handles_by_slice):
                env = self._manager._per_slice_envs[sl]
                payload = (self._cls, self._init_args, self._init_kwargs, name, args, kwargs)
                outer = [
                    h.run_remote_fn.remote(
                        _class_method_entry,
                        f_args=payload,
                        runtime_env=self._manager.runtime_env,
                        env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                    )
                    for host_idx, h in enumerate(hosts)
                ]
                outer_refs_per_slice.append(outer)

            inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
            if self._flatten:
                return [ref for refs in inner_per_slice for ref in refs]
            else:
                return inner_per_slice

        return _remote_impl

    def close(self):
        """Close the manager and clean up TPU resources.

        Delegates to the underlying TpuRemoteManager to terminate all actors
        and free TPU resources. Safe to call multiple times.
        """
        self._manager.close()


class _RemoteClassWrapper:
    """Wrapper that converts a regular class into a TPU-remote class.

    Internal class used by the tpu_remote decorator to wrap classes for
    remote execution. When the wrapped class is instantiated, it returns
    a _BoundRemoteClass that broadcasts method calls to TPU hosts.

    Attributes:
        _manager: TpuRemoteManager instance for resource management
        _flatten: Whether to flatten results from multiple hosts
    """

    def __init__(self, manager: TpuRemoteManager, flatten: bool):
        self._manager = manager
        self._flatten = flatten

    def __call__(self, cls: type):
        @functools.wraps(cls)
        def ctor(*args, **kwargs):
            return _BoundRemoteClass(self._manager, cls, args, kwargs, self._flatten)

        return ctor


class _RemoteFunctionWrapper:
    """Wrapper that converts a regular function into a TPU-remote function.

    Internal class used by the tpu_remote decorator to wrap functions for
    remote execution. The wrapped function broadcasts calls to all TPU hosts
    and supports both synchronous and asynchronous execution modes.

    Attributes:
        _manager: TpuRemoteManager instance for resource management
        _flatten: Whether to flatten results from multiple hosts
    """

    def __init__(self, manager: TpuRemoteManager, flatten: bool):
        self._manager = manager
        self._flatten = flatten

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self._manager.run_function(fn, *args, flatten=self._flatten, **kwargs)

        def _remote(*args, **kwargs):
            self._manager.ensure()
            outer_refs_per_slice = []
            for sl, hosts in enumerate(self._manager._host_handles_by_slice):
                env = self._manager._per_slice_envs[sl]
                outer = [
                    h.run_remote_fn.remote(
                        fn,
                        f_args=args,
                        f_kwargs=kwargs,
                        runtime_env=self._manager.runtime_env,
                        env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                    )
                    for host_idx, h in enumerate(hosts)
                ]
                outer_refs_per_slice.append(outer)
            inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
            if self._flatten:
                return [ref for refs in inner_per_slice for ref in refs]
            else:
                return inner_per_slice

        wrapper.remote = _remote  # type: ignore[attr-defined]
        return wrapper


def device_remote(*, accelerator_config: TpuAcceleratorConfig, flatten: bool = True):
    """Decorator for TPU-remote execution of functions or classes.

    Transforms a regular Python function or class into a TPU-remote version
    that automatically broadcasts execution across all TPU hosts in the
    specified configuration. Each execution runs in an ephemeral worker
    process to avoid TPU handle leaks.

    Args:
        accelerator_config: TPU configuration specifying version, pod count,
            and other execution parameters.
        flatten: If True (default), returns flat list of results from all
            hosts. If False, returns nested lists where outer list represents
            slices and inner lists contain results from hosts within each slice.

    Returns:
        Decorator function that wraps the target function or class.

    Example for functions:
        >>> @device_remote(accelerator_config=tpu_config)
        >>> def compute(x, y):
        ...     return jax.numpy.dot(x, y)
        >>>
        >>>
        >>> results = compute(array1, array2)
        >>>
        >>>
        >>> refs = compute.remote(array1, array2)
        >>> results = ray.get(refs)

    Example for classes:
        >>> @tpu_remote(accelerator_config=tpu_config)
        >>> class Model:
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        ...     def train(self, data):
        ...
        ...         return metrics
        >>>
        >>> model = Model(config)
        >>> metrics = model.train(data)
        >>> refs = model.train.remote(data)

    Note:
        - Functions/classes must be pickleable for Ray serialization
        - Each method call creates new worker processes on TPU hosts
        - The decorator manages slice scaling and resource allocation
        - Call model.close() to clean up resources when done
    """
    full_runtime_env = dict(accelerator_config.execution_env or {})
    env_vars = dict(full_runtime_env.get("env_vars", {}))
    full_runtime_env["env_vars"] = env_vars

    mgr = TpuRemoteManager(
        tpu_version=accelerator_config.tpu_version,
        pod_count=accelerator_config.pod_count,
        base_env=env_vars,
        runtime_env=full_runtime_env,
        coord_port=8081,
    )

    def decorator(obj: Callable | type):
        if isinstance(obj, type):
            return _RemoteClassWrapper(mgr, flatten)(obj)
        elif callable(obj):
            return _RemoteFunctionWrapper(mgr, flatten)(obj)
        else:
            raise TypeError("tpu_remote can only decorate a function or a class.")

    return decorator

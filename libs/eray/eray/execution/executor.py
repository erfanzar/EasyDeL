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


"""Ray-based distributed execution engine."""

import logging
import os
import time

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from ..core.cluster import SliceInfo
from ..core.exceptions import handle_ray_error
from ..core.status import JobError, JobFailed, JobInfo, JobPreempted, JobStatus, JobSucceeded
from ..pool.base import InsufficientSlicesError
from ..pool.slice import SlicePoolManager
from ..resources.configs import AcceleratorConfigType, TpuAcceleratorConfig
from ..resources.ray_resources import RayResources

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
            JobStatus: The execution result status.

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
            port = int(os.getenv("ERAY_COORD_PORT", str(MEGASCALE_DEFAULT_PORT)))
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
                time.sleep(int(os.getenv("ERAY_SCALE_RETRY_SLEEP", "60")))
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

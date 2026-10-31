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


"""Swarm execution — pack multiple runtimes onto TPU hosts with a chip-split plan.

A swarm runs several independent workloads ("runs") on the same TPU hosts by
carving each host's chips between them: e.g. three runs on a 4-chip v5p host
with the plan ``(2, 1, 1)``. :class:`SwarmConfig` mirrors
:class:`~eray.resources.configs.TpuAcceleratorConfig` — it takes the chip
type — plus the number of runs and an optional per-run chip plan. Runs are
isolated runtimes with chips assigned disjointly by Ray (see
:mod:`eray.resources.topology`).

Example:
    Even split — four copies of one function, one chip each on a v5p host::

        import eray

        config = eray.SwarmConfig(tpu_version="v5p-8", num_runs=4)
        status = eray.swarm_execute(serve_replica, config)

    A plan — three different workloads, chips split ``(2, 1, 1)``::

        config = eray.SwarmConfig(tpu_version="v5p-8", chip_split=(2, 1, 1))
        status = eray.swarm_execute(
            [
                eray.SwarmRun(train_probe, name="train"),
                eray.SwarmRun(serve_small, name="serve"),
                eray.SwarmRun(eval_loop, name="eval"),
            ],
            config,
        )

Each run sees ``ERAY_RUN_ID`` / ``ERAY_NUM_RUNS`` / ``ERAY_RUN_CHIPS`` (and
``ERAY_RUN_NAME`` when named) in its environment. With ``pod_count > 1`` or
multi-host slices, the same swarm layout is repeated on every host.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .core.exceptions import handle_ray_error
from .core.status import JobFailed, JobInfo, JobPreempted, JobStatus, JobSucceeded
from .pool.base import InsufficientSlicesError
from .pool.device_host import _fn_runner
from .pool.slice import SlicePoolManager
from .resources.configs import GpuAcceleratorConfig, TpuAcceleratorConfig
from .resources.ray_resources import RayResources

logger = logging.getLogger("ray")


def _unwrap_remote_fn(fn):
    """Extract the plain Python function from a Ray remote function.

    Args:
        fn: A Ray RemoteFunction or plain callable.

    Returns:
        The underlying Python callable.
    """
    try:
        from ray.remote_function import RemoteFunction

        return fn._function if isinstance(fn, RemoteFunction) else fn
    except Exception:
        return fn


def _merge_env_vars(runtime_env: dict | None, env_vars: dict) -> dict:
    """Merge environment variables into a Ray runtime environment dict.

    Args:
        runtime_env: Base runtime environment configuration.
        env_vars: Environment variables to merge in.

    Returns:
        Merged runtime environment dictionary.
    """
    merged = dict(runtime_env or {})
    ev = dict(merged.get("env_vars", {}))
    ev.update({str(k): str(v) for k, v in env_vars.items() if v is not None})
    merged["env_vars"] = ev
    return merged


@dataclass(frozen=True)
class SwarmRun:
    """Specification of one run (workload) inside a swarm.

    ``fn`` may be a plain **function** (runs to completion, its return value
    is the run's result) or a plain **class** (instantiated as a long-lived,
    detached Ray actor on the run's chip share). Class runs require a
    ``name`` and a :attr:`SwarmConfig.namespace`; once the swarm launches,
    the live runtime is addressable from any driver via
    ``ray.get_actor(name, namespace=...)`` and its constructor receives
    ``f_args`` / ``f_kwargs``. Pass the plain class — eray applies
    ``ray.remote`` itself.

    Attributes:
        fn: Function to execute, or class to instantiate, once per host. May
            be omitted only under the :func:`swarmed` decorator, which binds
            the decorated function to every bare run.
        chips: Devices this run owns — TPU chips under :class:`SwarmConfig`,
            GPUs under :class:`GpuSwarmConfig` (fractions below one GPU are
            allowed there). None takes the share from the config's split plan
            (or an even split / per-run default when no plan is given).
        name: Human-readable name, surfaced to the run as ``ERAY_RUN_NAME``.
            Required for class runs — it becomes the actor name.
        num_cores: CPU cores reserved for this run, or None for the config's
            default (``cpu_count`` on GPU swarms; the launcher default on TPU
            swarms).
        f_args: Positional arguments passed to ``fn`` (constructor arguments
            for class runs).
        f_kwargs: Keyword arguments passed to ``fn`` (constructor arguments
            for class runs).
        env: Extra environment variables for this run only.
    """

    fn: Callable | None = None
    chips: int | float | None = None
    name: str | None = None
    num_cores: float | None = None
    f_args: tuple = ()
    f_kwargs: dict | None = None
    env: dict | None = None


@dataclass(frozen=True)
class SwarmConfig(TpuAcceleratorConfig):
    """TPU accelerator configuration for swarm execution.

    Everything from :class:`TpuAcceleratorConfig` (chip type, pod count,
    execution env, ...) plus the swarm layout: how many runs share each host
    and how the host's chips are split between them.

    Attributes:
        num_runs: Number of runs per host. May be omitted when ``chip_split``
            is given (the plan's length defines it) or when the runs are
            passed as an explicit list to :func:`swarm_execute`.
        chip_split: Optional plan for how to split each host's chips between
            runs, index-aligned with run id — e.g. ``(2, 1, 1)`` gives run 0
            two chips and runs 1 and 2 one chip each. Omitted: chips are
            split evenly. Each entry must be 1, 2, or the whole host (a Ray
            fractional-TPU scheduling constraint), and the plan must not
            oversubscribe the host.
        namespace: Ray namespace for class runs' named actors. Required when
            any run is a class; the launched runtimes are then reachable from
            any driver via ``ray.get_actor(name, namespace=namespace)`` and
            can be torn down together with :func:`shutdown_swarm`.

    Example:
        >>> config = SwarmConfig(tpu_version="v5p-8", chip_split=(2, 1, 1))
        >>> config.resolved_num_runs()
        3
    """

    num_runs: int | None = None
    chip_split: tuple[int, ...] | None = field(default=None)
    namespace: str | None = None

    def __post_init__(self):
        """Validate layout consistency at construction time.

        Raises:
            ValueError: If ``num_runs`` and ``chip_split`` are both given but
                disagree, or either is non-positive/empty.
        """
        if self.chip_split is not None:
            if len(self.chip_split) == 0:
                raise ValueError("chip_split must not be empty")
            if self.num_runs is not None and self.num_runs != len(self.chip_split):
                raise ValueError(f"num_runs={self.num_runs} disagrees with chip_split of length {len(self.chip_split)}")
        elif self.num_runs is not None and self.num_runs <= 0:
            raise ValueError(f"num_runs must be positive, got {self.num_runs}")

    def resolved_num_runs(self) -> int | None:
        """Number of runs implied by the config, or None if undetermined.

        Returns:
            ``len(chip_split)`` when a plan is given, else ``num_runs``.
        """
        if self.chip_split is not None:
            return len(self.chip_split)
        return self.num_runs

    def resolved_split(self) -> tuple[int | float, ...] | None:
        """The per-run device plan, or None when no plan was given.

        Returns:
            The ``chip_split`` tuple, or None.
        """
        return self.chip_split


def _validate_gpu_count(count) -> int | float:
    """Validate a per-run GPU count against Ray's scheduling rules.

    Args:
        count: Requested GPUs for one run.

    Returns:
        The count as int (whole GPUs) or float (fraction of one GPU).

    Raises:
        ValueError: If the count is non-positive, or fractional above one
            GPU (Ray schedules whole GPUs beyond one).
    """
    c = float(count)
    if c <= 0:
        raise ValueError(f"GPU counts must be positive, got {count}")
    if c > 1 and not c.is_integer():
        raise ValueError(f"GPU counts above one must be whole GPUs (Ray scheduling rule), got {count}")
    return int(c) if c.is_integer() else c


@dataclass(frozen=True)
class GpuSwarmConfig(GpuAcceleratorConfig):
    """GPU accelerator configuration for swarm execution.

    Everything from :class:`~eray.resources.configs.GpuAcceleratorConfig`
    (``gpu_model``, ``cpu_count``, ``execution_env``, ...) plus the swarm
    layout. Unlike the TPU path, GPU swarms need no slice pool or chip-bounds
    environment: Ray natively assigns disjoint GPUs per ``num_gpus`` request
    and isolates them via ``CUDA_VISIBLE_DEVICES``, so runs are launched
    directly and any GPU count is allowed (fractions of a single GPU too).

    Attributes:
        num_runs: Number of runs in the swarm. May be omitted when
            ``gpu_split`` is given or the runs are passed as an explicit list.
        gpu_split: Optional plan for GPUs per run, index-aligned with run id —
            e.g. ``(2, 1, 1)``. Omitted: each run gets ``device_count`` GPUs
            (the inherited per-task default, 1) unless its
            :attr:`SwarmRun.chips` says otherwise. Entries above one GPU must
            be whole numbers; fractions of one GPU (e.g. ``0.5``) share a
            device between runs.
        namespace: Ray namespace for class runs' named actors (see
            :class:`SwarmConfig`).
        colocate: If True, pack all runs onto one node via a STRICT_PACK
            placement group — the GPU analogue of splitting a single host.
            Not supported together with class runs (a detached actor would
            die with the placement group).

    Example:
        >>> config = GpuSwarmConfig(gpu_model="A100", gpu_split=(2, 1, 1))
        >>> config.resolved_num_runs()
        3
    """

    num_runs: int | None = None
    gpu_split: tuple[int | float, ...] | None = None
    namespace: str | None = None
    colocate: bool = False

    def __post_init__(self):
        """Validate layout consistency at construction time.

        Raises:
            ValueError: If ``num_runs`` and ``gpu_split`` disagree, either is
                non-positive/empty, or a split entry violates Ray's GPU
                scheduling rules.
        """
        if self.gpu_split is not None:
            if len(self.gpu_split) == 0:
                raise ValueError("gpu_split must not be empty")
            if self.num_runs is not None and self.num_runs != len(self.gpu_split):
                raise ValueError(f"num_runs={self.num_runs} disagrees with gpu_split of length {len(self.gpu_split)}")
            for c in self.gpu_split:
                _validate_gpu_count(c)
        elif self.num_runs is not None and self.num_runs <= 0:
            raise ValueError(f"num_runs must be positive, got {self.num_runs}")

    def resolved_num_runs(self) -> int | None:
        """Number of runs implied by the config, or None if undetermined.

        Returns:
            ``len(gpu_split)`` when a plan is given, else ``num_runs``.
        """
        if self.gpu_split is not None:
            return len(self.gpu_split)
        return self.num_runs

    def resolved_split(self) -> tuple[int | float, ...] | None:
        """The per-run device plan, or None when no plan was given.

        Returns:
            The ``gpu_split`` tuple, or None.
        """
        return self.gpu_split


SwarmConfigT = SwarmConfig | GpuSwarmConfig


def _normalize_runs(
    runs: Callable | SwarmRun | Sequence[Callable | SwarmRun],
    config: SwarmConfigT,
) -> list[dict]:
    """Resolve user-facing run specs against the config into run payloads.

    Args:
        runs: A single callable or :class:`SwarmRun` (replicated
            ``config.num_runs`` times), or a sequence of them (one per run).
        config: Swarm configuration providing the layout (TPU or GPU).

    Returns:
        List of payload dicts, index-aligned with run id. Device precedence
        per run: the config's split-plan entry if given, else the run's own
        ``chips``, else None (resolved by the launcher — even split on TPU
        hosts, ``device_count`` GPUs per run on GPU swarms).

    Raises:
        ValueError: If the number of runs cannot be determined, disagrees
            with the config layout, a run spec is not callable, a class run
            is unnamed or duplicates a name, class runs are given without a
            config namespace, or a run is already a Ray-decorated actor
            class.
    """
    if callable(runs) or isinstance(runs, SwarmRun):
        count = config.resolved_num_runs()
        if count is None:
            raise ValueError("a single run needs the config's num_runs or a split plan to define the swarm size")
        run_list = [runs] * count
    else:
        run_list = list(runs)
        expected = config.resolved_num_runs()
        if expected is not None and expected != len(run_list):
            raise ValueError(f"{len(run_list)} runs given but the config defines {expected}")
    if not run_list:
        raise ValueError("no runs given")

    payloads = []
    class_names: list[str] = []
    for i, run in enumerate(run_list):
        spec = run if isinstance(run, SwarmRun) else SwarmRun(fn=run)
        if spec.fn is None:
            raise ValueError(f"run {i} has no fn — bare SwarmRun(...) specs are only valid under the @swarmed decorator")
        if isinstance(spec.fn, ray.actor.ActorClass):
            raise ValueError(
                f"run {i} is already a Ray actor class (@ray.remote); pass the plain class — "
                f"eray applies ray.remote itself"
            )
        if not callable(spec.fn):
            raise ValueError(f"run {i} is not callable")
        is_class = inspect.isclass(spec.fn)
        if is_class:
            if not spec.name:
                raise ValueError(f"run {i} is a class run and needs a name (it becomes the actor name)")
            if config.namespace is None:
                raise ValueError("class runs need SwarmConfig.namespace so the named actors are addressable")
            class_names.append(spec.name)
        split = config.resolved_split()
        chips = split[i] if split is not None else spec.chips
        if chips is not None:
            chips = float(chips)
            chips = int(chips) if chips.is_integer() else chips
        payloads.append(
            {
                "fn": spec.fn,
                "is_class": is_class,
                "chips": chips,
                "name": spec.name,
                "num_cores": spec.num_cores,
                "f_args": spec.f_args,
                "f_kwargs": spec.f_kwargs,
                "env": spec.env,
            }
        )
    duplicates = {n for n in class_names if class_names.count(n) > 1}
    if duplicates:
        raise ValueError(f"class run names must be unique within a swarm, duplicated: {sorted(duplicates)}")
    return payloads


def _gpu_colocate_bundles(payloads: list[dict], gpu_counts: list[int | float], config: GpuSwarmConfig) -> list[dict]:
    """Build the placement-group bundles for a colocated GPU swarm.

    Args:
        payloads: Normalized run payloads.
        gpu_counts: Validated GPUs per run, index-aligned with payloads.
        config: GPU swarm configuration.

    Returns:
        One bundle per run with its GPU/CPU reservation. When
        ``config.gpu_model`` is set, each bundle also carries Ray's implicit
        ``accelerator_type:<model>`` resource (weight 0.001) — the
        ``accelerator_type`` task option is implemented as exactly that
        implicit demand, and a bundle without it can never admit the task.
    """
    bundles = []
    for count, p in zip(gpu_counts, payloads, strict=True):
        bundle: dict = {"GPU": count, "CPU": p["num_cores"] if p["num_cores"] is not None else config.cpu_count}
        if config.gpu_model:
            bundle[f"accelerator_type:{config.gpu_model}"] = 0.001
        bundles.append(bundle)
    return bundles


def _swarm_execute_gpu(payloads: list[dict], config: GpuSwarmConfig, flatten: bool) -> JobStatus:
    """Launch a GPU swarm directly through Ray's native GPU scheduling.

    No slice pool and no chip-bounds environment are involved: each run is a
    Ray task (or detached named actor) requesting ``num_gpus``; Ray assigns
    disjoint devices and isolates them via ``CUDA_VISIBLE_DEVICES``. With
    ``config.colocate`` the runs are packed onto one node via a STRICT_PACK
    placement group.

    Args:
        payloads: Normalized run payloads from :func:`_normalize_runs`.
        config: GPU swarm configuration.
        flatten: Kept for signature symmetry with the TPU path; a GPU swarm
            is a single layout, so the result is one flat list either way
            (``flatten=False`` nests it one level as ``[results]``).

    Returns:
        JobStatus mirroring :func:`swarm_execute`.
    """
    info = JobInfo(str(config.runtime_name), "running", config.resource_name)

    gpu_counts = [_validate_gpu_count(p["chips"] if p["chips"] is not None else config.device_count) for p in payloads]
    if config.colocate and any(p["is_class"] for p in payloads):
        raise ValueError(
            "colocate=True is not supported with class runs: a detached actor would die "
            "with the placement group. Launch class runs without colocation."
        )

    pg = None
    awaitables: list[ray.ObjectRef] = []
    try:
        if config.colocate:
            pg = placement_group(_gpu_colocate_bundles(payloads, gpu_counts, config), strategy="STRICT_PACK")
            ray.get(pg.ready())

        for run_id, (payload, count) in enumerate(zip(payloads, gpu_counts, strict=True)):
            run_env = {
                "ERAY_RUN_ID": str(run_id),
                "ERAY_NUM_RUNS": str(len(payloads)),
                "ERAY_RUN_GPUS": str(count),
            }
            if payload["name"]:
                run_env["ERAY_RUN_NAME"] = str(payload["name"])
            merged_runtime_env = _merge_env_vars(config.execution_env, {**run_env, **(payload.get("env") or {})})
            num_cpus = payload["num_cores"] if payload["num_cores"] is not None else config.cpu_count
            options = dict(
                num_gpus=count,
                num_cpus=num_cpus,
                runtime_env=merged_runtime_env,
            )
            if config.gpu_model:
                options["accelerator_type"] = config.gpu_model
            if pg is not None:
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    pg, placement_group_bundle_index=run_id
                )

            if payload["is_class"]:
                actor = (
                    ray.remote(payload["fn"])
                    .options(
                        name=str(payload["name"]),
                        namespace=config.namespace,
                        lifetime="detached",
                        **options,
                    )
                    .remote(*(payload.get("f_args") or ()), **(payload.get("f_kwargs") or {}))
                )
                awaitables.append(actor.__ray_ready__.remote())
            else:
                awaitables.append(
                    _fn_runner.options(max_retries=0, **options).remote(
                        _unwrap_remote_fn(payload["fn"]),
                        payload.get("f_args") or (),
                        payload.get("f_kwargs") or {},
                    )
                )

        raw = ray.get(awaitables)
        results = []
        for run_id, (payload, value) in enumerate(zip(payloads, raw, strict=True)):
            if payload["is_class"]:
                results.append(
                    {
                        "kind": "actor",
                        "name": str(payload["name"]),
                        "namespace": config.namespace,
                        "run_id": run_id,
                        "run_name": payload["name"],
                    }
                )
            else:
                results.append(value)

        return JobSucceeded(info, results if flatten else [results])

    except (
        ray.exceptions.RayError,
        ray.exceptions.RayTaskError,
        ray.exceptions.TaskUnschedulableError,
    ) as e:
        try:
            RayResources.cancel_all_futures(awaitables)
        except Exception:
            logger.debug("Failed to cancel GPU swarm futures after Ray error.", exc_info=True)
        s = str(e).lower()
        if "preempt" in s:
            return JobPreempted(info, e)
        return handle_ray_error(info, e)
    except Exception as e:
        try:
            RayResources.cancel_all_futures(awaitables)
        except Exception:
            logger.debug("Failed to cancel GPU swarm futures after failure.", exc_info=True)
        return JobFailed(info, e)
    finally:
        if pg is not None:
            try:
                remove_placement_group(pg)
            except Exception:
                logger.debug("Failed to remove GPU swarm placement group.", exc_info=True)


def swarm_execute(
    runs: Callable | SwarmRun | Sequence[Callable | SwarmRun],
    config: SwarmConfigT,
    flatten: bool = True,
) -> JobStatus:
    """Execute a swarm of workloads across TPU hosts with chip-level splits.

    Acquires ``config.pod_count`` slices, then runs the swarm layout on every
    host: each run gets its planned chip share of that host as an isolated
    runtime. Mirrors ``RayExecutor.autoscale_execute`` semantics (job status
    result, preemption mapping, pool drain) but launches ``num_runs`` tasks
    per host instead of one.

    Args:
        runs: A single callable or :class:`SwarmRun` replicated over the
            swarm, or a sequence with one entry per run.
        config: Swarm configuration (chip type, pod count, layout plan).
        flatten: If True (default), the success result is a flat list ordered
            slice-major, then host, then run id. If False, nested lists
            ``[slice][host][run]``.

    Returns:
        JobStatus: JobSucceeded with results, or JobFailed / JobPreempted /
        JobError mirroring autoscale_execute.

    Raises:
        InsufficientSlicesError: If the requested slices cannot be allocated.
        ValueError: If the run specs disagree with the config layout.

    Note:
        Chip-plan validity against the actual host chip count is checked on
        each host (chip counts differ per TPU generation); an invalid plan
        surfaces as JobFailed. TPU chip partitioning is hardware-validated on
        v5p-8 for 1-chip, aligned 2-chip, and cooperative layouts (see
        :mod:`eray.resources.topology`).
    """
    payloads = _normalize_runs(runs, config)

    if isinstance(config, GpuSwarmConfig):
        return _swarm_execute_gpu(payloads, config, flatten)

    pool_manager = SlicePoolManager(tpu_type=config.tpu_version)
    info = JobInfo(config.runtime_name, "running", config.resource_name)
    per_host_run_refs: list[list[ray.ObjectRef]] = []
    try:
        pool_manager.scale_multislice(config.pod_count)
        pool_manager.prepare_all_slices()

        members = pool_manager.get_all_pool_members()
        if not members:
            raise RuntimeError("No SliceActors available after scaling.")
        ray.get([m.actor.ensure_host_pool.remote() for m in members])

        slice_infos = ray.get([m.actor.get_info.remote() for m in members])
        base_env = dict(
            TPU_NAME=os.getenv("TPU_NAME", "EMPTY"),
            TPU_VERSION=config.tpu_version,
            TPU_ZONE=os.getenv("TPU_ZONE", "EMPTY"),
            TPU_POD_COUNT=str(len(members)),
        )
        if config.execution_env:
            base_env.update({str(k): str(v) for k, v in config.execution_env.items() if v is not None})

        # Gather all host handles first: actor names must be suffixed when the
        # same swarm layout repeats on more than one host.
        handles_by_slice = [ray.get(m.actor.get_all_actors_in_pool.remote()) for m in members]
        total_hosts = sum(len(handles) for handles in handles_by_slice)

        refs_by_slice_host: list[list[ray.ObjectRef]] = []
        suffix_by_slice_host: list[list[str]] = []
        for slice_id, _member in enumerate(members):
            env_for_slice = dict(
                base_env,
                EXECUTOR_CALL_SLICE=str(slice_id),
                TPU_SLICE_NAME=slice_infos[slice_id].slice_name,
            )
            host_refs = []
            host_suffixes = []
            for host_idx, handle in enumerate(handles_by_slice[slice_id]):
                env_for_host = dict(env_for_slice, EXECUTOR_CALL_INDEX=str(host_idx))
                suffix = "" if total_hosts == 1 else f"-s{slice_id}h{host_idx}"
                host_suffixes.append(suffix)
                host_refs.append(
                    handle.run_swarm_remote_fn.remote(
                        payloads,
                        runtime_env=config.execution_env,
                        env=env_for_host,
                        namespace=config.namespace,
                        actor_name_suffix=suffix,
                    )
                )
            refs_by_slice_host.append(host_refs)
            suffix_by_slice_host.append(host_suffixes)

        # Each host call returns a list of per-run ObjectRefs; resolve the
        # outer actor calls first, then gather the run results. For class
        # runs the ref resolves when the actor is ready; the reported result
        # is its registration (name + namespace). Record each host's refs the
        # moment they are known so the error paths can cancel runs launched
        # on earlier hosts when a later host fails.
        runs_by_slice_host = []
        for host_refs in refs_by_slice_host:
            host_lists = []
            for h in host_refs:
                run_refs = ray.get(h)
                per_host_run_refs.append(run_refs)
                host_lists.append(run_refs)
            runs_by_slice_host.append(host_lists)

        results_nested = []
        for slice_id, host_lists in enumerate(runs_by_slice_host):
            host_results = []
            for host_idx, run_refs in enumerate(host_lists):
                raw = ray.get(run_refs)
                suffix = suffix_by_slice_host[slice_id][host_idx]
                run_results = []
                for run_id, (payload, value) in enumerate(zip(payloads, raw, strict=True)):
                    if payload["is_class"]:
                        run_results.append(
                            {
                                "kind": "actor",
                                "name": f"{payload['name']}{suffix}",
                                "namespace": config.namespace,
                                "run_id": run_id,
                                "run_name": payload["name"],
                            }
                        )
                    else:
                        run_results.append(value)
                host_results.append(run_results)
            results_nested.append(host_results)

        if flatten:
            results = [r for host_lists in results_nested for run_results in host_lists for r in run_results]
        else:
            results = results_nested

        return JobSucceeded(info, results)

    except InsufficientSlicesError:
        raise
    except (
        ray.exceptions.RayError,
        ray.exceptions.RayTaskError,
        ray.exceptions.TaskUnschedulableError,
    ) as e:
        for run_refs in per_host_run_refs:
            try:
                RayResources.cancel_all_futures(run_refs)
            except Exception:
                logger.debug("Failed to cancel swarm run futures after Ray error.", exc_info=True)
        s = str(e).lower()
        if ("preempt" in s) or ("unhealthy or preempted" in s) or ("owner died" in s) or ("owner has exited" in s):
            return JobPreempted(info, e)
        return handle_ray_error(info, e)
    except Exception as e:
        for run_refs in per_host_run_refs:
            try:
                RayResources.cancel_all_futures(run_refs)
            except Exception:
                logger.debug("Failed to cancel swarm run futures after failure.", exc_info=True)
        return JobFailed(info, e)
    finally:
        try:
            pool_manager.drain_actor_pool()
        except Exception:
            logger.debug("Failed to drain swarm actor pool.", exc_info=True)


def swarmed(
    config: SwarmConfigT,
    runs: Sequence[SwarmRun] | None = None,
) -> Callable:
    """Decorator form of :func:`swarm_execute` (TPU or GPU config).

    Decorates a function as a swarm workload; calling the decorated function
    launches the swarm and returns the :class:`~eray.core.status.JobStatus`.
    Bare run specs — ``SwarmRun(...)`` without ``fn`` — are bound to the
    decorated function; runs that carry their own ``fn`` (or class) keep it.

    Call-time arguments broadcast to every run: positional arguments are used
    where a run has no ``f_args`` of its own, and keyword arguments merge
    under each run's ``f_kwargs`` (the run's own entries win).

    Args:
        config: Swarm configuration (chip type, pod count, layout plan).
        runs: Optional per-run specs, one per run. Omitted: the decorated
            function is replicated according to the config's layout
            (``num_runs`` / ``chip_split``).

    Returns:
        A decorator producing a wrapper whose calls launch the swarm.

    Example:
        Sweep four learning-rate/warmup pairs, one chip each::

            @eray.swarmed(
                eray.SwarmConfig(tpu_version="v5p-8"),
                runs=[
                    eray.SwarmRun(name="lr3e4-w100", f_kwargs={"lr": 3e-4, "warmup": 100}),
                    eray.SwarmRun(name="lr1e4-w100", f_kwargs={"lr": 1e-4, "warmup": 100}),
                    eray.SwarmRun(name="lr3e4-w500", f_kwargs={"lr": 3e-4, "warmup": 500}),
                    eray.SwarmRun(name="lr1e4-w500", f_kwargs={"lr": 1e-4, "warmup": 500}),
                ],
            )
            def train(lr, warmup):
                ...

            status = train()

        Or replicate one function with broadcast arguments::

            @eray.swarmed(eray.SwarmConfig(tpu_version="v5p-8", num_runs=4))
            def serve(model_id):
                ...

            status = serve(model_id="qwen3.6-0.8b")
    """

    def decorator(fn: Callable) -> Callable:
        """Bind the decorated function into the swarm's run specs.

        Args:
            fn: The workload function (or class) the swarm runs.

        Returns:
            Wrapper that launches the swarm when called.
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """Launch the swarm with call-time arguments broadcast to runs.

            Args:
                *args: Positional arguments for runs without their own f_args.
                **kwargs: Keyword arguments merged under each run's f_kwargs.

            Returns:
                JobStatus from swarm_execute.
            """
            if runs is None:
                bound = [SwarmRun(fn=fn, f_args=args, f_kwargs=kwargs or None)]
                count = config.resolved_num_runs()
                if count is None:
                    raise ValueError("swarmed without runs needs SwarmConfig.num_runs or chip_split")
                bound = bound * count
            else:
                bound = [
                    replace(
                        run,
                        fn=run.fn if run.fn is not None else fn,
                        f_args=run.f_args if run.f_args else args,
                        f_kwargs={**kwargs, **(run.f_kwargs or {})} or None,
                    )
                    for run in runs
                ]
            return swarm_execute(bound, config)

        return wrapper

    return decorator


def shutdown_swarm(namespace: str, names: Sequence[str] | None = None) -> int:
    """Kill a swarm's named actors.

    Class runs launch detached actors, which outlive both the swarm's pool
    and the driver that created them — they hold their chips until killed.
    This tears down a swarm's runtimes in one call.

    Warning:
        Without ``names``, this kills **every** named actor in the namespace,
        including any that were not launched by the swarm. Give each swarm a
        dedicated :attr:`SwarmConfig.namespace` (never the driver's own Ray
        namespace), or pass the actor names from the swarm's registration
        results to scope the kill.

    Args:
        namespace: The :attr:`SwarmConfig.namespace` the swarm was launched
            with.
        names: Optional exact actor names to kill (e.g. the ``name`` fields
            of the swarm's registration results). None kills all named
            actors in the namespace.

    Returns:
        Number of actors killed.
    """
    from ray.util import list_named_actors

    wanted = set(names) if names is not None else None
    killed = 0
    for entry in list_named_actors(all_namespaces=True):
        if entry.get("namespace") != namespace:
            continue
        if wanted is not None and entry["name"] not in wanted:
            continue
        try:
            ray.kill(ray.get_actor(entry["name"], namespace=namespace), no_restart=True)
            killed += 1
        except Exception:
            logger.warning(f"Failed to kill swarm actor {entry['name']} in namespace {namespace}", exc_info=True)
    return killed

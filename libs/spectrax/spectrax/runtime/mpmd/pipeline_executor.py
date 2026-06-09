# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host wavefront executor for forward-only ``sxjit`` MPMD pipelines.

The regular ``sxjit`` forward path is intentionally direct: prepare an MPMD
plan, execute stage 0, pass its outputs to stage 1, and continue until the final
stage returns the user-facing pytree. That path is correct and simple, but an
inference server has a stronger requirement: while later stages are finishing
microbatch ``N``, earlier stages should be able to enqueue microbatch ``N + 1``.

This module provides that server-oriented execution layer. It consumes the same
private plan exposed by ``sxjit._mpmd_prepare`` and replays it as a host-side
pipeline wavefront. The executor does not invent model-specific semantics:
request packing, KV-cache layout, sampling, and final token routing remain the
caller/runtime's job. Spectrax owns the generic pieces that are independent of a
model family: prepared-stage reuse, activation transport, stage-local carries,
submesh placement, optional stage worker threads, and result reconstruction.

The executor is deliberately forward-only. Training schedules with true MPMD
forward/backward execution use ``sxcall``/``sxgrad`` and the schedule library;
this class targets low-latency decode/prefill pipelines where each microbatch is
a normal forward ``sxjit`` call.
"""

from __future__ import annotations

import contextlib
import dataclasses
import queue
import threading
import time
import typing as tp
from concurrent.futures import Future

import jax
from jax.sharding import Mesh, Sharding

from ..._internal.logging import get_logger
from ..types import MpMdMesh
from .runtime import (
    _apply_out_shardings,
    _assemble_invars,
    _assemble_invars_from_plan,
    _assemble_outputs,
    _InvarAssemblyPlan,
    _ordered_schedule_transport_scope,
    _OrderedScheduleTransportGate,
    _pipeline_transport_task_name,
    _prepare_invar_assembly_plan,
    _restore_result_treedef,
)

__all__ = [
    "MpmdPipelineDispatchStats",
    "MpmdPipelineExecutor",
]

PyTree: tp.TypeAlias = object
StageOutputs: tp.TypeAlias = tuple[PyTree, ...]
StageCallable: tp.TypeAlias = tp.Callable[..., StageOutputs]
InvarSource: tp.TypeAlias = tuple[str, int] | tuple[str, int, int] | tuple[str, int, int, PyTree]
InvarMap: tp.TypeAlias = list[InvarSource]
_CompiledStage: tp.TypeAlias = tuple[tp.Any, ...]


_PIPELINE_PROGRESS_DIAGNOSTICS: dict[str, int] = {}
_PIPELINE_DISPATCH_DIAGNOSTICS: dict[str, int] = {}
_PIPELINE_PROGRESS_LOCK = threading.Lock()
logger = get_logger("MPMD-Pipeline-Executor")


def _worker_jax_context() -> contextlib.AbstractContextManager[object]:
    """Return the worker-side JAX context manager."""
    return contextlib.nullcontext()


def _unpack_compiled_stage(
    entry: _CompiledStage,
    default_rank: int,
) -> tuple[StageCallable, Mesh, Sharding, Sharding | None, InvarMap, int]:
    """Return the stable fields from legacy and current sxjit stage plans."""
    if len(entry) >= 6:
        stage_jit, submesh, my_sh, next_sharding, invar_map, physical_rank = entry[:6]
        return (
            tp.cast(StageCallable, stage_jit),
            tp.cast(Mesh, submesh),
            tp.cast(Sharding, my_sh),
            tp.cast(Sharding | None, next_sharding),
            tp.cast(InvarMap, invar_map),
            int(physical_rank),
        )
    stage_jit, submesh, my_sh, next_sharding, invar_map = entry
    return (
        tp.cast(StageCallable, stage_jit),
        tp.cast(Mesh, submesh),
        tp.cast(Sharding, my_sh),
        tp.cast(Sharding | None, next_sharding),
        tp.cast(InvarMap, invar_map),
        default_rank,
    )


def _compiled_stage_consts(entry: _CompiledStage) -> tuple[PyTree, ...] | None:
    """Return runtime-passed stage constants for current plans, or ``None`` for legacy plans."""
    if len(entry) > 8:
        return tp.cast(tuple[PyTree, ...], entry[8])
    return None


def _stage_call_args(placed_consts: tuple[PyTree, ...] | None, invars: list[PyTree]) -> tuple[PyTree, ...]:
    """Build positional arguments for legacy and current compiled stage ABIs."""
    if placed_consts is None:
        return tuple(invars)
    return (placed_consts, *invars)


class _MpmdState(tp.TypedDict, total=False):
    """Prepared forward-only state exposed by ``sxjit._mpmd_prepare``.

    The state is intentionally a partially-typed mapping because it is a private
    handoff between ``sxjit`` and the host pipeline executor. The fixed entries
    below are the keys consumed by this executor; optional entries are present
    only when callers request explicit output shardings or modern dynamic-leaf
    remapping.
    """

    compiled: list[_CompiledStage]
    placed: dict[tuple[int, int], PyTree]
    dynamic: set[int]
    explicit_in_sh: dict[int, PyTree]
    fn_outvar_map: PyTree
    mpmd_mesh: MpMdMesh
    dynamic_flat_to_orig_flat: dict[int, int]
    out_shardings: PyTree
    result_treedef: PyTree


class _MpmdPreparedCallable(tp.Protocol):
    """Forward callable produced by ``sxjit`` with an exposed MPMD plan.

    ``MpmdPipelineExecutor`` intentionally depends on a tiny structural
    protocol instead of a concrete wrapper class. Tests can provide lightweight
    fakes, while production callers pass the object returned by ``sxjit``. The
    callable behavior is the semantic fallback; ``_mpmd_prepare`` is the fast
    path that exposes compiled per-stage executables and routing metadata.
    """

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> PyTree:
        """Execute the wrapped function with normal ``sxjit`` semantics.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result of invoking the wrapped callable or module.
        """
        ...

    def _mpmd_prepare(self, *args: PyTree) -> _MpmdState:
        """Return the prepared forward-only MPMD state for ``args``.

        The returned mapping is produced by ``sxjit`` and contains the compiled
        per-rank stage functions, input/output routing maps, explicit input and
        output shardings, pre-placed static leaves, and the original result
        treedef. The executor treats this state as immutable for a stable shape
        bucket so repeated decode calls can reuse it.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Return the prepared forward-only MPMD state for ``args``.
        """
        ...


@dataclasses.dataclass(frozen=True)
class MpmdPipelineDispatchStats:
    """Telemetry captured for the most recent pipeline dispatch.

    The fields are intentionally coarse. They are cheap enough to update during
    every decode step and are meant for runtime logging, regression checks, and
    quick performance triage. They do not replace XProf/XLA traces; they simply
    reveal whether time is going into prepare, host submission, final assembly,
    or waiting on stage worker futures.

    Attributes:
        stage_launches: Total number of compiled stage executable invocations.
            For a successful wavefront this is ``microbatches * num_stages``.
        microbatches: Number of same-shaped argument batches executed in this
            dispatch.
        stage_dispatch_time: Cumulative wall-clock seconds spent between
            enqueueing a worker-backed stage task and receiving its result. This
            is primarily meaningful when ``use_workers=True``.
        queue_wait_time: Currently the same lower-bound measurement as
            ``stage_dispatch_time``; kept separate for callers that already
            display queue-specific telemetry.
        prepare_time: Wall-clock seconds spent preparing ``sxjit`` state or
            flattening runtime arguments from a cached prepare entry.
        assemble_time: Wall-clock seconds spent rebuilding user-facing output
            pytrees from per-stage flat outputs.
        submit_time: Wall-clock seconds spent assembling stage inputs and
            enqueueing/executing stage work on the host.
        stage_submit_times_ms: Per-stage wall-clock milliseconds spent from
            stage input assembly through the ``stage_jit`` enqueue/worker
            submission call. JAX execution is asynchronous, so this is a
            boundary/launch timing signal, not a device-compute measurement.
        stage_assemble_times_ms: Per-stage wall-clock milliseconds spent
            materializing the positional input list for the stage call.
        stage_execute_times_ms: Per-stage wall-clock milliseconds spent inside
            the actual ``stage_jit`` enqueue call or resident-worker submit.
            This is still host dispatch time, not device execution time.
    """

    stage_launches: int
    microbatches: int
    stage_dispatch_time: float
    queue_wait_time: float
    prepare_time: float = 0.0
    assemble_time: float = 0.0
    submit_time: float = 0.0
    stage_submit_times_ms: tuple[float, ...] = ()
    stage_assemble_times_ms: tuple[float, ...] = ()
    stage_execute_times_ms: tuple[float, ...] = ()


@dataclasses.dataclass(frozen=True)
class _PreparedCall:
    """Prepared ``sxjit`` state plus flattened runtime argument leaves.

    Attributes:
        state: Private state returned by ``sxjit._mpmd_prepare``. It includes
            compiled stage functions, input/output maps, placed static values,
            sharding metadata, and result reconstruction metadata.
        flat_args: Positional call arguments flattened with ``jax.tree.leaves``.
            Stage input assembly indexes this list using Spectrax's flat-leaf
            routing maps.
    """

    state: _MpmdState
    flat_args: list[PyTree]


@dataclasses.dataclass
class _PrepareCacheEntry:
    """Shape-stable prepare result reused across same-bucket microbatches.

    Decode servers repeatedly call the same compiled bucket with different token
    and cache leaves but identical graph/weight/static arguments. Re-running
    ``_mpmd_prepare`` and re-flattening very large static pytrees for every
    token adds visible host overhead. This cache stores the prepared state plus
    enough flattening metadata to rebuild only the dynamic leaves on later calls.

    Attributes:
        state: Cached private ``sxjit`` MPMD state for the bucket.
        flat_args_template: Flattened argument list from the first call. Leaves
            belonging to runtime-static arguments are reused as-is; leaves for
            dynamic arguments are overwritten on each dispatch. Keeping this
            template avoids appending hundreds of stable graph/state leaves on
            every decode token.
        runtime_static_cache: Per-stage device-placed dynamic leaves that the
            caller declares stable for the bucket. Keys are ``(stage, flat_idx)``
            because each stage may require a different destination sharding.
        runtime_static_flat_args: Original flattened leaves for runtime-static
            positional arguments.
        arg_offsets: Starting flat-leaf index for each positional argument.
        arg_leaf_counts: Number of flat leaves contributed by each positional
            argument.
        invar_plans: Pre-classified stage input assembly plans derived from the
            cached state.
        runtime_static_invar_plan_cache: Invar plans with runtime-static dynamic
            slots already folded into the template. The key is the set of flat
            argument leaves treated as runtime-static for this bucket.
    """

    state: _MpmdState
    flat_args_template: list[PyTree]
    runtime_static_cache: dict[tuple[int, int], PyTree]
    runtime_static_flat_args: dict[int, PyTree]
    arg_offsets: tuple[int, ...]
    arg_leaf_counts: tuple[int, ...]
    invar_plans: tuple[_InvarAssemblyPlan, ...]
    runtime_static_invar_plan_cache: dict[frozenset[int], tuple[_InvarAssemblyPlan, ...]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class _StageTask:
    """One queued stage invocation for a worker-backed pipeline dispatch.

    Attributes:
        stage_jit: Compiled JAX callable for one physical pipeline rank.
        submesh: Rank-local mesh context used while invoking ``stage_jit``.
        invars: Fully materialized positional inputs for the stage.
    input_factory: Optional worker-side input assembly closure. Worker mode
            uses this to keep inter-stage activation transport off the caller
            thread while still serializing launches per physical rank.
        dependencies: Futures that must be complete before the worker should
            run ``input_factory``. Workers may receive tasks out of dependency
            order when one physical rank owns multiple virtual stages; delaying
            dequeue until dependencies are ready avoids head-of-line blocking.
        transport_gate: Optional deterministic transport gate shared by this
            dispatch.
        transport_task_names: Ordered transport names the input factory may
            issue while assembling this stage's inputs.
        future: Future completed by the worker with the stage's flat outputs.
    """

    stage_jit: StageCallable
    submesh: Mesh
    placed_consts: tuple[PyTree, ...] | None
    invars: list[PyTree] | None
    input_factory: tp.Callable[[], list[PyTree]] | None
    microbatch_idx: int | None
    stage_idx: int | None
    dependencies: tuple[Future[StageOutputs], ...]
    transport_gate: _OrderedScheduleTransportGate | None
    transport_task_names: tuple[str, ...]
    future: Future[StageOutputs]


class _StageWorker:
    """Daemon worker pinned logically to one physical pipeline rank.

    Worker mode is optional. The default inline path relies on JAX returning
    asynchronous device futures, which is usually enough to enqueue a wavefront
    quickly. Worker mode is useful for experiments where each physical stage
    should have an independent host thread, for example when caller-side Python
    work or mesh context management becomes visible in traces.
    """

    def __init__(self, *, rank: int) -> None:
        """Start a resident worker for ``rank``.

        Args:
            rank: Physical pipeline rank this worker represents. The value is
                used for naming/debugging only; device placement is still driven
                by the ``submesh`` carried on each submitted task.
        """
        self.rank = int(rank)
        self._queue: queue.Queue[_StageTask | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"spx-mpmd-stage-{rank}", daemon=True)
        self._thread.start()

    def submit(
        self,
        *,
        stage_jit: StageCallable,
        submesh: Mesh,
        placed_consts: tuple[PyTree, ...] | None,
        invars: list[PyTree] | None = None,
        input_factory: tp.Callable[[], list[PyTree]] | None = None,
        microbatch_idx: int | None = None,
        stage_idx: int | None = None,
        dependencies: tuple[Future[StageOutputs], ...] = (),
        transport_gate: _OrderedScheduleTransportGate | None = None,
        transport_task_names: tuple[str, ...] = (),
    ) -> Future[StageOutputs]:
        """Queue one stage invocation and return its completion future.

        Args:
            stage_jit: Compiled per-rank executable.
            submesh: Mesh context to enter before calling ``stage_jit``.
            invars: Positional inputs already placed for the destination stage.
            input_factory: Optional callable that materializes the positional
                inputs inside this worker thread. Exactly one of ``invars`` or
                ``input_factory`` must be supplied.
            microbatch_idx: Optional microbatch index for diagnostics.
            stage_idx: Optional logical stage index for diagnostics.
            dependencies: Futures whose outputs feed this stage task.
            transport_gate: Deterministic transport gate for multi-controller
                pair collectives.
            transport_task_names: Ordered transfer names this task may issue.
        Returns:
            A future that resolves to the stage's flat output tuple, or carries
            the raised exception from the worker thread.
        """
        future: Future[StageOutputs] = Future()
        self._queue.put(
            _StageTask(
                stage_jit=stage_jit,
                submesh=submesh,
                placed_consts=placed_consts,
                invars=invars,
                input_factory=input_factory,
                microbatch_idx=microbatch_idx,
                stage_idx=stage_idx,
                dependencies=dependencies,
                transport_gate=transport_gate,
                transport_task_names=transport_task_names,
                future=future,
            )
        )
        return future

    @staticmethod
    def _task_dependencies_ready(task: _StageTask) -> bool:
        """Return whether all upstream stage futures have completed."""
        return all(dependency.done() for dependency in task.dependencies)

    @staticmethod
    def _task_transport_ready(task: _StageTask) -> bool:
        """Return whether the task's first ordered transport is ready."""
        if task.transport_gate is None:
            return True
        return task.transport_gate.ready_for(task.transport_task_names)

    @staticmethod
    def _ready_task_priority(task: _StageTask, queue_index: int) -> tuple[int, int, int, int]:
        """Return a priority key for ready tasks sharing one physical worker."""
        stage_idx = task.stage_idx if task.stage_idx is not None else -1
        microbatch_idx = task.microbatch_idx if task.microbatch_idx is not None else 0
        event_position = task.transport_gate.position_for(task.transport_task_names) if task.transport_gate else None
        event_priority = -event_position if event_position is not None else -(10**12)
        return (event_priority, stage_idx, -microbatch_idx, -queue_index)

    def _maybe_log_timing(
        self,
        task: _StageTask,
        *,
        assemble_time: float,
        execute_time: float,
        total_time: float,
    ) -> None:
        del task, assemble_time, execute_time, total_time

    def shutdown(self) -> None:
        """Ask the worker to exit and wait briefly for its thread to finish."""
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        """Worker event loop that executes queued stage tasks under their submesh."""
        pending: list[_StageTask] = []
        while True:
            try:
                item = self._queue.get(timeout=0.01)
            except queue.Empty:
                item = None
                received_item = False
            else:
                received_item = True

            if received_item:
                if item is None:
                    self._queue.task_done()
                    for task in pending:
                        task.future.cancel()
                    return
                pending.append(item)
                self._queue.task_done()

            ready_index: int | None = None
            ready_priority: tuple[int, int, int] | None = None
            for index, candidate in enumerate(pending):
                if not self._task_dependencies_ready(candidate):
                    continue
                if not self._task_transport_ready(candidate):
                    continue
                priority = self._ready_task_priority(candidate, index)
                if ready_priority is None or priority > ready_priority:
                    ready_index = index
                    ready_priority = priority
            if ready_index is None:
                continue
            task = pending.pop(ready_index)
            if task.future.set_running_or_notify_cancel():
                try:
                    t_total = time.time()
                    with jax.named_scope(f"spectrax/mpmd/pipeline/worker/stage_{self.rank}"):
                        t_assemble = time.time()
                        if task.input_factory is not None:
                            invars = task.input_factory()
                        elif task.invars is not None:
                            invars = task.invars
                        else:
                            raise RuntimeError(
                                f"internal error: worker stage {self.rank} received no input assembly source"
                            )
                        assemble_time = time.time() - t_assemble
                        t_execute = time.time()
                        with _worker_jax_context():
                            with task.submesh:
                                stage_jit = task.stage_jit
                                placed_consts = task.placed_consts
                                stage_invars = invars

                                out = stage_jit(*_stage_call_args(placed_consts, stage_invars))
                        execute_time = time.time() - t_execute
                    self._maybe_log_timing(
                        task,
                        assemble_time=assemble_time,
                        execute_time=execute_time,
                        total_time=time.time() - t_total,
                    )
                    task.future.set_result(out)
                except BaseException as exc:
                    task.future.set_exception(exc)


class MpmdPipelineExecutor:
    """Host wavefront executor for forward-only ``sxjit`` MPMD plans.

    ``dispatch`` executes one call with the same semantics as invoking the
    wrapped ``sxjit`` function directly. ``dispatch_many`` accepts same-shaped
    microbatches and runs the physical pipeline as a host-side wavefront:
    stage 0 can enqueue microbatch 1 before the final stage has finished
    microbatch 0. JAX/XLA still owns device-side dependency ordering; the host
    only sequences launches whose inputs are available.

    The executor is intentionally small and reusable. It has no opinion about
    request queues, paged attention, logits processors, or samplers. Callers pass
    already-packed argument tuples, optional stage-local carry wiring for KV
    state, and an optional prepare-cache key for a stable decode bucket.
    """

    def __init__(self, *, stage_meshes: tp.Sequence[Mesh] | None = None, use_workers: bool = False) -> None:
        """Create an executor.

        Args:
            stage_meshes: Optional physical stage meshes used as a mesh lookup
                fallback for tests or older prepared states. Modern ``sxjit``
                plans carry the owning MPMD mesh directly.
            use_workers: When ``False`` stage calls are enqueued inline on the
                caller thread. When ``True`` the executor creates one resident
                daemon worker per physical stage and submits stage calls through
                futures.
        """
        self.stage_meshes = tuple(stage_meshes or ())
        self.use_workers = bool(use_workers)
        self._workers: list[_StageWorker] = []
        self._worker_count = 0
        self._last_stats = MpmdPipelineDispatchStats(0, 0, 0.0, 0.0)
        self._prepare_cache: dict[tp.Hashable, _PrepareCacheEntry] = {}

    @property
    def last_stats(self) -> MpmdPipelineDispatchStats:
        """Return telemetry from the most recent ``dispatch``/``dispatch_many`` call.

        Returns:
            Return telemetry from the most recent ``dispatch``/``dispatch_many`` call.
        """
        return self._last_stats

    def _shutdown_workers(self) -> None:
        """Stop resident workers while preserving prepared-plan caches."""
        for worker in self._workers:
            worker.shutdown()
        self._workers = []
        self._worker_count = 0

    def shutdown(self) -> None:
        """Stop resident workers and clear bucket-local prepare caches."""
        self._shutdown_workers()
        self._prepare_cache.clear()

    def clear_prepare_cache(self) -> None:
        """Drop cached ``sxjit`` prepare state while keeping resident workers alive.

        Use this when a caller invalidates a decode bucket, swaps weights, or
        changes a static call argument but wants to keep the executor object and
        any worker threads around.
        """
        self._prepare_cache.clear()

    def dispatch(self, sxjit_fn: _MpmdPreparedCallable, *args: PyTree) -> PyTree:
        """Execute one ``sxjit`` call through the pipeline executor.

        This is a convenience wrapper around ``dispatch_many`` for callers that
        do not need wavefront overlap. It still uses the same stage input
        assembly and result reconstruction path, which makes it useful for
        validating pipeline executor correctness against direct ``sxjit`` calls.

        Args:
            sxjit_fn: Sxjit fn value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        outputs = self.dispatch_many(sxjit_fn, (args,))
        return outputs[0]

    def dispatch_many(
        self,
        sxjit_fn: _MpmdPreparedCallable,
        arg_batches: tp.Iterable[tuple[PyTree, ...]],
        *,
        carry_input_output_map: tp.Mapping[int, tp.Mapping[int, int]] | None = None,
        prepare_cache_key: tp.Hashable | None = None,
        runtime_static_argnums: tp.Iterable[int] | None = None,
    ) -> list[PyTree]:
        """Execute same-shaped microbatches as a pipeline wavefront.

        The executor consumes SpectraX's private forward-only MPMD plan. For a
        single call it is semantically equivalent to invoking the ``sxjit``
        function normally; for multiple calls it walks the physical pipeline in
        wave order:

        ``(mb0, stage0) -> (mb1, stage0), (mb0, stage1) -> ...``

        When a stage-local carry map is supplied, recurrent leaves such as KV
        cache pages are sourced from the previous output of the same stage. That
        keeps cache ownership local to each pipeline rank while allowing the
        host to overlap independent stage launches.

        Args:
            sxjit_fn: A forward-only ``spectrax.sxjit`` callable exposing
                ``_mpmd_prepare``.
            arg_batches: Iterable of positional-argument tuples, one per
                microbatch. All microbatches must resolve to compatible
                forward-only stage plans.
            carry_input_output_map: Optional stage-local recurrent-state
                mapping. Keys are stage indices. Values map original flat input
                leaf indices to output positions produced by that same stage in
                the previous microbatch. This lets callers pipeline stateful
                decode: stage 0 consumes stage 0's prior KV output while stage 1
                independently consumes stage 1's prior KV output.

            prepare_cache_key: Optional hashable bucket key. When supplied, the
                executor caches the prepared stage plan, static flattening
                metadata, and stage input assembly plans for reuse on later
                calls with the same bucket shape.
            runtime_static_argnums: Positional argument indices whose leaves are
                dynamic from Spectrax's tracing perspective but stable for the
                runtime bucket, such as graph definitions or weights in an
                inference engine. Those leaves are flattened and placed once per
                cache entry.

        Returns:
            One output pytree per microbatch, in input order.
        """
        t_prepare = time.time()
        arg_batches = list(arg_batches)
        cached_entry = self._prepare_cache.get(prepare_cache_key) if prepare_cache_key is not None else None
        if cached_entry is None:
            prepared = [self._prepare_call(sxjit_fn, args) for args in arg_batches]
            if prepare_cache_key is not None and prepared:
                self._prepare_cache[prepare_cache_key] = self._make_cache_entry(
                    prepared[0].state,
                    arg_batches[0],
                    prepared[0].flat_args,
                    runtime_static_argnums=runtime_static_argnums,
                )
                cached_entry = self._prepare_cache[prepare_cache_key]
        else:
            prepared = [
                _PreparedCall(
                    state=cached_entry.state,
                    flat_args=self._flatten_args_with_runtime_static_cache(
                        cached_entry,
                        args,
                        runtime_static_argnums=runtime_static_argnums,
                    ),
                )
                for args in arg_batches
            ]
        prepare_time = time.time() - t_prepare
        if not prepared:
            self._last_stats = MpmdPipelineDispatchStats(0, 0, 0.0, 0.0)
            return []

        compiled: list[_CompiledStage] = prepared[0].state["compiled"]
        for call in prepared[1:]:
            other = call.state["compiled"]
            if other is not compiled:
                raise ValueError(
                    "MpmdPipelineExecutor.dispatch_many requires every microbatch to use the same compiled plan. "
                    "Bucket or pad microbatches to the same shape before calling it."
                )

        carry_map = self._normalize_carry_map(carry_input_output_map, len(compiled), len(prepared[0].flat_args))
        runtime_static_flat_indices = self._runtime_static_flat_indices(
            cached_entry,
            runtime_static_argnums,
            len(arg_batches[0]),
        )
        runtime_static_cache = cached_entry.runtime_static_cache if cached_entry is not None else None
        invar_plans = cached_entry.invar_plans if cached_entry is not None else None
        runtime_static_plan_key: frozenset[int] | None = None
        runtime_static_plan_updates: list[_InvarAssemblyPlan | None] | None = None
        if cached_entry is not None and runtime_static_flat_indices:
            runtime_static_plan_key = frozenset(runtime_static_flat_indices)
            resolved_plans = cached_entry.runtime_static_invar_plan_cache.get(runtime_static_plan_key)
            if resolved_plans is not None:
                invar_plans = resolved_plans
            elif invar_plans is not None:
                runtime_static_plan_updates = [None] * len(invar_plans)
        if invar_plans is None:
            invar_plans = self._make_invar_plans(prepared[0].state)
        compiled_stage_info = [_unpack_compiled_stage(stage, idx) for idx, stage in enumerate(compiled)]
        physical_ranks = [int(stage_info[5]) for stage_info in compiled_stage_info]
        physical_rank_count = max(physical_ranks, default=-1) + 1
        rank_submeshes: list[Mesh] = []
        if physical_rank_count > 0:
            by_physical: list[Mesh | None] = [None] * physical_rank_count
            for stage_info in compiled_stage_info:
                physical_rank = int(stage_info[5])
                if by_physical[physical_rank] is None:
                    by_physical[physical_rank] = stage_info[1]
            rank_submeshes = [
                submesh if submesh is not None else compiled_stage_info[rank][1]
                for rank, submesh in enumerate(by_physical)
            ]
        rank_device_sets = [set(submesh.devices.flat) for submesh in rank_submeshes]
        use_worker_dispatch = self.use_workers
        if use_worker_dispatch:
            self._log_deterministic_worker_launch_if_needed(rank_device_sets)
        if use_worker_dispatch:
            self._ensure_workers(physical_rank_count)
        mpmd_mesh = self._resolve_mpmd_mesh(prepared[0].state, rank_submeshes, sxjit_fn=sxjit_fn)
        futures: list[list[Future[StageOutputs] | None]] = [[None] * len(compiled) for _ in prepared]
        cluster_outputs: list[list[StageOutputs | None]] = [[None] * len(compiled) for _ in prepared]
        stage_dispatch_time = 0.0
        queue_wait_time = 0.0
        submit_time = 0.0
        stage_submit_times_ms = [0.0] * len(compiled)
        stage_assemble_times_ms = [0.0] * len(compiled)
        stage_execute_times_ms = [0.0] * len(compiled)

        if not use_worker_dispatch and len(prepared) == 1:
            call = prepared[0]
            single_cluster_outputs: list[StageOutputs | None] = [None] * len(compiled)
            for stage_idx, stage_entry in enumerate(compiled):
                stage_jit, submesh, my_sh, _, invar_map, physical_rank = _unpack_compiled_stage(
                    stage_entry,
                    stage_idx,
                )
                placed_consts = _compiled_stage_consts(stage_entry)
                t_submit = time.time()
                prev_outputs = single_cluster_outputs[stage_idx - 1] if stage_idx > 0 else ()
                if prev_outputs is None:
                    raise RuntimeError(f"internal error: missing previous output for stage {stage_idx}")

                t_assemble_stage = time.time()
                with jax.named_scope(f"spectrax/mpmd/pipeline/single/stage_{stage_idx}/assemble"):
                    invar_plan = invar_plans[stage_idx] if invar_plans is not None else None
                    invars = self._assemble_stage_invars(
                        call=call,
                        stage_idx=physical_rank,
                        logical_stage_idx=stage_idx,
                        microbatch_idx=0,
                        invar_map=invar_map,
                        invar_plan=invar_plan,
                        my_sh=my_sh,
                        rank_devices=rank_device_sets[physical_rank],
                        rank_submeshes=rank_submeshes,
                        mpmd_mesh=mpmd_mesh,
                        prev_outputs=prev_outputs,
                        all_cluster_outputs=single_cluster_outputs,
                        runtime_static_flat_indices=runtime_static_flat_indices,
                        runtime_static_cache=runtime_static_cache,
                    )
                    if (
                        runtime_static_plan_updates is not None
                        and invar_plan is not None
                        and runtime_static_flat_indices is not None
                    ):
                        runtime_static_plan_updates[stage_idx] = self._fold_runtime_static_slots_into_plan(
                            invar_plan,
                            invars,
                            runtime_static_flat_indices,
                        )
                stage_assemble_elapsed = time.time() - t_assemble_stage
                stage_assemble_times_ms[stage_idx] += stage_assemble_elapsed * 1000.0

                t_execute_stage = time.time()
                with jax.named_scope(f"spectrax/mpmd/pipeline/single/stage_{stage_idx}/execute"):
                    with submesh:
                        single_cluster_outputs[stage_idx] = stage_jit(*_stage_call_args(placed_consts, invars))
                stage_execute_elapsed = time.time() - t_execute_stage
                stage_execute_times_ms[stage_idx] += stage_execute_elapsed * 1000.0

                stage_submit_elapsed = time.time() - t_submit
                submit_time += stage_submit_elapsed
                stage_submit_times_ms[stage_idx] += stage_submit_elapsed * 1000.0

            t_assemble = time.time()
            result = self._assemble_result(call, single_cluster_outputs)
            assemble_time = time.time() - t_assemble
            if (
                cached_entry is not None
                and runtime_static_plan_key is not None
                and runtime_static_plan_updates is not None
                and all(plan is not None for plan in runtime_static_plan_updates)
            ):
                cached_entry.runtime_static_invar_plan_cache[runtime_static_plan_key] = tuple(
                    runtime_static_plan_updates
                )
            self._last_stats = MpmdPipelineDispatchStats(
                stage_launches=len(compiled),
                microbatches=1,
                stage_dispatch_time=0.0,
                queue_wait_time=0.0,
                prepare_time=prepare_time,
                assemble_time=assemble_time,
                submit_time=submit_time,
                stage_submit_times_ms=tuple(stage_submit_times_ms),
                stage_assemble_times_ms=tuple(stage_assemble_times_ms),
                stage_execute_times_ms=tuple(stage_execute_times_ms),
            )
            return [result]

        def wait_stage(mb_idx: int, stage_idx: int) -> StageOutputs:
            nonlocal stage_dispatch_time, queue_wait_time
            with output_lock:
                ready = cluster_outputs[mb_idx][stage_idx]
            if ready is not None:
                return ready
            if not use_worker_dispatch:
                raise RuntimeError(f"internal error: stage {stage_idx} microbatch {mb_idx} was not dispatched")
            future = futures[mb_idx][stage_idx]
            if future is None:
                raise RuntimeError(f"internal error: stage {stage_idx} microbatch {mb_idx} was not submitted")
            t_wait = time.time()
            ready = future.result()
            elapsed = time.time() - t_wait
            with stats_lock:
                stage_dispatch_time += elapsed
                queue_wait_time += max(0.0, elapsed)
            with output_lock:
                if cluster_outputs[mb_idx][stage_idx] is None:
                    cluster_outputs[mb_idx][stage_idx] = ready
            return ready

        num_microbatches = len(prepared)
        num_stages = len(compiled)
        retained_outputs = self._retained_output_stages(prepared[0].state, num_stages)
        output_ref_counts = self._output_ref_counts(
            num_microbatches=num_microbatches,
            num_stages=num_stages,
            invar_plans=invar_plans,
            carry_map=carry_map,
            retained_outputs=retained_outputs,
        )
        output_lock = threading.Lock()
        stats_lock = threading.Lock()
        transport_gate = self._pipeline_transport_gate(
            num_microbatches=num_microbatches,
            num_stages=num_stages,
            invar_plans=invar_plans,
            rank_device_sets=rank_device_sets,
            logical_physical_ranks=physical_ranks,
            use_worker_dispatch=use_worker_dispatch,
        )

        def release_output(mb_idx: int, stage_idx: int, *, count: int = 1) -> None:
            if count <= 0 or stage_idx < 0 or stage_idx >= num_stages:
                return
            key = (mb_idx, stage_idx)
            with output_lock:
                remaining = output_ref_counts.get(key)
                if remaining is None:
                    return
                remaining -= count
                if remaining > 0:
                    output_ref_counts[key] = remaining
                    return
                output_ref_counts.pop(key, None)
                if stage_idx not in retained_outputs:
                    cluster_outputs[mb_idx][stage_idx] = None

        def release_consumed_inputs(mb_idx: int, stage_idx: int, invar_plan: _InvarAssemblyPlan | None) -> None:
            if invar_plan is None:
                return
            if invar_plan.prev_slots:
                release_output(mb_idx, stage_idx - 1, count=len(invar_plan.prev_slots))
            stage_uses: dict[int, int] = {}
            for _out_pos, src_rank, _src_pos, _edge_sharding, _src_physical_rank in invar_plan.stage_slots:
                stage_uses[src_rank] = stage_uses.get(src_rank, 0) + 1
            for src_rank, count in stage_uses.items():
                release_output(mb_idx, src_rank, count=count)

        def stage_dependencies(
            *,
            mb_idx: int,
            stage_idx: int,
            invar_plan: _InvarAssemblyPlan | None,
        ) -> tuple[Future[StageOutputs], ...]:
            if not use_worker_dispatch:
                return ()

            dependencies: list[Future[StageOutputs]] = []
            seen: set[tuple[int, int]] = set()

            def add_dependency(dep_mb_idx: int, dep_stage_idx: int) -> None:
                key = (dep_mb_idx, dep_stage_idx)
                if key in seen:
                    return
                if dep_mb_idx < 0 or dep_mb_idx >= num_microbatches or dep_stage_idx < 0 or dep_stage_idx >= num_stages:
                    raise RuntimeError(
                        "internal error: pipeline stage dependency is out of range "
                        f"(microbatch={mb_idx}, stage={stage_idx}, dependency={key})."
                    )
                future = futures[dep_mb_idx][dep_stage_idx]
                if future is None:
                    raise RuntimeError(
                        "internal error: pipeline stage was submitted before an upstream dependency "
                        f"(microbatch={mb_idx}, stage={stage_idx}, dependency={key})."
                    )
                seen.add(key)
                dependencies.append(future)

            if invar_plan is not None:
                if invar_plan.prev_slots:
                    add_dependency(mb_idx, stage_idx - 1)
                for _out_pos, src_rank, _src_pos, _edge_sharding, _src_physical_rank in invar_plan.stage_slots:
                    add_dependency(mb_idx, int(src_rank))
            elif stage_idx > 0:
                for src_stage in range(stage_idx):
                    add_dependency(mb_idx, src_stage)

            if carry_map.get(stage_idx) and mb_idx > 0:
                add_dependency(mb_idx - 1, stage_idx)

            return tuple(dependencies)

        for wave_idx in range(num_microbatches + num_stages - 1):
            hi_stage = min(wave_idx, num_stages - 1)
            for stage_idx in range(0, hi_stage + 1):
                mb_idx = wave_idx - stage_idx
                if mb_idx < 0 or mb_idx >= num_microbatches:
                    continue
                if futures[mb_idx][stage_idx] is not None:
                    continue

                t_submit = time.time()
                stage_entry = compiled[stage_idx]
                stage_jit, submesh, my_sh, _, invar_map, physical_rank = _unpack_compiled_stage(
                    stage_entry,
                    stage_idx,
                )
                placed_consts = _compiled_stage_consts(stage_entry)
                rank_devices = rank_device_sets[physical_rank]

                invar_plan = invar_plans[stage_idx] if invar_plans is not None else None

                def build_stage_invars(
                    *,
                    mb_idx: int = mb_idx,
                    stage_idx: int = stage_idx,
                    physical_rank: int = physical_rank,
                    invar_map: InvarMap = invar_map,
                    invar_plan: _InvarAssemblyPlan | None = invar_plan,
                    my_sh: Sharding = my_sh,
                    rank_devices: set[jax.Device] = rank_devices,
                ) -> list[PyTree]:
                    t_assemble_stage = time.time()
                    prev_outputs: StageOutputs = ()
                    if invar_plan is not None:
                        if invar_plan.prev_slots:
                            prev_outputs = wait_stage(mb_idx, stage_idx - 1)
                        for _out_pos, src_rank, _src_pos, _edge_sharding, _src_physical_rank in invar_plan.stage_slots:
                            wait_stage(mb_idx, src_rank)
                    elif stage_idx > 0:
                        prev_outputs = wait_stage(mb_idx, stage_idx - 1)
                        for src_stage in range(stage_idx):
                            wait_stage(mb_idx, src_stage)

                    call = prepared[mb_idx]
                    stage_carries = carry_map.get(stage_idx, {})
                    if stage_carries and mb_idx > 0:
                        previous_stage_outputs = wait_stage(mb_idx - 1, stage_idx)
                        flat_args = list(call.flat_args)
                        for orig_flat_idx, stage_out_pos in stage_carries.items():
                            flat_args[orig_flat_idx] = previous_stage_outputs[stage_out_pos]
                        call = _PreparedCall(state=call.state, flat_args=flat_args)
                        release_output(mb_idx - 1, stage_idx, count=len(stage_carries))

                    with jax.named_scope(f"spectrax/mpmd/pipeline/microbatch_{mb_idx}/stage_{stage_idx}/assemble"):
                        with _worker_jax_context():
                            with _ordered_schedule_transport_scope(transport_gate):
                                invars = self._assemble_stage_invars(
                                    call=call,
                                    stage_idx=physical_rank,
                                    logical_stage_idx=stage_idx,
                                    microbatch_idx=mb_idx,
                                    invar_map=invar_map,
                                    invar_plan=invar_plan,
                                    my_sh=my_sh,
                                    rank_devices=rank_devices,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                    prev_outputs=prev_outputs,
                                    all_cluster_outputs=cluster_outputs[mb_idx],
                                    runtime_static_flat_indices=runtime_static_flat_indices,
                                    runtime_static_cache=runtime_static_cache,
                                )
                        release_consumed_inputs(mb_idx, stage_idx, invar_plan)
                        if (
                            runtime_static_plan_updates is not None
                            and mb_idx == 0
                            and invar_plan is not None
                            and runtime_static_flat_indices is not None
                        ):
                            runtime_static_plan_updates[stage_idx] = self._fold_runtime_static_slots_into_plan(
                                invar_plan,
                                invars,
                                runtime_static_flat_indices,
                            )
                    stage_assemble_elapsed = time.time() - t_assemble_stage
                    with stats_lock:
                        stage_assemble_times_ms[stage_idx] += stage_assemble_elapsed * 1000.0
                    return invars

                t_execute_stage = time.time()
                if use_worker_dispatch:
                    futures[mb_idx][stage_idx] = self._workers[physical_rank].submit(
                        stage_jit=stage_jit,
                        submesh=submesh,
                        placed_consts=placed_consts,
                        input_factory=build_stage_invars,
                        microbatch_idx=mb_idx,
                        stage_idx=stage_idx,
                        dependencies=stage_dependencies(
                            mb_idx=mb_idx,
                            stage_idx=stage_idx,
                            invar_plan=invar_plan,
                        ),
                        transport_gate=transport_gate,
                        transport_task_names=self._pipeline_task_event_names(
                            mb_idx=mb_idx,
                            stage_idx=stage_idx,
                            invar_plan=invar_plan,
                        ),
                    )
                else:
                    invars = build_stage_invars()
                    with jax.named_scope(f"spectrax/mpmd/pipeline/microbatch_{mb_idx}/stage_{stage_idx}/execute"):
                        with submesh:
                            cluster_outputs[mb_idx][stage_idx] = stage_jit(*_stage_call_args(placed_consts, invars))
                stage_execute_elapsed = time.time() - t_execute_stage
                stage_execute_times_ms[stage_idx] += stage_execute_elapsed * 1000.0
                stage_submit_elapsed = time.time() - t_submit
                submit_time += stage_submit_elapsed
                stage_submit_times_ms[stage_idx] += stage_submit_elapsed * 1000.0

        final_stage = num_stages - 1
        for mb_idx in range(num_microbatches):
            wait_stage(mb_idx, final_stage)

        t_assemble = time.time()
        results = [self._assemble_result(call, outputs) for call, outputs in zip(prepared, cluster_outputs, strict=True)]
        assemble_time = time.time() - t_assemble
        if (
            cached_entry is not None
            and runtime_static_plan_key is not None
            and runtime_static_plan_updates is not None
            and all(plan is not None for plan in runtime_static_plan_updates)
        ):
            cached_entry.runtime_static_invar_plan_cache[runtime_static_plan_key] = tuple(runtime_static_plan_updates)
        self._last_stats = MpmdPipelineDispatchStats(
            stage_launches=len(compiled) * len(prepared),
            microbatches=len(prepared),
            stage_dispatch_time=stage_dispatch_time,
            queue_wait_time=queue_wait_time,
            prepare_time=prepare_time,
            assemble_time=assemble_time,
            submit_time=submit_time,
            stage_submit_times_ms=tuple(stage_submit_times_ms),
            stage_assemble_times_ms=tuple(stage_assemble_times_ms),
            stage_execute_times_ms=tuple(stage_execute_times_ms),
        )
        return results

    def _normalize_carry_map(
        self,
        carry_input_output_map: tp.Mapping[int, tp.Mapping[int, int]] | None,
        num_stages: int,
        num_flat_args: int,
    ) -> dict[int, dict[int, int]]:
        """Validate and normalize caller-provided stage-local carry wiring.

        The carry map is intentionally expressed in flat-leaf positions because
        it sits below model/runtime abstractions. Each stage map says: before
        dispatching stage ``S`` for microbatch ``N > 0``, replace original flat
        input ``I`` with output position ``O`` from stage ``S`` of microbatch
        ``N - 1``. This preserves stage-local cache ownership and avoids routing
        one stage's KV state through another stage.

        Args:
            carry_input_output_map: Carry input output map value consumed by this operation.
            num_stages: Num stages value consumed by this operation.
            num_flat_args: Num flat args value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not carry_input_output_map:
            return {}
        normalized: dict[int, dict[int, int]] = {}
        for stage_idx_raw, stage_map_raw in carry_input_output_map.items():
            stage_idx = int(stage_idx_raw)
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(f"carry_input_output_map contains invalid stage index {stage_idx}.")
            stage_map: dict[int, int] = {}
            for orig_flat_idx_raw, stage_out_pos_raw in stage_map_raw.items():
                orig_flat_idx = int(orig_flat_idx_raw)
                stage_out_pos = int(stage_out_pos_raw)
                if orig_flat_idx < 0 or orig_flat_idx >= num_flat_args:
                    raise ValueError(
                        f"carry_input_output_map stage {stage_idx} references invalid flat input {orig_flat_idx}."
                    )
                if stage_out_pos < 0:
                    raise ValueError(
                        f"carry_input_output_map stage {stage_idx} references invalid output {stage_out_pos}."
                    )
                stage_map[orig_flat_idx] = stage_out_pos
            if stage_map:
                normalized[stage_idx] = stage_map
        return normalized

    def _prepare_call(self, sxjit_fn: _MpmdPreparedCallable, args: tuple[PyTree, ...]) -> _PreparedCall:
        """Ask ``sxjit`` for its MPMD stage plan and flatten runtime leaves.

        Args:
            sxjit_fn: A callable implementing ``_mpmd_prepare``.
            args: Positional arguments for one microbatch.

        Returns:
            A prepared call containing the immutable stage plan plus the flat
            dynamic leaves for this specific microbatch.

        Raises:
            TypeError: If the callable is not an ``sxjit`` MPMD wrapper or the
                prepared state is not a forward-only compiled plan.
        """
        prepare = getattr(sxjit_fn, "_mpmd_prepare", None)
        if prepare is None:
            raise TypeError("MpmdPipelineExecutor requires a SpectraX sxjit function with _mpmd_prepare.")
        state = tp.cast(_MpmdState, dict(prepare(*args)))
        if "compiled" not in state:
            raise TypeError("MpmdPipelineExecutor only supports forward-only sxjit plans.")
        return _PreparedCall(state=state, flat_args=jax.tree.leaves(args))

    def _make_cache_entry(
        self,
        state: _MpmdState,
        args: tuple[PyTree, ...],
        flat_args: list[PyTree],
        *,
        runtime_static_argnums: tp.Iterable[int] | None,
    ) -> _PrepareCacheEntry:
        """Build a reusable prepare-cache entry for a stable bucket shape.

        The entry records positional-argument leaf offsets so later calls can
        flatten only non-static arguments while replaying cached leaves for
        runtime-static ones. It also precomputes stage input assembly plans from
        the prepared ``sxjit`` state so decode steps avoid repeatedly scanning
        full invar maps.

        Args:
            state: SpectraX state tree or transform state passed into the operation.
            args: Positional arguments forwarded to the wrapped callable.
            flat_args: Flat args value consumed by this operation.
            runtime_static_argnums: Runtime static argnums value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        offsets: list[int] = []
        counts: list[int] = []
        offset = 0
        for arg in args:
            count = len(jax.tree.leaves(arg))
            offsets.append(offset)
            counts.append(count)
            offset += count
        static_argnums = self._normalize_runtime_static_argnums(runtime_static_argnums, len(args))
        static_flat_args: dict[int, PyTree] = {}
        for argnum in static_argnums:
            start = offsets[argnum]
            count = counts[argnum]
            for flat_idx in range(start, start + count):
                static_flat_args[flat_idx] = flat_args[flat_idx]
        return _PrepareCacheEntry(
            state=state,
            flat_args_template=list(flat_args),
            runtime_static_cache={},
            runtime_static_flat_args=static_flat_args,
            arg_offsets=tuple(offsets),
            arg_leaf_counts=tuple(counts),
            invar_plans=self._make_invar_plans(state),
        )

    def _fold_runtime_static_slots_into_plan(
        self,
        plan: _InvarAssemblyPlan,
        invars: list[PyTree],
        runtime_static_flat_indices: set[int],
    ) -> _InvarAssemblyPlan:
        """Return an invar plan with stable dynamic slots pre-filled.

        Runtime integrations such as eSurge pass graph definitions and weights
        as normal dynamic arguments so they are not JAX compile-time constants.
        They are nevertheless stable for a decode bucket. The generic hot path
        used to visit every such slot on every token, doing set membership and
        cache lookups for roughly a hundred weight leaves per stage. After the
        first call has materialized those leaves with the correct stage
        placement, this helper folds them into the plan template. Later decode
        steps only iterate over truly changing slots: KV/cache leaves, metadata,
        and inter-stage activations.

        Args:
            plan: Plan value consumed by this operation.
            invars: Invars value consumed by this operation.
            runtime_static_flat_indices: Runtime static flat indices value consumed by this operation.

        Returns:
            Return an invar plan with stable dynamic slots pre-filled.
        """
        template = list(plan.template)
        dynamic_slots: list[tuple[int, int]] = []
        for out_pos, orig_idx in plan.dynamic_slots:
            if int(orig_idx) in runtime_static_flat_indices:
                template[int(out_pos)] = invars[int(out_pos)]
            else:
                dynamic_slots.append((int(out_pos), int(orig_idx)))
        return type(plan)(
            template=tuple(template),
            dynamic_slots=tuple(dynamic_slots),
            stage_slots=plan.stage_slots,
            prev_slots=plan.prev_slots,
        )

    def _normalize_runtime_static_argnums(
        self,
        runtime_static_argnums: tp.Iterable[int] | None,
        num_args: int,
    ) -> set[int]:
        """Convert Python-style argnums, including negatives, into a checked set.

        Args:
            runtime_static_argnums: User-provided argument indices. Negative
                values follow normal Python indexing from the end.
            num_args: Number of positional arguments in the call signature.

        Returns:
            A deduplicated set of valid non-negative argument indices.

        Raises:
            ValueError: If an index falls outside the positional argument range.
        """
        if runtime_static_argnums is None:
            return set()
        normalized: set[int] = set()
        for argnum_raw in runtime_static_argnums:
            argnum = int(argnum_raw)
            if argnum < 0:
                argnum += num_args
            if argnum < 0 or argnum >= num_args:
                raise ValueError(f"runtime_static_argnums contains invalid arg index {argnum_raw}.")
            normalized.add(argnum)
        return normalized

    def _flatten_args_with_runtime_static_cache(
        self,
        entry: _PrepareCacheEntry,
        args: tuple,
        *,
        runtime_static_argnums: tp.Iterable[int] | None,
    ) -> list[PyTree]:
        """Flatten args while reusing selected static leaves from the cache.

        EasyDeL passes graph definitions and weight pytrees through the same
        Python call signature for every decode step, but those values are static
                for a compiled bucket. Reusing their flattened leaves avoids repeatedly
        walking very large graph/state pytrees on the host.

        The method still verifies leaf counts for non-static arguments. A count
        mismatch means the caller changed the bucket shape or argument treedef
        while reusing the same prepare-cache key, which would make Spectrax's
        flat-leaf routing maps invalid.

        Args:
            entry: Entry value consumed by this operation.
            args: Positional arguments forwarded to the wrapped callable.
            runtime_static_argnums: Runtime static argnums value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        static_argnums = self._normalize_runtime_static_argnums(runtime_static_argnums, len(args))
        flat_args = list(entry.flat_args_template)
        for argnum, arg in enumerate(args):
            start = entry.arg_offsets[argnum]
            expected_count = entry.arg_leaf_counts[argnum]
            if argnum in static_argnums:
                continue
            leaves = jax.tree.leaves(arg)
            if len(leaves) != expected_count:
                raise ValueError(
                    "MpmdPipelineExecutor cached prepare shape changed: "
                    f"arg {argnum} had {expected_count} leaves, now has {len(leaves)}."
                )
            flat_args[start : start + expected_count] = leaves
        return flat_args

    def _retained_output_stages(self, state: _MpmdState, num_stages: int) -> set[int]:
        """Return stage outputs that must survive until result assembly."""
        retained: set[int] = set()
        for mapping in state.get("fn_outvar_map", ()):
            if not isinstance(mapping, tuple) or not mapping:
                continue
            src_rank = mapping[0]
            if isinstance(src_rank, int) and 0 <= src_rank < num_stages:
                retained.add(src_rank)
        return retained

    def _output_ref_counts(
        self,
        *,
        num_microbatches: int,
        num_stages: int,
        invar_plans: tuple[_InvarAssemblyPlan, ...] | None,
        carry_map: dict[int, dict[int, int]],
        retained_outputs: set[int],
    ) -> dict[tuple[int, int], int]:
        """Count non-result consumers of each stage output.

        Forward-only pipeline dispatch used to keep every intermediate stage
        output for every microbatch until the final pytree was assembled. On
        large meshes those hidden activations dominate memory even though most
        are consumed exactly once by the next stage. The ref-count here tracks
        stage-input and carry consumers so the hot loop can clear intermediate
        handles as soon as their last consumer has assembled its inputs.
        """
        if invar_plans is None:
            return {}

        counts: dict[tuple[int, int], int] = {}
        for mb_idx in range(num_microbatches):
            for stage_idx in range(num_stages):
                count = 0
                if stage_idx in retained_outputs:
                    count += 1
                if stage_idx in carry_map and mb_idx + 1 < num_microbatches:
                    count += len(carry_map[stage_idx])
                for dst_stage, plan in enumerate(invar_plans):
                    if dst_stage == stage_idx + 1:
                        count += len(plan.prev_slots)
                    for _out_pos, src_rank, _src_pos, _edge_sharding, _src_physical_rank in plan.stage_slots:
                        if src_rank == stage_idx:
                            count += 1
                if count:
                    counts[(mb_idx, stage_idx)] = count
        return counts

    def _runtime_static_flat_indices(
        self,
        entry: _PrepareCacheEntry | None,
        runtime_static_argnums: tp.Iterable[int] | None,
        num_args: int,
    ) -> set[int] | None:
        """Return flat-leaf indices covered by runtime-static positional args.

        Args:
            entry: Active prepare cache entry, or ``None`` when no cache is in
                use.
            runtime_static_argnums: Positional arguments declared static for the
                current runtime bucket.
            num_args: Number of positional arguments in the call.

        Returns:
            ``None`` when there is no cache/static declaration, otherwise the
            flat-leaf indices that should use stage-local placement caching.
        """
        if entry is None or runtime_static_argnums is None:
            return None
        indices: set[int] = set()
        for argnum in self._normalize_runtime_static_argnums(runtime_static_argnums, num_args):
            start = entry.arg_offsets[argnum]
            count = entry.arg_leaf_counts[argnum]
            indices.update(range(start, start + count))
        return indices

    def _pipeline_transport_gate(
        self,
        *,
        num_microbatches: int,
        num_stages: int,
        invar_plans: tuple[_InvarAssemblyPlan, ...] | None,
        rank_device_sets: list[set[jax.Device]],
        logical_physical_ranks: tp.Sequence[int],
        use_worker_dispatch: bool,
    ) -> _OrderedScheduleTransportGate | None:
        """Return a deterministic transport gate for multi-controller worker pipelines."""
        if not use_worker_dispatch or invar_plans is None:
            return None
        try:
            if jax.process_count() <= 1:
                return None
        except Exception:
            return None
        if len({frozenset(devices) for devices in rank_device_sets}) <= 1:
            return None

        task_order: list[str] = []
        for wave_idx in range(num_microbatches + num_stages - 1):
            hi_stage = min(wave_idx, num_stages - 1)
            for stage_idx in range(0, hi_stage + 1):
                mb_idx = wave_idx - stage_idx
                if mb_idx < 0 or mb_idx >= num_microbatches:
                    continue
                plan = invar_plans[stage_idx]
                dst_physical_rank = int(logical_physical_ranks[stage_idx])
                for out_pos, src_logical_stage, _src_pos, _edge_sharding, src_physical_rank in plan.stage_slots:
                    if int(src_physical_rank) == dst_physical_rank:
                        continue
                    name = _pipeline_transport_task_name(
                        (mb_idx, stage_idx),
                        src_logical_stage=int(src_logical_stage),
                        input_pos=out_pos,
                    )
                    if name is not None:
                        task_order.append(name)
                for out_pos, _prev_idx in plan.prev_slots:
                    src_logical_stage = stage_idx - 1
                    if src_logical_stage < 0:
                        continue
                    src_physical_rank = int(logical_physical_ranks[src_logical_stage])
                    if src_physical_rank == dst_physical_rank:
                        continue
                    name = _pipeline_transport_task_name(
                        (mb_idx, stage_idx),
                        src_logical_stage=src_logical_stage,
                        input_pos=out_pos,
                    )
                    if name is not None:
                        task_order.append(name)
        if not task_order:
            return None
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        if process_index == 0:
            logger.warning(
                "SpectraX MPMD pipeline executor using deterministic async transport ordering for %d "
                "forward-pipeline transfer(s) across %d microbatch(es) and %d stage(s).",
                len(task_order),
                num_microbatches,
                num_stages,
            )
        return _OrderedScheduleTransportGate(tuple(task_order))

    def _log_deterministic_worker_launch_if_needed(self, rank_device_sets: list[set[jax.Device]]) -> None:
        """Log when worker dispatch is protected by deterministic transport ordering.

        Multi-controller TPU programs must enter JAX launches in the same order
        on every controller. Resident Python stage workers are kept enabled, but
        cross-rank pair-mesh transport launches are gated by the deterministic
        task order built in :meth:`_pipeline_transport_gate`.
        """
        if not self.use_workers:
            return
        try:
            if jax.process_count() <= 1:
                return
        except Exception:
            return
        if len({frozenset(devices) for devices in rank_device_sets}) <= 1:
            return
        try:
            process_index = jax.process_index()
        except Exception:
            process_index = -1
        with _PIPELINE_PROGRESS_LOCK:
            logged = _PIPELINE_DISPATCH_DIAGNOSTICS.get("deterministic_host_launch", 0)
            if process_index == 0 and logged < 1:
                logger.warning(
                    "SpectraX MPMD pipeline executor keeping worker dispatch enabled with deterministic "
                    "transport ordering because multi-controller stage meshes use different device sets. "
                    "This preserves JAX launch order while still relying on asynchronous device execution."
                )
            _PIPELINE_DISPATCH_DIAGNOSTICS["deterministic_host_launch"] = logged + 1
        return None

    def _pipeline_task_event_names(
        self,
        *,
        mb_idx: int,
        stage_idx: int,
        invar_plan: _InvarAssemblyPlan | None,
    ) -> tuple[str, ...]:
        """Return ordered cross-rank transport names for one pipeline task."""
        return self._pipeline_transport_task_names(mb_idx=mb_idx, stage_idx=stage_idx, invar_plan=invar_plan)

    def _pipeline_transport_task_names(
        self,
        *,
        mb_idx: int,
        stage_idx: int,
        invar_plan: _InvarAssemblyPlan | None,
    ) -> tuple[str, ...]:
        """Return ordered transport names issued while assembling a stage task."""
        if invar_plan is None:
            return ()
        names: list[str] = []
        for out_pos, src_rank, _src_pos, _edge_sharding, _src_physical_rank in invar_plan.stage_slots:
            name = _pipeline_transport_task_name(
                (mb_idx, stage_idx),
                src_logical_stage=int(src_rank),
                input_pos=out_pos,
            )
            if name is not None:
                names.append(name)
        for out_pos, _prev_idx in invar_plan.prev_slots:
            name = _pipeline_transport_task_name(
                (mb_idx, stage_idx),
                src_logical_stage=stage_idx - 1,
                input_pos=out_pos,
            )
            if name is not None:
                names.append(name)
        return tuple(names)

    def _assemble_stage_invars(
        self,
        *,
        call: _PreparedCall,
        stage_idx: int,
        logical_stage_idx: int,
        microbatch_idx: int,
        invar_map: InvarMap,
        invar_plan: _InvarAssemblyPlan | None,
        my_sh: Sharding,
        rank_devices: set[jax.Device],
        rank_submeshes: list[Mesh],
        mpmd_mesh: MpMdMesh | None,
        prev_outputs: StageOutputs,
        all_cluster_outputs: list[StageOutputs | None],
        runtime_static_flat_indices: set[int] | None,
        runtime_static_cache: dict[tuple[int, int], PyTree] | None,
    ) -> list[PyTree]:
        """Materialize one stage's positional inputs for one microbatch.

        SpectraX's compiled stage plan describes each input as one of three
        sources: an original function argument, the immediately previous stage,
        or an earlier cluster output. This helper delegates to the fast prepared
        invar plan when available and otherwise falls back to the generic
        assembler from ``runtime.py``.

        Runtime-static cache information is passed through to the runtime helper
        so graph/weight-like dynamic leaves are placed once per stage and reused
        across decode steps in the same bucket.

        Args:
            call: Call value consumed by this operation.
            stage_idx: Stage idx value consumed by this operation.
            invar_map: Invar map value consumed by this operation.
            invar_plan: Invar plan value consumed by this operation.
            my_sh: My sh value consumed by this operation.
            rank_devices: Rank devices value consumed by this operation.
            rank_submeshes: Rank submeshes value consumed by this operation.
            mpmd_mesh: Mpmd mesh value consumed by this operation.
            prev_outputs: Prev outputs value consumed by this operation.
            all_cluster_outputs: All cluster outputs value consumed by this operation.
            runtime_static_flat_indices: Runtime static flat indices value consumed by this operation.
            runtime_static_cache: Runtime static cache value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        state = call.state
        if invar_plan is not None:
            return _assemble_invars_from_plan(
                invar_plan,
                call.flat_args,
                state["explicit_in_sh"],
                prev_outputs,
                all_cluster_outputs,
                stage_idx,
                my_sh,
                rank_devices,
                rank_submeshes,
                mpmd_mesh,
                runtime_static_flat_indices=runtime_static_flat_indices,
                runtime_static_cache=runtime_static_cache,
                transport_context=(microbatch_idx, logical_stage_idx),
            )
        return _assemble_invars(
            invar_map,
            call.flat_args,
            state["placed"],
            state["dynamic"],
            state["explicit_in_sh"],
            prev_outputs,
            all_cluster_outputs,
            stage_idx,
            my_sh,
            rank_devices,
            rank_submeshes,
            mpmd_mesh,
            dynamic_flat_to_orig_flat=state.get("dynamic_flat_to_orig_flat"),
            runtime_static_flat_indices=runtime_static_flat_indices,
            runtime_static_cache=runtime_static_cache,
            transport_context=(microbatch_idx, logical_stage_idx),
        )

    def _make_invar_plans(self, state: _MpmdState) -> tuple[_InvarAssemblyPlan, ...]:
        """Precompute stage-input routing plans for the cached prepare state.

        Each compiled stage carries an ``invar_map`` describing where every
        positional input should come from. This method turns those maps into
        compact templates with holes for only dynamic arguments and inter-stage
        transfers, shaving Python branching out of hot decode dispatch.

        Args:
            state: SpectraX state tree or transform state passed into the operation.

        Returns:
            Result described by this helper.
        """
        compiled: list[_CompiledStage] = state["compiled"]
        return tuple(
            _prepare_invar_assembly_plan(
                invar_map,
                state["placed"],
                state["dynamic"],
                physical_rank,
                dynamic_flat_to_orig_flat=state.get("dynamic_flat_to_orig_flat"),
            )
            for stage_idx, stage_entry in enumerate(compiled)
            for _, _, _, _, invar_map, physical_rank in (_unpack_compiled_stage(stage_entry, stage_idx),)
        )

    def _assemble_result(self, call: _PreparedCall, outputs: list[StageOutputs | None]) -> PyTree:
        """Rebuild the user-facing pytree from per-stage flat outputs.

        Args:
            call: Prepared call whose state contains the output routing map and
                result treedef.
            outputs: Per-stage flat output tuples collected by the wavefront.

        Returns:
            The exact pytree shape that a direct ``sxjit`` call would return,
            with optional ``out_shardings`` applied by Spectrax's normal rules.
        """
        required_stages = {
            int(mapping[0]) for mapping in call.state["fn_outvar_map"] if mapping and isinstance(mapping[0], int)
        }
        ready_outputs: list[StageOutputs] = []
        for stage_idx, value in enumerate(outputs):
            if value is None:
                if stage_idx in required_stages:
                    raise RuntimeError(f"internal error: missing output for required stage {stage_idx}")
                value = ()
            ready_outputs.append(value)
        result = _assemble_outputs(
            call.state["fn_outvar_map"],
            ready_outputs,
            call.flat_args,
            dynamic_flat_to_orig_flat=call.state.get("dynamic_flat_to_orig_flat"),
        )
        result = _apply_out_shardings(result, call.state.get("out_shardings"))
        return _restore_result_treedef(result, call.state.get("result_treedef"))

    def _resolve_mpmd_mesh(
        self,
        state: _MpmdState,
        rank_submeshes: list[Mesh],
        sxjit_fn: _MpmdPreparedCallable | None,
    ) -> MpMdMesh | None:
        """Find the owning MPMD mesh for device placement decisions.

        The forward plan normally carries this directly in ``state``. The
        fallbacks exist because older prepared states and tests may only expose
        the mesh through submeshes or the wrapped sxjit function.

        Returning ``None`` is allowed for host-only/unit-test fakes where the
        transfer helper never needs to resolve a real marker edge sharding.

        Args:
            state: SpectraX state tree or transform state passed into the operation.
            rank_submeshes: Rank submeshes value consumed by this operation.
            sxjit_fn: Sxjit fn value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        mpmd_mesh = state.get("mpmd_mesh")
        if mpmd_mesh is not None:
            return mpmd_mesh
        if rank_submeshes:
            mpmd_mesh = getattr(getattr(rank_submeshes[0], "spmd_mesh", None), "mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
        if self.stage_meshes:
            mpmd_mesh = getattr(getattr(self.stage_meshes[0], "spmd_mesh", None), "mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
            mpmd_mesh = getattr(self.stage_meshes[0], "mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
        if sxjit_fn is not None:
            mpmd_mesh = getattr(sxjit_fn, "_mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
        return None

    def _ensure_workers(self, worker_count: int) -> None:
        """Create exactly one resident worker per physical pipeline stage.

        The worker pool is rebuilt when the stage count changes because each
        worker is named and logically associated with one rank. Prepared-plan
        cache entries remain valid for their own cache keys; if a later call
        reuses a different cached plan, this method will resize the workers to
        that plan's physical pipeline before dispatch.

        Args:
            worker_count: Worker count value consumed by this operation.
        """
        if self._worker_count == worker_count and len(self._workers) == worker_count:
            return
        self._shutdown_workers()
        self._workers = [_StageWorker(rank=rank) for rank in range(worker_count)]
        self._worker_count = worker_count

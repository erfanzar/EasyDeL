# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Resident per-stage worker runtime for eSurge pipeline-parallel inference.

SpectraX's default ``sxjit`` MPMD path executes a forward-only clustering plan
through a single host-side dispatcher loop that walks every stage in sequence.
For inference we instead pin one daemon thread per pipeline rank — a
:class:`_PipelineStageWorker` — and feed each thread its own activation queue.
The dispatcher (:class:`PipelineStageRuntime`) reuses the SpectraX-prepared
compile plan (``_mpmd_prepare``) for invar/outvar wiring, but routes every
stage call through ``submit() → Future`` on the matching worker thread, then
threads the resolved output activations back into the next stage's invars via
SpectraX's private edge-sharding helpers (``_assemble_invars`` /
``_assemble_outputs``).

Public contract: a caller still submits one bucketed model-step pytree and
receives the same output pytree; the runtime only changes *how* the per-stage
launches reach the device, not what they compute. The win is twofold —
per-stage Python overhead is amortized across the daemon thread, and the
scheduler thread is freed to do prefetch work while a step is in flight.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
import typing as tp
from concurrent.futures import Future

import jax
from eformer.loggings import get_logger
from spectrax.runtime.mpmd.runtime import (
    _apply_out_shardings,
    _assemble_invars,
    _assemble_outputs,
    _restore_result_treedef,
)

from easydel.infra.sharding import MeshLike

from .pipeline_plan import PipelineInferencePlan

logger = get_logger("eSurge-PipelineRuntime")


@dataclasses.dataclass(frozen=True)
class PipelineDispatchStats:
    """Per-call dispatch counters captured by :class:`PipelineStageRuntime`.

    Refreshed once per :meth:`PipelineStageRuntime.dispatch` invocation and
    surfaced to the model runner via the ``last_pipeline_stats`` channel for
    inclusion in step-level perf logs.

    Attributes:
        stage_launches (int): Number of stage submits issued during the call;
            equals ``len(compiled)`` from the SpectraX plan.
        stage_dispatch_time (float): Cumulative wall-clock seconds the
            dispatcher spent in ``submit()`` plus blocking ``future.result()``
            across all stages. Includes both queue wait and on-device
            execution time because the dispatcher waits for each stage
            result inline before moving on.
        queue_wait_time (float): Lower-bounded estimate of time spent
            waiting for worker threads, as opposed to actual device time.
            Currently equal to ``stage_dispatch_time`` (clamped to
            non-negative); kept as a separate field so future work can split
            the two without breaking callers.
    """

    stage_launches: int
    stage_dispatch_time: float
    queue_wait_time: float


@dataclasses.dataclass
class _StageTask:
    """Producer→worker handoff record for one pipeline stage call.

    Constructed by :meth:`_PipelineStageWorker.submit` (the producer side runs
    on the dispatcher thread) and consumed by :meth:`_PipelineStageWorker._run`
    on the worker thread. The task's ``future`` is the rendezvous: the
    dispatcher blocks on ``future.result()`` to fetch stage activations before
    assembling the next stage's invars.

    Attributes:
        stage_jit (Callable): The compiled per-stage jit returned by
            SpectraX's compile plan. Closed over the layer subset that lives
            on this rank.
        submesh (MeshLike): Stage submesh entered as a context manager during
            execution so XLA places the call on the right slice of devices.
        invars (list[Any]): Flat list of stage inputs in plan order
            (graphdef/state, kv pages, transferred activations from the
            previous stage, …). Pre-assembled by :func:`_assemble_invars`.
        future (Future): Result handle the dispatcher blocks on. Resolved
            with the stage output pytree on success, or with the raised
            exception on failure.
        enqueued_at (float): Wall-clock seconds at submit time. Reserved
            for future queue-latency telemetry; not currently consumed.
    """

    stage_jit: tp.Callable[..., tp.Any]
    submesh: MeshLike
    invars: list[tp.Any]
    future: Future
    enqueued_at: float


class _PipelineStageWorker:
    """Resident daemon thread serving one physical pipeline-parallel rank.

    The constructor immediately starts a daemon thread that consumes
    :class:`_StageTask` items from an unbounded :class:`queue.Queue` and
    executes each ``stage_jit(*invars)`` under the task's submesh context.
    There is exactly one worker per PP rank, instantiated lazily by
    :meth:`PipelineStageRuntime._ensure_workers` after the SpectraX plan is
    first observed. Workers outlive individual ``dispatch()`` calls; only
    :meth:`PipelineStageRuntime.shutdown` (or a plan-shape change) tears them
    down.

    The dispatcher and worker threads communicate strictly through the queue
    and the per-task :class:`Future`: enqueue, then block on ``future.result``.
    No shared mutable state lives on the worker.
    """

    def __init__(self, *, rank: int) -> None:
        """Spawn the daemon thread for ``rank`` and start polling the queue.

        Args:
            rank (int): Pipeline-stage rank this worker is dedicated to.
                Encoded into the thread name (``eSurge-pp-stage-{rank}``)
                for ``ps`` / profiler visibility; not otherwise consulted —
                stage targeting is enforced by the ``submesh`` carried on
                each task.
        """
        self.rank = int(rank)
        self._queue: queue.Queue[_StageTask | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"eSurge-pp-stage-{rank}", daemon=True)
        self._thread.start()

    def submit(self, *, stage_jit: tp.Callable[..., tp.Any], submesh: MeshLike, invars: list[tp.Any]) -> Future:
        """Enqueue a stage call from the dispatcher thread.

        Producer side of the producer/consumer handshake: builds a fresh
        :class:`_StageTask`, copies ``invars`` defensively (the caller may
        reuse the list to assemble the next stage), and puts it on the
        worker's queue. Returns immediately without waiting for the worker
        to pick the task up.

        Args:
            stage_jit (Callable): Compiled per-stage callable. Must accept
                exactly the unpacked ``invars`` and return the pytree of
                stage output activations.
            submesh (MeshLike): Stage submesh activated as a context manager
                around the call so XLA places it on the correct device slice.
            invars (list[Any]): Flat list of stage inputs in the order
                expected by ``stage_jit``. Copied before being attached to
                the task.

        Returns:
            Future: Future the dispatcher should ``.result()`` on. Resolves
            with the stage output pytree on success, or re-raises any
            exception thrown inside ``stage_jit`` on the worker thread.
        """
        future: Future = Future()
        self._queue.put(
            _StageTask(
                stage_jit=stage_jit,
                submesh=submesh,
                invars=list(invars),
                future=future,
                enqueued_at=time.time(),
            )
        )
        return future

    def shutdown(self) -> None:
        """Send the sentinel and join the worker thread with a short timeout.

        Pushes ``None`` onto the queue (interpreted by :meth:`_run` as the
        exit signal) and waits up to five seconds for the daemon thread to
        return. Idempotent — call as many times as you like.
        """
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        """Consume the task queue until a ``None`` sentinel arrives.

        Worker-thread main loop. For each task: claim the future via
        ``set_running_or_notify_cancel``, enter the task's submesh as a
        context manager, invoke ``stage_jit(*invars)``, and either resolve
        the future with the result or set its exception so the dispatcher
        can re-raise on the originating thread. Always calls ``task_done``
        so callers waiting on ``queue.join()`` (e.g. teardown paths) make
        progress.
        """
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            if task.future.set_running_or_notify_cancel():
                try:
                    with task.submesh:
                        out = task.stage_jit(*task.invars)
                    task.future.set_result(out)
                except BaseException as exc:
                    task.future.set_exception(exc)
            self._queue.task_done()


class PipelineStageRuntime:
    """Queue-backed dispatcher over a SpectraX forward-only stage compile plan.

    Owns one :class:`_PipelineStageWorker` per pipeline rank and replaces the
    default in-line SpectraX dispatcher with a producer (this object's
    :meth:`dispatch`) plus N consumer threads (one per PP rank). The actual
    activation transfer between stages still uses SpectraX's private edge-
    sharding helpers (``_assemble_invars`` / ``_assemble_outputs`` /
    ``_apply_out_shardings``); we only change *who* calls each stage's
    compiled jit.

    Lifecycle:

    1. Construct with an enabled :class:`PipelineInferencePlan` — workers are
       not spawned yet.
    2. The first :meth:`dispatch` call observes the SpectraX compile plan
       (``_mpmd_prepare``), reads ``len(compiled)``, and provisions exactly
       that many workers via :meth:`_ensure_workers`. Subsequent calls reuse
       the same workers as long as the stage count is stable.
    3. :meth:`shutdown` (called from :class:`ModelStepExecutor.shutdown` and
       transitively from engine teardown) joins all worker threads.

    Attributes:
        plan: The enabled :class:`PipelineInferencePlan` whose ``stage_meshes``
            and ``mpmd_dim`` are consulted as a fallback for the
            ``mpmd_mesh`` lookup when the SpectraX plan does not surface one
            directly.
    """

    def __init__(self, *, plan: PipelineInferencePlan) -> None:
        """Construct an idle runtime; workers are spawned lazily on first dispatch.

        Args:
            plan (PipelineInferencePlan): Enabled pipeline plan. The runtime
                does not consult ``stage_meshes`` until a dispatch needs to
                resolve ``mpmd_mesh``, so the plan only needs to satisfy
                ``is_enabled=True`` here.

        Raises:
            ValueError: If ``plan.is_enabled`` is ``False`` — pipeline-stage
                workers make no sense without an MPMD topology.
        """
        if not plan.is_enabled:
            raise ValueError("PipelineStageRuntime requires an enabled PipelineInferencePlan.")
        self.plan = plan
        self._workers: list[_PipelineStageWorker] = []
        self._worker_count = 0
        self._last_stats = PipelineDispatchStats(0, 0.0, 0.0)

    @property
    def last_stats(self) -> PipelineDispatchStats:
        """Counters captured by the most recent :meth:`dispatch` call.

        Reset to a zero-valued :class:`PipelineDispatchStats` at construction
        time. Read by ``ModelStepExecutor.last_pipeline_stats`` for inclusion
        in the per-step perf log line.
        """
        return self._last_stats

    def shutdown(self) -> None:
        """Join every worker thread and clear the pool.

        Sends a sentinel to each worker and resets internal state so a fresh
        :meth:`dispatch` would re-provision a new pool. Idempotent.
        """
        for worker in self._workers:
            worker.shutdown()
        self._workers = []
        self._worker_count = 0

    def dispatch(self, sxjit_fn: tp.Callable[..., tp.Any], *args: tp.Any) -> tp.Any:
        """Execute one bucketed model-step pytree through the resident workers.

        Walks the SpectraX compile plan rank-by-rank. For each stage the
        method (a) materializes its invars by gathering original inputs and
        previously-computed activations through ``_assemble_invars``, (b)
        ``submit``s the call to the rank's worker, (c) blocks on the future
        before moving to the next stage so cross-stage transfers respect the
        plan's edge-sharding metadata, and finally (d) stitches the
        per-stage outputs back into the original result pytree using the
        plan's ``fn_outvar_map`` and out-sharding specs.

        Args:
            sxjit_fn (Callable): The SpectraX ``@spx.jit`` function. Must
                expose the private ``_mpmd_prepare`` callable used to lower
                the inputs to a per-rank compile plan; this attribute is
                only present on forward-only sxjit functions (i.e. those
                that do not use the gradient/scheduler MPMD path).
            *args: The same positional arguments the caller would have
                passed to ``sxjit_fn`` directly. Forwarded verbatim into
                ``_mpmd_prepare`` for tracing and used as the source of
                ``flat_args`` during invar assembly.

        Returns:
            The reassembled output pytree, with each leaf placed under the
            sharding specified by the plan's ``out_shardings``. The result
            structure matches what calling ``sxjit_fn(*args)`` directly
            would have produced.

        Raises:
            TypeError: If ``sxjit_fn`` lacks ``_mpmd_prepare`` (not a
                SpectraX MPMD function), or if the plan dictionary doesn't
                contain a forward-only ``compiled`` entry (i.e. the function
                was compiled under a backward/scheduler MPMD plan that this
                dispatcher cannot drive).
        """

        prepare = getattr(sxjit_fn, "_mpmd_prepare", None)
        if prepare is None:
            raise TypeError("PipelineStageRuntime requires a SpectraX sxjit function with _mpmd_prepare.")
        state = prepare(*args)
        if "compiled" not in state:
            raise TypeError("PipelineStageRuntime only supports forward-only sxjit plans.")

        compiled = state["compiled"]
        self._ensure_workers(compiled)

        placed = state["placed"]
        dynamic = state["dynamic"]
        explicit_in_sh = state["explicit_in_sh"]
        rank_submeshes = [stage[1] for stage in compiled]
        mpmd_mesh = state.get("mpmd_mesh")
        if mpmd_mesh is None:
            mpmd_mesh = getattr(getattr(rank_submeshes[0], "spmd_mesh", None), "mpmd_mesh", None)
        if mpmd_mesh is None:
            # SpectraX's private assembler only needs the object for
            # edge-sharding helpers; the sxjit state always has compatible
            # submeshes, so keep a direct reference from the model mesh if set.
            mpmd_mesh = getattr(getattr(self.plan.stage_meshes[0], "spmd_mesh", None), "mpmd_mesh", None)
        if mpmd_mesh is None:
            mpmd_mesh = getattr(self.plan.stage_meshes[0], "mpmd_mesh", None)
        if mpmd_mesh is None:
            # Fallback for current SpectraX MpMdMesh.submesh objects: the
            # private assembler only passes it through to edge transfer helpers.
            mpmd_mesh = getattr(sxjit_fn, "_mpmd_mesh", None)

        flat_args = jax.tree.leaves(args)
        all_cluster_outputs: list[tuple] = []
        prev_outputs: tuple = ()
        stage_dispatch_time = 0.0
        queue_wait_time = 0.0

        for ri, (_, _, my_sh, _, invar_map) in enumerate(compiled):
            stage_jit, submesh, *_ = compiled[ri]
            rank_devices = set(rank_submeshes[ri].devices.flat)
            invars = _assemble_invars(
                invar_map,
                flat_args,
                placed,
                dynamic,
                explicit_in_sh,
                prev_outputs,
                all_cluster_outputs,
                ri,
                my_sh,
                rank_devices,
                rank_submeshes,
                mpmd_mesh,
                dynamic_flat_to_orig_flat=state.get("dynamic_flat_to_orig_flat"),
            )
            t0 = time.time()
            future = self._workers[ri].submit(stage_jit=stage_jit, submesh=submesh, invars=invars)
            prev_outputs = future.result()
            elapsed = time.time() - t0
            stage_dispatch_time += elapsed
            queue_wait_time += max(0.0, elapsed)
            all_cluster_outputs.append(prev_outputs)

        result = _assemble_outputs(
            state["fn_outvar_map"],
            all_cluster_outputs,
            flat_args,
            dynamic_flat_to_orig_flat=state.get("dynamic_flat_to_orig_flat"),
        )
        result = _apply_out_shardings(result, state.get("out_shardings"))
        result = _restore_result_treedef(result, state.get("result_treedef"))
        self._last_stats = PipelineDispatchStats(
            stage_launches=len(compiled),
            stage_dispatch_time=stage_dispatch_time,
            queue_wait_time=queue_wait_time,
        )
        return result

    def _ensure_workers(self, compiled: list[tuple]) -> None:
        """Provision exactly ``len(compiled)`` workers, replacing any stale pool.

        Called at the start of every :meth:`dispatch` so that re-tracing a
        plan with a different number of pipeline stages (e.g. after a hot
        weight update that altered layer assignment) transparently rebuilds
        the worker pool. The fast path is a no-op when the existing pool
        already matches the requested count.

        Args:
            compiled (list[tuple]): SpectraX compiled stage list — the
                ``compiled`` entry of the ``_mpmd_prepare`` plan. Only its
                length matters here; one worker is spawned per element.
        """
        worker_count = len(compiled)
        if self._worker_count == worker_count and len(self._workers) == worker_count:
            return
        self.shutdown()
        self._workers = [_PipelineStageWorker(rank=rank) for rank in range(worker_count)]
        self._worker_count = worker_count
        logger.info("Started %s PP stage workers.", len(compiled))

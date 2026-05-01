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

"""Resident pipeline-stage runtime for eSurge PP inference.

This module reuses SpectraX's forward-only ``sxjit`` clustering/placement plan,
but replaces the default host dispatcher with long-lived per-stage workers and
explicit activation queues.  The first implementation keeps the public eSurge
step contract intact: a caller submits one bucketed model step and receives the
same output pytree, while internally the hidden-state activations flow through
rank-local worker queues instead of one monolithic dispatcher loop.
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
    """Host dispatch bookkeeping for one pipeline item."""

    stage_launches: int
    stage_dispatch_time: float
    queue_wait_time: float


@dataclasses.dataclass
class _StageTask:
    stage_jit: tp.Callable[..., tp.Any]
    submesh: MeshLike
    invars: list[tp.Any]
    future: Future
    enqueued_at: float


class _PipelineStageWorker:
    """One resident host worker for one physical PP stage."""

    def __init__(self, *, rank: int) -> None:
        self.rank = int(rank)
        self._queue: queue.Queue[_StageTask | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"eSurge-pp-stage-{rank}", daemon=True)
        self._thread.start()

    def submit(self, *, stage_jit: tp.Callable[..., tp.Any], submesh: MeshLike, invars: list[tp.Any]) -> Future:
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
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
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
    """Queue-backed dispatcher over SpectraX forward-only stage jits."""

    def __init__(self, *, plan: PipelineInferencePlan) -> None:
        if not plan.is_enabled:
            raise ValueError("PipelineStageRuntime requires an enabled PipelineInferencePlan.")
        self.plan = plan
        self._workers: list[_PipelineStageWorker] = []
        self._worker_count = 0
        self._last_stats = PipelineDispatchStats(0, 0.0, 0.0)

    @property
    def last_stats(self) -> PipelineDispatchStats:
        return self._last_stats

    def shutdown(self) -> None:
        for worker in self._workers:
            worker.shutdown()
        self._workers = []
        self._worker_count = 0

    def dispatch(self, sxjit_fn: tp.Callable[..., tp.Any], *args: tp.Any) -> tp.Any:
        """Run one bucketed backbone item through resident stage queues."""

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
        worker_count = len(compiled)
        if self._worker_count == worker_count and len(self._workers) == worker_count:
            return
        self.shutdown()
        self._workers = [_PipelineStageWorker(rank=rank) for rank in range(worker_count)]
        self._worker_count = worker_count
        logger.info("Started %s PP stage workers.", len(compiled))

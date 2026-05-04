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

"""eSurge adapter over SpectraX's resident MPMD pipeline executor."""

from __future__ import annotations

import dataclasses
import typing as tp

import spectrax as spx
from eformer.loggings import get_logger

from .pipeline_plan import PipelineInferencePlan

logger = get_logger("eSurge-PipelineRuntime")


@dataclasses.dataclass(frozen=True)
class PipelineDispatchStats:
    """Per-call SpectraX PP dispatch counters surfaced in eSurge perf logs.

    The execution manager treats these values as host-side scheduling metrics:
    they describe how many physical stage calls were launched and how much wall
    time was spent preparing, submitting, waiting for, and assembling an MPMD
    pipeline call. They intentionally do not replace device-side XLA profiling;
    they make the eSurge decode loop explainable from normal verbose logs.
    """

    stage_launches: int
    stage_dispatch_time: float
    queue_wait_time: float
    prepare_time: float = 0.0
    assemble_time: float = 0.0
    submit_time: float = 0.0
    stage_submit_times_ms: tuple[float, ...] = ()
    stage_assemble_times_ms: tuple[float, ...] = ()
    stage_execute_times_ms: tuple[float, ...] = ()


class PipelineStageRuntime:
    """EasyDeL-facing adapter over ``spx.MpmdPipelineExecutor``.

    SpectraX owns the generic MPMD mechanics: preparing an ``sxjit`` stage
    plan, routing activations, entering stage submeshes, and running resident
    per-rank workers. EasyDeL owns the inference-specific semantics around it:
    which model function to call, how KV-cache state is carried, and how the
    final hidden states are projected and sampled.

    The runtime keeps two executors:
        * an inline executor for a single logical decode batch, where worker
          futures cannot create overlap and only add a host rendezvous;
        * a resident-worker executor for true multi-microbatch wavefronts,
          where each physical stage needs its own queue to overlap with the
          other stages.
    """

    def __init__(self, *, plan: PipelineInferencePlan) -> None:
        if not plan.is_enabled:
            raise ValueError("PipelineStageRuntime requires an enabled PipelineInferencePlan.")
        self.plan = plan
        self._inline_executor = spx.MpmdPipelineExecutor(stage_meshes=plan.stage_meshes, use_workers=False)
        self._wavefront_executor = spx.MpmdPipelineExecutor(stage_meshes=plan.stage_meshes, use_workers=True)
        self._last_stats = PipelineDispatchStats(0, 0.0, 0.0)
        self._logged_plan_keys: set[tuple[str, tp.Hashable]] = set()

    @property
    def last_stats(self) -> PipelineDispatchStats:
        """Metrics for the most recent single-call or wavefront dispatch."""
        return self._last_stats

    def shutdown(self) -> None:
        """Stop resident SpectraX stage workers and reset visible metrics."""
        self._inline_executor.shutdown()
        self._wavefront_executor.shutdown()
        self._last_stats = PipelineDispatchStats(0, 0.0, 0.0)

    def clear_prepare_cache(self) -> None:
        """Drop cached SpectraX prepare plans after graph/cache shape changes."""
        self._inline_executor.clear_prepare_cache()
        self._wavefront_executor.clear_prepare_cache()

    def dispatch(
        self,
        sxjit_fn: tp.Callable[..., tp.Any],
        *args: tp.Any,
        prepare_cache_key: tp.Hashable | None = None,
        runtime_static_argnums: tp.Iterable[int] | None = None,
    ) -> tp.Any:
        """Run one prepared ``sxjit`` backbone call through the PP executor."""
        result = self._inline_executor.dispatch_many(
            sxjit_fn,
            (args,),
            prepare_cache_key=prepare_cache_key,
            runtime_static_argnums=runtime_static_argnums,
        )[0]
        stats = self._inline_executor.last_stats
        self._last_stats = PipelineDispatchStats(
            stage_launches=int(stats.stage_launches),
            stage_dispatch_time=float(stats.stage_dispatch_time),
            queue_wait_time=float(stats.queue_wait_time),
            prepare_time=float(stats.prepare_time),
            assemble_time=float(stats.assemble_time),
            submit_time=float(stats.submit_time),
            stage_submit_times_ms=tuple(float(x) for x in getattr(stats, "stage_submit_times_ms", ())),
            stage_assemble_times_ms=tuple(float(x) for x in getattr(stats, "stage_assemble_times_ms", ())),
            stage_execute_times_ms=tuple(float(x) for x in getattr(stats, "stage_execute_times_ms", ())),
        )
        self._maybe_log_cached_plan("single", self._inline_executor, prepare_cache_key)
        return result

    def dispatch_many(
        self,
        sxjit_fn: tp.Callable[..., tp.Any],
        arg_batches: tp.Iterable[tuple],
        *,
        carry_input_output_map: tp.Mapping[int, tp.Mapping[int, int]] | None = None,
        prepare_cache_key: tp.Hashable | None = None,
        runtime_static_argnums: tp.Iterable[int] | None = None,
    ) -> list[tp.Any]:
        """Run same-shaped microbatches through SpectraX's wavefront executor.

        This is the building block for overlapped PP decode. The caller is
        still responsible for ensuring KV-cache updates are independent or
        stage-local before submitting multiple microbatches concurrently.
        """
        arg_batches = tuple(arg_batches)
        executor = self._inline_executor if len(arg_batches) <= 1 else self._wavefront_executor
        results = executor.dispatch_many(
            sxjit_fn,
            arg_batches,
            carry_input_output_map=carry_input_output_map,
            prepare_cache_key=prepare_cache_key,
            runtime_static_argnums=runtime_static_argnums,
        )
        stats = executor.last_stats
        self._last_stats = PipelineDispatchStats(
            stage_launches=int(stats.stage_launches),
            stage_dispatch_time=float(stats.stage_dispatch_time),
            queue_wait_time=float(stats.queue_wait_time),
            prepare_time=float(stats.prepare_time),
            assemble_time=float(stats.assemble_time),
            submit_time=float(stats.submit_time),
            stage_submit_times_ms=tuple(float(x) for x in getattr(stats, "stage_submit_times_ms", ())),
            stage_assemble_times_ms=tuple(float(x) for x in getattr(stats, "stage_assemble_times_ms", ())),
            stage_execute_times_ms=tuple(float(x) for x in getattr(stats, "stage_execute_times_ms", ())),
        )
        logger.debug(
            "SpectraX MPMD wavefront dispatched %s microbatches over %s stage launches",
            int(stats.microbatches),
            int(stats.stage_launches),
        )
        self._maybe_log_cached_plan("wavefront", executor, prepare_cache_key)
        return results

    def _maybe_log_cached_plan(
        self,
        mode: str,
        executor: spx.MpmdPipelineExecutor,
        prepare_cache_key: tp.Hashable | None,
    ) -> None:
        """Log one compact audit of a cached SpectraX PP plan.

        eSurge's split backbone calls pass
        ``(graphdef, graphstate, graphother, kv_pages, metadata)`` while the
        fused PP model-step path passes ``(graphstate, graphother, kv_pages,
        metadata)``. Decode performance depends on SpectraX keeping KV/cache
        leaves stage-local and treating metadata as ordinary dynamic inputs, so
        the audit chooses the correct argument slots per cached executable
        shape and reports only compact counts.
        """
        if prepare_cache_key is None:
            return
        logged_key = (str(mode), prepare_cache_key)
        if logged_key in self._logged_plan_keys:
            return
        cache = getattr(executor, "_prepare_cache", None)
        if not isinstance(cache, dict):
            return
        entry = cache.get(prepare_cache_key)
        if entry is None:
            return
        self._logged_plan_keys.add(logged_key)

        arg_offsets = tuple(getattr(entry, "arg_offsets", ()))
        arg_leaf_counts = tuple(getattr(entry, "arg_leaf_counts", ()))
        invar_plans = tuple(getattr(entry, "invar_plans", ()))
        state = getattr(entry, "state", {})
        fn_outvar_map = state.get("fn_outvar_map", ()) if isinstance(state, dict) else ()
        donated_by_stage = tuple(state.get("donate_argnums_per_stage", ())) if isinstance(state, dict) else ()

        def _arg_span(argnum: int) -> range:
            if argnum >= len(arg_offsets) or argnum >= len(arg_leaf_counts):
                return range(0, 0)
            start = int(arg_offsets[argnum])
            count = int(arg_leaf_counts[argnum])
            return range(start, start + count)

        if (
            isinstance(prepare_cache_key, tuple)
            and len(prepare_cache_key) >= 1
            and prepare_cache_key[0] == "model_step"
        ):
            kv_argnum = 2
            metadata_argnum = 3
        else:
            kv_argnum = 3
            metadata_argnum = 4

        kv_span = _arg_span(kv_argnum)
        metadata_span = _arg_span(metadata_argnum)
        kv_indices = set(kv_span)
        metadata_indices = set(metadata_span)
        dynamic_kv_slots: list[int] = []
        dynamic_metadata_slots: list[int] = []
        dynamic_slots: list[int] = []
        stage_slots: list[int] = []
        prev_slots: list[int] = []
        total_invars: list[int] = []

        for plan in invar_plans:
            plan_dynamic = tuple(getattr(plan, "dynamic_slots", ()))
            dynamic_slots.append(len(plan_dynamic))
            dynamic_kv_slots.append(sum(1 for _, orig_idx in plan_dynamic if int(orig_idx) in kv_indices))
            dynamic_metadata_slots.append(sum(1 for _, orig_idx in plan_dynamic if int(orig_idx) in metadata_indices))
            stage_slots.append(len(tuple(getattr(plan, "stage_slots", ()))))
            prev_slots.append(len(tuple(getattr(plan, "prev_slots", ()))))
            total_invars.append(len(tuple(getattr(plan, "template", ()))))

        kv_outputs_by_stage: dict[int | str, int] = {}
        for mapping in tuple(fn_outvar_map)[: len(kv_indices)]:
            if not mapping:
                continue
            owner = mapping[0]
            if isinstance(owner, int):
                key: int | str = int(owner)
            else:
                key = str(owner)
            kv_outputs_by_stage[key] = kv_outputs_by_stage.get(key, 0) + 1

        logger.info(
            "PP plan audit mode=%s key=%s arg_leaf_counts=%s kv_arg=%d metadata_arg=%d stages=%d "
            "kv_leaves=%d metadata_leaves=%d dynamic/stage=%s dynamic_kv/stage=%s "
            "dynamic_metadata/stage=%s stage_edges/stage=%s prev_edges/stage=%s "
            "total_invars/stage=%s donate_argnums/stage=%s kv_outputs_by_stage=%s",
            mode,
            prepare_cache_key,
            arg_leaf_counts,
            kv_argnum,
            metadata_argnum,
            len(invar_plans),
            len(kv_indices),
            len(metadata_indices),
            dynamic_slots,
            dynamic_kv_slots,
            dynamic_metadata_slots,
            stage_slots,
            prev_slots,
            total_invars,
            donated_by_stage,
            kv_outputs_by_stage,
        )

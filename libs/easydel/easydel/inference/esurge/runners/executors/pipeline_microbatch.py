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

"""Pipeline-parallel decode microbatching (wavefront execution) for eSurge.

Extracted verbatim from ``ExecutionManager`` (phase 11.2 of the eSurge
modularization). The :class:`PipelineMicrobatchExecutor` owns the reusable
CPU scratch slots and device index/scalar caches for the SpectraX PP decode
wavefront path, and exposes :meth:`PipelineMicrobatchExecutor.try_execute`,
which either runs the wavefront and returns the full ``execute()`` result
tuple or returns ``None`` so the manager falls back to the fused full-window
path. All manager-owned state (kv pages, rng key, sampler runtime, compiled
executors) is reached through a back-reference; mutation stays on the
scheduler thread, so the single-writer discipline for ``kv_pages`` /
``rng_key`` / sampler token counts is unchanged.
"""

from __future__ import annotations

import os
import time
import typing as tp
from functools import partial

import jax
import numpy
from ejkernel.ops import forward_autotune_only  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp

from easydel.infra.sharding import replicate_on_array_mesh

from ..async_types import DeviceInputTokenHandoff
from ..execution_types import BatchMetadata
from .sampler_executor import _get_padded_num_reqs_with_upper_limit

if tp.TYPE_CHECKING:
    from ..execution_manager import ExecutionManager

_PP_MB_DEBUG = bool(os.environ.get("ESURGE_PP_MB_DEBUG"))


class _PipelineMicrobatchScratchSlot(tp.TypedDict):
    """Reusable host buffers for one PP decode microbatch.

    eSurge builds decode microbatches by slicing an active request window into
    several smaller logical batches. Allocating fresh NumPy arrays on every
    token would show up directly in the decode loop, so each slot owns one set
    of CPU buffers and is repopulated in-place for the next step.
    """

    scheduled: numpy.ndarray
    active: numpy.ndarray
    token_ids: numpy.ndarray
    num_computed: numpy.ndarray
    temperature: numpy.ndarray
    top_p: numpy.ndarray
    top_k: numpy.ndarray
    min_p: numpy.ndarray
    frequency: numpy.ndarray
    presence: numpy.ndarray
    repetition: numpy.ndarray
    page_table: numpy.ndarray
    prev_count: int


_PipelineExecuteResult = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, tp.Any],
]


@partial(jax.jit, static_argnames=("padded_num_reqs",))
def _combine_pipeline_hidden_parts(
    hidden_parts: tuple[jax.Array, ...],
    logits_index_parts: tuple[jax.Array, ...],
    position_parts: tuple[jax.Array, ...],
    *,
    padded_num_reqs: int,
) -> jax.Array:
    """Scatter per-microbatch backbone hidden states back into a single window.

    Used by the pipeline-parallel decode path, where the active decode rows of a
    scheduler window are split across multiple PP microbatches. Each microbatch
    runs the backbone and produces its own ``hidden_parts[i]`` plus the
    ``logits_index_parts[i]`` (the index of the last token per row inside that
    microbatch's packed token buffer) and the ``position_parts[i]`` (the row
    indices in the full window where those rows belong). This helper scatters
    the gathered ``hidden_parts[i][logits_index_parts[i]]`` slices into a
    ``[padded_num_reqs, hidden_dim]`` output so the LM head can be run once on
    the combined window instead of per-microbatch.

    Args:
        hidden_parts: Tuple of per-microbatch hidden-state tensors, one per
            microbatch, with shape ``[num_tokens_i, hidden_dim]``.
        logits_index_parts: Tuple of per-microbatch logits indices selecting
            the last token row from each microbatch's hidden tensor.
        position_parts: Tuple of window-row indices marking where each
            microbatch's rows land in the combined output.
        padded_num_reqs: Padded request count for the current window; defines
            the row dimension of the returned tensor.

    Returns:
        A ``[padded_num_reqs, hidden_dim]`` array with each microbatch's last
        token hidden states placed at its corresponding window row position.
    """
    with jax.named_scope("easydel/esurge/pp_hidden_parts/combine"):
        hidden = hidden_parts[0]
        combined = jnp.zeros((int(padded_num_reqs), int(hidden.shape[-1])), dtype=hidden.dtype)
        for part_hidden, part_indices, part_positions in zip(
            hidden_parts,
            logits_index_parts,
            position_parts,
            strict=True,
        ):
            combined = combined.at[part_positions].set(part_hidden[part_indices])
        return combined


class PipelineMicrobatchExecutor:
    """SpectraX PP decode-wavefront executor for ``ExecutionManager``.

    Constructed only when the manager's :class:`PipelineInferencePlan` is
    enabled. Splits large decode-only windows into same-shaped microbatches,
    launches them as a host wavefront through the resident pipeline runtime,
    then combines hidden states, runs the LM head, and samples — returning
    the same tuple shape as :meth:`ExecutionManager.execute`.
    """

    def __init__(self, manager: "ExecutionManager") -> None:
        """Bind the owning manager and initialize reusable caches.

        Args:
            manager: The :class:`ExecutionManager` whose compiled executors,
                cache pages, rng key, and sampler runtime this executor
                drives. Held as a back-reference; no state is copied.
        """
        self._manager = manager
        self._pipeline_microbatch_scratch: list[_PipelineMicrobatchScratchSlot] = []
        self._pipeline_microbatch_scratch_signature: tuple[tuple[tuple[int, ...], str], ...] | None = None
        self._pipeline_logits_index_cache: dict[tuple[int, str], jax.Array] = {}
        self._pipeline_handoff_scalar_cache: dict[tuple[int, int], tuple[jax.Array, jax.Array]] = {}

    def clear_cache(self) -> None:
        """Drop the CPU scratch pool and cached device indices/scalars."""
        self._pipeline_microbatch_scratch.clear()
        self._pipeline_microbatch_scratch_signature = None
        self._pipeline_logits_index_cache.clear()
        self._pipeline_handoff_scalar_cache.clear()

    @staticmethod
    def _skip(reason: str, **ctx: tp.Any) -> None:
        """Log (when ``ESURGE_PP_MB_DEBUG`` is set) why the wavefront was skipped.

        Diagnostic-only: every ``try_execute`` bailout routes through here so a
        single instrumented decode run reveals which gate keeps the PP wavefront
        overlap from engaging. Returns ``None`` so callers can ``return
        self._skip(...)`` directly.
        """
        if _PP_MB_DEBUG:
            extra = " ".join(f"{k}={v}" for k, v in ctx.items())
            print(f"[pp-mb-skip] {reason} {extra}".rstrip(), flush=True)
        return None

    def _get_pipeline_handoff_scalars(self, *, offset: int, count: int) -> tuple[jax.Array, jax.Array]:
        """Return cached device scalars for a PP microbatch handoff span.

        The exact decode microbatch path uses the same full-window sampled-token
        handoff for every microbatch. Only the live slice changes. Creating
        those scalar leaves once avoids putting tiny host arrays on every token,
        while also avoiding the much more expensive device scatter that used to
        build a fresh handoff vector per microbatch.
        """
        key = (int(offset), int(count))
        cached = self._pipeline_handoff_scalar_cache.get(key)
        if cached is not None:
            return cached
        offset_dev = jax.device_put(numpy.asarray(int(offset), dtype=numpy.int32), self._manager._empty_sharding)
        count_dev = jax.device_put(numpy.asarray(int(count), dtype=numpy.int32), self._manager._empty_sharding)
        cached = (offset_dev, count_dev)
        self._pipeline_handoff_scalar_cache[key] = cached
        return cached

    def _ensure_pipeline_microbatch_scratch(
        self,
        count: int,
        *,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        token_ids_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,
    ) -> list[_PipelineMicrobatchScratchSlot]:
        """Return reusable CPU scratch slots matching the current window shapes.

        The signature is the shape/dtype contract for all host-side arrays used
        to build a decode microbatch. Any shape change invalidates the scratch
        pool; otherwise slots are reused in-place across decode iterations.
        """
        signature = tuple(
            (tuple(arr.shape), str(arr.dtype))
            for arr in (
                scheduled_full_cpu,
                active_mask_full_cpu,
                token_ids_cpu,
                num_computed_tokens_cpu,
                temperature_cpu,
                top_p_cpu,
                top_k_cpu,
                min_p_cpu,
                frequency_penalties_cpu,
                presence_penalties_cpu,
                repetition_penalties_cpu,
                page_table_cpu,
            )
        )
        if self._pipeline_microbatch_scratch_signature != signature:
            self._pipeline_microbatch_scratch = []
            self._pipeline_microbatch_scratch_signature = signature

        def _new_slot() -> _PipelineMicrobatchScratchSlot:
            """Allocate one zero-initialized scratch slot for a PP microbatch.

            Each slot mirrors the per-request CPU buffers required to assemble
            a microbatch (scheduled tokens, active mask, token ids, computed
            tokens, sampling parameters, page table) plus a ``prev_count`` sentinel
            used by repopulation logic. Buffers are sized like the corresponding
            window-level CPU arrays so repeated microbatches can write into them
            without reallocating.
            """
            return {
                "scheduled": numpy.zeros_like(scheduled_full_cpu),
                "active": numpy.zeros_like(active_mask_full_cpu),
                "token_ids": numpy.zeros_like(token_ids_cpu),
                "num_computed": numpy.zeros_like(num_computed_tokens_cpu),
                "temperature": numpy.ones_like(temperature_cpu),
                "top_p": numpy.ones_like(top_p_cpu),
                "top_k": numpy.zeros_like(top_k_cpu),
                "min_p": numpy.zeros_like(min_p_cpu),
                "frequency": numpy.zeros_like(frequency_penalties_cpu),
                "presence": numpy.zeros_like(presence_penalties_cpu),
                "repetition": numpy.ones_like(repetition_penalties_cpu),
                "page_table": numpy.zeros_like(page_table_cpu),
                "prev_count": 0,
            }

        while len(self._pipeline_microbatch_scratch) < int(count):
            self._pipeline_microbatch_scratch.append(_new_slot())
        return self._pipeline_microbatch_scratch[: int(count)]

    @staticmethod
    def _populate_pipeline_microbatch_scratch(
        slot: _PipelineMicrobatchScratchSlot,
        *,
        chunk: numpy.ndarray,
        scheduled_full_cpu: numpy.ndarray,
        token_ids_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,
    ) -> tuple[
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
    ]:
        """Fill one scratch slot with a compact request chunk.

        ``chunk`` contains row indices from the full active window. The first
        ``len(chunk)`` rows of each scratch array become a dense microbatch;
        stale rows from the previous use of the slot are cleared so JAX sees a
        stable padded shape with no leftover request metadata.
        """
        real_count = len(chunk)
        prev_count = int(slot["prev_count"])
        clear_count = max(prev_count, real_count)

        mb_scheduled = slot["scheduled"]
        mb_active = slot["active"]
        mb_token_ids = slot["token_ids"]
        mb_num_computed = slot["num_computed"]
        mb_temperature = slot["temperature"]
        mb_top_p = slot["top_p"]
        mb_top_k = slot["top_k"]
        mb_min_p = slot["min_p"]
        mb_frequency = slot["frequency"]
        mb_presence = slot["presence"]
        mb_repetition = slot["repetition"]
        mb_page_table = slot["page_table"]

        if clear_count:
            mb_scheduled[:clear_count] = 0
            mb_active[:clear_count] = False
            mb_num_computed[:clear_count] = 0
            mb_temperature[:clear_count] = 1.0
            mb_top_p[:clear_count] = 1.0
            mb_top_k[:clear_count] = 0
            mb_min_p[:clear_count] = 0.0
            mb_frequency[:clear_count] = 0.0
            mb_presence[:clear_count] = 0.0
            mb_repetition[:clear_count] = 1.0
        if prev_count > real_count:
            mb_token_ids[real_count:prev_count] = 0
            mb_page_table[real_count:prev_count] = 0

        if real_count:
            mb_scheduled[:real_count] = scheduled_full_cpu[chunk]
            mb_active[:real_count] = True
            mb_token_ids[:real_count] = token_ids_cpu[chunk]
            mb_num_computed[:real_count] = num_computed_tokens_cpu[chunk]
            mb_temperature[:real_count] = temperature_cpu[chunk]
            mb_top_p[:real_count] = top_p_cpu[chunk]
            mb_top_k[:real_count] = top_k_cpu[chunk]
            mb_min_p[:real_count] = min_p_cpu[chunk]
            mb_frequency[:real_count] = frequency_penalties_cpu[chunk]
            mb_presence[:real_count] = presence_penalties_cpu[chunk]
            mb_repetition[:real_count] = repetition_penalties_cpu[chunk]
            mb_page_table[:real_count] = page_table_cpu[chunk]

        slot["prev_count"] = real_count
        return (
            mb_scheduled,
            mb_active,
            mb_token_ids,
            mb_num_computed,
            mb_temperature,
            mb_top_p,
            mb_top_k,
            mb_min_p,
            mb_frequency,
            mb_presence,
            mb_repetition,
            mb_page_table,
        )

    def _get_pipeline_logits_indices(self, count: int, like: jax.Array) -> jax.Array:
        """Return cached decode logits indices placed like a PP hidden-state part."""
        count = int(count)
        sharding_key = repr(like.sharding)
        key = (count, sharding_key)
        cached = self._pipeline_logits_index_cache.get(key)
        if cached is not None:
            return cached
        indices = jnp.arange(count, dtype=jnp.int32)
        indices = replicate_on_array_mesh(indices, like)
        self._pipeline_logits_index_cache[key] = indices
        return indices

    @staticmethod
    def _use_pipeline_microbatching(
        *,
        active_count: int,
        num_stages: int,
        microbatch_count: int,
        microbatch_req_count: int,
        min_token_bucket: int,
        has_compiled_handoff: bool,
    ) -> bool:
        """Choose PP wavefront microbatching when decode rows can fill stages.

        Decode microbatching overlaps pipeline stages by splitting active
        requests into a wavefront. One active request cannot fill a PP pipeline,
        but ``num_stages`` active decode requests can: while stage 3 finishes
        microbatch 0, stage 2 can process microbatch 1, and so on.

        Keep the fused full-window path for underfilled batches because it uses
        fewer launches and is better latency-wise. Without compiled token
        handoff, also require each microbatch to carry enough real decode rows
        to fill the smallest token bucket; otherwise the loop-carried sampled
        token dependency comes back as a large repair/synchronization before
        the next launch. With compiled handoff, a one-token microbatch is a
        legitimate PP wavefront unit because the previous sampled token is
        consumed inside the compiled model step.
        """
        if int(active_count) < int(num_stages):
            return False
        if int(microbatch_count) < int(num_stages):
            return False
        if bool(has_compiled_handoff):
            return int(microbatch_req_count) >= 1
        return int(microbatch_req_count) >= max(1, int(min_token_bucket))

    @staticmethod
    def _resolve_pipeline_microbatch_shape(
        *,
        active_count: int,
        num_stages: int,
        pp_microbatch_count: int | str | None,
        pp_microbatch_size: int | str | None,
    ) -> tuple[int, int] | None:
        """Resolve the PP wavefront shape for the current active decode rows.

        Returns ``(microbatch_count, rows_per_microbatch)``. ``None`` means the
        user disabled PP microbatch wavefronting for this runner.
        """

        def _policy_value(value: int | str | None) -> int | str | None:
            """Normalize a microbatch policy knob to ``"auto"`` / ``int`` / ``None``.

            Accepts the public surface (``"auto"``, ``"none"``/``"off"``,
            integer strings, plain ints, ``None``) and collapses it to the
            three internal forms used by :meth:`_resolve_pipeline_microbatch_shape`.
            Zero is folded to ``None`` (disabled), and unknown strings raise
            via the inner ``int(...)`` cast.
            """
            if value is None:
                return None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == "auto":
                    return "auto"
                if lowered in {"none", "off", "disable", "disabled"}:
                    return None
                value = int(lowered)
            value = int(value)
            return None if value == 0 else value

        active_count = int(active_count)
        num_stages = int(num_stages)
        pp_microbatch_count = _policy_value(pp_microbatch_count)
        pp_microbatch_size = _policy_value(pp_microbatch_size)
        if active_count <= 0 or num_stages <= 0:
            return None
        if pp_microbatch_count is None or pp_microbatch_size is None:
            return None

        count_is_auto = pp_microbatch_count == "auto"
        size_is_auto = pp_microbatch_size == "auto"
        if not count_is_auto and not size_is_auto:
            raise ValueError("Only one of pp_microbatch_count or pp_microbatch_size may be a positive integer.")

        if not size_is_auto:
            rows_per_microbatch = max(1, int(pp_microbatch_size))
            microbatch_count = (active_count + rows_per_microbatch - 1) // rows_per_microbatch
        elif not count_is_auto:
            microbatch_count = max(1, min(int(pp_microbatch_count), active_count))
            rows_per_microbatch = (active_count + microbatch_count - 1) // microbatch_count
        else:
            microbatch_count = min(num_stages, active_count)
            rows_per_microbatch = (active_count + microbatch_count - 1) // microbatch_count

        return int(microbatch_count), int(rows_per_microbatch)

    def try_execute(
        self,
        *,
        num_tokens: int,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        window_row_indices_cpu: numpy.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        padded_num_reqs: int,
        req_num_tokens_full_cpu: numpy.ndarray,
        token_ids_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
        page_table_cpu: numpy.ndarray,
        mrope_position_ids_cpu: numpy.ndarray | None,
        prefill_embeds_cpu: numpy.ndarray | None,
        prefill_embeds_mask_cpu: numpy.ndarray | None,
        visual_pos_masks_cpu: numpy.ndarray | None,
        deepstack_visual_embeds_cpu: list[numpy.ndarray] | None,
        pixel_values: numpy.ndarray | None,
        image_grid_thw: numpy.ndarray | None,
        pixel_values_videos: numpy.ndarray | None,
        video_grid_thw: numpy.ndarray | None,
        device_token_handoff: DeviceInputTokenHandoff | None,
        wait_for_outputs: bool,
    ) -> _PipelineExecuteResult | None:
        """Try the SpectraX PP wavefront path for decode-only microbatches.

        Architecture:
            * The normal full-window PP path launches each physical stage once.
              That is best for small active windows because it minimizes launch
              count.
            * This path is selected only for larger decode windows. It splits
              active rows into same-shaped microbatches, asks SpectraX to launch
              them as a host wavefront, and carries each stage's local KV-cache
              leaves from microbatch N to N+1.
            * EasyDeL then gathers the sampled hidden states, runs the LM head,
              and scatters sampler output back to the original request rows.

        Returns:
            The same tuple shape as :meth:`execute` when the wavefront path is
            applicable; ``None`` when the caller should use the fused
            full-window path instead.
        """
        pipeline_plan = self._manager._model_executor.pipeline_plan
        pipeline_runtime = self._manager._model_executor.pipeline_runtime
        if pipeline_plan is None or not pipeline_plan.is_enabled or pipeline_runtime is None:
            return self._skip("plan_disabled")
        if any(
            x is not None
            for x in (
                mrope_position_ids_cpu,
                prefill_embeds_cpu,
                prefill_embeds_mask_cpu,
                visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            )
        ):
            return self._skip("multimodal_inputs")

        active_window = numpy.asarray(active_mask_full_cpu[:padded_num_reqs], dtype=numpy.bool_)
        scheduled_window = numpy.asarray(scheduled_full_cpu[:padded_num_reqs], dtype=numpy.int32)
        active_positions = numpy.flatnonzero(active_window & (scheduled_window > 0))
        active_count = int(active_positions.size)
        if active_count < 2:
            return self._skip("active_count<2", active_count=active_count)
        if not numpy.all(scheduled_window[active_positions] == 1):
            return self._skip("not_decode_only")

        num_stages = len(pipeline_plan.stage_meshes)
        if num_stages < 2:
            return self._skip("num_stages<2", num_stages=num_stages)
        microbatch_shape = self._resolve_pipeline_microbatch_shape(
            active_count=active_count,
            num_stages=num_stages,
            pp_microbatch_count=self._manager.pp_microbatch_count,
            pp_microbatch_size=self._manager.pp_microbatch_size,
        )
        if microbatch_shape is None:
            return self._skip(
                "microbatch_shape_disabled",
                pp_microbatch_count=self._manager.pp_microbatch_count,
                pp_microbatch_size=self._manager.pp_microbatch_size,
            )
        microbatch_count, microbatch_req_count = microbatch_shape
        if not self._use_pipeline_microbatching(
            active_count=active_count,
            num_stages=num_stages,
            microbatch_count=microbatch_count,
            microbatch_req_count=microbatch_req_count,
            min_token_bucket=int(self._manager.min_input_pad),
            has_compiled_handoff=device_token_handoff is not None,
        ):
            return self._skip(
                "use_microbatching=False",
                active_count=active_count,
                num_stages=num_stages,
                microbatch_count=microbatch_count,
                microbatch_req_count=microbatch_req_count,
                min_input_pad=int(self._manager.min_input_pad),
                has_handoff=device_token_handoff is not None,
            )
        microbatch_padded_reqs = _get_padded_num_reqs_with_upper_limit(
            microbatch_req_count,
            upper_limit=int(padded_num_reqs),
            min_input_pad=int(self._manager.min_input_pad),
        )
        token_paddings = self._manager._model_num_tokens_paddings
        if token_paddings:
            candidates = [int(x) for x in token_paddings if int(x) >= int(microbatch_req_count)]
            if not candidates:
                return self._skip(
                    "no_token_bucket",
                    microbatch_req_count=microbatch_req_count,
                    token_paddings=list(token_paddings),
                )
            microbatch_num_tokens = min(candidates)
        else:
            microbatch_num_tokens = max(int(num_tokens), int(microbatch_req_count), int(self._manager.min_input_pad))
        if not self._manager._has_backbone_variant(microbatch_num_tokens, use_pipeline_runtime=True):
            return self._skip(
                "no_pp_backbone_variant",
                microbatch_num_tokens=microbatch_num_tokens,
                microbatch_req_count=microbatch_req_count,
            )
        if not self._manager._model_executor.has_lm_head(padded_num_reqs):
            return self._skip("no_lm_head", padded_num_reqs=padded_num_reqs)

        start_prep = time.time()
        chunks = [active_positions[i : i + microbatch_req_count] for i in range(0, active_count, microbatch_req_count)]
        if len(chunks) < 2:
            return self._skip("chunks<2", num_chunks=len(chunks), microbatch_req_count=microbatch_req_count)
        if device_token_handoff is not None and int(device_token_handoff.token_ids.shape[0]) < active_count:
            return self._skip(
                "handoff_too_small",
                handoff_rows=int(device_token_handoff.token_ids.shape[0]),
                active_count=active_count,
            )

        input_batches: list[tuple[tp.Any, tp.Any, tp.Any, BatchMetadata]] = []
        metadata_batches: list[BatchMetadata] = []
        original_positions: list[numpy.ndarray] = []
        prep_stats_accum: dict[str, float] = {}
        microbatch_scratch_time = 0.0
        microbatch_metadata_time = 0.0
        microbatch_handoff_time = 0.0
        sampler_window_time = 0.0
        ensure_variants_time = 0.0
        scratch_slots = self._ensure_pipeline_microbatch_scratch(
            len(chunks),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            frequency_penalties_cpu=frequency_penalties_cpu,
            presence_penalties_cpu=presence_penalties_cpu,
            repetition_penalties_cpu=repetition_penalties_cpu,
            page_table_cpu=page_table_cpu,
        )

        handoff_offset = 0
        for chunk, scratch_slot in zip(chunks, scratch_slots, strict=True):
            scratch_start = time.time()
            (
                mb_scheduled,
                mb_active,
                mb_token_ids,
                mb_num_computed,
                mb_temperature,
                mb_top_p,
                mb_top_k,
                mb_min_p,
                mb_frequency,
                mb_presence,
                mb_repetition,
                mb_page_table,
            ) = self._populate_pipeline_microbatch_scratch(
                scratch_slot,
                chunk=chunk,
                scheduled_full_cpu=scheduled_full_cpu,
                token_ids_cpu=token_ids_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                temperature_cpu=temperature_cpu,
                top_p_cpu=top_p_cpu,
                top_k_cpu=top_k_cpu,
                min_p_cpu=min_p_cpu,
                frequency_penalties_cpu=frequency_penalties_cpu,
                presence_penalties_cpu=presence_penalties_cpu,
                repetition_penalties_cpu=repetition_penalties_cpu,
                page_table_cpu=page_table_cpu,
            )
            microbatch_scratch_time += time.time() - scratch_start

            metadata_start = time.time()
            batch_metadata, input_ids_buf, position_ids_buf, _scheduled_dev, _active_dev = (
                self._manager._batch_preparer.prepare_batch_metadata(
                    num_tokens_static=microbatch_num_tokens,
                    scheduled_full_cpu=mb_scheduled,
                    active_mask_full_cpu=mb_active,
                    input_ids_buf=input_ids_buf,
                    position_ids_buf=position_ids_buf,
                    token_ids_cpu=mb_token_ids,
                    num_computed_tokens_cpu=mb_num_computed,
                    temperature_cpu=mb_temperature,
                    top_p_cpu=mb_top_p,
                    top_k_cpu=mb_top_k,
                    min_p_cpu=mb_min_p,
                    frequency_penalties_cpu=mb_frequency,
                    presence_penalties_cpu=mb_presence,
                    repetition_penalties_cpu=mb_repetition,
                    page_table_cpu=mb_page_table,
                    page_table_version=None,
                    padded_num_reqs_in=microbatch_padded_reqs,
                )
            )
            microbatch_metadata_time += time.time() - metadata_start
            local_count = int(chunk.shape[0])
            if device_token_handoff is not None and local_count > 0:
                handoff_start = time.time()
                handoff_offset_dev, handoff_count_dev = self._get_pipeline_handoff_scalars(
                    offset=handoff_offset,
                    count=local_count,
                )
                local_handoff = DeviceInputTokenHandoff(
                    input_positions=device_token_handoff.input_positions,
                    token_ids=device_token_handoff.token_ids,
                    count=handoff_count_dev,
                    offset=handoff_offset_dev,
                )
                batch_metadata = self._manager._apply_device_token_handoff(batch_metadata, local_handoff)
                microbatch_handoff_time += time.time() - handoff_start
            handoff_offset += local_count
            for key, value in self._manager._batch_preparer.last_prep_stats.items():
                prep_stats_accum[key] = prep_stats_accum.get(key, 0.0) + float(value)

            input_batches.append((self._manager.graphstate, self._manager.graphother, self._manager.kv_pages, batch_metadata))
            metadata_batches.append(batch_metadata)
            original_positions.append(numpy.asarray(chunk, dtype=numpy.int32))

        sampler_window_start = time.time()
        sampler_num_reqs, sampler_padded_num_reqs, sampler_total_tokens = self._manager._prepare_compact_sampler_window(
            padded_num_reqs=padded_num_reqs,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            window_row_indices_cpu=window_row_indices_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            frequency_penalties_cpu=frequency_penalties_cpu,
            presence_penalties_cpu=presence_penalties_cpu,
            repetition_penalties_cpu=repetition_penalties_cpu,
        )
        sampler_window_time += time.time() - sampler_window_start
        gather_positions_cpu = self._manager._sampler_gather_positions_cpu.copy()
        sampling_seeds_cpu = self._manager._sampler_sampling_seeds_cpu.copy()
        scatter_positions_cpu = self._manager._sampler_scatter_positions_cpu.copy()
        compact_window_row_indices_cpu = self._manager._sampler_window_row_indices_cpu.copy()
        compact_scheduled_cpu = self._manager._sampler_scheduled_cpu.copy()
        compact_seq_lens_cpu = self._manager._sampler_seq_lens_cpu.copy()
        compact_active_mask_cpu = self._manager._sampler_active_mask_cpu.copy()
        compact_temperature_cpu = self._manager._sampler_temperature_cpu.copy()
        compact_top_p_cpu = self._manager._sampler_top_p_cpu.copy()
        compact_top_k_cpu = self._manager._sampler_top_k_cpu.copy()
        compact_min_p_cpu = self._manager._sampler_min_p_cpu.copy()
        compact_frequency_penalties_cpu = self._manager._sampler_frequency_penalties_cpu.copy()
        compact_presence_penalties_cpu = self._manager._sampler_presence_penalties_cpu.copy()
        compact_repetition_penalties_cpu = self._manager._sampler_repetition_penalties_cpu.copy()
        need_penalties = bool(
            numpy.any(frequency_penalties_cpu[:padded_num_reqs] != 0.0)
            or numpy.any(presence_penalties_cpu[:padded_num_reqs] != 0.0)
            or numpy.any(repetition_penalties_cpu[:padded_num_reqs] != 1.0)
        )
        ensure_start = time.time()
        self._manager._require_precompiled_variants(
            num_tokens=microbatch_num_tokens,
            padded_num_reqs=padded_num_reqs,
            sampler_padded_num_reqs=sampler_padded_num_reqs,
            use_pipeline_runtime=True,
        )
        ensure_variants_time += time.time() - ensure_start

        prep_took = time.time() - start_prep

        start_backbone = time.time()
        with forward_autotune_only():
            backbone_outputs_list = self._manager._model_executor.execute_backbones_many(
                num_tokens=microbatch_num_tokens,
                input_batches=input_batches,
            )
        self._manager.kv_pages = backbone_outputs_list[-1].kv_pages
        backbone_took = time.time() - start_backbone

        start_combine = time.time()
        hidden_parts: list[jax.Array] = []
        logits_index_parts: list[jax.Array] = []
        position_parts: list[jax.Array] = []
        positions_are_prefix = bool(
            active_count <= int(padded_num_reqs)
            and active_positions.shape[0] == active_count
            and numpy.array_equal(active_positions, numpy.arange(active_count, dtype=active_positions.dtype))
        )
        for metadata, backbone_out, positions in zip(
            metadata_batches,
            backbone_outputs_list,
            original_positions,
            strict=True,
        ):
            local_count = int(positions.shape[0])
            if positions_are_prefix:
                logits_indices = self._get_pipeline_logits_indices(local_count, backbone_out.hidden_states)
            else:
                logits_indices = metadata.logits_indices[:local_count]
                logits_indices = replicate_on_array_mesh(logits_indices, backbone_out.hidden_states)
            hidden_parts.append(backbone_out.hidden_states)
            logits_index_parts.append(logits_indices)
            if not positions_are_prefix:
                scatter_idx = replicate_on_array_mesh(
                    jnp.asarray(positions, dtype=jnp.int32), backbone_out.hidden_states
                )
                position_parts.append(scatter_idx)

        if positions_are_prefix:
            combined_hidden = None
        else:
            combined_hidden = _combine_pipeline_hidden_parts(
                tuple(hidden_parts),
                tuple(logits_index_parts),
                tuple(position_parts),
                padded_num_reqs=int(padded_num_reqs),
            )
        combine_took = time.time() - start_combine

        start_lm_head = time.time()
        if positions_are_prefix:
            lm_head_fn = self._manager._model_executor.get_pipeline_lm_head(
                padded_num_reqs=padded_num_reqs,
                part_rows=tuple(int(positions.shape[0]) for positions in original_positions),
            )
            logits = lm_head_fn(self._manager.graphstate, self._manager.graphother, tuple(hidden_parts), tuple(logits_index_parts))
        else:
            combined_hidden = self._manager._model_executor._place_lm_head_hidden(combined_hidden)
            lm_head_fn = self._manager._model_executor.get_lm_head(padded_num_reqs=padded_num_reqs)
            logits = lm_head_fn(self._manager.graphstate, self._manager.graphother, combined_hidden)
        lm_head_took = time.time() - start_lm_head

        start_sampler = time.time()
        rng_key_out, out_tokens_full, valid_mask_full, token_counts_out = self._manager.sample_tokens(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            sampler_padded_num_reqs=sampler_padded_num_reqs,
            sampler_num_reqs=sampler_num_reqs,
            sampler_total_tokens=sampler_total_tokens,
            req_num_tokens_full_cpu=req_num_tokens_full_cpu,
            logits=logits,
            rng_key=self._manager.rng_key,
            gather_positions_cpu=gather_positions_cpu,
            sampling_seeds_cpu=sampling_seeds_cpu,
            scatter_positions_cpu=scatter_positions_cpu,
            compact_window_row_indices_cpu=compact_window_row_indices_cpu,
            compact_scheduled_cpu=compact_scheduled_cpu,
            compact_seq_lens_cpu=compact_seq_lens_cpu,
            compact_active_mask_cpu=compact_active_mask_cpu,
            compact_temperature_cpu=compact_temperature_cpu,
            compact_top_p_cpu=compact_top_p_cpu,
            compact_top_k_cpu=compact_top_k_cpu,
            compact_min_p_cpu=compact_min_p_cpu,
            compact_frequency_penalties_cpu=compact_frequency_penalties_cpu,
            compact_presence_penalties_cpu=compact_presence_penalties_cpu,
            compact_repetition_penalties_cpu=compact_repetition_penalties_cpu,
            need_penalties=need_penalties,
        )
        self._manager.rng_key = rng_key_out
        self._manager._sampler_token_counts = token_counts_out
        sampler_enqueue_took = time.time() - start_sampler
        exec_took = backbone_took + combine_took + lm_head_took + sampler_enqueue_took

        if wait_for_outputs:
            jax.block_until_ready(logits)
            start_sample = time.time()
            jax.block_until_ready(out_tokens_full)
            sample_took = time.time() - start_sample
        else:
            sample_took = 0.0

        metrics = {
            "exec_time": exec_took,
            "sample_time": sample_took,
            "prep_time": prep_took,
            "execute_overhead_time": 0.0,
            "buckets_processed": int(microbatch_num_tokens) * len(chunks),
            "token_bucket": int(microbatch_num_tokens),
            "padded_num_reqs": int(microbatch_padded_reqs),
            "sampler_padded_num_reqs": int(sampler_padded_num_reqs),
            "sampler_num_reqs": int(active_count),
            "pp_microbatches": len(chunks),
            "pp_backbone_time": backbone_took,
            "pp_combine_time": combine_took,
            "pp_lm_head_time": lm_head_took,
            "pp_sampler_enqueue_time": sampler_enqueue_took,
            "pp_microbatch_scratch_time": microbatch_scratch_time,
            "pp_microbatch_metadata_time": microbatch_metadata_time,
            "pp_microbatch_handoff_time": microbatch_handoff_time,
            "pp_sampler_window_time": sampler_window_time,
            "pp_ensure_variants_time": ensure_variants_time,
        }
        metrics.update(self._manager._model_executor.last_pipeline_stats())
        metrics.update(prep_stats_accum)

        return (
            out_tokens_full,
            valid_mask_full,
            input_ids_buf,
            position_ids_buf,
            backbone_outputs_list[-1].hidden_states,
            logits,
            metrics,
        )

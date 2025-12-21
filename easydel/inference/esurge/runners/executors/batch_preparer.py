# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""CPU-first batch metadata preparation for eSurge execution.

This module extracts the host-side batch preparation logic from the runner's
execution manager. It performs fast NumPy computations on CPU and moves a single
consolidated payload to device via `jax.device_put`.
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np

from easydel.layers.caching import RaggedPagesCacheView
from easydel.layers.caching._metadatabuilder import AttentionMetadataBuilder

from ...page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from ..execution_types import BatchMetadata


class BatchMetadataPreparer:
    """Prepare and transfer per-step metadata for model execution."""

    def __init__(
        self,
        *,
        metadata: RaggedPagesCacheView,
        empty_sharding: jax.sharding.Sharding,
        max_num_tokens: int,
        max_num_reqs: int,
        max_model_len: int,
        min_input_pad: int,
    ) -> None:
        self.metadata = metadata
        self._empty_sharding = empty_sharding

        self.max_num_tokens = int(max_num_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.max_model_len = int(max_model_len)
        self.min_input_pad = int(min_input_pad)

        self._metadata_version = metadata.version
        self._use_slot_mapping = self._metadata_version == "v2"
        self._use_request_distribution = not self._use_slot_mapping

        # Ragged paging shapes (compile-stable).
        num_reqs_max_model_len = (
            min(int(metadata.get_max_num_seqs()), self.max_num_reqs) if metadata is not None else self.max_num_reqs
        )
        self._num_reqs_max_model_len = max(int(num_reqs_max_model_len), 1)

        max_pages_per_req = int(getattr(metadata, "max_num_pages_per_req", 0)) if metadata is not None else 0
        self._max_pages_per_req = max(int(max_pages_per_req), 1)

        # Pre-allocate CPU buffers for fast batch metadata preparation
        self._input_ids_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._positions_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._query_start_loc_cpu = np.zeros((self.max_num_reqs + 1,), dtype=np.int32)
        self._seq_lens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._logits_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._scheduled_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._arange_cpu = np.arange(self.max_num_tokens, dtype=np.int32)
        self._pages_tables_cpu = np.full(
            (self._num_reqs_max_model_len, self._max_pages_per_req),
            PAGE_TABLE_PADDING_VAL,
            dtype=np.int32,
        )
        self._request_distribution_placeholder = np.zeros((3,), dtype=np.int32)
        self._slot_mapping_placeholder = np.zeros((3, 1), dtype=np.int32)
        self._num_kv_update_placeholder = np.zeros((1,), dtype=np.int32)
        self._async_slot_mapping_placeholder = np.zeros((3, 1), dtype=np.int32)

        # Async staging buffers (avoid per-call allocations in overlap mode).
        self._async_input_ids_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._async_positions_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._async_query_start_loc_cpu = np.zeros((self.max_num_reqs + 1,), dtype=np.int32)
        self._async_seq_lens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_logits_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_scheduled_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_pages_tables_cpu = np.full(
            (self._num_reqs_max_model_len, self._max_pages_per_req),
            PAGE_TABLE_PADDING_VAL,
            dtype=np.int32,
        )
        self._async_request_distribution_cpu = np.zeros((3,), dtype=np.int32)
        self._async_num_kv_update_cpu = np.zeros((1,), dtype=np.int32)

        self._async_num_computed_tokens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_scheduled_full_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_active_mask_full_cpu = np.zeros((self.max_num_reqs,), dtype=bool)
        self._async_temperature_cpu = np.zeros((self.max_num_reqs,), dtype=np.float32)
        self._async_top_p_cpu = np.zeros((self.max_num_reqs,), dtype=np.float32)
        self._async_top_k_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_min_p_cpu = np.zeros((self.max_num_reqs,), dtype=np.float32)
        self._async_page_table_cpu = np.zeros((self.max_num_reqs, self._max_pages_per_req), dtype=np.int32)

        if self._use_slot_mapping and metadata is not None:
            self._slices_per_page = metadata.num_slices_per_kv_cache_update_page
            self._max_padded_slices = int(metadata.get_padded_num_slices(self.max_num_tokens, self.max_num_reqs))
            self._slot_mapping_cpu = np.full(
                (3, self._max_padded_slices),
                SLOT_MAPPING_PADDING_VAL,
                dtype=np.int32,
            )
            self._async_slot_mapping_cpu = np.full(
                (3, self._max_padded_slices),
                SLOT_MAPPING_PADDING_VAL,
                dtype=np.int32,
            )
            self._slot_mapping_indices = np.arange(self._max_padded_slices, dtype=np.int32)
        else:
            self._max_padded_slices = None
            self._slot_mapping_cpu = None
            self._slices_per_page = None
            self._slot_mapping_indices = None
            self._async_slot_mapping_cpu = None

        # Double buffering: store pending async device transfer
        self._pending_transfer: tuple | None = None
        self._pending_transfer_metadata: dict | None = None

    @property
    def uses_slot_mapping(self) -> bool:
        return self._use_slot_mapping

    @property
    def uses_request_distribution(self) -> bool:
        return self._use_request_distribution

    def _compute_slot_mapping_v2(
        self,
        *,
        num_requests: int,
        scheduled: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        slot_mapping_out: np.ndarray | None = None,
        copy_out: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Rebuild slot_mapping tensor for ragged-page attention v2."""

        slot_mapping = self._slot_mapping_cpu if slot_mapping_out is None else slot_mapping_out
        if slot_mapping is None:
            return np.zeros((3, 1), dtype=np.int32), 0

        slot_mapping.fill(SLOT_MAPPING_PADDING_VAL)

        if num_requests <= 0:
            return (slot_mapping.copy(), 0) if copy_out else (slot_mapping, 0)

        scheduled_active = np.asarray(scheduled[:num_requests], dtype=np.int32)
        num_computed_active = np.asarray(num_computed_tokens_cpu[:num_requests], dtype=np.int32)
        if not np.any(scheduled_active):
            return (slot_mapping.copy(), 0) if copy_out else (slot_mapping, 0)

        page_size = int(self.metadata.page_size)
        max_pages_per_req = int(self.metadata.max_num_pages_per_req)
        slices_per_page = int(self._slices_per_page or 1)

        start_tokens = num_computed_active
        end_tokens = num_computed_active + scheduled_active
        lps = start_tokens // page_size
        lpe = (np.maximum(end_tokens, 1) - 1) // page_size
        page_lens = np.where(scheduled_active > 0, lpe - lps + 1, 0).astype(np.int32)

        if not np.any(page_lens):
            return (slot_mapping.copy(), 0) if copy_out else (slot_mapping, 0)

        page_cum = np.cumsum(page_lens, dtype=np.int32)
        total_pages = int(page_cum[-1])

        max_padded_slices = int(self._max_padded_slices or 1)
        padded_num_slices = min(
            ((total_pages + slices_per_page - 1) // slices_per_page) * slices_per_page,
            max_padded_slices,
        )

        indices = self._slot_mapping_indices
        if indices is None:
            return slot_mapping.copy(), total_pages

        slice_active_mask = (indices < total_pages) & (indices < padded_num_slices)
        active_positions = indices[slice_active_mask]
        if active_positions.size == 0:
            return (slot_mapping.copy(), total_pages) if copy_out else (slot_mapping, total_pages)

        page_cum_prev = np.concatenate(([0], page_cum[:-1]))
        req_for_slice = np.searchsorted(page_cum, active_positions, side="right").astype(np.int32)
        local_off = active_positions - page_cum_prev[req_for_slice]

        pt = np.asarray(page_table_cpu[:num_requests, :max_pages_per_req], dtype=np.int32)
        pt_flat = pt.reshape(-1)
        gather_idx = req_for_slice * max_pages_per_req + lps[req_for_slice] + local_off
        np.clip(gather_idx, 0, pt_flat.size - 1, out=gather_idx)
        page_numbers = pt_flat[gather_idx]

        s_mod = start_tokens % page_size
        e_mod = ((np.maximum(end_tokens, 1) - 1) % page_size) + 1
        lens_rep = page_lens

        is_first = local_off == 0
        is_last = local_off == (lens_rep[req_for_slice] - 1)

        kv_local_st = np.where(is_first, s_mod[req_for_slice], 0)
        kv_local_en = np.where(is_last, e_mod[req_for_slice], page_size)
        slice_lens = np.maximum(kv_local_en - kv_local_st, 0).astype(np.int32)
        kv_cache_start = kv_local_st + page_numbers * page_size

        new_kv_start = np.cumsum(slice_lens, dtype=np.int32)
        if new_kv_start.size:
            new_kv_start = np.roll(new_kv_start, 1)
            new_kv_start[0] = 0

        slot_mapping[0, active_positions] = kv_cache_start
        slot_mapping[1, active_positions] = new_kv_start
        slot_mapping[2, active_positions] = slice_lens

        return (slot_mapping.copy(), total_pages) if copy_out else (slot_mapping, total_pages)

    def _build_host_payload(
        self,
        *,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        copy_slot_mapping: bool,
        use_async_buffers: bool = False,
    ) -> tuple[tuple, int]:
        """Build host payload using preallocated CPU buffers (hot path)."""

        max_num_reqs = int(self.max_num_reqs)
        num_requests = int(np.sum(active_mask_full_cpu[:max_num_reqs]))
        num_requests = min(num_requests, max_num_reqs)

        padded_num_reqs = AttentionMetadataBuilder.compute_padded_num_reqs(
            num_requests=num_requests,
            max_num_reqs=max_num_reqs,
            min_input_pad=int(self.min_input_pad),
            padded_num_reqs_in=int(padded_num_reqs_in),
        )

        if use_async_buffers:
            scheduled = self._async_scheduled_cpu
            qsl = self._async_query_start_loc_cpu
            input_ids = self._async_input_ids_cpu[: int(num_tokens_static)]
            positions = self._async_positions_cpu[: int(num_tokens_static)]
            seq_lens = self._async_seq_lens_cpu
            logits_indices = self._async_logits_indices_cpu
            pages_tables = self._async_pages_tables_cpu
            request_distribution = self._async_request_distribution_cpu
            num_kv_update_cpu = self._async_num_kv_update_cpu
            slot_mapping_buf = self._async_slot_mapping_cpu
        else:
            scheduled = self._scheduled_cpu
            qsl = self._query_start_loc_cpu
            input_ids = self._input_ids_cpu[: int(num_tokens_static)]
            positions = self._positions_cpu[: int(num_tokens_static)]
            seq_lens = self._seq_lens_cpu
            logits_indices = self._logits_indices_cpu
            pages_tables = self._pages_tables_cpu
            request_distribution = self._request_distribution_placeholder
            num_kv_update_cpu = self._num_kv_update_placeholder
            slot_mapping_buf = self._slot_mapping_cpu

        # scheduled: only active prefix is meaningful.
        scheduled.fill(0)
        if num_requests > 0:
            scheduled[:num_requests] = scheduled_full_cpu[:num_requests]

        # query_start_loc: prefix sums of scheduled tokens.
        qsl.fill(0)
        if num_requests > 0:
            np.cumsum(scheduled[:num_requests], out=qsl[1 : num_requests + 1])
            qsl[num_requests + 1 :] = qsl[num_requests]

        actual_num_tokens = int(qsl[num_requests]) if num_requests > 0 else 0
        if actual_num_tokens > int(num_tokens_static):
            raise ValueError(
                f"Scheduled {actual_num_tokens} tokens but `num_tokens_static`={num_tokens_static}; "
                "select a larger token bucket."
            )

        # Contiguous token batch: fill only the current bucket size.
        input_ids.fill(0)
        positions.fill(0)

        off = 0
        for req_idx in range(num_requests):
            n = int(scheduled[req_idx])
            if n <= 0:
                continue
            start = int(num_computed_tokens_cpu[req_idx])
            end = start + n
            if end > token_ids_cpu.shape[1]:
                raise ValueError(
                    f"Request {req_idx} scheduled [{start}:{end}] exceeds token_ids width {token_ids_cpu.shape[1]}."
                )
            positions[off : off + n] = start + self._arange_cpu[:n]
            input_ids[off : off + n] = token_ids_cpu[req_idx, start:end]
            off += n

        # seq_lens: computed tokens after this forward (start + scheduled).
        seq_lens.fill(0)
        if num_requests > 0:
            np.add(
                num_computed_tokens_cpu[:num_requests],
                scheduled[:num_requests],
                out=seq_lens[:num_requests],
                dtype=np.int32,
            )

        # logits_indices: last token index per request in the packed batch.
        logits_indices.fill(0)
        if num_requests > 0:
            logits_indices[:num_requests] = qsl[1 : num_requests + 1] - 1

        # pages_tables: copy active rows, pad inactive.
        pages_tables.fill(int(PAGE_TABLE_PADDING_VAL))
        rows_to_copy = min(num_requests, int(self._num_reqs_max_model_len), int(page_table_cpu.shape[0]))
        if rows_to_copy > 0:
            pages_tables[:rows_to_copy, :] = page_table_cpu[:rows_to_copy, : pages_tables.shape[1]]

        # request_distribution (v3) / slot_mapping (v2)
        request_distribution.fill(0)
        slot_mapping_cpu = None
        num_kv_update_cpu.fill(0)

        if self._use_request_distribution:
            if num_requests > 0:
                is_decode = (scheduled[:num_requests] == 1) & (num_computed_tokens_cpu[:num_requests] > 0)
                decode_count = int(np.sum(is_decode))
            else:
                decode_count = 0
            request_distribution[0] = decode_count
            request_distribution[1] = decode_count
            request_distribution[2] = num_requests

        if self._use_slot_mapping:
            slot_mapping_cpu, total_pages = self._compute_slot_mapping_v2(
                num_requests=num_requests,
                scheduled=scheduled,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                page_table_cpu=page_table_cpu,
                slot_mapping_out=slot_mapping_buf,
                copy_out=copy_slot_mapping,
            )
            num_kv_update_cpu[0] = int(total_pages)

        slot_mapping_placeholder = (
            self._async_slot_mapping_placeholder if use_async_buffers else self._slot_mapping_placeholder
        )

        host_payload = (
            input_ids,
            positions,
            qsl,
            seq_lens,
            logits_indices[:padded_num_reqs],
            pages_tables,
            scheduled[:padded_num_reqs],
            temperature_cpu[:padded_num_reqs],
            top_p_cpu[:padded_num_reqs],
            top_k_cpu[:padded_num_reqs],
            min_p_cpu[:padded_num_reqs],
            np.int32(num_requests),
            np.int32(padded_num_reqs),
            request_distribution,
            slot_mapping_cpu if slot_mapping_cpu is not None else slot_mapping_placeholder,
            num_kv_update_cpu,
            scheduled_full_cpu,
            active_mask_full_cpu,
        )

        return host_payload, padded_num_reqs

    def prepare_batch_metadata(
        self,
        *,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        # VLM prefill helpers (optional)
        mrope_position_ids_cpu: np.ndarray | None = None,
        prefill_embeds_cpu: np.ndarray | None = None,
        prefill_embeds_mask_cpu: np.ndarray | None = None,
        # DeepStack-style visual injection (optional)
        visual_pos_masks_cpu: np.ndarray | None = None,
        deepstack_visual_embeds_cpu: list[np.ndarray] | None = None,
        # Vision-language model data (optional)
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        pixel_values_videos: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
    ) -> tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Precompute batch metadata using CPU-first approach."""
        host_payload, _padded_num_reqs = self._build_host_payload(
            num_tokens_static=int(num_tokens_static),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            page_table_cpu=page_table_cpu,
            padded_num_reqs_in=int(padded_num_reqs_in),
            copy_slot_mapping=False,
        )

        (
            input_ids_buf,
            position_ids_buf,
            qsl,
            seq_lens,
            logits_indices,
            pt,
            scheduled_dev,
            temperature_dev,
            top_p_dev,
            top_k_dev,
            min_p_dev,
            num_requests_dev,
            padded_num_reqs_dev,
            req_dist_dev,
            slot_mapping_dev,
            num_kv_update_dev,
            scheduled_full_dev,
            active_mask_full_dev,
        ) = jax.device_put(host_payload, self._empty_sharding)

        mrope_position_ids_dev = None
        prefill_embeds_dev = None
        prefill_embeds_mask_dev = None
        if mrope_position_ids_cpu is not None:
            mrope_position_ids_dev = jax.device_put(mrope_position_ids_cpu, self._empty_sharding)
        if prefill_embeds_cpu is not None:
            prefill_embeds_dev = jax.device_put(prefill_embeds_cpu, self._empty_sharding)
        if prefill_embeds_mask_cpu is not None:
            prefill_embeds_mask_dev = jax.device_put(prefill_embeds_mask_cpu, self._empty_sharding)

        visual_pos_masks_dev = None
        deepstack_visual_embeds_dev = None
        if visual_pos_masks_cpu is not None:
            visual_pos_masks_dev = jax.device_put(visual_pos_masks_cpu, self._empty_sharding)
        if deepstack_visual_embeds_cpu is not None:
            deepstack_visual_embeds_dev = tuple(
                jax.device_put(arr, self._empty_sharding) for arr in deepstack_visual_embeds_cpu
            )

        pixel_values_dev = None
        image_grid_thw_dev = None
        pixel_values_videos_dev = None
        video_grid_thw_dev = None
        if pixel_values is not None:
            pixel_values_dev = jax.device_put(pixel_values, self._empty_sharding)
        if image_grid_thw is not None:
            image_grid_thw_dev = jax.device_put(image_grid_thw, self._empty_sharding)
        if pixel_values_videos is not None:
            pixel_values_videos_dev = jax.device_put(pixel_values_videos, self._empty_sharding)
        if video_grid_thw is not None:
            video_grid_thw_dev = jax.device_put(video_grid_thw, self._empty_sharding)

        metadata = BatchMetadata(
            scheduled=scheduled_dev,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            pages_tables=pt,
            padded_num_reqs=padded_num_reqs_dev,
            logits_indices=logits_indices,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            num_requests=num_requests_dev,
            temperature=temperature_dev,
            top_p=top_p_dev,
            top_k=top_k_dev,
            min_p=min_p_dev,
            positions=position_ids_buf,
            request_distribution=req_dist_dev,
            slot_mapping=slot_mapping_dev if self._use_slot_mapping else None,
            num_kv_update_slices=num_kv_update_dev if self._use_slot_mapping else None,
            pixel_values=pixel_values_dev,
            image_grid_thw=image_grid_thw_dev,
            pixel_values_videos=pixel_values_videos_dev,
            video_grid_thw=video_grid_thw_dev,
            mrope_position_ids=mrope_position_ids_dev,
            prefill_embeds=prefill_embeds_dev,
            prefill_embeds_mask=prefill_embeds_mask_dev,
            visual_pos_masks=visual_pos_masks_dev,
            deepstack_visual_embeds=deepstack_visual_embeds_dev,
        )

        return (
            metadata,
            input_ids_buf,
            position_ids_buf,
            scheduled_full_dev,
            active_mask_full_dev,
        )

    def start_async_prep(
        self,
        *,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
    ) -> None:
        """Start async device transfer for the next batch (double buffering)."""
        if self._pending_transfer is not None:
            raise RuntimeError("Async prep already in-flight; call get_async_prep_result() before starting another.")

        # Snapshot mutable CPU state into staging buffers (avoid per-call allocations).
        np.copyto(self._async_num_computed_tokens_cpu, num_computed_tokens_cpu)
        np.copyto(self._async_scheduled_full_cpu, scheduled_full_cpu)
        np.copyto(self._async_active_mask_full_cpu, active_mask_full_cpu)
        np.copyto(self._async_temperature_cpu, temperature_cpu)
        np.copyto(self._async_top_p_cpu, top_p_cpu)
        np.copyto(self._async_top_k_cpu, top_k_cpu)
        np.copyto(self._async_min_p_cpu, min_p_cpu)
        np.copyto(self._async_page_table_cpu, page_table_cpu)

        host_payload, padded_num_reqs = self._build_host_payload(
            num_tokens_static=int(num_tokens_static),
            scheduled_full_cpu=self._async_scheduled_full_cpu,
            active_mask_full_cpu=self._async_active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=self._async_num_computed_tokens_cpu,
            temperature_cpu=self._async_temperature_cpu,
            top_p_cpu=self._async_top_p_cpu,
            top_k_cpu=self._async_top_k_cpu,
            min_p_cpu=self._async_min_p_cpu,
            page_table_cpu=self._async_page_table_cpu,
            padded_num_reqs_in=int(padded_num_reqs_in),
            copy_slot_mapping=False,
            use_async_buffers=True,
        )

        self._pending_transfer = jax.device_put(host_payload, self._empty_sharding)
        self._pending_transfer_metadata = {"num_tokens_static": num_tokens_static, "padded_num_reqs": padded_num_reqs}

    def get_async_prep_result(
        self,
    ) -> (
        tuple[
            tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array],
            dict,
        ]
        | None
    ):
        """Get the result of a previously started async prep."""

        if self._pending_transfer is None:
            return None

        (
            input_ids_buf,
            position_ids_buf,
            qsl,
            seq_lens,
            logits_indices,
            pt,
            scheduled_dev,
            temperature_dev,
            top_p_dev,
            top_k_dev,
            min_p_dev,
            num_requests_dev,
            padded_num_reqs_dev,
            req_dist_dev,
            slot_mapping_dev,
            num_kv_update_dev,
            scheduled_full_dev,
            active_mask_full_dev,
        ) = self._pending_transfer

        metadata = BatchMetadata(
            scheduled=scheduled_dev,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            pages_tables=pt,
            padded_num_reqs=padded_num_reqs_dev,
            logits_indices=logits_indices,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            num_requests=num_requests_dev,
            temperature=temperature_dev,
            top_p=top_p_dev,
            top_k=top_k_dev,
            min_p=min_p_dev,
            positions=position_ids_buf,
            request_distribution=req_dist_dev,
            slot_mapping=slot_mapping_dev if self._use_slot_mapping else None,
            num_kv_update_slices=num_kv_update_dev if self._use_slot_mapping else None,
        )

        self._pending_transfer = None
        transfer_meta = tp.cast(dict, self._pending_transfer_metadata or {})
        self._pending_transfer_metadata = None

        return (
            (
                metadata,
                input_ids_buf,
                position_ids_buf,
                scheduled_full_dev,
                active_mask_full_dev,
            ),
            transfer_meta,
        )

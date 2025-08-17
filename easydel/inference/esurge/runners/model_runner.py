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
from __future__ import annotations

import bisect
import time
import typing
from typing import cast

import jax
from eformer import escale as es
from flax import nnx as nn
from jax import numpy as jnp

from easydel.layers.caching import PagesCache, PagesMetadata
from easydel.utils import capture_time, ejit, get_logger

from ...vsurge.core.functions import sample_top_p_efficient
from ..metrics import get_metrics_collector
from ..outputs import LogprobsTensors, ModelRunnerOutput
from ..page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from ..scheduler import SchedulerOutput
from .sequence_buffer import ModelRunnerSamplingMetadata, SequenceBuffer
from .states import CachedRequestState

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurge")


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings), f"Token length {x} exceeds maximum padding {paddings[-1]}"
    return paddings[index]


def _get_padded_num_kv_cache_update_slices(
    num_tokens: int,
    max_num_reqs: int,
    page_size: int,
    num_slices_per_kv_cache_update_page: int,
) -> int:
    """Calculate padded number of KV cache update slices to avoid recompilation."""
    return (
        (min(2 * max_num_reqs + num_tokens // page_size, num_tokens) + num_slices_per_kv_cache_update_page - 1)
        // num_slices_per_kv_cache_update_page
        * num_slices_per_kv_cache_update_page
    )


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    """Get padded number of requests with upper limit."""
    res = 8 if x <= 8 else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


class eSurgeRunner:
    """Handles model execution with efficient batching and KV cache management."""

    def __init__(
        self,
        model: EasyDeLBaseModule,
        hbm_utilization: float = 0.5,
        page_size: int = 128,
        max_model_len: int = 2**13,
        max_num_seqs: int = 8,
    ):
        """Initialize the model runner.

        Args:
            model: The EasyDeL model to run inference on
            hbm_utilization: Fraction of HBM to use for KV cache
            page_size: Size of each page in the paged attention mechanism
            max_model_len: Maximum model sequence length
            max_num_seqs: Maximum number of sequences to process in parallel
        """
        self.model = model
        self.metadata = model.create_paged_metadata(
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_model_length=max_model_len,
        )
        self.max_num_seqs = max_num_seqs
        self.max_num_reqs = max_num_seqs
        self.max_model_len = max_model_len

        self.kv_pages = model.init_pages(self.metadata)
        self.graphdef, self.graphstate, self.graphother = model.split_module()
        self.rng_key = jax.random.PRNGKey(0)

        self.page_size = int(self.metadata.page_size)
        self.max_pages_per_req = int(self.metadata.max_num_pages_per_req)

        self._setup_variables()
        self._setup_model()

    @property
    def mesh(self):
        """Get the device mesh."""
        return self.model.mesh

    @property
    def _empty_sharding(self):
        """Get empty sharding specification."""
        return jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

    @staticmethod
    def _get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int) -> list[int]:
        """Generate padding sizes for efficient compilation.

        Args:
            min_token_size: Minimum token size (must be power of 2)
            max_token_size: Maximum token size to cover
            padding_gap: Gap between padding sizes (0 for exponential growth)

        Returns:
            List of padding sizes
        """
        assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
        paddings = []
        num = min_token_size

        if padding_gap == 0:
            while num <= max_token_size:
                paddings.append(num)
                num *= 2
        else:
            while num <= padding_gap:
                paddings.append(num)
                num *= 2
            num //= 2
            while num < max_token_size:
                num += padding_gap
                paddings.append(num)

        return paddings

    def _setup_variables(self):
        """Initialize internal variables and preallocate reusable buffers."""
        self.num_reqs_max_model_len = min(self.metadata.get_max_num_seqs(), self.max_num_reqs)
        self.num_reqs_most_model_len = self.num_reqs_max_model_len
        self.num_tokens_paddings = self._get_token_paddings(
            min_token_size=16,
            max_token_size=self.max_model_len,
            padding_gap=0,
        )
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.requests: dict[str, CachedRequestState] = {}

        self.sequence_buffer = SequenceBuffer(
            self.max_num_reqs,
            self.max_model_len,
            self.max_num_tokens,
            self.model.config.vocab_size,
            [self.metadata.page_size],
        )

        # Common helpers
        self.arange = jnp.arange(self.max_num_tokens, dtype=jnp.int32)

        self.input_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32)
        self.position_ids_buf = jnp.zeros((self.max_num_tokens,), dtype=jnp.int32)
        self.query_start_loc_buf = jnp.zeros((self.max_num_reqs + 1,), dtype=jnp.int32)
        self.seq_lens_buf = jnp.zeros((self.max_num_reqs,), dtype=jnp.int32)

        self.pages_tables_buf = jnp.full(
            (self.num_reqs_max_model_len, self.max_pages_per_req),
            fill_value=PAGE_TABLE_PADDING_VAL,
            dtype=jnp.int32,
        )

        self.max_padded_slices = int(self.metadata.get_padded_num_slices(self.max_num_tokens, self.max_num_reqs))
        self.slot_mapping_buf = jnp.full(
            (3, self.max_padded_slices),
            fill_value=SLOT_MAPPING_PADDING_VAL,
            dtype=jnp.int32,
        )

    def _setup_model(self):
        """Set up JIT-compiled model execution functions."""

        # Fused forward + select logits to avoid an extra device round-trip
        @ejit(
            static_argnums=(0,),
            donate_argnames=["input_ids", "position_ids", "cache"],  # donate
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                self._empty_sharding,  # input_ids
                self._empty_sharding,  # position_ids
                es.extract_shardings(self.kv_pages, self.mesh),  # cache
                self._empty_sharding,  # cache_metadata
                self._empty_sharding,  # logits_indices
                self._empty_sharding,  # sampling_params
                self._empty_sharding,  # rng_key
            ),
            out_shardings=(self._empty_sharding, es.extract_shardings(self.kv_pages, self.mesh), self._empty_sharding),
        )
        def _fowrad_and_compute_tokens_fn(
            graphdef,
            graphstate,
            graphother,
            input_ids: jax.Array,
            position_ids: jax.Array,
            cache: PagesCache,
            cache_metadata: PagesMetadata,
            logits_indices: jax.Array,
            sampling_params: ModelRunnerSamplingMetadata,
            rng_key: jax.random.PRNGKey,
        ):
            model: EasyDeLBaseModule = nn.merge(graphdef, graphstate, graphother)
            with model.mesh:
                output = model(
                    input_ids=jnp.expand_dims(input_ids, 0),
                    position_ids=jnp.expand_dims(position_ids, 0),
                    past_key_values=cache,
                    cache_metadata=cache_metadata,
                    apply_lm_head=False,
                )
                logits = model.apply_lm_head(output.last_hidden_state.squeeze(0)[logits_indices])
                keys = jax.random.split(rng_key, logits.shape[0] + 1)

                samples = jax.vmap(
                    sample_top_p_efficient,
                    in_axes=(0, 0, 0, 0, None),
                    out_axes=0,
                )(
                    logits,
                    sampling_params.top_p.astype(logits.dtype),
                    sampling_params.temperature.astype(logits.dtype),
                    keys[1:],
                    64,
                )
                return samples.reshape(-1, 1), output.past_key_values, keys[0]

        def _fowrad_and_compute_tokens(
            input_ids: jax.Array,
            position_ids: jax.Array,
            cache_metadata: PagesMetadata,
            logits_indices: jax.Array,
            sampling_params: ModelRunnerSamplingMetadata,
        ) -> jax.Array:
            next_token, self.kv_pages, self.rng_key = _fowrad_and_compute_tokens_fn(
                self.graphdef,
                self.graphstate,
                self.graphother,
                input_ids,
                position_ids,
                self.kv_pages,
                cache_metadata,
                logits_indices,
                sampling_params,
                self.rng_key,
            )
            return next_token

        self.fowrad_and_compute_tokens = _fowrad_and_compute_tokens

    def _step_compile(self, num_tokens: int, num_reqs: int, num_pages: int) -> bool:
        """Compile a single step configuration."""

        actual_num_reqs = min(num_tokens, num_reqs)
        padded_num_slices = self.metadata.get_padded_num_slices(num_tokens, self.max_num_reqs)
        query_lens = [1] * num_reqs
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(actual_num_reqs, self.max_num_reqs)

        self.fowrad_and_compute_tokens(
            jnp.zeros((num_tokens,), dtype=jnp.int32),
            jnp.zeros(num_tokens, dtype=jnp.int32),
            PagesMetadata(
                pages_tables=jnp.full((num_reqs, num_pages), fill_value=PAGE_TABLE_PADDING_VAL, dtype=jnp.int32),
                context_lens=jnp.ones((num_reqs,), dtype=jnp.int32),
                query_start_loc=jnp.cumsum(jnp.array([0, *query_lens], dtype=jnp.int32), axis=0, dtype=jnp.int32),
                num_seqs=jnp.array([actual_num_reqs], dtype=jnp.int32),
                slot_mapping=jnp.full((3, padded_num_slices), fill_value=SLOT_MAPPING_PADDING_VAL, dtype=jnp.int32),
                num_kv_update_slices=jnp.array([padded_num_slices], dtype=jnp.int32),
                num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
                page_size=self.metadata.page_size,
            ),
            jnp.arange(padded_num_reqs, dtype=jnp.int32),
            ModelRunnerSamplingMetadata(
                top_p=jnp.ones((padded_num_reqs,), dtype=jnp.float32),
                temperature=jnp.ones((padded_num_reqs,), dtype=jnp.float32),
                min_p=jnp.zeros((padded_num_reqs,), dtype=jnp.float32),
                top_k=jnp.zeros((padded_num_reqs,), dtype=jnp.int32),
            ),
        )
        return True

    def compile(self):
        """Compile the model for all token padding sizes."""
        for num_tokens in self.num_tokens_paddings:
            logger.info(f"Compiling for {num_tokens} tokens")
            with capture_time() as took:
                self._step_compile(
                    num_tokens,
                    self.num_reqs_max_model_len,
                    self.metadata.max_num_pages_per_req,
                )
            logger.info(f"  Compilation took: {took():.2f}s")

    @staticmethod
    def _vectorized_slot_mapping(
        num_computed_tokens: jax.Array,  # [>=num_reqs]
        num_scheduled_tokens_per_req: jax.Array,  # [num_reqs]
        page_table_flat: jax.Array,  # [num_reqs * max_num_pages_per_req]
        num_reqs: int,
        page_size: int,
        max_num_pages_per_req: int,
    ) -> jax.Array:
        """Fully vectorized slot-mapping: returns [total_slices, 3]."""
        s = num_computed_tokens[:num_reqs]
        e = s + num_scheduled_tokens_per_req
        lps = s // page_size
        lpe = (e - 1) // page_size
        page_lens = lpe - lps + 1
        total_pages = int(jnp.sum(page_lens))
        if total_pages == 0:
            return jnp.zeros((0, 3), dtype=jnp.int32)

        req_ids = jnp.repeat(jnp.arange(num_reqs, dtype=jnp.int32), page_lens)
        cum_pages = jnp.cumsum(page_lens)
        starts = cum_pages - page_lens
        local_page_offsets = jnp.arange(total_pages, dtype=jnp.int32) - jnp.repeat(starts, page_lens)
        local_starts_rep = lps[req_ids]
        global_page_indices = req_ids * max_num_pages_per_req + local_starts_rep + local_page_offsets
        page_numbers = page_table_flat[global_page_indices]
        s_mod = s % page_size
        e_mod = ((e - 1) % page_size) + 1
        s_mod_rep = s_mod[req_ids]
        e_mod_rep = e_mod[req_ids]
        lens_rep = page_lens[req_ids]

        is_first = local_page_offsets == 0
        is_last = local_page_offsets == (lens_rep - 1)

        kv_local_st = jnp.where(is_first, s_mod_rep, jnp.int32(0))
        kv_local_en = jnp.where(is_last, e_mod_rep, jnp.int32(page_size))
        slice_lens = kv_local_en - kv_local_st
        kv_cache_start = kv_local_st + page_numbers * page_size
        new_kv_start = jnp.cumsum(jnp.pad(slice_lens[:-1], (1, 0)), dtype=jnp.int32)

        return jnp.stack([kv_cache_start, new_kv_start, slice_lens], axis=1)

    def _get_slot_mapping_metadata(self, num_reqs: int, num_scheduled_tokens_per_req: jax.Array) -> jax.Array:
        """Compute metadata mapping slices to KV pages. Returns [total_slices, 3]."""
        page_table_flat = self.sequence_buffer.page_table[0].get_array().flatten()
        return self._vectorized_slot_mapping(
            self.sequence_buffer.num_computed_tokens,
            num_scheduled_tokens_per_req,
            page_table_flat,
            num_reqs,
            self.metadata.page_size,
            self.metadata.max_num_pages_per_req,
        )

    @staticmethod
    def _prepare_inputs_optimized(
        num_scheduled_tokens_per_req: jax.Array,  # [num_reqs]
        num_computed_tokens: jax.Array,  # [>=num_reqs]
        token_ids_flat: jax.Array,  # [max_num_reqs * token_ids_shape_1]
        token_ids_shape_1: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
        """Vectorized input preparation (no Python ragged concatenation)."""
        num_reqs = int(num_scheduled_tokens_per_req.shape[0])
        total = int(jnp.sum(num_scheduled_tokens_per_req))
        if total == 0:
            # Should never happen in call-sites; guard for safety
            zero = jnp.zeros((0,), dtype=jnp.int32)
            return zero, zero, jnp.zeros((num_reqs + 1,), dtype=jnp.int32), jnp.zeros((num_reqs,), dtype=jnp.int32), 0

        # req_indices per token and token offsets within each request
        req_ids = jnp.repeat(jnp.arange(num_reqs, dtype=jnp.int32), num_scheduled_tokens_per_req)  # [total]
        cum = jnp.cumsum(num_scheduled_tokens_per_req)
        starts = cum - num_scheduled_tokens_per_req
        offs = jnp.arange(total, dtype=jnp.int32) - jnp.repeat(starts, num_scheduled_tokens_per_req)  # [total]

        positions = num_computed_tokens[req_ids] + offs
        token_indices = positions + req_ids * token_ids_shape_1
        input_ids = jnp.take(token_ids_flat, token_indices)

        # per-request metadata
        query_start_loc = jnp.zeros((num_reqs + 1,), dtype=jnp.int32).at[1:].set(cum)
        seq_lens = num_computed_tokens[:num_reqs] + num_scheduled_tokens_per_req

        return input_ids, positions, query_start_loc, seq_lens, total

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Update internal states based on scheduler output.

        Returns:
            True if there were unscheduled requests or new requests added
        """
        # Remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove unscheduled ones currently in buffer
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.sequence_buffer.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids

        for req_id in unscheduled_req_ids:
            logger.debug(f"Removing unscheduled request {req_id} from sequence buffer")
            req_index = self.sequence_buffer.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        # Add new requests
        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None, "Pooling not supported in TPU"
            req_id = new_req_data.req_id
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                generator=None,
                page_ids=new_req_data.page_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update cached reqs metadata
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                logger.warning(f"Cached request {req_id} not found in requests dict")
                continue

            num_computed_tokens = req_data.num_computed_tokens[i]
            new_page_ids = req_data.new_page_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]
            req_state.num_computed_tokens = num_computed_tokens

            if not resumed_from_preemption:
                for page_ids, new_ids in zip(req_state.page_ids, new_page_ids, strict=False):
                    page_ids.extend(new_ids)
            else:
                req_state.page_ids = new_page_ids

            req_index = self.sequence_buffer.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue
            self.sequence_buffer.num_computed_tokens = self.sequence_buffer.num_computed_tokens.at[req_index].set(
                num_computed_tokens
            )
            self.sequence_buffer.page_table.append_row(new_page_ids, req_index)

        # Add new or reinsert
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            req_index = removed_req_indices.pop() if removed_req_indices else None
            self.sequence_buffer.add_request(req_state, req_index)

        if removed_req_indices:
            self.sequence_buffer.condense(removed_req_indices)

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        start_index: int,
    ) -> tuple[PagesMetadata, jax.Array, int, int, int, int]:
        """Prepare inputs for model execution."""
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs_total = self.sequence_buffer.num_reqs
        assert num_reqs_total > 0
        assert start_index < num_reqs_total

        # We currently always use the max-model-len path to keep shapes stable.
        num_scheduled_tokens_per_req: list[int] = []
        for i in range(start_index, num_reqs_total):
            req_id = self.sequence_buffer.req_ids[i]
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue
            num_scheduled_tokens_per_req.append(num_tokens)

        # Clip to capacity
        if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
            num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[: self.num_reqs_max_model_len]
            end_index = start_index + self.num_reqs_max_model_len
        else:
            end_index = num_reqs_total

        num_scheduled_tokens_per_req = jnp.array(num_scheduled_tokens_per_req, dtype=jnp.int32)
        num_reqs = int(num_scheduled_tokens_per_req.shape[0])

        # Build packed inputs
        (
            input_ids,
            positions_np,
            query_start_loc_partial,
            seq_lens_partial,
            total_num_scheduled_tokens,
        ) = self._prepare_inputs_optimized(
            num_scheduled_tokens_per_req,
            self.sequence_buffer.num_computed_tokens,
            self.sequence_buffer.token_ids.flatten(),
            self.sequence_buffer.token_ids.shape[1],
        )

        padded_total = _get_padded_token_len(self.num_tokens_paddings, total_num_scheduled_tokens)

        # Fill preallocated buffers (avoid allocating new arrays)
        self.input_ids_buf = self.input_ids_buf.at[:padded_total].set(0)
        self.input_ids_buf = self.input_ids_buf.at[:total_num_scheduled_tokens].set(input_ids)
        input_ids_view = self.input_ids_buf[:padded_total]

        self.position_ids_buf = self.position_ids_buf.at[:padded_total].set(0)
        self.position_ids_buf = self.position_ids_buf.at[:total_num_scheduled_tokens].set(positions_np)
        position_ids_view = self.position_ids_buf[:padded_total]

        self.query_start_loc_buf = self.query_start_loc_buf.at[: num_reqs + 1].set(query_start_loc_partial)
        self.query_start_loc_buf = self.query_start_loc_buf.at[num_reqs + 1 :].set(1)  # doesn't matter; kept stable
        query_start_loc = self.query_start_loc_buf[: self.num_reqs_max_model_len + 1]

        self.seq_lens_buf = self.seq_lens_buf.at[:num_reqs].set(seq_lens_partial)
        seq_lens = self.seq_lens_buf[: self.num_reqs_max_model_len]

        # Pages table (pad rows)
        seq_page_table = self.sequence_buffer.page_table[0].get_array()
        self.pages_tables_buf = self.pages_tables_buf.at[:, :].set(PAGE_TABLE_PADDING_VAL)
        self.pages_tables_buf = self.pages_tables_buf.at[:num_reqs, :].set(seq_page_table[:num_reqs])
        pages_tables = self.pages_tables_buf  # [num_reqs_max_model_len, max_pages_per_req]

        # Slot mapping
        slot_mapping_metadata = self._get_slot_mapping_metadata(num_reqs, num_scheduled_tokens_per_req)
        num_kv_update_slices = int(slot_mapping_metadata.shape[0])

        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            padded_total,
            self.max_num_reqs,
            self.metadata.page_size,
            self.metadata.num_slices_per_kv_cache_update_page,
        )

        # Fill slot mapping buffer (transpose to [3, N]) and pad to compiled max
        self.slot_mapping_buf = self.slot_mapping_buf.at[:, :].set(SLOT_MAPPING_PADDING_VAL)
        if num_kv_update_slices > 0:
            self.slot_mapping_buf = self.slot_mapping_buf.at[:, :num_kv_update_slices].set(
                jnp.transpose(slot_mapping_metadata)
            )
        slot_mapping = self.slot_mapping_buf[:, :padded_num_slices]

        attn_metadata = PagesMetadata(
            pages_tables=pages_tables,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_reqs], dtype=jnp.int32),
            num_kv_update_slices=jnp.array([num_kv_update_slices], dtype=jnp.int32),
            num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
            page_size=self.metadata.page_size,
        )

        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(num_reqs, self.max_num_reqs)
        logits_indices = self.query_start_loc_buf[1 : padded_num_reqs + 1] - 1

        # Return views of inputs to avoid extra allocations
        self._current_input_ids_view = input_ids_view
        self._current_position_ids_view = position_ids_view

        return attn_metadata, logits_indices, padded_num_reqs, num_reqs, end_index, padded_total

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute the model on scheduled requests."""
        execution_start_time = time.time()

        self._update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                finished_sending=None,
                finished_recving=None,
                num_nans_in_logits=None,
            )

        start_index = 0
        combined_selected_tokens: list[jax.Array] = []

        while start_index < self.sequence_buffer.num_reqs:
            (
                cache_metadata,
                logits_indices,
                padded_num_reqs,
                num_reqs,
                end_index,
                _,
            ) = self._prepare_inputs(scheduler_output, start_index)

            selected_token_ids = self.fowrad_and_compute_tokens(
                self._current_input_ids_view,
                self._current_position_ids_view,
                cache_metadata,
                logits_indices,
                ModelRunnerSamplingMetadata.from_sequence_buffer(self.sequence_buffer, padded_num_reqs),
            )[:num_reqs]
            combined_selected_tokens.append(selected_token_ids)

            start_index = end_index

        selected_token_ids = jnp.concatenate(combined_selected_tokens, axis=0)

        return self._process_sampled_tokens(
            selected_token_ids,
            scheduler_output,
            execution_start_time,
        )

    @staticmethod
    def _update_token_buffers_optimized(
        token_ids: jax.Array,  # [max_reqs, max_len]
        num_tokens: jax.Array,  # [max_reqs]
        update_indices: jax.Array,  # [K]
        new_token_ids: jax.Array,  # [K] or [K, 1]
        seq_lens: jax.Array,  # [K]
    ) -> tuple[jax.Array, jax.Array]:
        """Vectorized token buffer updates."""
        new_token_ids = new_token_ids.squeeze(-1) if new_token_ids.ndim == 2 else new_token_ids
        token_ids = token_ids.at[(update_indices, seq_lens)].set(new_token_ids)
        num_tokens = num_tokens.at[update_indices].add(1)
        return token_ids, num_tokens

    def _process_sampled_tokens(
        self,
        selected_token_ids: jax.Array,  # [num_reqs, 1] typically
        scheduler_output: SchedulerOutput,
        execution_start_time: float,
    ) -> ModelRunnerOutput:
        """Process sampled tokens and update buffers."""
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices: list[int] = []
        num_reqs = self.sequence_buffer.num_reqs

        for i, req_id in enumerate(self.sequence_buffer.req_ids[:num_reqs]):
            assert req_id is not None
            req_state = self.requests[req_id]
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if scheduled_tokens == 0:
                continue
            seq_len = req_state.num_computed_tokens + scheduled_tokens

            # If scheduled token extends past current num_tokens, we accept it; else discard.
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                discard_sampled_tokens_req_indices.append(i)

        req_ids = cast(list[str], self.sequence_buffer.req_ids[:num_reqs])
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {req_id: None for req_id in req_ids}

        max_gen_len = int(selected_token_ids.shape[-1])

        if max_gen_len == 1:
            # Vectorized path (typical decoding)
            sampled_flat = selected_token_ids.squeeze(-1)  # [num_reqs]
            # Discard indices: just don't update buffer, and clear returned list entry
            update_rows: list[int] = []
            update_tokens: list[int] = []
            update_seq_lens: list[int] = []

            valid_sampled_token_ids = [[int(sampled_flat[i])] for i in range(num_reqs)]
            for idx in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[idx].clear()

            for i, req_state, seq_len in request_seq_lens:
                token_id = int(sampled_flat[i])
                update_rows.append(i)
                update_tokens.append(token_id)
                update_seq_lens.append(seq_len)
                req_state.output_token_ids.append(token_id)

            if update_rows:
                rows = jnp.array(update_rows, dtype=jnp.int32)
                toks = jnp.array(update_tokens, dtype=jnp.int32)
                lens = jnp.array(update_seq_lens, dtype=jnp.int32)
                self.sequence_buffer.token_ids, self.sequence_buffer.num_tokens = self._update_token_buffers_optimized(
                    self.sequence_buffer.token_ids, self.sequence_buffer.num_tokens, rows, toks, lens
                )

        else:
            # Rare ragged multi-token case (keep original logic)
            valid_mask = selected_token_ids != -1
            gen_lens = valid_mask.sum(axis=1).tolist()
            valid_sampled_token_ids = [seq.tolist() for seq in selected_token_ids[valid_mask].split(gen_lens)]
            self.sequence_buffer.num_tokens = self.sequence_buffer.num_tokens.at[:num_reqs].add(jnp.array(gen_lens))

            for i, req_state, seq_len in request_seq_lens:
                target_slice = slice(seq_len - gen_lens[i] + 1, seq_len + 1)
                self.sequence_buffer.token_ids = self.sequence_buffer.token_ids.at[i, target_slice].set(
                    jnp.array(valid_sampled_token_ids[i], dtype=jnp.int32)
                )
                req_state.output_token_ids.extend(valid_sampled_token_ids[i])

        # Log runner metrics
        execution_time = time.time() - execution_start_time
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_runner_metrics(
                execution_time=execution_time,
                batch_size=num_reqs,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
            )

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.sequence_buffer.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            finished_sending=None,
            finished_recving=None,
        )

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
            kvdtype: Data type for KV cache
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
        """Initialize internal variables and buffers."""
        self.num_reqs_max_model_len = min(self.metadata.get_max_num_seqs(), self.max_num_reqs)
        self.num_reqs_most_model_len = self.num_reqs_max_model_len

        self.num_tokens_paddings = self._get_token_paddings(
            min_token_size=16,
            max_token_size=self.max_model_len,
            padding_gap=0,
        )
        self.max_num_tokens = self.num_tokens_paddings[-1]

        self.requests = {}
        self.sequence_buffer = SequenceBuffer(
            self.max_num_reqs,
            self.max_model_len,
            self.max_num_tokens,
            self.model.config.vocab_size,
            [self.metadata.page_size],
        )

        self.arange = jnp.arange(self.max_num_tokens, dtype=jnp.int32)
        self.input_ids = jnp.zeros(self.max_num_tokens, dtype=jnp.int32)
        self.positions = jnp.zeros(self.max_num_tokens, dtype=jnp.int32)
        self.query_start_loc = jnp.zeros(self.max_num_tokens + 1, dtype=jnp.int32)
        self.seq_lens = jnp.zeros(self.max_num_tokens, dtype=jnp.int32)

    def _setup_model(self):
        """Set up JIT-compiled model execution functions."""

        @ejit(
            static_argnums=(0,),
            donate_argnums=(5,),
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                self._empty_sharding,
                self._empty_sharding,
                es.extract_shardings(self.kv_pages, self.mesh),
                self._empty_sharding,
            ),
            out_shardings=(self._empty_sharding, es.extract_shardings(self.kv_pages, self.mesh)),
        )
        def _execute_forward_fn(
            graphdef,
            graphstate,
            graphother,
            input_ids: jax.Array,
            position_ids: jax.Array,
            cache: PagesCache,
            cache_metadata: PagesMetadata,
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
            return output.last_hidden_state.squeeze(0), output.past_key_values

        @ejit(
            static_argnums=(0,),
            in_shardings=(
                es.extract_shardings(self.graphstate, self.mesh),
                es.extract_shardings(self.graphother, self.mesh),
                self._empty_sharding,
                self._empty_sharding,
            ),
            out_shardings=(self._empty_sharding),
        )
        def _apply_logits_and_select_fn(graphdef, graphstate, graphother, hidden_states, indices_do_sample):
            return nn.merge(graphdef, graphstate, graphother).apply_lm_head(hidden_states[indices_do_sample])

        @ejit(
            in_shardings=(self._empty_sharding, self._empty_sharding, self._empty_sharding),
            out_shardings=(self._empty_sharding, self._empty_sharding),
        )
        def _sample_from_logits_func(logits, sampling_params: ModelRunnerSamplingMetadata, rng_key):
            batch_size = logits.shape[0]
            keys = jax.random.split(rng_key, batch_size + 1)
            new_rng = keys[0]
            sample_keys = keys[1:]

            batched_sample = jax.vmap(sample_top_p_efficient, in_axes=(0, 0, 0, 0, None), out_axes=0)

            samples = batched_sample(
                logits,
                sampling_params.top_p[:batch_size],
                sampling_params.temperature[:batch_size],
                sample_keys,
                64,
            )
            return samples.reshape(-1, 1), new_rng

        def _apply_logits_and_select(hidden_states, indices_do_sample):
            return _apply_logits_and_select_fn(
                self.graphdef,
                self.graphstate,
                self.graphother,
                hidden_states,
                indices_do_sample,
            )

        def _execute_forward(
            input_ids: jax.Array,
            position_ids: jax.Array,
            cache_metadata: PagesMetadata,
        ) -> jax.Array:
            states, self.kv_pages = _execute_forward_fn(
                self.graphdef,
                self.graphstate,
                self.graphother,
                input_ids,
                position_ids,
                self.kv_pages,
                cache_metadata,
            )
            return states

        self.sample_from_logits_func = _sample_from_logits_func
        self.execute_forward = _execute_forward
        self.apply_logits_and_select = _apply_logits_and_select

    def _step_compile(self, num_tokens: int, num_reqs: int, num_pages: int) -> bool:
        """Compile a single step configuration."""
        input_ids = jnp.zeros((num_tokens,), dtype=jnp.int32)
        actual_num_reqs = min(num_tokens, num_reqs)
        position_ids = jnp.zeros(num_tokens, dtype=jnp.int32)

        padded_num_slices = self.metadata.get_padded_num_slices(num_tokens, self.max_num_reqs)
        num_kv_update_slices = jnp.array([padded_num_slices], dtype=jnp.int32)
        slot_mapping = jnp.full((3, padded_num_slices), fill_value=-1, dtype=jnp.int32)
        pages_tables = jnp.full((num_reqs, num_pages), fill_value=-1, dtype=jnp.int32)

        query_lens = [1] * num_reqs
        query_start_loc = jnp.cumsum(jnp.array([0, *query_lens], dtype=jnp.int32), axis=0, dtype=jnp.int32)
        context_lens = jnp.ones((num_reqs,), dtype=jnp.int32)
        num_seqs = jnp.array([actual_num_reqs], dtype=jnp.int32)

        cache_metadata = PagesMetadata(
            pages_tables=pages_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            slot_mapping=slot_mapping,
            num_kv_update_slices=num_kv_update_slices,
            num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
            page_size=self.metadata.page_size,
        )
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(actual_num_reqs, self.max_num_reqs)
        dummy_indices = jnp.arange(padded_num_reqs, dtype=jnp.int32)
        self.apply_logits_and_select(self.execute_forward(input_ids, position_ids, cache_metadata), dummy_indices)
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

    def _get_slot_mapping_metadata(self, num_reqs: int, num_scheduled_tokens_per_req: jax.Array) -> jax.Array:
        """Compute metadata for mapping slots to pages in KV cache.

        Returns:
            Array of shape (total_page_len, 3) containing:
            - kv_cache_start_index
            - new_kv_start_index
            - slice_len
        """
        slices_start = self.sequence_buffer.num_computed_tokens[:num_reqs]
        slices_end = slices_start + num_scheduled_tokens_per_req

        local_page_start_idx = slices_start // self.metadata.page_size
        local_page_end_idx = (slices_end - 1) // self.metadata.page_size

        no_repeat_req_indices = self.arange[:num_reqs]
        global_page_start_idx = no_repeat_req_indices * self.metadata.max_num_pages_per_req + local_page_start_idx

        page_lens = local_page_end_idx - local_page_start_idx + 1

        global_page_start_idx = jnp.repeat(global_page_start_idx, page_lens)
        slice_arange = jnp.concatenate([self.arange[:n] for n in page_lens])
        global_page_indices = global_page_start_idx + slice_arange

        page_table = self.sequence_buffer.page_table[0].get_array()
        page_numbers = page_table.flatten()[global_page_indices]

        total_page_len = jnp.sum(page_lens)
        slot_mapping_slices = jnp.repeat(
            jnp.array([[0, self.metadata.page_size]], dtype=jnp.int32),
            total_page_len,
            axis=0,
        )

        cu_page_lens = jnp.zeros(len(page_lens) + 1, dtype=jnp.int32)
        cu_page_lens = cu_page_lens.at[1:].set(jnp.cumsum(page_lens))

        for req_idx in range(num_reqs):
            slot_mapping_slices = slot_mapping_slices.at[cu_page_lens[req_idx], 0].set(
                slices_start[req_idx] % self.metadata.page_size
            )
            slot_mapping_slices = slot_mapping_slices.at[cu_page_lens[req_idx + 1] - 1, 1].set(
                (slices_end[req_idx] - 1) % self.metadata.page_size + 1
            )

        slice_lens = slot_mapping_slices[:, 1] - slot_mapping_slices[:, 0]
        cu_slices_lens = jnp.zeros(len(slice_lens) + 1, dtype=jnp.int32)
        cu_slices_lens = cu_slices_lens.at[1:].set(jnp.cumsum(slice_lens))

        kv_cache_start_indices = slot_mapping_slices[:, 0] + (page_numbers * self.metadata.page_size)
        new_kv_start_indices = cu_slices_lens[:-1]

        return jnp.stack([kv_cache_start_indices, new_kv_start_indices, slice_lens], axis=1)

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Update internal states based on scheduler output.

        Returns:
            True if there were unscheduled requests or new requests added
        """
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.sequence_buffer.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids

        for req_id in unscheduled_req_ids:
            logger.debug(f"Removing unscheduled request {req_id} from sequence buffer")
            req_index = self.sequence_buffer.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

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
    ) -> tuple[PagesMetadata, jax.Array, int, int, int]:
        """Prepare inputs for model execution."""
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs = self.sequence_buffer.num_reqs
        assert num_reqs > 0
        assert start_index < num_reqs

        use_max_model_len = True
        num_scheduled_tokens_per_req = []
        end_index = start_index

        for i in range(start_index, num_reqs):
            req_id = self.sequence_buffer.req_ids[i]
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            if not use_max_model_len and num_tokens > self.most_model_len:
                use_max_model_len = True
            num_scheduled_tokens_per_req.append(num_tokens)

        if use_max_model_len:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[: self.num_reqs_max_model_len]
                end_index = start_index + self.num_reqs_max_model_len
            else:
                end_index = num_reqs
        else:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_most_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[: self.num_reqs_most_model_len]
                end_index = start_index + self.num_reqs_most_model_len
            else:
                end_index = num_reqs

        num_scheduled_tokens_per_req = jnp.array(num_scheduled_tokens_per_req, dtype=jnp.int32)
        total_num_scheduled_tokens = sum(num_scheduled_tokens_per_req)
        num_reqs = len(num_scheduled_tokens_per_req)

        req_indices = jnp.repeat(self.arange[:num_reqs], num_scheduled_tokens_per_req)
        arange = jnp.concatenate([self.arange[:n] for n in num_scheduled_tokens_per_req])

        positions_np = self.positions[:total_num_scheduled_tokens]
        positions_np = self.sequence_buffer.num_computed_tokens[req_indices] + arange

        token_indices = positions_np + req_indices * self.sequence_buffer.token_ids.shape[1]
        input_ids = jnp.take(self.sequence_buffer.token_ids.flatten(), token_indices)

        padded_total_num_scheduled_tokens = _get_padded_token_len(self.num_tokens_paddings, total_num_scheduled_tokens)

        pad_token_id = -1
        padded_input_ids = jnp.full(padded_total_num_scheduled_tokens, pad_token_id, dtype=jnp.int32)
        padded_input_ids = padded_input_ids.at[:total_num_scheduled_tokens].set(input_ids)
        self.input_ids = padded_input_ids

        padded_position_ids = jnp.zeros(padded_total_num_scheduled_tokens, dtype=jnp.int32)
        padded_position_ids = padded_position_ids.at[:total_num_scheduled_tokens].set(positions_np)
        self.position_ids = padded_position_ids

        self.query_start_loc = self.query_start_loc.at[0].set(0)
        self.query_start_loc = self.query_start_loc.at[1 : num_reqs + 1].set(jnp.cumsum(num_scheduled_tokens_per_req))
        self.query_start_loc = self.query_start_loc.at[num_reqs + 1 :].set(1)

        self.seq_lens = self.seq_lens.at[:num_reqs].set(
            self.sequence_buffer.num_computed_tokens[:num_reqs] + num_scheduled_tokens_per_req
        )
        seq_page_table = self.sequence_buffer.page_table[0].get_array()
        if use_max_model_len:
            pages_tables = jnp.full(
                (self.num_reqs_max_model_len, self.metadata.max_num_pages_per_req),
                fill_value=-1,
                dtype=jnp.int32,
            )
            pages_tables = pages_tables.at[:num_reqs].set(seq_page_table[:num_reqs])
            query_start_loc = self.query_start_loc[: self.num_reqs_max_model_len + 1]
            seq_lens = self.seq_lens[: self.num_reqs_max_model_len]
        else:
            pages_tables = jnp.full(
                (self.num_reqs_most_model_len, self.num_pages_per_most_len_req),
                fill_value=-1,
                dtype=jnp.int32,
            )
            pages_tables = pages_tables.at[:num_reqs, : self.num_pages_per_most_len_req].set(
                seq_page_table[:num_reqs, : self.num_pages_per_most_len_req]
            )
            query_start_loc = self.query_start_loc[: self.num_reqs_most_model_len + 1]
            seq_lens = self.seq_lens[: self.num_reqs_most_model_len]

        slot_mapping_metadata = self._get_slot_mapping_metadata(num_reqs, num_scheduled_tokens_per_req)
        num_kv_update_slices = slot_mapping_metadata.shape[0]

        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            padded_total_num_scheduled_tokens,
            self.max_num_reqs,
            self.metadata.page_size,
            self.metadata.num_slices_per_kv_cache_update_page,
        )

        slot_mapping_metadata = jnp.transpose(
            jnp.pad(
                slot_mapping_metadata,
                [[0, padded_num_slices - len(slot_mapping_metadata)], [0, 0]],
                constant_values=-1,
            )
        )

        attn_metadata = PagesMetadata(
            pages_tables=pages_tables,
            slot_mapping=slot_mapping_metadata,
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_reqs], dtype=jnp.int32),
            num_kv_update_slices=jnp.array([num_kv_update_slices], dtype=jnp.int32),
            num_slices_per_kv_cache_update_page=self.metadata.num_slices_per_kv_cache_update_page,
            page_size=self.metadata.page_size,
        )

        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(num_reqs, self.max_num_reqs)
        logits_indices = self.query_start_loc[1 : padded_num_reqs + 1] - 1

        return attn_metadata, logits_indices, padded_num_reqs, num_reqs, end_index

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute the model on scheduled requests."""
        import time

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
            ) = self._prepare_inputs(scheduler_output, start_index)
            hidden_states = self.execute_forward(self.input_ids, self.position_ids, cache_metadata)
            selected_token_ids, self.rng_key = self.sample_from_logits_func(
                self.apply_logits_and_select(hidden_states, logits_indices),
                ModelRunnerSamplingMetadata.from_sequence_buffer(
                    sequence_buffer=self.sequence_buffer,
                    padded_num_reqs=padded_num_reqs,
                ),
                self.rng_key,
            )
            selected_token_ids = jax.device_get(selected_token_ids)[:num_reqs]
            combined_selected_tokens.append(selected_token_ids)

            start_index = end_index

        selected_token_ids = jnp.concatenate(combined_selected_tokens, axis=0)

        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        num_reqs = self.sequence_buffer.num_reqs

        for i, req_id in enumerate(self.sequence_buffer.req_ids[:num_reqs]):
            assert req_id is not None
            req_state = self.requests[req_id]
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if scheduled_tokens == 0:
                continue
            seq_len = req_state.num_computed_tokens + scheduled_tokens

            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Token discarding case - skip this request
                discard_sampled_tokens_req_indices.append(i)

        req_ids = cast(list[str], self.sequence_buffer.req_ids[:num_reqs])
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {req_id: None for req_id in req_ids}

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            for i, req_state, seq_len in request_seq_lens:
                token_id = valid_sampled_token_ids[i][0]
                self.sequence_buffer.token_ids = self.sequence_buffer.token_ids.at[i, seq_len].set(token_id)
                req_state.output_token_ids.append(token_id)
                self.sequence_buffer.num_tokens = self.sequence_buffer.num_tokens.at[i].add(1)
        else:
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

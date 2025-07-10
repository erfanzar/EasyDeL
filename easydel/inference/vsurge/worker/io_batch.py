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

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ...sampling_params import SamplingParams
from .block_table import MultiGroupBlockTable

LogprobsTensors = tuple[jnp.ndarray, jnp.ndarray]
LogprobsLists = list[jnp.ndarray]


@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    spec_token_ids: list[list[int]] | None
    logprobs: LogprobsLists | None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None


@dataclass
class SamplingMetadata:
    temperature: jnp.ndarray | None
    all_greedy: bool
    all_random: bool
    top_p: jnp.ndarray | None
    top_k: jnp.ndarray | None
    min_p: jnp.ndarray | None
    generators: dict[int, jax.random.PRNGKey]
    max_num_logprobs: int | None
    prompt_token_ids: jnp.ndarray | None
    frequency_penalties: jnp.ndarray
    presence_penalties: jnp.ndarray
    repetition_penalties: jnp.ndarray
    output_token_ids: list[list[int]]
    min_tokens: dict[int, tuple[int, set[int]]]
    no_penalties: bool
    logit_bias: list[dict[int, float] | None]
    allowed_token_ids_mask: jnp.ndarray | None
    bad_words_token_ids: dict[int, list[list[int]]]


def swap_dict_values(d: dict, key1: Any, key2: Any) -> None:
    val1 = d.pop(key1, None)
    val2 = d.pop(key2, None)
    if val1 is not None:
        d[key2] = val1
    if val2 is not None:
        d[key1] = val2


@dataclass
class CachedRequestState:
    """Represents the state of a single request, compatible with JAX."""

    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[dict[str, jax.Array]]
    mm_positions: list[dict[str, jax.Array]]
    sampling_params: SamplingParams
    generator: jax.random.PRNGKey
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_positions: jnp.ndarray | None = None
    mrope_position_delta: int | None = None

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]


class InputBatch:
    """
    A batch of requests for the model, implemented using JAX.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        block_sizes: list[int],
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        self.token_ids = jnp.zeros((max_num_reqs, max_model_len), dtype=jnp.int32)
        self.num_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_tokens_no_spec = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_prompt_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_computed_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)

        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            block_sizes=block_sizes,
        )

        # Sampling-related arrays (host-side)
        self.temperature = jnp.empty((max_num_reqs,), dtype=jnp.float32)
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = jnp.empty((max_num_reqs,), dtype=jnp.float32)
        self.top_p_reqs: set[str] = set()

        self.top_k = jnp.empty((max_num_reqs,), dtype=jnp.int32)
        self.top_k_reqs: set[str] = set()

        self.min_p = jnp.empty((max_num_reqs,), dtype=jnp.float32)
        self.min_p_reqs: set[str] = set()

        self.frequency_penalties = jnp.empty((max_num_reqs,), dtype=jnp.float32)
        self.frequency_penalties_reqs: set[str] = set()

        self.presence_penalties = jnp.empty((max_num_reqs,), dtype=jnp.float32)
        self.presence_penalties_reqs: set[str] = set()

        self.repetition_penalties = jnp.empty((max_num_reqs,), dtype=jnp.float32)
        self.repetition_penalties_reqs: set[str] = set()

        self.min_tokens: dict[int, tuple[int, set[int]]] = {}
        self.generators: dict[int, jax.random.PRNGKey] = {}
        self.num_logprobs: dict[str, int] = {}
        self.num_prompt_logprobs: dict[str, int] = {}
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}
        self.logit_bias: list[dict[int, float] | None] = [None] * max_num_reqs
        self.has_allowed_token_ids: set[str] = set()
        self.allowed_token_ids_mask: jnp.ndarray | None = None
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}
        self.req_output_token_ids: list[list[int] | None] = []

        self.sampling_metadata = self._make_sampling_metadata()

    @property
    def req_ids(self) -> list[str]:
        return cast(list[str], [rid for rid in self._req_ids if rid is not None])

    def add_request(self, request: CachedRequestState, req_index: int | None = None) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Update arrays using JAX's immutable update pattern
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens = self.num_prompt_tokens.at[req_index].set(num_prompt_tokens)

        prompt_ids = jnp.array(request.prompt_token_ids, dtype=jnp.int32)
        self.token_ids = self.token_ids.at[req_index, :num_prompt_tokens].set(prompt_ids)

        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        output_ids = jnp.array(request.output_token_ids, dtype=jnp.int32)
        self.token_ids = self.token_ids.at[req_index, start_idx:end_idx].set(output_ids)

        self.num_tokens = self.num_tokens.at[req_index].set(request.num_tokens)
        self.num_tokens_no_spec = self.num_tokens_no_spec.at[req_index].set(request.num_tokens)
        self.num_computed_tokens = self.num_computed_tokens.at[req_index].set(request.num_computed_tokens)
        self.block_table.add_row(request.block_ids, req_index)

        sampling_params = request.sampling_params

        self.temperature = self.temperature.at[req_index].set(sampling_params.temperature)
        if sampling_params.temperature != -1:
            self.random_reqs.add(req_id)

        self.top_p = self.top_p.at[req_index].set(sampling_params.top_p)
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)

        top_k = sampling_params.top_k if 0 < sampling_params.top_k < self.vocab_size else self.vocab_size
        self.top_k = self.top_k.at[req_index].set(top_k)
        if 0 < sampling_params.top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)

        self.min_p = self.min_p.at[req_index].set(sampling_params.min_p)
        if sampling_params.min_p > 1e-5:
            self.min_p_reqs.add(req_id)

        self.frequency_penalties = self.frequency_penalties.at[req_index].set(sampling_params.frequency_penalty)
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)

        self.presence_penalties = self.presence_penalties.at[req_index].set(sampling_params.presence_penalty)
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)

        self.repetition_penalties = self.repetition_penalties.at[req_index].set(sampling_params.repetition_penalty)
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens, sampling_params.all_stop_token_ids)

        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs
        if sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask is None:
                self.allowed_token_ids_mask = jnp.zeros((self.max_num_reqs, self.vocab_size), dtype=jnp.bool_)

            mask = jnp.ones(self.vocab_size, dtype=jnp.bool_)
            allowed_indices = jnp.array(list(sampling_params.allowed_token_ids))
            mask = mask.at[allowed_indices].set(False)
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[req_index].set(mask)

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids

    def remove_request(self, req_id: str) -> int | None:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # Discard from sets
        for req_set in [
            self.greedy_reqs,
            self.random_reqs,
            self.top_p_reqs,
            self.top_k_reqs,
            self.min_p_reqs,
            self.frequency_penalties_reqs,
            self.presence_penalties_reqs,
            self.repetition_penalties_reqs,
            self.has_allowed_token_ids,
        ]:
            req_set.discard(req_id)

        # Pop from dicts
        for req_dict in [self.min_tokens, self.generators, self.bad_words_token_ids]:
            req_dict.pop(req_index, None)

        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

        self.logit_bias[req_index] = None
        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[req_index].set(False)

        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        # Swap list/dict elements
        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )

        old_id_i1, old_id_i2 = self._req_ids[i2], self._req_ids[i1]  # Note: swapped already
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = i1, i2

        # Swap JAX array rows
        arrays_to_swap = [
            "num_tokens",
            "num_tokens_no_spec",
            "num_prompt_tokens",
            "num_computed_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalties",
            "presence_penalties",
            "repetition_penalties",
            "min_p",
            "token_ids",
        ]
        if self.allowed_token_ids_mask is not None:
            arrays_to_swap.append("allowed_token_ids_mask")

        for arr_name in arrays_to_swap:
            arr = getattr(self, arr_name)
            v1, v2 = arr[i1], arr[i2]
            arr = arr.at[i1].set(v2)
            arr = arr.at[i2].set(v1)
            setattr(self, arr_name, arr)

        self.logit_bias[i1], self.logit_bias[i2] = self.logit_bias[i2], self.logit_bias[i1]
        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)
        self.block_table.swap_row(i1, i2)

    def condense(self, empty_req_indices: list[int]) -> None:
        if not self.num_reqs:
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        last_req_index = self.num_reqs + len(empty_req_indices) - 1
        for empty_index in sorted(empty_req_indices):
            while last_req_index >= 0 and self._req_ids[last_req_index] is None:
                last_req_index -= 1
            if empty_index >= last_req_index:
                break

            req_id = self._req_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self.req_id_to_index[req_id] = empty_index
            self.req_output_token_ids[empty_index] = self.req_output_token_ids[last_req_index]

            arrays_to_move = [
                "num_tokens",
                "num_tokens_no_spec",
                "num_prompt_tokens",
                "num_computed_tokens",
                "temperature",
                "top_p",
                "top_k",
                "frequency_penalties",
                "presence_penalties",
                "repetition_penalties",
                "min_p",
                "token_ids",
            ]
            if self.allowed_token_ids_mask is not None:
                arrays_to_move.append("allowed_token_ids_mask")

            for arr_name in arrays_to_move:
                arr = getattr(self, arr_name)
                arr = arr.at[empty_index].set(arr[last_req_index])
                setattr(self, arr_name, arr)

            self.block_table.move_row(last_req_index, empty_index)
            self.logit_bias[empty_index] = self.logit_bias[last_req_index]

            for d in [self.generators, self.min_tokens, self.bad_words_token_ids]:
                if last_req_index in d:
                    d[empty_index] = d.pop(last_req_index)

            self._req_ids[last_req_index] = None
            self.req_output_token_ids[last_req_index] = None
            last_req_index -= 1

        del self._req_ids[self.num_reqs :]
        del self.req_output_token_ids[self.num_reqs :]

    def refresh_sampling_metadata(self):
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs

        temperature = None if self.all_greedy else self.temperature
        prompt_token_ids = None if self.no_penalties else self._make_prompt_token_ids_tensor()
        allowed_token_ids_mask = None if self.no_allowed_token_ids else self.allowed_token_ids_mask

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p,
            top_k=None if self.no_top_k else self.top_k,
            min_p=None if self.no_min_p else self.min_p,
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties,
            presence_penalties=self.presence_penalties,
            repetition_penalties=self.repetition_penalties,
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            min_tokens=self.min_tokens,
            no_penalties=self.no_penalties,
            logit_bias=self.logit_bias[:num_reqs],
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
        )

    def _make_prompt_token_ids_tensor(self) -> jnp.ndarray:
        num_reqs = self.num_reqs
        if num_reqs == 0:
            return jnp.empty((0, 0), dtype=jnp.int64)

        num_prompts_np = np.asarray(self.num_prompt_tokens[:num_reqs])
        max_prompt_len = num_prompts_np.max() if num_reqs > 0 else 0

        if max_prompt_len == 0:
            return jnp.empty((num_reqs, 0), dtype=jnp.int64)

        prompt_token_ids = self.token_ids[:num_reqs, :max_prompt_len]
        indices = jnp.arange(max_prompt_len)
        mask = indices[None, :] >= jnp.asarray(num_prompts_np)[:, None]
        padded_token_ids = jnp.where(mask, self.vocab_size, prompt_token_ids)

        return jax.device_put(padded_token_ids.astype(jnp.int64), self.device)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return not self.random_reqs

    @property
    def all_random(self) -> bool:
        return not self.greedy_reqs

    @property
    def no_top_p(self) -> bool:
        return not self.top_p_reqs

    @property
    def no_top_k(self) -> bool:
        return not self.top_k_reqs

    @property
    def no_min_p(self) -> bool:
        return not self.min_p_reqs

    @property
    def no_penalties(self) -> bool:
        return not (self.presence_penalties_reqs or self.frequency_penalties_reqs or self.repetition_penalties_reqs)

    @property
    def max_num_logprobs(self) -> int | None:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_prompt_logprob(self) -> bool:
        return not self.num_prompt_logprobs

    @property
    def no_allowed_token_ids(self) -> bool:
        return not self.has_allowed_token_ids

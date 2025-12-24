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

"""Sampler compilation/execution for eSurge.

This module isolates the token-sampling + minimal state update portion of an
eSurge step. It compiles and caches variants keyed by (num_tokens, padded_num_reqs).
"""

from __future__ import annotations

import typing as tp
from collections import OrderedDict

import jax
from jax import numpy as jnp

from easydel.utils import ejit

from ...core.sampler import sample_tokens as sample_tokens_fn
from ...core.sampling_metadata import SamplingMetadata
from ..execution_types import BatchMetadata, StepFunctionInputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


class SamplerExecutor:
    """Compile/cache and execute the sampler/update step."""

    def __init__(
        self,
        *,
        model: "EasyDeLBaseModule",
        max_model_len: int,
        empty_sharding: jax.sharding.Sharding,
        use_aot_forward: bool,
        cache_capacity: int = 64,
        maybe_implicit: tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]] | None = None,
    ) -> None:
        self.model = model
        self.max_model_len = int(max_model_len)
        self._empty_sharding = empty_sharding
        self.use_aot_forward = bool(use_aot_forward)
        self._cache_capacity = int(cache_capacity)

        self._maybe_implicit = maybe_implicit or (lambda f: f)

        self._sampling_fn = self._build_sampling_fn()
        self._cache: OrderedDict[tuple[int, int, str, str], tp.Any] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_put(self, key: tuple[int, int, str, str], value: tp.Any) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)

    def _cache_get(self, key: tuple[int, int, str, str]) -> tp.Any:
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def cache_keys(self) -> list[tuple[int, int, str, str]]:
        return list(self._cache.keys())

    def has(self, key: tuple[int, int, str, str]) -> bool:
        return key in self._cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int) -> tp.Any:
        mode = "aot" if self.use_aot_forward else "jit"
        key = (int(num_tokens), int(padded_num_reqs), "sampler", mode)
        return self._cache_get(key)

    def compile(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
        metadata: BatchMetadata,
    ) -> None:
        mode = "aot" if self.use_aot_forward else "jit"
        key = (int(num_tokens), int(padded_num_reqs), "sampler", mode)
        if key in self._cache:
            return

        vocab_size = int(self.model.config.get_text_config().vocab_size)
        dummy_logits = jnp.zeros(
            (int(padded_num_reqs), vocab_size),
            dtype=self.model.dtype,
            out_sharding=self._empty_sharding,
        )

        sampler_args = (
            metadata,
            inputs.req_num_tokens_full,
            inputs.active_mask_full,
            dummy_logits,
            inputs.rng_key,
        )

        if self.use_aot_forward:
            compiled = self._sampling_fn.lower(*sampler_args).compile()
            self._cache_put(key, compiled)
            return

        _ = self._sampling_fn(*sampler_args)
        self._cache_put(key, self._sampling_fn)

    def _build_sampling_fn(self) -> tp.Callable[..., tp.Any]:
        @ejit
        @self._maybe_implicit
        def _sampling_fn(
            metadata: BatchMetadata,
            req_num_tokens_full: jax.Array,
            active_mask_full: jax.Array,
            logits: jax.Array,
            rng_key: jax.Array,
        ):
            batch_size = logits.shape[0]
            i_reqs = jnp.arange(batch_size, dtype=jnp.int32)

            active_mask = (i_reqs < metadata.num_requests) & active_mask_full[:batch_size]

            temp = metadata.temperature.reshape(-1, 1).astype(logits.dtype)
            temp = jnp.where(active_mask[:, None], temp, jnp.ones_like(temp))
            topp = metadata.top_p.astype(logits.dtype)
            topk = metadata.top_k.astype(jnp.int32)
            minp = metadata.min_p.astype(logits.dtype)

            is_all_greedy = jnp.all(jnp.where(active_mask[:, None], temp <= 0.0, True))
            need_min_p_sampling = jnp.any((minp > 0.0) & active_mask)

            sampling_metadata = SamplingMetadata(
                temperatures=temp,
                top_ps=topp,
                top_ks=topk,
                min_ps=minp,
                sampling_seeds=None,
                is_all_greedy=is_all_greedy,
                need_min_p_sampling=need_min_p_sampling,
                do_penalties=False,
                linear_penalty=None,
            )

            sampled_flat = sample_tokens_fn(logits, sampling_metadata, rng_key)
            total_tokens = metadata.query_start_loc[-1]
            rng_key = jax.random.fold_in(rng_key, jnp.int32(total_tokens))

            scheduled_slice = metadata.scheduled[:batch_size]
            seq_lens_now = metadata.seq_lens[:batch_size]
            req_num_tokens_slice = req_num_tokens_full[:batch_size]
            active_mask_slice = active_mask_full[:batch_size]
            meets_len = seq_lens_now >= req_num_tokens_slice
            valid_mask = (i_reqs < metadata.num_requests) & active_mask_slice & (scheduled_slice > 0) & meets_len
            out_tokens = jnp.where(valid_mask, sampled_flat, -1)
            return rng_key, out_tokens, valid_mask

        return _sampling_fn

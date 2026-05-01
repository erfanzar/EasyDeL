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

"""Sampler compilation/execution for eSurge.

This module isolates the token-sampling portion of an eSurge step, separating
it from the model forward pass. The SamplerExecutor compiles and caches
sampler function variants keyed by (num_tokens, padded_num_reqs).

The sampler handles:
    - Temperature-based sampling (greedy when temperature <= 0)
    - Top-k filtering (restricts to k most likely tokens)
    - Top-p (nucleus) sampling (restricts by cumulative probability)
    - Min-p filtering (filters tokens below a threshold)
    - RNG state management (fold_in based on total tokens)
    - Valid mask computation (determines which requests get tokens)

Separation Benefits:
    - Allows model forward pass and sampling to be independently optimized
    - Enables async scheduling where sampling overlaps with next forward pass
    - Reduces compilation variants (sampler compilation is lighter than model)
    - Simplifies debugging and profiling of sampling behavior

Classes:
    SamplerExecutor: Manages compilation, caching, and execution of the
        token sampling function.

Example:
    >>> executor = SamplerExecutor(
    ...     model=model,
    ...     max_model_len=8192,
    ...     empty_sharding=sharding,
    ...     use_aot_forward=True,
    ... )
    >>>
    >>> # Compile for expected configurations
    >>> executor.compile(
    ...     num_tokens=256,
    ...     padded_num_reqs=16,
    ...     inputs=dummy_inputs,
    ...     metadata=batch_metadata,
    ... )
    >>>
    >>> # Execute sampling
    >>> sampler_fn = executor.get_compiled(num_tokens=256, padded_num_reqs=16)
    >>> rng_key, tokens, valid_mask = sampler_fn(
    ...     metadata,
    ...     req_num_tokens,
    ...     active_mask,
    ...     logits,
    ...     rng_key,
    ...     token_history,
    ... )
"""

from __future__ import annotations

import typing as tp
from collections import OrderedDict

import jax
from jax import numpy as jnp

from easydel.utils import ejit

from ...core.sampler import apply_history_penalties_from_counts, update_token_counts
from ...core.sampler import sample_tokens as sample_tokens_fn
from ...core.sampling_metadata import SamplingMetadata
from ..execution_types import StepFunctionInputs

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


class SamplerExecutor:
    """Compile, cache, and execute the token sampling step.

    The SamplerExecutor manages compilation and execution of the sampler
    function, which converts model logits into sampled tokens. It maintains
    compiled function variants for different input dimensions until the cache
    is explicitly cleared.

    The sampler function performs:
        1. Extract sampling parameters (temperature, top_k, top_p, min_p)
        2. Apply temperature scaling to logits
        3. Apply filtering (top_k, top_p, min_p) as configured
        4. Sample from the filtered distribution (or argmax for greedy)
        5. Compute validity mask for active requests
        6. Update RNG state for next iteration

    Attributes:
        model (EasyDeLBaseModule): The model instance (used for dtype/config).
        max_model_len (int): Maximum sequence length for buffer sizing.
        use_aot_forward (bool): Whether to use AOT compilation.

    Example:
        >>> executor = SamplerExecutor(
        ...     model=model,
        ...     max_model_len=8192,
        ...     empty_sharding=sharding,
        ...     use_aot_forward=True,
        ... )
        >>> sampler_fn = executor.get_compiled(num_tokens=256, padded_num_reqs=16)
        >>> rng_key, tokens, valid = sampler_fn(
        ...     batch_metadata,
        ...     req_num_tokens,
        ...     active_mask,
        ...     logits,
        ...     rng_key,
    ...     token_counts_full,
    ...     window_row_indices,
    ... )
    """

    def __init__(
        self,
        *,
        model: "EasyDeLBaseModule",
        max_model_len: int,
        empty_sharding: jax.sharding.Sharding,
        use_aot_forward: bool,
        cache_capacity: int = 64,
    ) -> None:
        """Initialize the SamplerExecutor.

        Args:
            model: The EasyDeL model instance. Used to access dtype and
                configuration (especially vocab_size).
            max_model_len: Maximum sequence length supported. Used for
                buffer sizing and validation.
            empty_sharding: Default JAX sharding for replicated arrays.
                Applied to scalar outputs and intermediate buffers.
            use_aot_forward: If True, use ahead-of-time compilation via
                lower().compile() for predictable latency. If False, use
                JIT compilation on first call.
            cache_capacity: Deprecated compatibility argument. Compiled
                variants are retained until ``clear_cache()`` is called.
        """
        self.model = model
        self.max_model_len = int(max_model_len)
        self._empty_sharding = empty_sharding
        self.use_aot_forward = bool(use_aot_forward)
        del cache_capacity

        self._sampling_fn = self._build_sampling_fn()
        self._cache: OrderedDict[tuple[int, int, str, str], tp.Any] = OrderedDict()

    def clear_cache(self) -> None:
        """Clear all cached compiled functions.

        Removes all entries from the compilation cache, forcing recompilation
        on subsequent calls. Useful when model weights change or when memory
        needs to be freed.
        """
        self._cache.clear()

    def _cache_put(self, key: tuple[int, int, str, str], value: tp.Any) -> None:
        """Add a compiled function to the cache.

        Args:
            key: Cache key tuple (num_tokens, padded_num_reqs, "sampler", mode).
            value: Compiled function to cache.
        """

        self._cache[key] = value
        self._cache.move_to_end(key)

    def _cache_get(self, key: tuple[int, int, str, str]) -> tp.Any:
        """Retrieve a compiled function from the cache.

        Args:
            key: Cache key tuple (num_tokens, padded_num_reqs, "sampler", mode).

        Returns:
            The cached compiled function.

        Raises:
            KeyError: If the key is not in the cache.

        Note:
            This method updates insertion order by moving the accessed entry
            to the end.
        """
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def cache_keys(self) -> list[tuple[int, int, str, str]]:
        """Get all keys currently in the cache.

        Returns:
            List of cache key tuples.
        """
        return list(self._cache.keys())

    def cache_key(self, *, padded_num_reqs: int) -> tuple[int, int, str, str]:
        """Return the sampler cache key.

        Sampler executable shapes are independent of the model token bucket:
        every token bucket passes the same ``[padded_num_reqs, vocab]`` logits
        and scalar ``total_tokens``.  Keying by ``num_tokens`` creates many
        redundant compiles during startup.
        """
        mode = "aot" if self.use_aot_forward else "jit"
        return (0, int(padded_num_reqs), "sampler", mode)

    def has(self, key: tuple[int, int, str, str]) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key tuple to check.

        Returns:
            True if the key is in the cache, False otherwise.
        """
        return key in self._cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int) -> tp.Any:
        """Retrieve a pre-compiled sampler function for given dimensions.

        Args:
            num_tokens: Number of tokens (for bucket selection).
            padded_num_reqs: Padded request count (for bucket selection).

        Returns:
            Compiled sampler function matching the specified dimensions.

        Raises:
            KeyError: If no compiled function exists for this configuration.
                Call compile() first to create the cached entry.
        """
        del num_tokens
        key = self.cache_key(padded_num_reqs=padded_num_reqs)
        return self._cache_get(key)

    def compile(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
        metadata,
    ) -> None:
        """Compile and cache a sampler function for specific dimensions.

        Creates a compiled sampler function for the given token count and
        request count, caching it for later retrieval via get_compiled().
        Skips compilation if an entry already exists for this configuration.

        Args:
            num_tokens: Number of tokens for this compilation variant.
            padded_num_reqs: Padded request count for this variant.
            inputs: Step function inputs (used for shape/dtype inference).
            metadata: Batch metadata (used for shape/dtype inference).

        Note:
            For AOT compilation, uses lower().compile() to produce a fully
            compiled XLA executable. For JIT compilation, performs a warmup
            call to trigger tracing and compilation.
        """
        key = self.cache_key(padded_num_reqs=padded_num_reqs)
        if key in self._cache:
            return

        vocab_size = int(self.model.config.get_text_config().vocab_size)
        dummy_logits = jnp.zeros(
            (int(padded_num_reqs), vocab_size),
            dtype=self.model.dtype,
            out_sharding=self._empty_sharding,
        )

        sampler_args = (
            jnp.ones((int(padded_num_reqs), 1), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jnp.ones((int(padded_num_reqs),), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jnp.zeros((int(padded_num_reqs),), dtype=jnp.int32, out_sharding=self._empty_sharding),
            jnp.zeros((int(padded_num_reqs),), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jnp.zeros((int(padded_num_reqs),), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jnp.zeros((int(padded_num_reqs),), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jnp.ones((int(padded_num_reqs),), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jax.device_put(jnp.arange(int(padded_num_reqs), dtype=jnp.int32), self._empty_sharding),
            jnp.ones((int(padded_num_reqs),), dtype=jnp.int32, out_sharding=self._empty_sharding),
            jnp.ones((int(padded_num_reqs),), dtype=jnp.int32, out_sharding=self._empty_sharding),
            jax.device_put(jnp.int32(int(padded_num_reqs)), self._empty_sharding),
            jax.device_put(jnp.int32(int(num_tokens)), self._empty_sharding),
            jnp.ones((int(padded_num_reqs),), dtype=inputs.req_num_tokens_full.dtype, out_sharding=self._empty_sharding),
            jnp.ones((int(padded_num_reqs),), dtype=jnp.bool_, out_sharding=self._empty_sharding),
            dummy_logits,
            jax.device_put(inputs.rng_key, self._empty_sharding),
            jnp.zeros(
                (int(inputs.req_num_tokens_full.shape[0]), vocab_size),
                dtype=jnp.uint32,
                out_sharding=self._empty_sharding,
            ),
            jnp.zeros(
                (int(padded_num_reqs),),
                dtype=jnp.int32,
                out_sharding=self._empty_sharding,
            ),
        )

        if self.use_aot_forward:
            compiled = self._sampling_fn.lower(*sampler_args).compile()  # pyright: ignore[reportFunctionMemberAccess]
            self._cache_put(key, compiled)
            return

        _ = self._sampling_fn(*sampler_args)
        self._cache_put(key, self._sampling_fn)

    def _build_sampling_fn(self) -> tp.Callable[..., tp.Any]:
        """Build the JIT-compiled sampling function.

        Constructs the inner sampling function that will be compiled and
        cached. The function handles temperature scaling, filtering (top_k,
        top_p, min_p), sampling, and validity mask computation.

        Returns:
            JIT-decorated sampling function ready for compilation.

        Note:
            The returned function signature is:
            (
                temperatures,
                top_ps,
                top_ks,
                min_ps,
                frequency_penalties,
                presence_penalties,
                repetition_penalties,
                sampling_seeds,
                scheduled,
                seq_lens,
                num_requests,
                total_tokens,
                req_num_tokens,
                active_mask,
                logits,
                rng_key,
                token_counts_full,
                window_row_indices,
            ) -> (updated_rng_key, sampled_tokens, valid_mask, updated_token_counts_full)
        """

        @ejit
        def _sampling_fn(
            temperatures: jax.Array,
            top_ps: jax.Array,
            top_ks: jax.Array,
            min_ps: jax.Array,
            frequency_penalties: jax.Array,
            presence_penalties: jax.Array,
            repetition_penalties: jax.Array,
            sampling_seeds: jax.Array,
            scheduled: jax.Array,
            seq_lens: jax.Array,
            num_requests: jax.Array,
            total_tokens: jax.Array,
            req_num_tokens: jax.Array,
            active_mask: jax.Array,
            logits: jax.Array,
            rng_key: jax.Array,
            token_counts_full: jax.Array,
            window_row_indices: jax.Array,
        ):
            batch_size = logits.shape[0]
            i_reqs = jnp.arange(batch_size, dtype=jnp.int32)

            active_mask = (i_reqs < num_requests) & active_mask[:batch_size]

            temp = temperatures[:batch_size].reshape(-1, 1).astype(logits.dtype)
            temp = jnp.where(active_mask[:, None], temp, jnp.ones_like(temp))
            topp = top_ps[:batch_size].astype(logits.dtype)
            topk = top_ks[:batch_size].astype(jnp.int32)
            minp = min_ps[:batch_size].astype(logits.dtype)

            is_all_greedy = jnp.all(jnp.where(active_mask[:, None], temp <= 0.0, True))
            need_min_p_sampling = jnp.any((minp > 0.0) & active_mask)
            need_history_penalties = jnp.any(
                active_mask
                & (
                    (presence_penalties[:batch_size] != 0.0)
                    | (frequency_penalties[:batch_size] != 0.0)
                    | (repetition_penalties[:batch_size] != 1.0)
                )
            )
            window_rows = window_row_indices[:batch_size].astype(jnp.int32)

            logits = jax.lax.cond(
                need_history_penalties,
                lambda legi: apply_history_penalties_from_counts(
                    legi,
                    token_counts=token_counts_full[window_rows],
                    active_mask=active_mask,
                    presence_penalties=presence_penalties[:batch_size],
                    frequency_penalties=frequency_penalties[:batch_size],
                    repetition_penalties=repetition_penalties[:batch_size],
                ),
                lambda legi: legi,
                logits,
            )

            sampling_metadata = SamplingMetadata(
                temperatures=temp,
                top_ps=topp,
                top_ks=topk,
                min_ps=minp,
                sampling_seeds=sampling_seeds[:batch_size],
                is_all_greedy=is_all_greedy,
                need_min_p_sampling=need_min_p_sampling,
                do_penalties=False,
                linear_penalty=None,
            )

            sampled_flat = sample_tokens_fn(logits, sampling_metadata, rng_key)
            rng_key = jax.random.fold_in(rng_key, jnp.int32(total_tokens))

            scheduled_slice = scheduled[:batch_size]
            seq_lens_now = seq_lens[:batch_size]
            req_num_tokens_slice = req_num_tokens[:batch_size]
            meets_len = seq_lens_now >= req_num_tokens_slice
            valid_mask = (i_reqs < num_requests) & active_mask & (scheduled_slice > 0) & meets_len
            out_tokens = jnp.where(valid_mask, sampled_flat, -1)
            token_counts_full = jax.lax.cond(
                need_history_penalties,
                lambda counts: update_token_counts(
                    counts,
                    row_indices=window_rows,
                    sampled_tokens=sampled_flat,
                    valid_mask=valid_mask,
                ),
                lambda counts: counts,
                token_counts_full,
            )
            return rng_key, out_tokens, valid_mask, token_counts_full

        return _sampling_fn

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
    ...     metadata, req_num_tokens, active_mask, logits, rng_key
    ... )
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
    """Compile, cache, and execute the token sampling step.

    The SamplerExecutor manages compilation and execution of the sampler
    function, which converts model logits into sampled tokens. It maintains
    an LRU cache of compiled function variants for different input dimensions.

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
        ...     batch_metadata, req_num_tokens, active_mask, logits, rng_key
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
            cache_capacity: Maximum number of compiled variants to cache.
                Defaults to 64. Uses LRU eviction when full.
        """
        self.model = model
        self.max_model_len = int(max_model_len)
        self._empty_sharding = empty_sharding
        self.use_aot_forward = bool(use_aot_forward)
        self._cache_capacity = int(cache_capacity)

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
        """Add a compiled function to the cache with LRU eviction.

        Args:
            key: Cache key tuple (num_tokens, padded_num_reqs, "sampler", mode).
            value: Compiled function to cache.

        Note:
            If the cache exceeds capacity, the least recently used entry
            is evicted.
        """
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)

    def _cache_get(self, key: tuple[int, int, str, str]) -> tp.Any:
        """Retrieve a compiled function from the cache.

        Args:
            key: Cache key tuple (num_tokens, padded_num_reqs, "sampler", mode).

        Returns:
            The cached compiled function.

        Raises:
            KeyError: If the key is not in the cache.

        Note:
            This method updates the LRU ordering by moving the accessed
            entry to the end.
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
        """Build the JIT-compiled sampling function.

        Constructs the inner sampling function that will be compiled and
        cached. The function handles temperature scaling, filtering (top_k,
        top_p, min_p), sampling, and validity mask computation.

        Returns:
            JIT-decorated sampling function ready for compilation.

        Note:
            The returned function signature is:
            (metadata, req_num_tokens_full, active_mask_full, logits, rng_key)
            -> (updated_rng_key, sampled_tokens, valid_mask)
        """

        @ejit
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

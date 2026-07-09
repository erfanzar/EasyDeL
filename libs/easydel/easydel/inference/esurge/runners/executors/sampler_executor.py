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
    SamplerRuntime: Owns the host-side sampler scratch buffers, penalty-count
        state, and the per-step sampler wrapper used by ExecutionManager.

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
from functools import partial

import jax
import numpy
from jax import numpy as jnp
from jax.experimental import multihost_utils
from numpy.typing import DTypeLike

from easydel.utils import ejit

from ...core.sampler import apply_history_penalties_from_counts, build_history_token_counts, update_token_counts
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
        self._greedy_sampling_fn = self._build_greedy_sampling_fn()
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

    def cache_key(self, *, padded_num_reqs: int, greedy: bool = False) -> tuple[int, int, str, str]:
        """Return the sampler cache key.

        Sampler executable shapes are independent of the model token bucket:
        every token bucket passes the same ``[padded_num_reqs, vocab]`` logits
        and scalar ``total_tokens``.  Keying by ``num_tokens`` creates many
        redundant compiles during startup.
        """
        mode = "aot" if self.use_aot_forward else "jit"
        variant = "sampler_greedy" if greedy else "sampler"
        return (0, int(padded_num_reqs), variant, mode)

    def has(self, key: tuple[int, int, str, str]) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key tuple to check.

        Returns:
            True if the key is in the cache, False otherwise.
        """
        return key in self._cache

    def get_compiled(self, *, num_tokens: int, padded_num_reqs: int, greedy: bool = False) -> tp.Any:
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
        key = self.cache_key(padded_num_reqs=padded_num_reqs, greedy=greedy)
        return self._cache_get(key)

    def compile(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        inputs: StepFunctionInputs,
        metadata,
        greedy: bool = False,
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
        key = self.cache_key(padded_num_reqs=padded_num_reqs, greedy=greedy)
        if key in self._cache:
            return

        vocab_size = int(self.model.config.get_text_config().vocab_size)
        dummy_logits = jnp.zeros(
            (int(padded_num_reqs), vocab_size),
            dtype=self.model.dtype,
            out_sharding=self._empty_sharding,
        )

        sampler_args = (
            jnp.ones((6, int(padded_num_reqs)), dtype=jnp.float32, out_sharding=self._empty_sharding),
            jnp.zeros((7, int(padded_num_reqs)), dtype=jnp.int32, out_sharding=self._empty_sharding),
            jnp.array([int(padded_num_reqs), int(num_tokens)], dtype=jnp.int32, out_sharding=self._empty_sharding),
            dummy_logits,
            jax.device_put(inputs.rng_key, self._empty_sharding),
            jnp.zeros(
                (int(inputs.req_num_tokens_full.shape[0]), vocab_size),
                dtype=jnp.uint32,
                out_sharding=self._empty_sharding,
            ),
        )

        sampling_fn = self._greedy_sampling_fn if greedy else self._sampling_fn
        if self.use_aot_forward:
            compiled = sampling_fn.lower(*sampler_args).compile()  # pyright: ignore[reportFunctionMemberAccess]
            self._cache_put(key, compiled)
            return

        _ = sampling_fn(*sampler_args)
        self._cache_put(key, sampling_fn)

    def _build_greedy_sampling_fn(self) -> tp.Callable[..., tp.Any]:
        """Build an argmax-only sampler for all-greedy/no-penalty windows."""

        @ejit
        def _greedy_sampling_fn(
            packed_f32: jax.Array,
            packed_i32: jax.Array,
            packed_misc_i32: jax.Array,
            logits: jax.Array,
            rng_key: jax.Array,
            token_counts_full: jax.Array,
        ):
            del packed_f32
            with jax.named_scope("easydel/esurge/greedy_sampler_step"):
                batch_size = logits.shape[0]
                i_reqs = jnp.arange(batch_size, dtype=jnp.int32)

                scheduled = packed_i32[1]
                seq_lens = packed_i32[2]
                active_mask = packed_i32[4].astype(jnp.bool_)
                req_num_tokens = packed_i32[6]
                num_requests = packed_misc_i32[0]
                total_tokens = packed_misc_i32[1]

                active_mask = (i_reqs < num_requests) & active_mask[:batch_size]
                sampled_flat = jnp.argmax(logits, axis=-1).astype(jnp.int32)

                scheduled_slice = scheduled[:batch_size]
                seq_lens_now = seq_lens[:batch_size]
                req_num_tokens_slice = req_num_tokens[:batch_size]
                emits_next_token = seq_lens_now >= req_num_tokens_slice
                valid_mask = (i_reqs < num_requests) & active_mask & (scheduled_slice > 0) & emits_next_token
                out_tokens = jnp.where(valid_mask, sampled_flat, -1)
                rng_key = jax.random.fold_in(rng_key, jnp.int32(total_tokens))
                return rng_key, out_tokens, valid_mask, token_counts_full

        return _greedy_sampling_fn

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
                packed_f32,
                packed_i32,
                packed_misc_i32,
                logits,
                rng_key,
                token_counts_full,
            ) -> (updated_rng_key, sampled_tokens, valid_mask, updated_token_counts_full)
        """

        @ejit
        def _sampling_fn(
            packed_f32: jax.Array,
            packed_i32: jax.Array,
            packed_misc_i32: jax.Array,
            logits: jax.Array,
            rng_key: jax.Array,
            token_counts_full: jax.Array,
        ):
            """Compiled sampler core consuming packed sampler-side metadata.

            Unpacks the packed float (temperature / top-p / min-p /
            frequency / presence / repetition) and int (sampling seeds /
            scheduled / seq lens / window row indices / active mask /
            top-k / req_num_tokens) metadata rows, applies repetition /
            presence / frequency penalties from the live token-count buffer,
            runs ``sample_tokens_fn`` over the configured sampling
            distribution, and computes the per-row ``valid_mask`` (request
            actually emits a token this step). The token-count buffer is
            updated for sampled tokens, and the rng key is folded with the
            window's ``total_tokens`` so step-level randomness stays
            deterministic and unique per window.
            """
            with jax.named_scope("easydel/esurge/sampler_step"):
                batch_size = logits.shape[0]
                i_reqs = jnp.arange(batch_size, dtype=jnp.int32)

                with jax.named_scope("easydel/esurge/sampler_step/unpack_metadata"):
                    temperatures = packed_f32[0]
                    top_ps = packed_f32[1]
                    min_ps = packed_f32[2]
                    frequency_penalties = packed_f32[3]
                    presence_penalties = packed_f32[4]
                    repetition_penalties = packed_f32[5]
                    sampling_seeds = packed_i32[0]
                    scheduled = packed_i32[1]
                    seq_lens = packed_i32[2]
                    window_row_indices = packed_i32[3]
                    active_mask = packed_i32[4].astype(jnp.bool_)
                    top_ks = packed_i32[5]
                    req_num_tokens = packed_i32[6]
                    num_requests = packed_misc_i32[0]
                    total_tokens = packed_misc_i32[1]
                    active_mask = (i_reqs < num_requests) & active_mask[:batch_size]

                    temp = temperatures[:batch_size].reshape(-1, 1).astype(logits.dtype)
                    temp = jnp.where(active_mask[:, None], temp, jnp.ones_like(temp))
                    topp = top_ps[:batch_size].astype(logits.dtype)
                    topk = top_ks[:batch_size].astype(jnp.int32)
                    minp = min_ps[:batch_size].astype(logits.dtype)

                with jax.named_scope("easydel/esurge/sampler_step/penalty_flags"):
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

                with jax.named_scope("easydel/esurge/sampler_step/apply_history_penalties"):
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

                with jax.named_scope("easydel/esurge/sampler_step/sample_tokens"):
                    sampled_flat = sample_tokens_fn(logits, sampling_metadata, rng_key)
                    rng_key = jax.random.fold_in(rng_key, jnp.int32(total_tokens))

                with jax.named_scope("easydel/esurge/sampler_step/valid_mask"):
                    scheduled_slice = scheduled[:batch_size]
                    seq_lens_now = seq_lens[:batch_size]
                    req_num_tokens_slice = req_num_tokens[:batch_size]
                    emits_next_token = seq_lens_now >= req_num_tokens_slice
                    valid_mask = (i_reqs < num_requests) & active_mask & (scheduled_slice > 0) & emits_next_token
                    out_tokens = jnp.where(valid_mask, sampled_flat, -1)
                with jax.named_scope("easydel/esurge/sampler_step/update_token_counts"):
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


@partial(jax.jit, static_argnames=("padded_num_reqs",))
def _greedy_argmax_tokens(
    logits: jax.Array,
    valid_mask: jax.Array,
    rng_key: jax.Array,
    total_tokens: jax.Array,
    *,
    padded_num_reqs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return greedy sampled tokens for prefix-layout decode windows."""
    with jax.named_scope("easydel/esurge/greedy_argmax_tokens"):
        tokens = jnp.argmax(logits[:padded_num_reqs], axis=-1).astype(jnp.int32)
        valid_mask = valid_mask[:padded_num_reqs].astype(jnp.bool_)
        tokens = jnp.where(valid_mask, tokens, jnp.full_like(tokens, -1))
        rng_key = jax.random.fold_in(rng_key, jnp.asarray(total_tokens, dtype=jnp.int32))
        return rng_key, tokens, valid_mask


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int) -> int:
    """Calculate padded request count for compilation efficiency.

    Pads the number of requests to powers of 2 (up to min_input_pad) or the nearest
    power of 2 above min_input_pad. This reduces the number of unique compilations
    needed while maintaining good utilization.

    Args:
        x: Actual number of requests to pad.
        upper_limit: Maximum allowed requests, acts as a cap on the returned value.
        min_input_pad: Minimum padding value to use when x is small.

    Returns:
        Padded request count, capped at upper_limit.

    Examples:
        >>> _get_padded_num_reqs_with_upper_limit(3, 32, 8)   # Returns 8
        >>> _get_padded_num_reqs_with_upper_limit(10, 32, 8)  # Returns 16
        >>> _get_padded_num_reqs_with_upper_limit(20, 16, 8)  # Returns 16

    Note:
        This function helps reduce JAX compilation overhead by bucketing
        request counts into a smaller set of sizes.
    """
    res = min_input_pad if x <= min_input_pad else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _device_put_tree_uniform(tree, sharding):
    """Place every array leaf of ``tree`` on devices with the same sharding."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding) if hasattr(x, "dtype") else x,
        tree,
    )


def _cpu_vector_size(values: numpy.ndarray) -> int:
    """Return the first-dimension size of a CPU scheduler vector."""
    return int(numpy.asarray(values).shape[0])


def _padded_cpu_window(
    values: numpy.ndarray,
    *,
    padded_num_reqs: int,
    dtype: DTypeLike,
    fill_value: int | float | bool,
) -> numpy.ndarray:
    """Return a one-dimensional CPU request vector padded to ``padded_num_reqs``.

    Scheduler payloads are sometimes active-row sized while the compiled
    model bucket is padded. Host-side predicates and sampler compaction must
    do their NumPy math on arrays with the same length, but padded rows must
    remain neutral.
    """
    pnr = int(padded_num_reqs)
    array = numpy.asarray(values, dtype=dtype).reshape(-1)
    if array.shape[0] >= pnr:
        return array[:pnr]
    out = numpy.full((pnr,), fill_value, dtype=dtype)
    out[: array.shape[0]] = array
    return out


class SamplerRuntime:
    """Host-side sampler runtime state and step wrapper for the execution manager.

    Owns the CPU scratch buffers used to compact the sampler window, the
    incremental penalty-count device state and its rebuild jit, the
    compact-to-window scatter jit, and the per-step ``sample_tokens`` wrapper
    that drives a compiled :class:`SamplerExecutor` variant.

    All state here was extracted verbatim from ``ExecutionManager`` (phase
    11.1 of the eSurge modularization). Threading contract is unchanged: only
    the scheduler thread calls into this object, and the numpy scratch buffers
    are mutated in place (never rebound) so back-references stay valid.
    """

    _cpu_vector_size = staticmethod(_cpu_vector_size)
    _padded_cpu_window = staticmethod(_padded_cpu_window)

    def __init__(
        self,
        *,
        sampler_executor: SamplerExecutor,
        vocab_size: int,
        max_num_reqs: int,
        min_input_pad: int = 1,
        sampler_sharding: jax.sharding.Sharding,
        empty_sharding: jax.sharding.Sharding,
    ) -> None:
        """Initialize sampler runtime buffers and jit closures.

        Args:
            sampler_executor: Compiled-variant cache used by ``sample_tokens``.
            vocab_size: Text-config vocabulary size for penalty-count shapes.
            max_num_reqs: Maximum concurrent requests (buffer row count).
            min_input_pad: Minimum sampler request-bucket padding.
            sampler_sharding: Sharding for sampler-side device arrays.
            empty_sharding: Replicated sharding fallback.
        """
        self._sampler_executor = sampler_executor
        self._sampler_sharding = sampler_sharding
        self._empty_sharding = empty_sharding
        self.max_num_reqs = int(max_num_reqs)
        self._sampler_vocab_size = int(vocab_size)
        self._sampler_min_input_pad = int(min_input_pad)

        self._sampler_zero_token_counts = jnp.zeros(
            (self.max_num_reqs, self._sampler_vocab_size),
            dtype=jnp.uint32,
            out_sharding=self._sampler_sharding,
        )
        self._sampler_zero_window_row_indices = jnp.zeros(
            (self.max_num_reqs,),
            dtype=jnp.int32,
            out_sharding=self._sampler_sharding,
        )
        self._sampler_token_counts = self._sampler_zero_token_counts
        self._sampler_penalty_state_dirty = True
        self._sampler_penalty_state_ready = False
        self._sampler_penalty_rebuild_token_ids_cpu: numpy.ndarray | None = None
        self._sampler_penalty_rebuild_seq_lens_cpu: numpy.ndarray | None = None
        self._sampler_gather_positions_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_sampling_seeds_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_scatter_positions_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_window_row_indices_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_scheduled_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_seq_lens_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_active_mask_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.bool_)
        self._sampler_temperature_cpu = numpy.ones((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_top_p_cpu = numpy.ones((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_top_k_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.int32)
        self._sampler_min_p_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_frequency_penalties_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_presence_penalties_cpu = numpy.zeros((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_repetition_penalties_cpu = numpy.ones((self.max_num_reqs,), dtype=numpy.float32)
        self._sampler_packed_i32_cpu_by_reqs: dict[int, numpy.ndarray] = {}
        self._sampler_packed_f32_cpu_by_reqs: dict[int, numpy.ndarray] = {}
        self._sampler_packed_misc_i32_cpu = numpy.zeros((2,), dtype=numpy.int32)
        self._sampler_prefix_cpu_by_reqs: dict[int, numpy.ndarray] = {}

        @jax.jit
        def _rebuild_penalty_counts(token_history: jax.Array, seq_lens: jax.Array) -> jax.Array:
            """Rebuild per-request token-frequency counts for sampling penalties.

            Closes over ``self._sampler_vocab_size`` and runs the JIT-compiled
            :func:`build_history_token_counts` over a ``[max_num_reqs,
            max_model_len]`` token-history buffer, treating only rows with
            ``seq_lens > 0`` as active. Used to refresh the device-side counts
            after a state mutation (request adds/removes, sequence drops) so
            frequency / presence / repetition penalties stay correct.

            Args:
                token_history: Token-id buffer ``[max_num_reqs, max_model_len]``.
                seq_lens: Active token counts per request ``[max_num_reqs]``.

            Returns:
                Count tensor ``[max_num_reqs, vocab_size]`` with frequency
                of each token id seen in each request's history.
            """
            with jax.named_scope("easydel/esurge/sampler/rebuild_penalty_counts"):
                return build_history_token_counts(
                    token_history=token_history,
                    seq_lens=seq_lens.astype(jnp.int32),
                    active_mask=seq_lens > 0,
                    vocab_size=self._sampler_vocab_size,
                )

        self._rebuild_penalty_counts = _rebuild_penalty_counts

        @partial(jax.jit, static_argnames=("padded_num_reqs",))
        def _scatter_sampler_outputs(
            sampled_tokens: jax.Array,
            valid_mask: jax.Array,
            scatter_positions: jax.Array,
            padded_num_reqs: int,
        ) -> tuple[jax.Array, jax.Array]:
            """Scatter compact sampler outputs back to the padded window layout.

            The sampler runs on a compacted slice of the active window; this
            scatter inverts the gather by writing the compact ``sampled_tokens``
            and ``valid_mask`` into a full ``[padded_num_reqs]`` view at
            ``scatter_positions``. Out-of-range or padded positions stay
            ``-1`` / ``False`` to mark them invalid.

            Args:
                sampled_tokens: Compact sampled token ids ``[num_compact]``.
                valid_mask: Compact validity flags ``[num_compact]``.
                scatter_positions: Window-row indices for each compact entry.
                padded_num_reqs: Window padding size used for the output.

            Returns:
                Tuple ``(full_tokens, full_valid)`` of shape
                ``[padded_num_reqs]`` covering every window row.
            """
            with jax.named_scope("easydel/esurge/sampler/scatter_outputs"):
                spill = int(scatter_positions.shape[0])
                full_tokens = jnp.full((int(padded_num_reqs) + spill,), -1, dtype=sampled_tokens.dtype)
                full_valid = jnp.zeros((int(padded_num_reqs) + spill,), dtype=jnp.bool_)
                full_tokens = full_tokens.at[scatter_positions].set(jnp.where(valid_mask, sampled_tokens, -1))
                full_valid = full_valid.at[scatter_positions].set(valid_mask)
                return full_tokens[:padded_num_reqs], full_valid[:padded_num_reqs]

        self._scatter_sampler_outputs = _scatter_sampler_outputs

    def invalidate_sampler_penalty_state(
        self,
        token_ids_cpu: numpy.ndarray | None = None,
        seq_lens_cpu: numpy.ndarray | None = None,
    ) -> None:
        """Mark sampler penalty counters dirty after host-side row reorderings.

        The frequency / presence / repetition penalty kernels keep an
        incremental device-side count of how often each token has appeared
        per request. That counter must be rebuilt whenever the runner
        permutes rows in :class:`SequenceBuffer` (e.g. swap_rows /
        condense). This method records the new ground-truth host views
        and flips the dirty flag so the next sampler call goes through
        :meth:`_ensure_sampler_penalty_state`.

        Args:
            token_ids_cpu: ``(max_num_reqs, max_model_len)`` host array of
                token ids per slot — the source of truth used during the
                next rebuild. ``None`` keeps the previously-recorded view.
            seq_lens_cpu: ``(max_num_reqs,)`` host array of effective
                sequence lengths. ``None`` keeps the previous view.
        """
        if token_ids_cpu is not None:
            self._sampler_penalty_rebuild_token_ids_cpu = token_ids_cpu
        if seq_lens_cpu is not None:
            self._sampler_penalty_rebuild_seq_lens_cpu = seq_lens_cpu
        self._sampler_penalty_state_dirty = True
        self._sampler_penalty_state_ready = False

    def _ensure_sampler_penalty_state(self) -> None:
        """Recompute device-side per-token frequency counts when dirty.

        No-op when the cached counters are clean. On dirty, takes the
        host views captured by :meth:`invalidate_sampler_penalty_state`,
        broadcasts them across ranks under multi-host JAX, transfers
        them onto the sampler sharding, and runs ``_rebuild_penalty_counts``
        to rebuild the exact device-side per-token counts. Toggles the
        dirty / ready flags on success.

        Raises:
            RuntimeError: If a rebuild is requested without a previously
                recorded host source — indicates a missing
                :meth:`invalidate_sampler_penalty_state` call earlier in the
                step.
        """
        if self._sampler_penalty_state_ready and not self._sampler_penalty_state_dirty:
            return
        if self._sampler_penalty_rebuild_token_ids_cpu is None or self._sampler_penalty_rebuild_seq_lens_cpu is None:
            raise RuntimeError("Sampler penalty state rebuild requested without a full-sequence source.")

        _token_ids_cpu = self._sampler_penalty_rebuild_token_ids_cpu
        _seq_lens_cpu = self._sampler_penalty_rebuild_seq_lens_cpu
        if jax.process_count() > 1:
            _token_ids_cpu, _seq_lens_cpu = multihost_utils.broadcast_one_to_all((_token_ids_cpu, _seq_lens_cpu))
        token_history = jax.device_put(_token_ids_cpu, self._sampler_sharding)
        seq_lens = jax.device_put(_seq_lens_cpu, self._sampler_sharding)
        self._sampler_token_counts = self._rebuild_penalty_counts(token_history, seq_lens)
        self._sampler_penalty_state_dirty = False
        self._sampler_penalty_state_ready = True

    def _prepare_compact_sampler_window(
        self,
        *,
        padded_num_reqs: int,
        scheduled_full_cpu: numpy.ndarray,
        active_mask_full_cpu: numpy.ndarray,
        window_row_indices_cpu: numpy.ndarray,
        num_computed_tokens_cpu: numpy.ndarray,
        temperature_cpu: numpy.ndarray,
        top_p_cpu: numpy.ndarray,
        top_k_cpu: numpy.ndarray,
        min_p_cpu: numpy.ndarray,
        frequency_penalties_cpu: numpy.ndarray,
        presence_penalties_cpu: numpy.ndarray,
        repetition_penalties_cpu: numpy.ndarray,
    ) -> tuple[int, int, int]:
        """Compact the sampler workload to rows that can actually emit tokens.

        The model forward may still need a wider request window to preserve
        sparse row layout, especially for async scheduling. The sampler only
        needs rows that are both active and scheduled, so compacting here
        avoids burning top-k/top-p work on zero-token rows.
        """
        padded_num_reqs = int(padded_num_reqs)
        scheduled_window = self._padded_cpu_window(
            scheduled_full_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.int32,
            fill_value=0,
        )
        active_window = self._padded_cpu_window(
            active_mask_full_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.bool_,
            fill_value=False,
        )
        available_count = min(
            padded_num_reqs,
            self._cpu_vector_size(scheduled_full_cpu),
            self._cpu_vector_size(active_mask_full_cpu),
            self._cpu_vector_size(window_row_indices_cpu),
            self._cpu_vector_size(num_computed_tokens_cpu),
        )
        if available_count < padded_num_reqs:
            active_window[available_count:] = False
        sample_positions = numpy.flatnonzero(active_window & (scheduled_window > 0))
        sample_count = int(sample_positions.size)
        if sample_count <= 0:
            raise RuntimeError("Sampler compaction found no scheduled active rows for a non-empty execution window.")

        sampler_padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            sample_count,
            upper_limit=padded_num_reqs,
            min_input_pad=int(getattr(self, "_sampler_min_input_pad", 1)),
        )

        gather_positions = self._sampler_gather_positions_cpu
        sampling_seeds = self._sampler_sampling_seeds_cpu
        scatter_positions = self._sampler_scatter_positions_cpu
        window_rows = self._sampler_window_row_indices_cpu
        scheduled_out = self._sampler_scheduled_cpu
        seq_lens_out = self._sampler_seq_lens_cpu
        active_out = self._sampler_active_mask_cpu
        temperature_out = self._sampler_temperature_cpu
        top_p_out = self._sampler_top_p_cpu
        top_k_out = self._sampler_top_k_cpu
        min_p_out = self._sampler_min_p_cpu
        frequency_out = self._sampler_frequency_penalties_cpu
        presence_out = self._sampler_presence_penalties_cpu
        repetition_out = self._sampler_repetition_penalties_cpu

        gather_positions[:sampler_padded_num_reqs] = 0
        window_rows[:sampler_padded_num_reqs] = 0
        scheduled_out[:sampler_padded_num_reqs] = 0
        seq_lens_out[:sampler_padded_num_reqs] = 0
        active_out[:sampler_padded_num_reqs] = False
        temperature_out[:sampler_padded_num_reqs] = 1.0
        top_p_out[:sampler_padded_num_reqs] = 1.0
        top_k_out[:sampler_padded_num_reqs] = 0
        min_p_out[:sampler_padded_num_reqs] = 0.0
        frequency_out[:sampler_padded_num_reqs] = 0.0
        presence_out[:sampler_padded_num_reqs] = 0.0
        repetition_out[:sampler_padded_num_reqs] = 1.0

        padding_range = numpy.arange(sampler_padded_num_reqs, dtype=numpy.int32)
        sampling_seeds[:sampler_padded_num_reqs] = padded_num_reqs + padding_range
        scatter_positions[:sampler_padded_num_reqs] = padded_num_reqs + padding_range

        gather_positions[:sample_count] = sample_positions
        sampling_seeds[:sample_count] = sample_positions
        scatter_positions[:sample_count] = sample_positions
        window_rows[:sample_count] = self._padded_cpu_window(
            window_row_indices_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.int32,
            fill_value=0,
        )[sample_positions]
        scheduled_out[:sample_count] = scheduled_window[sample_positions]
        computed_window = self._padded_cpu_window(
            num_computed_tokens_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.int32,
            fill_value=0,
        )
        seq_lens_out[:sample_count] = computed_window[sample_positions] + scheduled_window[sample_positions]
        active_out[:sample_count] = True
        temperature_out[:sample_count] = self._padded_cpu_window(
            temperature_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.float32,
            fill_value=1.0,
        )[sample_positions]
        top_p_out[:sample_count] = self._padded_cpu_window(
            top_p_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.float32,
            fill_value=1.0,
        )[sample_positions]
        top_k_out[:sample_count] = self._padded_cpu_window(
            top_k_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.int32,
            fill_value=0,
        )[sample_positions]
        min_p_out[:sample_count] = self._padded_cpu_window(
            min_p_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.float32,
            fill_value=0.0,
        )[sample_positions]
        frequency_out[:sample_count] = self._padded_cpu_window(
            frequency_penalties_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.float32,
            fill_value=0.0,
        )[sample_positions]
        presence_out[:sample_count] = self._padded_cpu_window(
            presence_penalties_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.float32,
            fill_value=0.0,
        )[sample_positions]
        repetition_out[:sample_count] = self._padded_cpu_window(
            repetition_penalties_cpu,
            padded_num_reqs=padded_num_reqs,
            dtype=numpy.float32,
            fill_value=1.0,
        )[sample_positions]
        total_tokens = int(scheduled_out[:sample_count].sum())
        return sample_count, sampler_padded_num_reqs, total_tokens

    def sample_tokens(
        self,
        num_tokens: int,
        padded_num_reqs: int,
        *,
        sampler_padded_num_reqs: int,
        sampler_num_reqs: int,
        sampler_total_tokens: int,
        req_num_tokens_full_cpu: numpy.ndarray,
        logits: jax.Array,
        rng_key: jax.Array,
        gather_positions_cpu: numpy.ndarray,
        sampling_seeds_cpu: numpy.ndarray,
        scatter_positions_cpu: numpy.ndarray,
        compact_window_row_indices_cpu: numpy.ndarray,
        compact_scheduled_cpu: numpy.ndarray,
        compact_seq_lens_cpu: numpy.ndarray,
        compact_active_mask_cpu: numpy.ndarray,
        compact_temperature_cpu: numpy.ndarray,
        compact_top_p_cpu: numpy.ndarray,
        compact_top_k_cpu: numpy.ndarray,
        compact_min_p_cpu: numpy.ndarray,
        compact_frequency_penalties_cpu: numpy.ndarray,
        compact_presence_penalties_cpu: numpy.ndarray,
        compact_repetition_penalties_cpu: numpy.ndarray,
        need_penalties: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run the compiled sampler step over only the rows that need sampling.

        Executes the pre-compiled sampler function, converting model logits
        into sampled tokens based on per-request sampling parameters.

        Args:
            num_tokens: Number of tokens for bucket selection.
            padded_num_reqs: Model-side padded request count for the current window.
            sampler_padded_num_reqs: Sampler-side padded request count after
                compacting out zero-token rows.
            sampler_num_reqs: Actual number of compacted sampler rows.
            sampler_total_tokens: Total scheduled tokens in the compacted sampler batch.
            req_num_tokens_full_cpu: Target token count per request [max_num_reqs].
            logits: Model output logits [padded_num_reqs, vocab_size].
            rng_key: JAX random key for stochastic sampling.
            gather_positions_cpu: Model-row indices used to gather compact logits.
            sampling_seeds_cpu: Original model-row indices used to preserve
                exact per-row RNG fold-ins.
            scatter_positions_cpu: Output positions used to scatter compact
                results back to the model-window layout.
            compact_window_row_indices_cpu: Global request-row indices aligned
                with the compact sampler rows.
            compact_scheduled_cpu: Scheduled token counts for compact rows.
            compact_seq_lens_cpu: Sequence lengths after the forward pass.
            compact_active_mask_cpu: Active sampler rows.
            compact_temperature_cpu: Temperature per compact row.
            compact_top_p_cpu: Top-p per compact row.
            compact_top_k_cpu: Top-k per compact row.
            compact_min_p_cpu: Min-p per compact row.
            compact_frequency_penalties_cpu: Frequency penalty per compact row.
            compact_presence_penalties_cpu: Presence penalty per compact row.
            compact_repetition_penalties_cpu: Repetition penalty per compact row.
            need_penalties: Whether any request in the window uses penalties.

        Returns:
            Tuple of (updated_rng_key, sampled_tokens, valid_mask, updated_token_counts) where:
            - updated_rng_key: New RNG key for next step.
            - sampled_tokens: Generated token IDs [padded_num_reqs], -1 for invalid.
            - valid_mask: Boolean mask indicating valid samples [padded_num_reqs].
            - updated_token_counts: Exact device-side token counts [max_num_reqs, vocab_size].

        Note:
            This method does not block on completion, allowing the caller to
            overlap host work while the device executes. The caller should
            synchronize on the returned arrays when ready to use them.
        """
        sampler_active = compact_active_mask_cpu[:sampler_padded_num_reqs].astype(numpy.bool_, copy=False)
        use_greedy_sampler = bool(
            not need_penalties
            and numpy.all(
                numpy.where(
                    sampler_active,
                    compact_temperature_cpu[:sampler_padded_num_reqs] <= 0.0,
                    True,
                )
            )
        )
        sampler_fn = self._sampler_executor.get_compiled(
            num_tokens=num_tokens,
            padded_num_reqs=sampler_padded_num_reqs,
            greedy=use_greedy_sampler,
        )
        sampler_sharding = getattr(self, "_sampler_sharding", self._empty_sharding)
        if need_penalties:
            self._ensure_sampler_penalty_state()
        token_counts_full = (
            self._sampler_token_counts if self._sampler_penalty_state_ready else self._sampler_zero_token_counts
        )
        sampler_packed_i32 = self._sampler_packed_i32_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_packed_i32 is None:
            sampler_packed_i32 = numpy.zeros((7, int(sampler_padded_num_reqs)), dtype=numpy.int32)
            self._sampler_packed_i32_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_packed_i32
        sampler_packed_f32 = self._sampler_packed_f32_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_packed_f32 is None:
            sampler_packed_f32 = numpy.zeros((6, int(sampler_padded_num_reqs)), dtype=numpy.float32)
            self._sampler_packed_f32_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_packed_f32

        sampler_packed_f32[0] = compact_temperature_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[1] = compact_top_p_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[2] = compact_min_p_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[3] = compact_frequency_penalties_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[4] = compact_presence_penalties_cpu[:sampler_padded_num_reqs]
        sampler_packed_f32[5] = compact_repetition_penalties_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[0] = sampling_seeds_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[1] = compact_scheduled_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[2] = compact_seq_lens_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[3] = compact_window_row_indices_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[4] = compact_active_mask_cpu[:sampler_padded_num_reqs].astype(numpy.int32)
        sampler_packed_i32[5] = compact_top_k_cpu[:sampler_padded_num_reqs]
        sampler_packed_i32[6].fill(0)
        if int(sampler_num_reqs) > 0:
            live_gather_positions = gather_positions_cpu[: int(sampler_num_reqs)]
            sampler_packed_i32[6, : int(sampler_num_reqs)] = req_num_tokens_full_cpu[live_gather_positions]
        self._sampler_packed_misc_i32_cpu[0] = numpy.int32(sampler_num_reqs)
        self._sampler_packed_misc_i32_cpu[1] = numpy.int32(sampler_total_tokens)

        sampler_host_payload = (sampler_packed_f32, sampler_packed_i32, self._sampler_packed_misc_i32_cpu)
        if jax.process_count() > 1:
            sampler_host_payload = multihost_utils.broadcast_one_to_all(sampler_host_payload)
        with jax.named_scope("easydel/esurge/sampler/place_metadata"):
            packed_f32, packed_i32, packed_misc_i32 = _device_put_tree_uniform(sampler_host_payload, sampler_sharding)
        sampler_prefix = self._sampler_prefix_cpu_by_reqs.get(int(sampler_padded_num_reqs))
        if sampler_prefix is None:
            sampler_prefix = numpy.arange(sampler_padded_num_reqs, dtype=numpy.int32)
            self._sampler_prefix_cpu_by_reqs[int(sampler_padded_num_reqs)] = sampler_prefix
        prefix_gather_layout = numpy.array_equal(
            gather_positions_cpu[:sampler_padded_num_reqs],
            sampler_prefix,
        )
        prefix_scatter_layout = numpy.array_equal(
            scatter_positions_cpu[:sampler_padded_num_reqs],
            sampler_prefix,
        )
        with jax.named_scope("easydel/esurge/sampler/gather_logits"):
            if prefix_gather_layout:
                compact_logits = logits[:sampler_padded_num_reqs]
            else:
                logits_sharding = logits.sharding
                gather_positions = jax.device_put(gather_positions_cpu[:sampler_padded_num_reqs], sampler_sharding)
                gather_positions_for_logits = jax.device_put(gather_positions, logits_sharding)
                compact_logits = logits[gather_positions_for_logits]
            if getattr(compact_logits, "sharding", None) != sampler_sharding:
                compact_logits = jax.device_put(compact_logits, sampler_sharding)

        with jax.named_scope("easydel/esurge/sampler/run_jit"):
            rng_key, compact_tokens, compact_valid_mask, token_counts_full = sampler_fn(
                packed_f32,
                packed_i32,
                packed_misc_i32,
                compact_logits,
                rng_key,
                token_counts_full,
            )
        if prefix_scatter_layout:
            return rng_key, compact_tokens, compact_valid_mask, token_counts_full
        with jax.named_scope("easydel/esurge/sampler/scatter_to_model_rows"):
            scatter_positions = jax.device_put(scatter_positions_cpu[:sampler_padded_num_reqs], sampler_sharding)
            out_tokens_full, valid_mask_full = self._scatter_sampler_outputs(
                compact_tokens,
                compact_valid_mask,
                scatter_positions,
                padded_num_reqs=padded_num_reqs,
            )
        return rng_key, out_tokens_full, valid_mask_full, token_counts_full

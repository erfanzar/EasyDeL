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

"""Binary search algorithms for efficient top-k and top-p sampling.

This module provides optimized binary search implementations for probability
distribution filtering, including top-k and top-p (nucleus) sampling. The algorithms
perform bit-level search over float32 representations to avoid expensive sorting
operations, making them particularly efficient on TPU/accelerator hardware.

Key functions:
    - int32_bsearch: Core binary search over int32 bit patterns
    - apply_float32_bsearch: Binary search adapted for float32 values
    - apply_topk_mask: Fast top-k filtering using binary search
    - apply_topp_mask: Fast nucleus (top-p) sampling using binary search
    - apply_topk_mask_bf16/apply_topp_mask_bf16: BFloat16-optimized variants
"""

from collections.abc import Callable, Sequence

import jax
from jax import lax
from jax import numpy as jnp

from .sampling_metadata import SamplingMetadata


def int32_bsearch(batch_shape: Sequence[int], predicate: Callable[[jnp.ndarray], jnp.ndarray]):
    """Perform batched binary search over int32 bit patterns.

    Searches for the largest int32 value (closest to positive infinity) for which
    the predicate returns False. This forms the foundation for float32 search by
    operating on the underlying bit representations.

    Args:
        batch_shape: Shape of the batch dimension for parallel search.
        predicate: Monotonic function mapping int32 → bool. Must return False
            for all values ≤ threshold and True for all values > threshold.
            The threshold may differ per batch element.

    Returns:
        Array of shape `batch_shape` containing the largest int32 value for which
        the predicate returns False in each batch element.

    Note:
        This uses bit-level operations (31 iterations) to efficiently search the
        entire int32 space without materializing intermediate arrays.
    """
    current_bits = jnp.zeros(batch_shape, dtype=jnp.int32)

    # Bit 31 (sign bit) requires special handling due to two's complement
    midpoint = current_bits
    predicate_satisfied = predicate(midpoint)
    current_bits = current_bits | jnp.where(predicate_satisfied, jnp.uint32(1 << 31), jnp.uint32(0))
    del midpoint, predicate_satisfied

    def loop_body(i, current_bits):
        bit_index = 30 - i
        bit = jnp.int32(1 << bit_index)
        midpoint = current_bits | bit
        predicate_satisfied = predicate(midpoint)
        current_bits = current_bits | jnp.where(predicate_satisfied, jnp.int32(0), bit)
        return current_bits

    current_bits = lax.fori_loop(0, 31, loop_body, current_bits)
    return current_bits


def _monotonic_int32_to_float32_bit_pattern(x: int) -> int:
    """Convert int32 to float32 bit pattern preserving IEEE 754 total order.

    This mapping ensures that int32 ordering matches the IEEE 754 total-ordering
    predicate for float32 values, enabling binary search on float representations.

    Args:
        x: Int32 bit pattern to convert.

    Returns:
        Float32 bit pattern (as int32) that maintains ordering consistency.

    Reference:
        IEEE 754 total-ordering: https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate
    """
    non_sign_bits = jnp.int32((1 << 31) - 1)

    x = x ^ jnp.where(x < 0, non_sign_bits, jnp.int32(0))
    return x


def _monotonic_int32_to_float32(x: int) -> jax.Array:
    """Convert int32 to float32 preserving total order.

    Args:
        x: Int32 value to convert.

    Returns:
        Float32 value with consistent ordering relative to int32 input.
    """
    x = _monotonic_int32_to_float32_bit_pattern(x)
    return lax.bitcast_convert_type(x, jnp.float32)


def apply_float32_bsearch(batch_shape, predicate):
    """Perform batched binary search over finite float32 values.

    Searches for the largest finite non-NaN float32 for which the predicate
    returns False. Handles special values (infinity, NaN) at the extremes
    of the float32 range.

    Args:
        batch_shape: Shape of the batch dimension for parallel search.
        predicate: Monotonic function mapping float32 → bool, following
            IEEE 754 total-ordering. Must return False for values ≤ threshold
            and True for values > threshold.

    Returns:
        Array of shape `batch_shape` containing the largest float32 value
        for which the predicate returns False in each batch element.

    Note:
        The implementation uses int32 binary search on bit patterns, then
        converts back to float32 to handle the complexities of float ordering.
    """
    exponent_bits = jnp.int32((1 << 31) - (1 << (31 - 8)))

    def int32_predicate(x):
        x = _monotonic_int32_to_float32_bit_pattern(x)
        is_finite = (x & exponent_bits) != exponent_bits

        # Handle non-finite values at int32 extremes
        predicate_on_nonfinite = x >= 0
        x_float32 = lax.bitcast_convert_type(x, jnp.float32)
        return jnp.where(is_finite, predicate(x_float32), predicate_on_nonfinite)

    result = int32_bsearch(batch_shape, int32_predicate)
    return _monotonic_int32_to_float32(result)


def apply_topk_mask_bf16(x: jnp.ndarray, k: jax.Array, replace_val: float) -> jnp.ndarray:
    """Apply top-k masking with bfloat16 optimization.

    Efficiently filters logits to keep only the top-k values per batch element,
    minimizing dtype conversions for bfloat16 inputs.

    Args:
        x: Values to mask [batch..., vocab_size].
        k: Number of top values to retain (ties may result in more).
        replace_val: Value to use for masked positions.

    Returns:
        Masked array with same shape as input, dtype preserved.

    Performance:
        - 32 reductions over vocab_size dimension
        - Minimizes casting: only converts to float32 for binary search comparisons
        - Optimal when vocab_size dimension is unsharded

    Note:
        For bfloat16 inputs, performs final masking in original dtype to avoid
        unnecessary upcasting overhead.
    """
    batch_shape = tuple(list(x.shape)[:-1])
    original_dtype = x.dtype

    x_f32 = x.astype(jnp.float32)
    x_for_loop = x_f32
    reduce_axis = x_f32.ndim - 1
    if x_f32.ndim > 1:
        # Transpose to avoid expensive last-dimension reductions
        x_for_loop = jnp.swapaxes(x_for_loop, -1, -2)
        reduce_axis = x_f32.ndim - 2

    def predicate(threshold):
        threshold = -threshold
        threshold = lax.expand_dims(threshold, (reduce_axis,))
        count_gt = jnp.sum(x_for_loop > threshold, axis=reduce_axis)
        return count_gt >= k

    cutoff = apply_float32_bsearch(batch_shape, predicate)
    cutoff = -cutoff
    cutoff = lax.expand_dims(cutoff, (cutoff.ndim,))

    if original_dtype == jnp.bfloat16:
        cutoff_orig = cutoff.astype(original_dtype)
        return jnp.where(x >= cutoff_orig, x, jnp.full_like(x, replace_val))
    else:
        return jnp.where(x_f32 >= cutoff, x, jnp.full_like(x, replace_val))


def apply_topk_mask(x: jnp.ndarray, k: jax.Array, replace_val: float) -> jnp.ndarray:
    """Apply top-k masking using binary search.

    Retains only the top-k values per batch element, replacing all others
    with a specified value. Uses binary search over bit patterns for efficiency.

    Args:
        x: Values to mask [batch..., vocab_size].
        k: Number of top values to retain per batch element.
        replace_val: Value to use for masked positions.

    Returns:
        Masked array with same shape and dtype as input.

    Performance:
        - 32 reductions over vocab_size dimension
        - For best performance, ensure vocab_size is unsharded
        - Prefer batching along batch dimensions for parallelism

    Note:
        In presence of ties, more than k values may be returned.
    """
    batch_shape = tuple(list(x.shape)[:-1])

    x_for_loop = x
    reduce_axis = x.ndim - 1
    if x.ndim > 1:
        x_for_loop = jnp.swapaxes(x_for_loop, -1, -2)
        reduce_axis = x.ndim - 2

    def predicate(threshold):
        threshold = -threshold
        threshold = lax.expand_dims(threshold, (reduce_axis,))
        count_gt = jnp.sum(x_for_loop > threshold, axis=reduce_axis)
        return count_gt >= k

    cutoff = apply_float32_bsearch(batch_shape, predicate)
    cutoff = -cutoff
    cutoff = lax.expand_dims(cutoff, (cutoff.ndim,))
    return jnp.where(x >= cutoff, x, jnp.full_like(x, replace_val))


def apply_topp_mask_bf16(logits: jnp.ndarray, p: jax.Array, replace_val: float) -> jnp.ndarray:
    """Apply top-p (nucleus) masking with bfloat16 optimization.

    Filters logits to the smallest set of tokens whose cumulative probability
    mass is ≥ p, optimized for bfloat16 inputs.

    Args:
        logits: Logits before masking [batch..., vocab_size].
        p: Minimum probability mass to retain (0.0 to 1.0).
        replace_val: Value for masked logits.

    Returns:
        Masked logits with same shape and dtype as input.

    Performance:
        - 33 reductions over vocab_size (1 softmax + 32 binary search)
        - Keeps softmax in original dtype when possible
        - Only uses float32 for accumulation precision

    Note:
        The bfloat16 variant minimizes unnecessary casting while maintaining
        numerical precision where needed.
    """
    batch_shape = tuple(list(logits.shape)[:-1])
    original_dtype = logits.dtype

    probs = jax.nn.softmax(logits, axis=-1)

    probs_f32 = probs.astype(jnp.float32) if original_dtype == jnp.bfloat16 else probs
    probs_for_reduction = probs_f32
    reduce_axis = probs_for_reduction.ndim - 1
    if probs_for_reduction.ndim > 1:
        probs_for_reduction = jnp.swapaxes(probs_for_reduction, -1, -2)
        reduce_axis = probs_for_reduction.ndim - 2

    def predicate(threshold):
        threshold = lax.expand_dims(threshold, (reduce_axis,))
        probability_mass = jnp.sum(
            jnp.where(probs_for_reduction >= threshold, probs_for_reduction, 0.0),
            axis=reduce_axis,
        )
        return probability_mass < p

    threshold = apply_float32_bsearch(batch_shape, predicate)
    threshold = lax.expand_dims(threshold, (threshold.ndim,))

    if original_dtype == jnp.bfloat16:
        threshold_orig = threshold.astype(original_dtype)
        probs_orig = probs
        return jnp.where(probs_orig >= threshold_orig, logits, jnp.full_like(logits, replace_val))
    else:
        return jnp.where(probs_f32 >= threshold, logits, jnp.full_like(logits, replace_val))


def apply_topp_mask(logits: jnp.ndarray, p: jax.Array, replace_val: float) -> jnp.ndarray:
    """Apply top-p (nucleus) sampling mask using binary search.

    Masks logits to retain only the smallest set of tokens whose cumulative
    probability is ≥ p. This implements nucleus sampling efficiently without
    requiring explicit sorting.

    Args:
        logits: Logits before masking [batch..., vocab_size].
        p: Minimum probability mass to retain (0.0 to 1.0).
        replace_val: Value to use for masked logits.

    Returns:
        Masked logits with same shape and dtype as input.

    Performance:
        - 33 reductions over vocab_size dimension
        - Ensure vocab_size is unsharded for minimal latency
        - Consider microbatching for memory efficiency

    Example:
        >>> logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
        >>> masked = apply_topp_mask(logits, p=0.9, replace_val=-1e10)
        >>> # Retains top tokens covering 90% probability mass
    """
    batch_shape = tuple(list(logits.shape)[:-1])

    probs = jax.nn.softmax(logits, axis=-1)

    probs_for_reduction = probs
    reduce_axis = probs_for_reduction.ndim - 1
    if probs_for_reduction.ndim > 1:
        probs_for_reduction = jnp.swapaxes(probs_for_reduction, -1, -2)
        reduce_axis = probs_for_reduction.ndim - 2

    def predicate(threshold):
        threshold = lax.expand_dims(threshold, (reduce_axis,))
        probability_mass = jnp.sum(
            jnp.where(probs_for_reduction >= threshold, probs_for_reduction, 0.0),
            axis=reduce_axis,
        )
        return probability_mass < p

    threshold = apply_float32_bsearch(batch_shape, predicate)
    threshold = lax.expand_dims(threshold, (threshold.ndim,))
    return jnp.where(probs >= threshold, logits, jnp.full_like(logits, replace_val))


def apply_min_p_mask(logits: jax.Array, sampling_metadata: SamplingMetadata) -> jax.Array:
    """Apply min-p masking to logits.

    Min-p filtering keeps only tokens whose logit is at least min_p times
    the maximum logit. This helps filter out low-probability tail tokens
    more aggressively than top-p alone.

    Args:
        logits: Input logits [batch, vocab_size].
        sampling_metadata: Sampling configuration containing min_p values.

    Returns:
        Masked logits with min-p filtering applied [batch, vocab_size].

    Note:
        Min-p is applied as: keep tokens where logit >= min_p * max(logit).
        This is computed on logits directly, not on probabilities, to avoid
        issues with already-filtered distributions.
    """
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    threshold = sampling_metadata.min_ps[:, None] * max_logits
    mask = logits >= threshold
    return jnp.where(mask, logits, jnp.full_like(logits, -1e10))


def apply_penalties(logits: jax.Array, sampling_metadata: SamplingMetadata) -> jax.Array:
    """Apply linear penalties to logits.

    Penalties modify logit values to discourage repetition or enforce other
    constraints. Common penalties include frequency, presence, and repetition.

    Args:
        logits: Input logits [batch, vocab_size].
        sampling_metadata: Sampling configuration with optional penalty matrix.

    Returns:
        Logits with penalties applied [batch, vocab_size].

    Note:
        If no penalty is specified (None), returns logits unchanged.
    """
    if sampling_metadata.linear_penalty is None:
        return logits
    penalty = sampling_metadata.linear_penalty.astype(logits.dtype)
    return logits + penalty

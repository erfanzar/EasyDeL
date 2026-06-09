# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Pallas forward kernel implementation for ragged decode attention on GPU.

This module contains the low-level Pallas kernel implementation for computing
multi-head attention during the decoding phase with variable-length (ragged)
sequences. The implementation uses JAX's Pallas with Triton backend for
GPU-optimized execution.

The kernel uses a tiled approach with configurable block sizes for heads and
keys, enabling efficient parallel computation across GPU thread blocks. It
implements online softmax (FlashAttention-style) to maintain numerical stability
while minimizing memory access.

Key Components:
    AttentionConfigError: Custom exception for invalid kernel configurations.
    forward_kernel: The core Pallas kernel that computes attention scores.
    decode_attn_sequence: Orchestrates kernel execution for a single sequence.
    _ragged_decode_attention_call: JIT-compiled entry point for batched execution.

Implementation Details:
    - Uses block-wise tiling for memory efficiency
    - Implements online softmax for numerical stability
    - Supports variable-length sequences via sequence_start/sequence_end
    - Handles MHA, MQA, and GQA head configurations

Note:
    This is an internal module. Use the public API through
    `ragged_decode_attention` from the parent package.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas import primitives as pallas_primitives
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as ptriton
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams


class AttentionConfigError(ValueError):
    """Raised when attention configuration parameters are invalid."""

    pass


def forward_kernel(
    query_ref: Any,
    key_ref: Any,
    value_ref: Any,
    sequence_start_ref: Any | None,
    sequence_end_ref: Any | None,
    output_ref: Any,
    log_sum_exp_ref: Any,
    max_logit_ref: Any,
    softmax_scale: float,
    block_size_k: int,
    block_size_heads: int,
    total_num_heads: int,
):
    """Pallas forward kernel for one (head-group, key-split) tile of decode attention.

    Executed on a 2-D grid: ``(num_head_splits, num_key_splits)``.  Each
    invocation processes ``block_size_heads`` query heads against one contiguous
    slice of the key/value sequence (``split_key_seq_len`` tokens), accumulating
    an online-softmax running state.

    Tensor layout (inputs, per grid cell):
        query_ref:  [block_size_heads, head_dim]
        key_ref:    [split_key_seq_len, head_dim]  (one key-split slice)
        value_ref:  [split_key_seq_len, head_dim]  (matching value slice)
        sequence_start_ref: scalar int32 — global start of valid tokens (or None)
        sequence_end_ref:   scalar int32 — global end of valid tokens (or None)

    Outputs, per grid cell:
        output_ref:      [block_size_heads, head_dim] — partial weighted sum
        log_sum_exp_ref: [block_size_heads] — running log-sum-exp denominator
        max_logit_ref:   [block_size_heads] — running max logit (for correction)

    The kernel iterates over ``split_key_seq_len // block_size_k`` key blocks
    using ``lax.fori_loop``.  For each block it:
      1. Loads the key slice and computes raw dot-product scores.
      2. Applies ``softmax_scale``.
      3. Masks positions outside ``[sequence_start, sequence_end)`` with
         ``finfo.min`` (only when either ref is not None).
      4. Updates the running online-softmax state (max, log-sum-exp, output).

    When ``sequence_start_ref`` and ``sequence_end_ref`` are both None the
    masking branch is skipped entirely and a ``lax.cond`` guard is omitted.

    Args:
        query_ref: VMEM reference to the query tile, shape
            [block_size_heads, head_dim].
        key_ref: VMEM reference to the key split, shape
            [split_key_seq_len, head_dim].
        value_ref: VMEM reference to the value split, shape
            [split_key_seq_len, head_dim].
        sequence_start_ref: Scalar SMEM reference containing the global
            sequence start index, or None for no lower-bound masking.
        sequence_end_ref: Scalar SMEM reference containing the global sequence
            end index (exclusive), or None for no upper-bound masking.
        output_ref: VMEM output reference for the partial attention output,
            shape [block_size_heads, head_dim].
        log_sum_exp_ref: VMEM reference for the log-sum-exp accumulator,
            shape [block_size_heads].
        max_logit_ref: VMEM reference for the running maximum logit,
            shape [block_size_heads].
        softmax_scale: Scalar float multiplied into attention logits before
            the online-softmax step (typically ``1/sqrt(head_dim)``).
        block_size_k: Number of key tokens processed per inner loop iteration.
            Must evenly divide ``split_key_seq_len`` and be >= 16.
        block_size_heads: Number of query heads processed by this kernel
            invocation.  May be larger than the remaining heads; a mask guards
            the out-of-bounds rows.
        total_num_heads: Total number of query heads across all head-splits.
            Used to compute the valid-head mask for the last head tile.

    Raises:
        RuntimeError: Wraps any exception raised during kernel body execution
            with context about the failure site.
    """
    try:
        _, head_dimension = query_ref.shape
        split_key_seq_len, _ = key_ref.shape

        program_id_heads, program_id_splits = pl.program_id(0), pl.program_id(1)
        query_slice = pl.ds(0, block_size_heads)
        query_mask = (jnp.arange(block_size_heads) < total_num_heads - block_size_heads * program_id_heads)[:, None]

        def _compute_attention(seq_start, seq_end, output_accumulator, max_logits_prev, log_sum_prev):
            """Inner computation function for attention mechanism."""
            current_query = pallas_primitives.load(query_ref, (query_slice, pl.ds(None)), mask=query_mask)
            block_indices = jnp.arange(block_size_k)

            def attention_body(key_block_start, carry_state):
                """Body function for iterating over key blocks."""
                output_prev, max_prev, logsum_prev = carry_state
                current_key_slice = pl.ds(key_block_start * block_size_k, block_size_k)

                attention_scores = pl.dot(
                    current_query,
                    pallas_primitives.load(key_ref, (current_key_slice, slice(None))).T,
                )

                if softmax_scale != 1.0:
                    attention_scores *= softmax_scale
                if sequence_start_ref is not None or sequence_end_ref is not None:
                    global_indices = (
                        program_id_splits * split_key_seq_len + key_block_start * block_size_k + block_indices
                    )
                    sequence_mask = ((global_indices >= seq_start) & (global_indices < seq_end))[None, :]
                    attention_scores = jnp.where(sequence_mask, attention_scores, jnp.finfo(attention_scores.dtype).min)

                max_current = attention_scores.max(axis=-1)
                max_next = jnp.maximum(max_prev, max_current)
                correction_factor = jnp.exp(max_prev - max_next)
                logsum_prev_corrected = correction_factor * logsum_prev

                softmax_scores = jnp.exp(attention_scores - max_next[:, None])
                logsum_current = softmax_scores.sum(axis=-1)
                logsum_next = logsum_prev_corrected + logsum_current

                current_values = pallas_primitives.load(value_ref, (current_key_slice, slice(None)))
                output_current = pl.dot(softmax_scores.astype(current_values.dtype), current_values)
                output_next = correction_factor[:, None] * output_prev + output_current

                return output_next, max_next, logsum_next

            max_iterations = jnp.minimum(
                pl.cdiv((seq_end - program_id_splits * split_key_seq_len), block_size_k),
                split_key_seq_len // block_size_k,
            )

            final_output, final_max, final_logsum = lax.fori_loop(
                0, max_iterations, attention_body, (output_accumulator, max_logits_prev, log_sum_prev)
            )

            return final_output, final_max, final_logsum

        max_logits_init = jnp.zeros(block_size_heads, dtype=jnp.float32) + jnp.finfo(jnp.float32).min
        log_sum_init = jnp.zeros(block_size_heads, dtype=jnp.float32)
        output_init = jnp.zeros((block_size_heads, head_dimension), dtype=jnp.float32)

        sequence_start = split_key_seq_len * program_id_splits
        if sequence_start_ref is not None:
            sequence_start = jnp.maximum(sequence_start, pallas_primitives.load(sequence_start_ref, ()))

        sequence_end = (program_id_splits + 1) * split_key_seq_len
        if sequence_end_ref is not None:
            sequence_end = jnp.minimum(sequence_end, pallas_primitives.load(sequence_end_ref, ()))

        if sequence_start_ref is None and sequence_end_ref is None:
            final_output, final_max_logits, final_log_sum = _compute_attention(
                sequence_start, sequence_end, output_init, max_logits_init, log_sum_init
            )
        else:
            final_output, final_max_logits, final_log_sum = jax.lax.cond(
                sequence_start >= sequence_end,
                lambda: (output_init, max_logits_init, log_sum_init),
                lambda: _compute_attention(sequence_start, sequence_end, output_init, max_logits_init, log_sum_init),
            )

        vector_query_mask = query_mask.reshape(-1) if query_mask is not None else None
        pallas_primitives.store(log_sum_exp_ref, query_slice, final_log_sum, mask=vector_query_mask)
        pallas_primitives.store(max_logit_ref, query_slice, final_max_logits, mask=vector_query_mask)

        final_output = final_output.astype(output_ref.dtype)
        pallas_primitives.store(output_ref, (query_slice, pl.ds(None)), final_output, mask=query_mask)

    except Exception as e:
        raise RuntimeError(f"Error in forward_kernel execution: {e!s}") from e


def decode_attn_sequence(
    query_tensor: jnp.ndarray,
    key_tensor: jnp.ndarray,
    value_tensor: jnp.ndarray,
    sequence_start: jnp.ndarray | None,
    sequence_end: jnp.ndarray | None,
    softmax_scale: float,
    block_size_heads: int,
    block_size_keys: int,
    num_key_splits: int,
    num_warps: int | None,
    num_stages: int,
) -> jnp.ndarray:
    """Orchestrate a single-sequence ragged decode attention call via Pallas.

    Reshapes the flat key/value tensors into ``num_key_splits`` splits, builds
    the ``pallas_call`` grid, launches ``forward_kernel`` in parallel across
    head-tiles and key-splits, then reduces the per-split results with a final
    online-softmax correction to produce the single-sequence output.

    Grid layout:
        Axis 0 — head splits: ``ceil(num_heads / block_size_heads)``
        Axis 1 — key splits: ``num_key_splits``

    Output reduction:
        Each split produces a partial weighted sum, a log-sum-exp denominator,
        and a running max.  After the kernel, the per-split max values are
        globally reduced, correction factors are applied, and the partial sums
        are combined and normalized.

    Args:
        query_tensor: Query array of shape ``(num_heads, head_dim)``.
            Must be 2-D.
        key_tensor: Key array of shape ``(seq_len, head_dim)``.  Must be 2-D
            and have the same ``head_dim`` as ``query_tensor``.
        value_tensor: Value array of shape ``(seq_len, head_dim)``.  Must
            have the same shape as ``key_tensor``.
        sequence_start: Optional scalar-shaped array with the global start
            index of the valid token range for this sequence.  When None,
            no lower-bound masking is applied.
        sequence_end: Optional scalar-shaped array with the global end index
            (exclusive) of the valid token range.  When None, no upper-bound
            masking is applied.
        softmax_scale: Scalar multiplied into attention logits before softmax
            (typically ``1/sqrt(head_dim)``).
        block_size_heads: Number of heads per Pallas head-tile.  Must be >= 1.
        block_size_keys: Number of key tokens per inner kernel loop iteration.
            Must be >= 16 and evenly divide the per-split key length.
        num_key_splits: Number of groups to split the key sequence into.
            ``seq_len`` must be divisible by ``num_key_splits`` and each split
            must be >= 16 tokens.
        num_warps: Number of Triton warps for the GPU kernel.  Defaults to 4
            when None.
        num_stages: Number of software pipeline stages for memory prefetching.

    Returns:
        Attention output array of shape ``(num_heads, head_dim)`` with the
        same dtype as ``query_tensor``.

    Raises:
        ValueError: If any tensor is not 2-D, head dimensions do not match,
            or key/value shapes differ.
        AttentionConfigError: If ``seq_len`` is not divisible by
            ``num_key_splits``, the per-split length is < 16, or
            ``block_size_keys`` is < 16 or does not divide the per-split
            sequence length.
    """
    try:
        if query_tensor.ndim != 2:
            raise ValueError(f"Query tensor must be 2D, got shape {query_tensor.shape}")
        if key_tensor.ndim != 2:
            raise ValueError(f"Key tensor must be 2D, got shape {key_tensor.shape}")
        if value_tensor.ndim != 2:
            raise ValueError(f"Value tensor must be 2D, got shape {value_tensor.shape}")

        total_num_heads, head_dimension = query_tensor.shape
        key_sequence_length, key_head_dim = key_tensor.shape

        if key_head_dim != head_dimension:
            raise ValueError(f"Key head dimension {key_head_dim} must match query head dimension {head_dimension}")
        if value_tensor.shape != key_tensor.shape:
            raise ValueError(f"Value tensor shape {value_tensor.shape} must match key tensor shape {key_tensor.shape}")

        if key_sequence_length % num_key_splits != 0:
            raise AttentionConfigError(
                f"Key sequence length {key_sequence_length} must be divisible by num_key_splits {num_key_splits}"
            )

        split_key_sequence_length = key_sequence_length // num_key_splits
        if split_key_sequence_length < 16:
            raise AttentionConfigError(f"Split key sequence length {split_key_sequence_length} must be >= 16")

        if block_size_keys < 16:
            raise AttentionConfigError(f"block_size_keys {block_size_keys} must be >= 16")

        num_head_splits = pl.cdiv(total_num_heads, block_size_heads)
        computation_grid = (num_head_splits, num_key_splits)

        reshaped_keys = key_tensor.reshape(num_key_splits, split_key_sequence_length, head_dimension)
        reshaped_values = value_tensor.reshape(num_key_splits, split_key_sequence_length, head_dimension)

        effective_block_size_k = min(block_size_keys, split_key_sequence_length)
        if split_key_sequence_length % effective_block_size_k != 0:
            raise AttentionConfigError(
                f"Split sequence length {split_key_sequence_length} must be divisible by "
                f"effective block size {effective_block_size_k}"
            )

        effective_num_warps = num_warps if num_warps is not None else 4

        bound_kernel = functools.partial(
            forward_kernel,
            block_size_k=effective_block_size_k,
            block_size_heads=block_size_heads,
            softmax_scale=softmax_scale,
            total_num_heads=total_num_heads,
        )

        attention_output, log_sum_exp_values, max_logit_values = pl.pallas_call(
            bound_kernel,
            grid=computation_grid,
            in_specs=[
                pl.BlockSpec((block_size_heads, head_dimension), lambda i, j: (i, 0)),
                pl.BlockSpec((None, split_key_sequence_length, head_dimension), lambda i, j: (j, 0, 0)),
                pl.BlockSpec((None, split_key_sequence_length, head_dimension), lambda i, j: (j, 0, 0)),
                None if sequence_start is None else pl.BlockSpec((), lambda i, j: ()),
                None if sequence_end is None else pl.BlockSpec((), lambda i, j: ()),
            ],
            out_specs=[
                pl.BlockSpec((None, block_size_heads, head_dimension), lambda i, j: (j, i, 0)),
                pl.BlockSpec((None, block_size_heads), lambda i, j: (j, i)),
                pl.BlockSpec((None, block_size_heads), lambda i, j: (j, i)),
            ],
            compiler_params=ptriton.CompilerParams(
                num_warps=effective_num_warps,
                num_stages=num_stages,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(shape=(num_key_splits, *query_tensor.shape), dtype=query_tensor.dtype),
                jax.ShapeDtypeStruct(shape=(num_key_splits, total_num_heads), dtype=jnp.float32),
                jax.ShapeDtypeStruct(shape=(num_key_splits, total_num_heads), dtype=jnp.float32),
            ],
            name="mha_forward",
        )(
            query_tensor,
            reshaped_keys,
            reshaped_values,
            sequence_start,
            sequence_end,
        )

        max_logits_global = max_logit_values.max(axis=0)
        max_logits_global = lax.optimization_barrier(max_logits_global)

        correction_factors = jnp.exp(max_logit_values - max_logits_global[None])
        corrected_outputs = attention_output * correction_factors[:, :, None].astype(attention_output.dtype)

        corrected_log_sum_exp = (log_sum_exp_values * correction_factors).sum(axis=0)
        numerical_epsilon = jnp.finfo(corrected_log_sum_exp.dtype).eps

        final_output = corrected_outputs.sum(axis=0)
        out_scale = corrected_log_sum_exp[:, None].astype(final_output.dtype) + numerical_epsilon

        return final_output / out_scale

    except Exception as e:
        raise RuntimeError(f"Error in decode_attn_sequence: {e!s}") from e


@ejit(static_argnames=["softmax_scale", "fwd_params", "sliding_window", "logits_soft_cap"])
def _ragged_decode_attention_call(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """JIT-compiled entry point that dispatches the batched ragged decode kernel.

    Validates tensor shapes, computes the default ``softmax_scale``, groups
    query heads per KV head for MHA/MQA/GQA, broadcasts the per-batch
    ``sequence_start``/``sequence_end`` indices to ``(batch, kv_heads)``, and
    applies ``jax.vmap`` twice (over batch and KV-head dimensions) to
    ``decode_attn_sequence``.

    Kernel parameters (``block_size_heads``, ``kv_blocksize``,
    ``num_key_splits``, ``num_warps``, ``num_stages``) are taken from
    ``fwd_params`` (a :class:`ejkernel.ops.FwdParams` instance).  When
    ``fwd_params`` is ``None`` a ``FwdParams()`` with all defaults is used,
    which may raise ``AttributeError`` because the defaults are ``None``.
    Callers should always pass an explicit ``FwdParams``.

    Args:
        query: Query tensor of shape ``[batch, num_q_heads, head_dim]``.
        key: Key tensor of shape ``[batch, seq_len, num_kv_heads, head_dim]``.
        value: Value tensor of shape ``[batch, seq_len, num_kv_heads, head_dim]``.
        sequence_start: Per-batch start index of valid tokens, shape
            ``[batch]``.  Broadcast internally to ``[batch, kv_heads]``.
        sequence_end: Per-batch end index (exclusive) of valid tokens, shape
            ``[batch]``.  Broadcast internally to ``[batch, kv_heads]``.
        softmax_scale: Scaling factor for attention logits.  Defaults to
            ``1/sqrt(head_dim)`` when None.
        fwd_params: Forward kernel parameters.  Relevant fields:
            ``blocksize_heads`` — head tile size for the GPU kernel;
            ``kv_blocksize``    — key tile size per inner loop iteration;
            ``num_key_splits``  — number of key-sequence splits;
            ``num_warps``       — number of Triton warps (None → 4);
            ``num_stages``      — Triton software-pipeline stages.
        sliding_window: Accepted but currently unused.  Reserved for future
            sliding-window attention support.
        logits_soft_cap: Accepted but currently unused.  Reserved for future
            logit-capping support.
        softmax_aux: Accepted but currently unused.  Reserved for future
            attention-sink support.

    Returns:
        Attention output of shape ``[batch, num_q_heads, head_dim]``.

    Raises:
        ValueError: If ``key.shape[2] != value.shape[2]`` (KV head count
            mismatch) or if ``num_q_heads % num_kv_heads != 0`` (query heads
            not divisible by KV heads).
    """
    softmax_scale = softmax_scale if softmax_scale is not None else (query.shape[-1] ** -0.5)
    batch_size, q_heads, head_dim = query.shape
    kv_heads = key.shape[2]

    if kv_heads != value.shape[2]:
        raise ValueError(
            f"Key-Value head count mismatch: expected {kv_heads} heads based on key tensor, "
            f"but value tensor has {value.shape[2]} heads. "
            f"Key and Value tensors must have the same number of heads for attention computation. "
            f"Value tensor shape: {value.shape}, Key-Value heads: {kv_heads}"
        )

    if q_heads % kv_heads != 0:
        raise ValueError(
            f"Invalid head configuration for Multi-Query/Grouped-Query Attention: "
            f"Query heads ({q_heads}) must be evenly divisible by Key-Value heads ({kv_heads}). "
            f"This ensures proper head grouping where each KV head can attend to "
            f"{q_heads // kv_heads if kv_heads != 0 else 'undefined'} query heads. "
            f"Common valid configurations: "
            f"- Multi-Head: q_heads=kv_heads (e.g., 32=32) "
            f"- Multi-Query: kv_heads=1 (e.g., 32÷1=32) "
            f"- Grouped-Query: q_heads divisible by kv_heads (e.g., 32÷8=4)"
        )

    if sequence_start is not None:
        sequence_start = sequence_start.reshape(batch_size, 1)
        sequence_start = jnp.broadcast_to(sequence_start, (batch_size, kv_heads))
    if sequence_end is not None:
        sequence_end = sequence_end.reshape(batch_size, 1)
        sequence_end = jnp.broadcast_to(sequence_end, (batch_size, kv_heads))

    fn = functools.partial(
        decode_attn_sequence,
        softmax_scale=softmax_scale,
        block_size_heads=fwd_params.blocksize_heads,
        block_size_keys=fwd_params.kv_blocksize,
        num_key_splits=fwd_params.num_key_splits,
        num_warps=fwd_params.num_warps,
        num_stages=fwd_params.num_stages,
    )
    o = jax.vmap(jax.vmap(fn))(
        query.reshape(batch_size, kv_heads, q_heads // kv_heads, head_dim),
        jnp.swapaxes(key, 1, 2),
        jnp.swapaxes(value, 1, 2),
        sequence_start,
        sequence_end,
    )
    return o.reshape(batch_size, q_heads, head_dim)

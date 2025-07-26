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

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as ptriton


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
    """
    Forward kernel for multi-head attention computation.

    Args:
        query_ref: Reference to query tensor
        key_ref: Reference to key tensor
        value_ref: Reference to value tensor
        sequence_start_ref: Reference to sequence start positions (optional)
        sequence_end_ref: Reference to sequence end positions (optional)
        output_ref: Reference to output tensor
        log_sum_exp_ref: Reference to log-sum-exp values
        max_logit_ref: Reference to maximum logits
        softmax_scale: Scaling factor for softmax
        block_size_k: Block size for key dimension
        block_size_heads: Block size for heads dimension
        total_num_heads: Total number of attention heads
    """
    try:
        _, head_dimension = query_ref.shape
        split_key_seq_len, _ = key_ref.shape

        program_id_heads, program_id_splits = pl.program_id(0), pl.program_id(1)
        query_slice = pl.ds(0, block_size_heads)
        query_mask = (jnp.arange(block_size_heads) < total_num_heads - block_size_heads * program_id_heads)[:, None]

        def _compute_attention(seq_start, seq_end, output_accumulator, max_logits_prev, log_sum_prev):
            """Inner computation function for attention mechanism."""
            current_query = pl.load(query_ref, (query_slice, pl.ds(None)), mask=query_mask)
            block_indices = jnp.arange(block_size_k)

            def attention_body(key_block_start, carry_state):
                """Body function for iterating over key blocks."""
                output_prev, max_prev, logsum_prev = carry_state
                current_key_slice = pl.ds(key_block_start * block_size_k, block_size_k)

                attention_scores = pl.dot(current_query, pl.load(key_ref, (current_key_slice, slice(None))).T)

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

                current_values = pl.load(value_ref, (current_key_slice, slice(None)))
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
            sequence_start = jnp.maximum(sequence_start, pl.load(sequence_start_ref, ()))

        sequence_end = (program_id_splits + 1) * split_key_seq_len
        if sequence_end_ref is not None:
            sequence_end = jnp.minimum(sequence_end, pl.load(sequence_end_ref, ()))

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
        pl.store(log_sum_exp_ref, query_slice, final_log_sum, mask=vector_query_mask)
        pl.store(max_logit_ref, query_slice, final_max_logits, mask=vector_query_mask)

        final_output = final_output.astype(output_ref.dtype)
        pl.store(output_ref, (query_slice, pl.ds(None)), final_output, mask=query_mask)

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
    """
    Decode attention for sequence processing with improved error handling.

    Args:
        query_tensor: Query tensor of shape (num_heads, head_dim)
        key_tensor: Key tensor of shape (seq_len, head_dim)
        value_tensor: Value tensor of shape (seq_len, head_dim)
        sequence_start: Optional start positions for sequences
        sequence_end: Optional end positions for sequences
        softmax_scale: Scaling factor for attention scores
        block_size_heads: Block size for processing heads
        block_size_keys: Block size for processing keys
        num_key_splits: Number of splits for key tensor
        num_warps: Number of warps for GPU execution (optional)
        num_stages: Number of pipeline stages

    Returns:
        Output tensor from attention computation

    Raises:
        AttentionConfigError: If configuration parameters are invalid
        ValueError: If tensor shapes are incompatible
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

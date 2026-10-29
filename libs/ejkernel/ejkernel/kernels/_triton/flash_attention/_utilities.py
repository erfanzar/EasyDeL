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


"""Utility functions for Flash Attention Triton kernels.

This module provides helper functions used by both the forward and backward
pass implementations of Flash Attention, including:

    - ``padded_load``: Triton JIT helper for conditionally masked memory loads
    - ``calc_bias_strides``: Computes memory strides for bias tensors with broadcasting
    - ``attention_pack_with_static_shape``: Packs variable-length sequences using attention masks
    - ``attention_pack_from_cu_static``: Packs using cumulative sequence lengths
    - ``attention_unpack_with_static_shape``: Reverses packing to restore padded batches
    - ``basic_attention_refrence``: Reference JAX attention for testing and validation
"""

import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Bool, Float, Int

from ejkernel.callib import ejit
from ejkernel.utils import get_strides


@triton.jit
def padded_load(
    ptrs,
    offs_a,
    offs_b,
    PA0: tl.constexpr,
    PA1: tl.constexpr,
    LA0: tl.constexpr,
    LA1: tl.constexpr,
):
    """Load data from memory with optional padding for boundary conditions.

    Conditionally loads data with masking based on compile-time constants,
    optimizing for different padding scenarios.

    Args:
        ptrs: Pointer to memory location
        offs_a: Offsets for first dimension
        offs_b: Offsets for second dimension
        PA0: Whether first dimension needs padding check
        PA1: Whether second dimension needs padding check
        LA0: Actual length of first dimension
        LA1: Actual length of second dimension

    Returns:
        Loaded tensor with zeros for out-of-bounds elements
    """
    if PA0:
        if PA1:
            x = tl.load(
                ptrs,
                mask=(offs_a[:, None] < LA0) & (offs_b[None, :] < LA1),
                other=0.0,
            )
        else:
            x = tl.load(
                ptrs,
                mask=offs_a[:, None] < LA0,
                other=0.0,
            )
    else:
        if PA1:
            x = tl.load(
                ptrs,
                mask=offs_b[None, :] < LA1,
                other=0.0,
            )
        else:
            x = tl.load(ptrs)
    return x


def calc_bias_strides(
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None,
    batch: int,
    nheads_q: int,
    QSeq: int,
    KSeq: int,
) -> tuple[int, int, int]:
    """Calculate memory strides for bias tensor with broadcasting support.

    Validates bias tensor dimensions and computes appropriate strides
    for batch and head dimensions, supporting broadcasting when dimensions are 1.

    Args:
        bias: Optional bias tensor with shape [batch, heads, QSeq, KSeq]
        batch: Expected batch size
        nheads_q: Number of query attention heads
        QSeq: Query sequence length
        KSeq: Key sequence length

    Returns:
        tuple: (stride_bz, stride_bh, stride_bm) memory strides

    Raises:
        ValueError: If bias dimensions are incompatible with expected shapes
    """
    if bias is not None:
        if not hasattr(bias, "strides"):
            strides = tuple(map(lambda x: x * bias.itemsize, get_strides(bias)))
        else:
            strides = bias.strides
        if bias.shape[2] != QSeq or bias.shape[3] != KSeq:
            raise ValueError(
                f"Bias tensor has incompatible sequence dimensions. "
                f"Expected shape [..., {QSeq}, {KSeq}], but got [..., {bias.shape[2]}, {bias.shape[3]}]. "
                f"Full bias shape: {bias.shape}"
            )
        if bias.shape[0] == 1:
            stride_bz = 0
        elif bias.shape[0] == batch:
            stride_bz = strides[0] // bias.itemsize
        else:
            raise ValueError(
                f"Batch dimension mismatch in bias tensor. "
                f"Expected either 1 (for broadcasting) or {batch} (batch size), "
                f"but got {bias.shape[0]}. Consider reshaping your bias tensor."
            )
        if bias.shape[1] == 1:
            stride_bh = 0
        elif bias.shape[1] == nheads_q:
            stride_bh = strides[1] // bias.itemsize
        else:
            raise ValueError(
                f"Head dimension mismatch in bias tensor. "
                f"Expected either 1 (for broadcasting) or {nheads_q} (number of heads), "
                f"but got {bias.shape[1]}. Check that your bias tensor matches the model configuration."
            )

        stride_bm = strides[2] // bias.itemsize
    else:
        stride_bz, stride_bh, stride_bm = 0, 0, 0
    return stride_bz, stride_bh, stride_bm


@ejit(static_argnames=["max_tokens"])
def attention_pack_with_static_shape(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    attention_mask: Bool[Array, "batch seq_len"],
    max_tokens: int | None = None,
) -> Float[Array, "1 max_tokens num_heads head_dim"]:
    """Pack attention tensor by removing padding based on attention mask.

    Compacts a padded tensor into a contiguous representation by removing
    tokens where the attention mask is False. This enables efficient
    variable-length sequence processing.

    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
        attention_mask: Boolean mask indicating valid tokens [batch, seq_len]
        max_tokens: Maximum number of tokens in output (default: batch * seq_len)

    Returns:
        Packed tensor of shape [1, max_tokens, num_heads, head_dim] where
        valid tokens are contiguously stored at the beginning

    Note:
        Uses a static maximum shape to be compatible with JIT compilation.
        The actual number of valid tokens is sum(attention_mask).
    """
    batch_size, seqlen = attention_mask.shape
    num_heads, head_dim = x.shape[2], x.shape[3]

    if max_tokens is None:
        max_tokens = batch_size * seqlen

    seqlens = jnp.sum(attention_mask, axis=1).astype(jnp.int32)
    offsets = jnp.zeros((batch_size,), dtype=jnp.int32)
    offsets = offsets.at[1:].set(jnp.cumsum(seqlens[:-1]))
    packed = jnp.zeros((1, max_tokens, num_heads, head_dim), dtype=x.dtype)
    batch_idx, pos_idx = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(seqlen), indexing="ij")

    batch_idx_flat = batch_idx.reshape(-1)
    pos_idx_flat = pos_idx.reshape(-1)

    valid_mask = pos_idx < seqlens[:, None]
    target_idx = jnp.where(
        valid_mask,
        offsets[:, None] + pos_idx,
        jnp.zeros_like(pos_idx),
    )
    target_idx_flat = target_idx.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)

    def process_token(i, packed_acc):
        """Copy a single token from the padded input to its packed position.

        Args:
            i: Flat index into the (batch * seq_len) token grid.
            packed_acc: Running packed output accumulator
                [1, max_tokens, num_heads, head_dim].

        Returns:
            Updated packed accumulator with token i placed at its target
            position if valid, otherwise unchanged.
        """
        b = batch_idx_flat[i]
        p = pos_idx_flat[i]
        t = target_idx_flat[i]
        valid = valid_mask_flat[i]
        packed_acc = jnp.where(valid, packed_acc.at[0, t].set(x[b, p]), packed_acc)

        return packed_acc

    packed = jax.lax.fori_loop(0, batch_size * seqlen, process_token, packed)
    return packed


def basic_attention_refrence(
    q: Float[Array, "batch seq_len_q num_heads head_dim"],
    k: Float[Array, "batch seq_len_k num_heads_kv head_dim"],
    v: Float[Array, "batch seq_len_k num_heads_kv head_dim"],
    attn_bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    query_padding_mask: Bool[Array, "batch seq_len_q"] | None = None,
    key_padding_mask: Bool[Array, "batch seq_len_k"] | None = None,
    dropout_prob: float = 0.0,
    dropout_key: jax.Array | None = None,
    window_size: tuple[int, int] = (-1, -1),
    causal: bool = False,
    softcap: float = 0.0,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Reference implementation of attention for testing and validation.

    Provides a standard JAX implementation of scaled dot-product attention
    with support for various masking options, useful for validating the
    optimized Triton kernels.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len_k, num_heads_kv, head_dim]
        v: Value tensor [batch, seq_len_k, num_heads_kv, head_dim]
        attn_bias: Optional attention bias tensor
        query_padding_mask: Boolean mask for query positions
        key_padding_mask: Boolean mask for key positions
        dropout_prob: Dropout probability for attention weights
        dropout_key: JAX random key for dropout
        window_size: Local attention window (left, right)
        causal: Whether to apply causal masking
        softcap: Soft capping value for attention scores

    Returns:
        jnp.ndarray: Attention output with same shape as queries
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    QSeq, KSeq = q.shape[1], k.shape[1]
    repeats = q.shape[2] // k.shape[2]
    if repeats > 1:
        k = jnp.repeat(k, repeats=repeats, axis=2)
        v = jnp.repeat(v, repeats=repeats, axis=2)
    d = q.shape[-1]
    q_scaled = q / math.sqrt(d)
    scores = jnp.einsum("bthd,bshd->bhts", q_scaled, k)
    if softcap > 0:
        scores = scores / softcap
        scores = jnp.tanh(scores)
        scores = scores * softcap
    if key_padding_mask is not None:
        key_mask = (~key_padding_mask).reshape(key_padding_mask.shape[0], 1, 1, KSeq)
        scores = jnp.where(key_mask, jnp.finfo(scores.dtype).min, scores)
    if window_size[0] >= 0 or window_size[1] >= 0:
        row_idx = jnp.arange(QSeq).reshape(-1, 1)
        col_idx = jnp.arange(KSeq)
        if key_padding_mask is None:
            sk = KSeq
        else:
            sk = jnp.sum(key_padding_mask, axis=-1).reshape(-1, 1, 1, 1, 1)
        if query_padding_mask is None:
            sq = QSeq
        else:
            sq = jnp.sum(query_padding_mask, axis=-1).reshape(-1, 1, 1, 1, 1)
        if window_size[0] < 0:
            local_mask = col_idx > row_idx + sk - sq + window_size[1]
        else:
            if key_padding_mask is None:
                sk_full = jnp.full_like(col_idx, KSeq)
            else:
                sk_full = sk
            local_mask = jnp.logical_or(
                col_idx > jnp.minimum(row_idx + sk - sq + window_size[1], sk_full),
                col_idx < row_idx + sk - sq - window_size[0],
            )
        scores = jnp.where(local_mask, jnp.finfo(scores.dtype).min, scores)
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    if window_size[0] >= 0 or window_size[1] >= 0:
        all_masked = jnp.all(local_mask, axis=-1, keepdims=True)
        attention = jnp.where(all_masked, 0.0, attention)
    if query_padding_mask is not None:
        query_mask = (~query_padding_mask).reshape(query_padding_mask.shape[0], 1, QSeq, 1)
        attention = jnp.where(query_mask, 0.0, attention)
    dropout_scaling = 1.0 / (1 - dropout_prob)
    if dropout_prob > 0 and dropout_key is not None:
        dropout_mask = jax.random.bernoulli(dropout_key, p=1 - dropout_prob, shape=attention.shape)
        attention_drop = attention * dropout_mask * dropout_scaling
    else:
        attention_drop = attention
    output = jnp.einsum("bhts,bshd->bthd", attention_drop, v)
    if query_padding_mask is not None:
        query_mask_expanded = (~query_padding_mask).reshape(
            query_padding_mask.shape[0],
            QSeq,
            1,
            1,
        )
        output = jnp.where(query_mask_expanded, 0.0, output)
    return output.astype(dtype_og)


@ejit(static_argnames=["max_tokens"])
def attention_pack_from_cu_static(
    x: Float[Array, "batch seq_max num_heads head_dim"],
    cum_seqlens: Int[Array, "batch_plus_one"],
    max_tokens: int | None = None,
) -> Float[Array, "1 max_tokens num_heads head_dim"]:
    """Pack variable-length batch using cumulative sequence lengths.

    Compacts a padded tensor into a contiguous representation using
    cumulative sequence lengths to determine valid token ranges for
    each batch element. Used for variable-length sequence processing.

    Args:
        x: Input tensor of shape [batch, seq_max, num_heads, head_dim]
        cum_seqlens: Cumulative sequence lengths of shape [batch + 1].
            cum_seqlens[i] gives start index, cum_seqlens[i+1] gives end index.
        max_tokens: Maximum number of tokens in output (default: batch * seq_max)

    Returns:
        Packed tensor of shape [1, max_tokens, num_heads, head_dim] where
        only the first cum_seqlens[-1] tokens contain valid data

    Note:
        Uses JIT-compatible static shapes. Tokens beyond the last valid
        sequence end are left as zeros.
    """
    B, S_max, H, D = x.shape
    if max_tokens is None:
        max_tokens = B * S_max

    out = jnp.zeros((1, max_tokens, H, D), dtype=x.dtype)

    def body_b(b, out_acc):
        """Pack tokens for a single batch element into the output buffer.

        Args:
            b: Batch index being processed.
            out_acc: Running packed output accumulator
                [1, max_tokens, num_heads, head_dim].

        Returns:
            Updated accumulator with batch element b's valid tokens
            copied to contiguous positions starting at cum_seqlens[b].
        """
        start = cum_seqlens[b]
        end = cum_seqlens[b + 1]
        L = end - start

        def body_p(p, acc):
            """Copy position p from batch element b if within sequence length.

            Args:
                p: Position index within the padded sequence.
                acc: Running packed output accumulator.

            Returns:
                Updated accumulator with token at position p written to
                its destination if p < sequence length, otherwise unchanged.
            """
            valid = p < L
            dst = start + p
            acc = jnp.where(valid, acc.at[0, dst].set(x[b, p]), acc)
            return acc

        out_acc = jax.lax.fori_loop(0, S_max, body_p, out_acc)
        return out_acc

    out = jax.lax.fori_loop(0, B, body_b, out)
    return out


@ejit(static_argnames=["seqlen", "batch_size"])
def attention_unpack_with_static_shape(
    x: Float[Array, "1 max_tokens num_heads head_dim"],
    cum_seqlens: Int[Array, "batch_plus_one"],
    batch_size: int,
    seqlen: int,
) -> Float[Array, "batch seqlen num_heads head_dim"]:
    """Unpack contiguous tensor back to padded batch format.

    Reverses the packing operation by distributing contiguous tokens back
    to their original batch positions using cumulative sequence lengths.
    Used after variable-length sequence processing.

    Args:
        x: Packed tensor of shape [1, max_tokens, num_heads, head_dim]
        cum_seqlens: Cumulative sequence lengths of shape [batch + 1].
            cum_seqlens[i] gives start index, cum_seqlens[i+1] gives end index.
        batch_size: Number of sequences in the batch
        seqlen: Padded sequence length for output

    Returns:
        Unpacked tensor of shape [batch, seqlen, num_heads, head_dim] where
        each sequence is padded to the specified seqlen

    Note:
        Tokens beyond the actual sequence length for each batch element
        are left as zeros (padding).
    """
    H, D = x.shape[2], x.shape[3]
    out = jnp.zeros((batch_size, seqlen, H, D), dtype=x.dtype)

    def body_b(b, out_acc):
        """Unpack tokens for a single batch element from the packed buffer.

        Args:
            b: Batch index being processed.
            out_acc: Running unpacked output accumulator
                [batch_size, seqlen, num_heads, head_dim].

        Returns:
            Updated accumulator with batch element b's tokens copied from
            their contiguous packed positions back to the padded layout.
        """
        start = cum_seqlens[b]
        end = cum_seqlens[b + 1]
        L = end - start

        def body_p(p, acc):
            """Copy position p from packed buffer to batch element b if valid.

            Args:
                p: Position index within the padded output sequence.
                acc: Running unpacked output accumulator.

            Returns:
                Updated accumulator with the token at packed position
                (start + p) written to (b, p) if p < sequence length,
                otherwise unchanged.
            """
            valid = p < L
            src = start + p
            acc = jnp.where(valid, acc.at[b, p].set(x[0, src]), acc)
            return acc

        out_acc = jax.lax.fori_loop(0, seqlen, body_p, out_acc)
        return out_acc

    out = jax.lax.fori_loop(0, batch_size, body_b, out)
    return out

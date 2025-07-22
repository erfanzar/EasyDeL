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


import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from easydel.utils.compiling_utils import ejit

from .._utils import get_strides


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
    bias: jnp.ndarray | None,
    batch: int,
    nheads_q: int,
    QSeq: int,
    KSeq: int,
) -> tuple[int, ...]:
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
    x: jnp.ndarray,
    attention_mask: jnp.ndarray,
    max_tokens: int | None = None,
) -> jnp.ndarray:
    """
    Pack attention tensor by removing padding based on attention mask.
    Uses a static maximum shape to be compatible with JIT.
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
        b = batch_idx_flat[i]
        p = pos_idx_flat[i]
        t = target_idx_flat[i]
        valid = valid_mask_flat[i]
        packed_acc = jnp.where(valid, packed_acc.at[0, t].set(x[b, p]), packed_acc)

        return packed_acc

    packed = jax.lax.fori_loop(0, batch_size * seqlen, process_token, packed)
    return packed


@ejit(static_argnames=["seqlen", "batch_size"])
def attention_unpack_with_static_shape(
    x: jnp.ndarray,
    cum_seqlens: jnp.ndarray,
    batch_size: int,
    seqlen: int,
) -> jnp.ndarray:
    """
    Unpack attention tensor by redistributing the packed values to their original positions.

    Args:
        x: Packed tensor of shape [1, packed_tokens, num_heads, head_dim]
        cum_seqlens: Cumulative sequence lengths, shape [batch_size+1]
        batch_size: Number of batches
        seqlen: Maximum sequence length

    Returns:
        Unpacked tensor of shape [batch_size, seqlen, num_heads, head_dim]
    """
    num_heads, head_dim = x.shape[2], x.shape[3]

    # Create output with static shape
    unpacked = jnp.zeros((batch_size, seqlen, num_heads, head_dim), dtype=x.dtype)

    # Process each batch
    def process_batch(b, unpacked_acc):
        start_idx = cum_seqlens[b]
        end_idx = cum_seqlens[b + 1]
        seq_len = end_idx - start_idx

        # Process each position in the sequence
        def process_position(p, acc):
            # Only copy if within valid sequence length
            valid = p < seq_len
            src_idx = start_idx + p

            # Update conditionally
            acc = jnp.where(valid, acc.at[b, p].set(x[0, src_idx]), acc)

            return acc

        # Process all positions in this batch
        unpacked_acc = jax.lax.fori_loop(0, seqlen, process_position, unpacked_acc)

        return unpacked_acc

    # Process all batches
    unpacked = jax.lax.fori_loop(0, batch_size, process_batch, unpacked)

    return unpacked


def basic_attention_refrence(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_bias: jnp.ndarray | None = None,
    query_padding_mask: jnp.ndarray | None = None,
    key_padding_mask: jnp.ndarray | None = None,
    dropout_prob: float = 0.0,
    dropout_key: jax.random.PRNGKey = None,
    window_size: tuple[int, int] = (-1, -1),
    causal: bool = False,
    softcap: float = 0.0,
):
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

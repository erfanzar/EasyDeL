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

"""Pallas TPU kernel for compressed-window decode attention (DeepSeek-V4).

The KV axis (sliding-window ring + compressed entries) is small and bounded,
so the whole KV of each slot fits in VMEM. The kernel therefore does a single
fused pass per slot rather than a flash-style online scan:

    grid = (batch,)   # one program per request slot; parallel over megacore

    per slot b:
        q_b   [num_heads, q_len, head_dim]
        kv_b  [kv_len, head_dim]                 (K == V, shared over heads)
        bias  [q_len, kv_len]                    (broadcast over heads)
        sink  [num_heads]
        ->  logits = scale * q_b @ kv_b^T + bias           [H, S, L]
            sink-augmented fp32 softmax over L (sink adds to the denominator
            only, never to the output)
            out   = probs @ kv_b                            [H, S, D]

Sinks are implemented for real here (unlike ``ragged_decode_attention``'s TPU
kernel, which ignores ``softmax_aux``): the per-head sink logit seeds the
running max and the softmax normaliser exactly as the XLA reference does.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams

_NEG_INF = float(jnp.finfo(jnp.float32).min)


_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
_VMEM_BUDGET_BYTES = 16 * 1024 * 1024
_MAX_BLOCK_BATCH = 8


def _round_up(x: int, multiple: int) -> int:
    """Round ``x`` up to the nearest multiple of ``multiple``."""
    return ((x + multiple - 1) // multiple) * multiple


def _pick_block_batch(batch: int, num_heads: int, q_len: int, kv_pad: int, d_pad: int) -> int:
    """Choose how many slots one program handles.

    Bounded by what fits in VMEM once double-buffering is counted: Mosaic
    prefetches step ``i+1`` while computing step ``i``, so every streamed
    operand is resident twice. Operands are upcast to f32 inside the kernel, so
    the resident KV block is budgeted at 4 bytes per element regardless of the
    input dtype.

    Args:
        batch: Number of slots.
        num_heads: Query heads.
        q_len: Queries per slot (1 at decode).
        kv_pad: KV length.
        d_pad: Padded head dim.

    Returns:
        Slots per program, in ``[1, min(batch, _MAX_BLOCK_BATCH)]``.
    """
    for block in range(min(batch, _MAX_BLOCK_BATCH), 0, -1):
        streamed = block * (2 * num_heads * q_len * d_pad + kv_pad * d_pad) * 4
        scratch = block * num_heads * q_len * kv_pad * 4 * 2  # logits + probs, both live
        if 2 * streamed + scratch <= _VMEM_BUDGET_BYTES:
            return block
    return 1


def _slot_attend(q_hsd, kv_ld, bias_sl, sink_h, *, softmax_scale: float, prec, num_heads: int, q_len: int):
    """Sink-augmented shared-KV attention for one slot, in collapsed 2-D form.

    Args:
        q_hsd: Queries ``[num_heads, q_len, head_dim]`` (f32).
        kv_ld: Shared KV ``[kv_len, head_dim]`` (f32).
        bias_sl: Additive bias ``[q_len, kv_len]`` (f32).
        sink_h: Per-head sink logits ``[num_heads]`` (f32).
        softmax_scale: Logit scale.
        prec: Dot precision.
        num_heads: Query heads.
        q_len: Queries per slot.

    Returns:
        Attention output ``[num_heads, q_len, head_dim]`` (f32).
    """
    q = q_hsd.reshape(num_heads * q_len, -1)  # [H*S, D]
    bias = jnp.broadcast_to(bias_sl[None, :, :], (num_heads, q_len, bias_sl.shape[-1])).reshape(
        num_heads * q_len, -1
    )  # [H*S, L]
    sink = jnp.broadcast_to(sink_h[:, None], (num_heads, q_len)).reshape(num_heads * q_len, 1)  # [H*S, 1]

    # logits[hs, l] = scale * sum_d q[hs, d] * kv[l, d] + bias[hs, l]
    logits = jax.lax.dot_general(q, kv_ld, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    logits = logits * softmax_scale + bias  # [H*S, L]

    m = jnp.maximum(jnp.max(logits, axis=-1, keepdims=True), sink)  # [H*S, 1]
    p = jnp.exp(logits - m)  # [H*S, L]
    denom = jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(sink - m)  # [H*S, 1]
    denom = jnp.where(denom == 0.0, 1.0, denom)
    p = p / denom

    # out[hs, d] = sum_l p[hs, l] * kv[l, d]
    out = jax.lax.dot_general(p, kv_ld, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    return out.reshape(num_heads, q_len, -1)


def _compressed_window_decode_kernel(q_ref, kv_ref, bias_ref, sink_ref, o_ref, *, softmax_scale: float):
    """Sink-augmented shared-KV attention over ``block_batch`` slots.

    Slots are unrolled rather than expressed as one batched ``dot_general``:
    Mosaic lowers a dot with a leading batch dimension poorly here (measured
    0.85x -> 0.46x median), while unrolling keeps the efficient 2-D dot per slot
    and still amortises the fixed cost of a grid step across the block.

    Args:
        q_ref: Query block ``[block_batch, num_heads, q_len, head_dim]``.
        kv_ref: Shared KV block ``[block_batch, kv_len, head_dim]``.
        bias_ref: Additive fp32 bias ``[block_batch, q_len, kv_len]``.
        sink_ref: Per-head sink logits ``[num_heads]`` (fp32).
        o_ref: Output block ``[block_batch, num_heads, q_len, head_dim]``.
        softmax_scale: Logit scale.
    """
    block_batch, num_heads, q_len, _ = q_ref.shape
    prec = jax.lax.Precision.HIGHEST if q_ref.dtype == jnp.float32 else jax.lax.Precision.DEFAULT
    sink = sink_ref[...].astype(jnp.float32)  # [H]

    for i in range(block_batch):
        out = _slot_attend(
            q_ref[i].astype(jnp.float32),
            kv_ref[i].astype(jnp.float32),
            bias_ref[i].astype(jnp.float32),
            sink,
            softmax_scale=softmax_scale,
            prec=prec,
            num_heads=num_heads,
            q_len=q_len,
        )
        o_ref[i] = out.astype(o_ref.dtype)


@ejit(static_argnames=["softmax_scale", "fwd_params"])
def compressed_window_decode_tpu(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Pallas TPU compressed-window decode attention.

    Pads the head dim to a multiple of 128 (the TPU lane width) with zero
    feature columns, which cannot perturb the dot products, then launches one
    Pallas program per block of slots. The KV length is deliberately left
    unpadded.

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 bias ``[batch, q_len, kv_len]`` (head-broadcast).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Unused tiling hint (the KV axis is not blocked).

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.

    Note:
        The batch axis is padded up to a whole number of blocks. Pad slots get
        an all ``-inf`` bias, so their softmax rests on the sink alone and stays
        finite; their outputs are sliced off before returning.
    """
    del fwd_params
    batch, num_heads, q_len, head_dim = query.shape
    kv_len = kv.shape[1]
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)

    d_pad = _round_up(head_dim, 128)

    if softmax_aux is None:
        sink = jnp.full((num_heads,), _NEG_INF, dtype=jnp.float32)
    else:
        sink = softmax_aux.astype(jnp.float32).reshape(num_heads)

    q_p = query
    kv_p = kv
    if d_pad != head_dim:
        q_p = jnp.pad(q_p, ((0, 0), (0, 0), (0, 0), (0, d_pad - head_dim)))
        kv_p = jnp.pad(kv_p, ((0, 0), (0, 0), (0, d_pad - head_dim)))
    kv_pad = kv_len
    block_batch = _pick_block_batch(batch, num_heads, q_len, kv_pad, d_pad)
    pad_b = _round_up(batch, block_batch) - batch
    if pad_b:
        q_p = jnp.pad(q_p, ((0, pad_b), (0, 0), (0, 0), (0, 0)))
        kv_p = jnp.pad(kv_p, ((0, pad_b), (0, 0), (0, 0)))
        bias = jnp.pad(bias, ((0, pad_b), (0, 0), (0, 0)), constant_values=_NEG_INF)
    batch_p = batch + pad_b

    out = pl.pallas_call(
        functools.partial(_compressed_window_decode_kernel, softmax_scale=scale),
        grid=(batch_p // block_batch,),
        in_specs=[
            pl.BlockSpec((block_batch, num_heads, q_len, d_pad), lambda b: (b, 0, 0, 0)),
            pl.BlockSpec((block_batch, kv_pad, d_pad), lambda b: (b, 0, 0)),
            pl.BlockSpec((block_batch, q_len, kv_pad), lambda b: (b, 0, 0)),
            pl.BlockSpec((num_heads,), lambda b: (0,)),
        ],
        out_specs=pl.BlockSpec((block_batch, num_heads, q_len, d_pad), lambda b: (b, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((batch_p, num_heads, q_len, d_pad), kv.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
    )(q_p, kv_p, bias, sink)
    if pad_b:
        out = out[:batch]

    if d_pad != head_dim:
        out = out[..., :head_dim]
    return out

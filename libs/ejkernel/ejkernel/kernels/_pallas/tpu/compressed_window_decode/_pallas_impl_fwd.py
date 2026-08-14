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


def _round_up(x: int, multiple: int) -> int:
    """Round ``x`` up to the nearest multiple of ``multiple``."""
    return ((x + multiple - 1) // multiple) * multiple


def _compressed_window_decode_kernel(q_ref, kv_ref, bias_ref, sink_ref, o_ref, *, softmax_scale: float):
    """Single-pass sink-augmented shared-KV attention for one slot.

    Args:
        q_ref: Query block ``[num_heads, q_len, head_dim]`` (fp compute dtype).
        kv_ref: Shared KV block ``[kv_len, head_dim]``.
        bias_ref: Additive fp32 bias block ``[q_len, kv_len]``.
        sink_ref: Per-head sink logits ``[num_heads]`` (fp32).
        o_ref: Output block ``[num_heads, q_len, head_dim]`` (written in place).
        softmax_scale: Logit scale.
    """
    q = q_ref[...].astype(jnp.float32)  # [H, S, D]
    kv = kv_ref[...].astype(jnp.float32)  # [L, D]
    bias = bias_ref[...].astype(jnp.float32)  # [S, L]
    sink = sink_ref[...].astype(jnp.float32)  # [H]

    # Same dtype-aware policy as the full-sequence sibling kernel: the operands
    # are upcast to f32 above, and an f32 dot at DEFAULT precision is a single
    # bf16 MXU pass -- which cost this kernel ~3e-3 against the fp32 dense
    # reference. HIGHEST buys the multi-pass f32 the reference math assumes.
    # bf16 inputs keep DEFAULT: the data is already at the bf16 floor there, so
    # extra passes would be pure cost.
    prec = jax.lax.Precision.HIGHEST if q_ref.dtype == jnp.float32 else jax.lax.Precision.DEFAULT

    # logits[h, s, l] = scale * sum_d q[h, s, d] * kv[l, d] + bias[s, l]
    logits = jax.lax.dot_general(q, kv, (((2,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    logits = logits * softmax_scale + bias[None, :, :]  # [H, S, L]

    sink_hs = sink[:, None]  # [H, 1] broadcast over S
    m = jnp.maximum(jnp.max(logits, axis=-1), sink_hs)  # [H, S]
    p = jnp.exp(logits - m[..., None])  # [H, S, L]
    denom = jnp.sum(p, axis=-1) + jnp.exp(sink_hs - m)  # [H, S]
    denom = jnp.where(denom == 0.0, 1.0, denom)
    p = p / denom[..., None]

    # out[h, s, d] = sum_l p[h, s, l] * kv[l, d]
    out = jax.lax.dot_general(p, kv, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    o_ref[...] = out.astype(o_ref.dtype)


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

    Pads the KV length and head dim up to a multiple of 128 (the TPU lane
    width) — the pad KV rows carry a ``-inf`` bias so they cannot contribute,
    and the pad feature columns are zeros so they do not perturb the dot
    products — then launches one Pallas program per slot.

    Args:
        query: Queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared KV axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 bias ``[batch, q_len, kv_len]`` (head-broadcast).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Unused tiling hint (the KV axis is not blocked).

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    del fwd_params
    batch, num_heads, q_len, head_dim = query.shape
    kv_len = kv.shape[1]
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)

    kv_pad = _round_up(kv_len, 128)
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
    if kv_pad != kv_len:
        kv_p = jnp.pad(kv_p, ((0, 0), (0, kv_pad - kv_len), (0, 0)))
        bias = jnp.pad(bias, ((0, 0), (0, 0), (0, kv_pad - kv_len)), constant_values=_NEG_INF)

    out = pl.pallas_call(
        functools.partial(_compressed_window_decode_kernel, softmax_scale=scale),
        grid=(batch,),
        in_specs=[
            pl.BlockSpec((None, num_heads, q_len, d_pad), lambda b: (b, 0, 0, 0)),
            pl.BlockSpec((None, kv_pad, d_pad), lambda b: (b, 0, 0)),
            pl.BlockSpec((None, q_len, kv_pad), lambda b: (b, 0, 0)),
            pl.BlockSpec((num_heads,), lambda b: (0,)),
        ],
        out_specs=pl.BlockSpec((None, num_heads, q_len, d_pad), lambda b: (b, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, num_heads, q_len, d_pad), kv.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(q_p, kv_p, bias, sink)

    if d_pad != head_dim:
        out = out[..., :head_dim]
    return out

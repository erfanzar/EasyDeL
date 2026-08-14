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

"""XLA reference for compressed-window decode attention (DeepSeek-V4).

Shared-KV (``K == V``) attention with an explicit per-``(slot, query, kv)``
additive bias and per-head learnable attention sinks (gpt-oss style: an extra
softmax logit column whose probability mass is discarded before weighting the
values). This is the numerical reference every backend must match; it is a
faithful transcription of the dense math in
``easydel/modules/deepseek_v4/modeling_deepseek_v4.py::DeepseekV4Attention._attend``.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float


def compressed_window_decode_xla(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Sink-augmented shared-KV attention (XLA reference).

    Computes, for every batch row ``b``, head ``h``, and query position ``s``::

        logits[b, h, s, l] = scale * (q[b, h, s, :] . kv[b, l, :]) + bias[b, s, l]
        p                  = softmax over l of [logits, sink[h]]   (fp32)
        out[b, h, s, :]    = sum_l p[..., l] * kv[b, l, :]         (sink dropped)

    The single shared KV head is broadcast over all query heads (MQA). The
    softmax and the sink normalisation run in float32.

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``);
            the sliding-window ring concatenated with compressed entries.
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` broadcast over
            heads (ring-slot validity, compressed-entry causal thresholds, and
            indexer top-k gating already folded in).
        softmax_aux: Optional per-head sink logits ``[num_heads]``. ``None``
            disables sinks.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]`` in ``kv``'s
        dtype (matching the reference einsum promotion).
    """
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    # HIGHEST precision so this reference is the accurate f32 attention on TPU
    # (XLA's default matmul precision uses single-pass bf16 on the MXU, ~1e-2
    # error), which is what the Pallas kernel is validated against. The
    # full-sequence sibling has carried this since it was written; leaving it
    # off here put the decode reference ~2.6e-3 from the dense math, and since
    # this XLA path is the *default* decode platform, that surfaced as
    # DeepSeek-V4's cached decode diverging from its own stateless forward.
    logits = jnp.einsum("bhsd,bld->bhsl", query, kv, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
    logits = logits * scale + bias[:, None, :, :].astype(jnp.float32)

    kv_len = logits.shape[-1]
    if softmax_aux is not None:
        num_heads = logits.shape[1]
        sinks = jnp.broadcast_to(
            softmax_aux.astype(jnp.float32).reshape(1, num_heads, 1, 1),
            (*logits.shape[:3], 1),
        )
        combined = jnp.concatenate([logits, sinks], axis=-1)
    else:
        combined = logits
    combined = combined - jnp.max(combined, axis=-1, keepdims=True)
    probs = jax.nn.softmax(combined, axis=-1)
    scores = probs[..., :kv_len].astype(kv.dtype)
    return jnp.einsum("bhsl,bld->bhsd", scores, kv, precision=jax.lax.Precision.HIGHEST)

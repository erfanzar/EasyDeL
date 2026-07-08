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

"""XLA reference for compressed-window (full-sequence) attention (DeepSeek-V4).

This is the differentiable dense reference for DeepSeek-V4's training / prefill
forward: shared-KV (``K == V``) attention over the full sliding-window token KV
axis concatenated with the compressed entries, with an explicit additive bias
and per-head learnable attention sinks (gpt-oss style: an extra softmax logit
column whose probability mass is discarded before weighting the values).

It is a faithful transcription of the dense math in
``easydel/modules/deepseek_v4/modeling_deepseek_v4.py::DeepseekV4Attention._attend``
and is the correctness *and* gradient reference every backend must match. Unlike
the decode reference (which targets a single query per slot), this reference is
written to be cleanly differentiable so that ``jax.grad`` of it defines the
gradient contract for the Pallas kernel's ``custom_vjp``.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float


def compressed_window_attention_xla(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Sink-augmented shared-KV attention (dense, differentiable XLA reference).

    Computes, for every batch row ``b``, head ``h``, and query position ``s``::

        logits[b, h, s, l] = scale * (q[b, h, s, :] . kv[b, l, :]) + bias[b, s, l]
        p                  = softmax over l of [logits, sink[h]]   (fp32)
        out[b, h, s, :]    = sum_l p[..., l] * kv[b, l, :]         (sink dropped)

    The single shared KV head is broadcast over all query heads (MQA). The
    softmax and the sink normalisation run in float32. The additive ``bias`` is
    a structural / indexer-gated mask and is treated as a stop-gradient input by
    the differentiable kernel contract (it carries no learnable signal); this
    reference simply adds it, so ``jax.grad`` w.r.t. ``bias`` is defined but the
    Pallas ``custom_vjp`` returns a zero cotangent for it.

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``): the
            sliding-window token KVs concatenated with the compressed entries.
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` broadcast over heads
            (sliding-window causal validity, compressed-entry causal thresholds,
            and indexer top-k gating already folded in by the caller).
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
    # error), which is what the Pallas kernel is validated against.
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

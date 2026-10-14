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
gradient contract for the Pallas kernel's ``custom_jvp`` (including its
reverse-mode transpose).
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
    structural / indexer-gated input that may carry learnable indexer scores.
    This reference and the Pallas ``custom_jvp`` differentiate it, along with
    queries, shared KVs, and optional sinks. Structural masking does not imply
    a stop-gradient contract or a zero bias cotangent.

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
    # Preserve the bf16-dot/fp32-softmax boundary under TPU compiled AD.
    # Fusing across it produces NaN JVPs for masked bf16 inputs.
    logits = jax.lax.optimization_barrier(logits)
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


_MASK_MIN = float(jnp.finfo(jnp.float32).min)


def dedupe_entry_topk(entry_topk: Array) -> Array:
    """Sort each query's selected entry ids and turn repeats / negatives into ``-1`` (dense-mask semantics)."""
    ids = jnp.sort(entry_topk.astype(jnp.int32), axis=-1)
    repeat = jnp.concatenate([jnp.zeros_like(ids[..., :1], dtype=bool), ids[..., 1:] == ids[..., :-1]], axis=-1)
    return jnp.where(repeat | (ids < 0), -1, ids)


def compressed_window_attention_topk_xla(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    entry_topk: Array,
    softmax_aux: Float[Array, "num_heads"],
    token_valid: Array,
    *,
    softmax_scale: float | None,
    window: int,
    block_q: int = 64,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Sparse top-k form of the sink-augmented attention: window band + each query's selected entries only.

    Same value as :func:`compressed_window_attention_xla` with the dense bias ``[token mask | selection mask]``
    where token key ``k`` is visible to query ``q`` iff ``q - window < k <= q`` and ``token_valid[k]``, and entry
    ``t`` iff ``t`` is in ``entry_topk[q]`` (``-1`` = none; repeats count once). Queries are processed in
    rematerialized tiles that touch ``block_q + window - 1`` token keys and ``k`` gathered entries, so time and
    memory are ``O(S * (window + k))`` in both directions and no ``[S, T]`` bias or score tensor exists.
    Requires a sink (sink-free fully masked rows are uniform over *all* KVs, which a sparse attend cannot see).

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
        kv: ``[batch, q_len + entries, head_dim]``: the sequence's own token KVs, then the compressed entries.
        entry_topk: Selected entry ids ``[batch, q_len, k]`` (int, ``-1`` padded).
        softmax_aux: Per-head sink logits ``[num_heads]``.
        token_valid: Per-key token validity ``[batch, q_len]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        window: Sliding-window size.
        block_q: Queries per rematerialized tile.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]`` in ``kv``'s dtype.
    """
    batch, heads, seq, dim = query.shape
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    n_entries = kv.shape[1] - seq
    ids = dedupe_entry_topk(entry_topk)
    has_entries = n_entries > 0 and ids.shape[-1] > 0
    tile = max(1, min(block_q, seq))
    span = min(tile + window - 1, seq)
    tiles = -(-seq // tile)
    pad = tiles * tile - seq
    tokens = kv[:, :seq]
    entries = kv[:, seq:]
    q_p = jnp.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0)))
    ids_p = jnp.pad(ids, ((0, 0), (0, pad), (0, 0)), constant_values=-1)
    valid = token_valid > 0
    sink = softmax_aux.astype(jnp.float32).reshape(1, heads, 1, 1)
    hi = jax.lax.Precision.HIGHEST

    @jax.checkpoint
    def attend_tile(index):
        start = index * tile
        lo = jnp.clip(start - window + 1, 0, seq - span)
        q_t = jax.lax.dynamic_slice_in_dim(q_p, start, tile, axis=2)
        k_t = jax.lax.dynamic_slice_in_dim(tokens, lo, span, axis=1)
        key_ok = jax.lax.dynamic_slice_in_dim(valid, lo, span, axis=1)
        q_pos = start + jnp.arange(tile, dtype=jnp.int32)[:, None]
        k_pos = lo + jnp.arange(span, dtype=jnp.int32)[None, :]
        allowed = ((k_pos <= q_pos) & (k_pos > q_pos - window))[None] & key_ok[:, None, :]
        tok_logits = jnp.einsum("bhqd,btd->bhqt", q_t, k_t, precision=hi, preferred_element_type=jnp.float32)
        tok_logits = jax.lax.optimization_barrier(tok_logits)
        parts = [jnp.where(allowed[:, None], tok_logits * scale, _MASK_MIN)]
        gathered = None
        if has_entries:
            sel = jax.lax.dynamic_slice_in_dim(ids_p, start, tile, axis=1)  # [B, tile, k]
            gathered = jnp.take_along_axis(entries[:, None], jnp.maximum(sel, 0)[..., None], axis=2)
            ent_logits = jnp.einsum("bhqd,bqkd->bhqk", q_t, gathered, precision=hi, preferred_element_type=jnp.float32)
            ent_logits = jax.lax.optimization_barrier(ent_logits)
            parts.append(jnp.where((sel >= 0)[:, None], ent_logits * scale, _MASK_MIN))
        parts.append(jnp.broadcast_to(sink, (batch, heads, tile, 1)))
        combined = jnp.concatenate(parts, axis=-1)
        probs = jax.nn.softmax(combined - jnp.max(combined, axis=-1, keepdims=True), axis=-1).astype(kv.dtype)
        out = jnp.einsum("bhqt,btd->bhqd", probs[..., :span], k_t, precision=hi, preferred_element_type=jnp.float32)
        if gathered is not None:
            out = out + jnp.einsum(
                "bhqk,bqkd->bhqd", probs[..., span:-1], gathered, precision=hi, preferred_element_type=jnp.float32
            )
        return out.astype(kv.dtype)

    blocks = jax.lax.map(attend_tile, jnp.arange(tiles, dtype=jnp.int32))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, tiles * tile, dim)[:, :, :seq]

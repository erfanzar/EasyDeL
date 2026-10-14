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

"""Compressed-window attention interface (XLA reference, differentiable)."""

import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import compressed_window_attention_topk_xla, compressed_window_attention_xla

_MASK_MIN = float(jnp.finfo(jnp.float32).min)


def _selection_bias(entry_topk: Array, num_entries: int) -> Array:
    """Dense ``[batch, q_len, entries]`` bias opening each query's selected entries (``-1`` = none)."""
    batch, seq, _ = entry_topk.shape
    target = jnp.where(entry_topk >= 0, entry_topk, num_entries)
    opened = jnp.zeros((batch, seq, num_entries), dtype=bool)
    opened = opened.at[jnp.arange(batch)[:, None, None], jnp.arange(seq)[None, :, None], target].set(True, mode="drop")
    return jnp.where(opened, 0.0, _MASK_MIN).astype(jnp.float32)


def _expand_implicit_token_bias(entry_bias: Array, token_valid: Array, window: int) -> Array:
    """Dense ``[batch, q_len, q_len + entries]`` bias from the implicit token mask.

    Token key ``k`` is visible to query ``q`` iff ``k <= q``, ``k > q - window`` and ``token_valid[k]``; visible
    keys get ``0`` and the rest the finite ``_MASK_MIN`` sentinel, identical to a caller-built dense bias.
    """
    q_len = token_valid.shape[1]
    qi = jnp.arange(q_len, dtype=jnp.int32)[:, None]
    ki = jnp.arange(q_len, dtype=jnp.int32)[None, :]
    allowed = ((ki <= qi) & (ki > qi - window))[None] & (token_valid > 0)[:, None, :]
    token_bias = jnp.where(allowed, 0.0, _MASK_MIN).astype(jnp.float32)
    return jnp.concatenate([token_bias, entry_bias.astype(jnp.float32)], axis=2)


@kernel_registry.register("compressed_window_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def compressed_window_attention(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len bias_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    window: int = 0,
    token_kv_len: int = 0,
    token_valid: Array | None = None,
    entry_topk: Array | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Sink-augmented shared-KV full-sequence attention with a bias (XLA).

    Registered under ``"compressed_window_attention"`` for ``Platform.XLA`` /
    ``Backend.ANY`` (priority 0). This is the mandatory fallback and the
    numerical + gradient reference for DeepSeek-V4's training / prefill forward;
    the Pallas TPU kernel must match it in both value and gradient. Plain dense
    XLA math, so ``jax.grad`` differentiates it natively.

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``): the
            sliding-window token KVs concatenated with the compressed entries.
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` broadcast over heads
            (sliding-window causal, compressed-entry causal, indexer gating).
        softmax_aux: Optional per-head sink logits ``[num_heads]``. ``None``
            disables sinks.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Unused tiling hint (interface parity; the reference is dense).
        window: Accepted and ignored — the dense reference masks via ``bias``
            regardless, so it fully supports the band-skip call signature (it is
            deliberately NOT ``del``'d, which would mark it unsupported).
        token_kv_len: Accepted and ignored (interface parity) unless
            ``token_valid`` is given.
        token_valid: Optional per-key token validity ``[batch, token_kv_len]``.
            When given, ``bias`` holds only the compressed-entry columns and the
            token-region window / causal / padding mask is expanded here (same
            values as a caller-built dense bias). Requires ``window > 0`` and
            ``token_kv_len == q_len``.
        entry_topk: Optional selected entry ids ``[batch, q_len, k]`` (``-1`` padded;
            repeats count once), replacing the entry columns of ``bias`` (pass a
            zero-width ``bias``). Requires ``token_valid``. With a sink this runs
            the sparse ``O(S * (window + k))`` attend; without one the selection is
            expanded to the dense mask.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    del fwd_params  # window / token_kv_len are supported no-ops for the dense path
    if entry_topk is not None:
        if token_valid is None:
            raise ValueError("entry_topk requires token_valid (the implicit token mask).")
        if window <= 0 or token_kv_len != query.shape[2]:
            raise ValueError("token_valid requires window > 0 and token_kv_len == q_len.")
        if softmax_aux is not None:
            return compressed_window_attention_topk_xla(
                query, kv, entry_topk, softmax_aux, token_valid, softmax_scale=softmax_scale, window=window
            )
        bias = _selection_bias(entry_topk, kv.shape[1] - query.shape[2])
    if token_valid is not None:
        if window <= 0 or token_kv_len != query.shape[2]:
            raise ValueError("token_valid requires window > 0 and token_kv_len == q_len.")
        bias = _expand_implicit_token_bias(bias, token_valid, window)
    return compressed_window_attention_xla(
        query=query,
        kv=kv,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
    )

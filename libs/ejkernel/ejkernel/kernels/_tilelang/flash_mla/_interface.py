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

"""TileLang Flash Multi-head Latent Attention."""

from __future__ import annotations

import math

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ..deepseek_attn._impl import _kv_recon, _mla_attention_tilelang


@kernel_registry.register("flash_mla", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_mla(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
    logits_soft_cap: float | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    dropout_prob: float = 0.0,
    sliding_window: int | tuple[int, int] | None = None,
    softmax_dtype: DTypeLike | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Flash Multi-head Latent Attention (MLA) via native TileLang kernels.

    Projects the compressed KV latent ``key_value`` to full-rank ``K`` and
    ``V`` matrices via :func:`_kv_recon` (native tile-lang GEMM with VJP),
    optionally packs RoPE score-tail dimensions via
    :func:`_pack_shared_tail`, then runs :func:`_mla_attention_tilelang`.

    The three head-dim cases are:

    * **No RoPE** (``b_k=None``): ``score_dim = q_head_dim = qk_nope_head_dim``.
    * **RoPE, split query** (``b_k`` provided, ``b_q=None``):
      ``score_dim = q_head_dim = qk_nope_head_dim + qk_rope_head_dim``.
      The shared RoPE tail ``b_k`` is packed with the key latent.
    * **RoPE, separate query** (both ``b_q`` and ``b_k`` provided):
      ``score_dim = qk_nope_head_dim + qk_rope_head_dim``.
      ``b_q`` is packed with the query and ``b_k`` with the key.

    Args:
        query: ``(batch, seq_len, q_heads, q_head_dim)``.
        key_value: compressed KV latent ``(batch, seq_len, kv_lora_rank)``.
        w_kc: key projection weight ``(kv_lora_rank, kv_heads, qk_nope_head_dim)``.
        w_vc: value projection weight ``(kv_lora_rank, kv_heads, v_head_dim)``.
        b_q: optional RoPE query tail ``(batch, seq_len, qk_rope_head_dim)``.
        b_k: optional shared RoPE key tail ``(batch, seq_len, qk_rope_head_dim)``.
        softmax_scale: ``QK^T`` multiplier; defaults to
            ``1/sqrt(qk_nope_head_dim + rope_dim)`` when RoPE is present,
            else ``1/sqrt(qk_nope_head_dim)``.
        causal: apply upper-triangular causal mask.
        cu_seqlens: not yet supported (raises ``EjkernelRuntimeError``).
        attention_mask: optional boolean/int keep-mask; if ``bias`` is also
            provided ``attention_mask`` is set to ``None`` (bias takes
            precedence, matching the XLA reference behaviour).
        bias: optional additive logit bias.
        softmax_aux: optional attention-sink logits.
        logits_soft_cap: optional ``cap * tanh(logits / cap)`` soft cap.
        deterministic: if False and ``dropout_rng`` is provided, apply
            attention dropout with probability ``dropout_prob``.
        dropout_rng: PRNG key for dropout.
        dropout_prob: dropout probability.
        sliding_window: local-attention window.
        softmax_dtype: must be ``None`` or ``jnp.float32`` (the kernel always
            accumulates in float32); any other value raises
            ``EjkernelRuntimeError``.

    Returns:
        ``(batch, seq_len, q_heads, v_head_dim)`` attention output.

    Raises:
        EjkernelRuntimeError: on shape/rank inconsistencies, unsupported
            ``cu_seqlens`` / ``softmax_dtype``, or missing ``dropout_rng``.
    """
    if cu_seqlens is not None:
        raise EjkernelRuntimeError("tile-lang flash_mla does not yet support cu_seqlens.")
    if softmax_dtype is not None and jnp.dtype(softmax_dtype) != jnp.dtype(jnp.float32):
        raise EjkernelRuntimeError("tile-lang flash_mla accumulates softmax in float32.")
    if not deterministic and dropout_prob > 0.0 and dropout_rng is None:
        raise EjkernelRuntimeError("dropout_rng is required when deterministic=False and dropout_prob > 0.")
    if query.ndim != 4 or key_value.ndim != 3 or w_kc.ndim != 3 or w_vc.ndim != 3:
        raise EjkernelRuntimeError("tile-lang flash_mla expects rank-4 query, rank-3 key_value and rank-3 weights.")
    if query.shape[0] != key_value.shape[0] or query.shape[1] != key_value.shape[1]:
        raise EjkernelRuntimeError("query and key_value must share batch and sequence dimensions.")
    if key_value.shape[-1] != w_kc.shape[0] or key_value.shape[-1] != w_vc.shape[0]:
        raise EjkernelRuntimeError("key_value last dimension must match both projection ranks.")
    if w_kc.shape[1] != w_vc.shape[1]:
        raise EjkernelRuntimeError("w_kc and w_vc must use the same number of KV heads.")
    if query.shape[2] % w_kc.shape[1] != 0:
        raise EjkernelRuntimeError("q_heads must be divisible by kv_heads.")

    q_head_dim = query.shape[-1]
    qk_nope_dim = w_kc.shape[-1]
    v_head_dim = w_vc.shape[-1]
    if b_k is None and q_head_dim != qk_nope_dim:
        raise EjkernelRuntimeError("without RoPE, query head_dim must equal w_kc qk_nope_head_dim.")
    if b_k is not None:
        rope_dim = b_k.shape[-1]
        if b_q is None and q_head_dim != qk_nope_dim + rope_dim:
            raise EjkernelRuntimeError(
                "with b_k, query head_dim must equal qk_nope_head_dim + qk_rope_head_dim when b_q is omitted."
            )
        if b_q is not None and q_head_dim != qk_nope_dim:
            raise EjkernelRuntimeError("with b_q/b_k, query head_dim must equal w_kc qk_nope_head_dim.")
    if v_head_dim <= 0:
        raise EjkernelRuntimeError("tile-lang flash_mla requires a positive value head dimension.")

    effective_dim = qk_nope_dim + (0 if b_k is None else b_k.shape[-1])
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(effective_dim)
    effective_dropout = 0.0 if deterministic else float(dropout_prob)
    mask = None if bias is not None else attention_mask
    k, v = _kv_recon(key_value, w_kc, w_vc)
    return _mla_attention_tilelang(
        query,
        k,
        v,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=scale,
        causal=causal,
        bias=bias,
        attention_mask=mask,
        softmax_aux=softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        dropout_prob=effective_dropout,
        dropout_key=dropout_rng,
    )


__all__ = ["flash_mla"]

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

"""Flash MLA Pallas TPU kernel - public interface and registration."""

from __future__ import annotations

import math

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from ejkernel.errors import EjkernelRuntimeError

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import (
    ROPE_DECOUPLED,
    ROPE_FUSED,
    ROPE_NONE,
    flash_mla_impl,
)


@kernel_registry.register("flash_mla", Platform.PALLAS, Backend.TPU)
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
    """Flash Multi-head Latent Attention on TPU using Pallas.

    Computes MLA attention with on-the-fly KV reconstruction from a
    compressed latent representation.  The Pallas kernel tiles the
    computation to avoid materialising the full K/V tensors in HBM.

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim].
        key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
        w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
        w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
        b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
        b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
        softmax_scale: Scaling factor (default: 1/sqrt(effective_head_dim)).
        causal: Whether to apply causal masking.
        cu_seqlens: Cumulative seq lengths (not supported on TPU).
        attention_mask: Optional boolean mask (converted to additive bias).
        bias: Optional additive attention bias.
        softmax_aux: Attention sinks (not supported on TPU).
        logits_soft_cap: Optional tanh soft-cap for logits.
        deterministic: If True, disables dropout (default).
        dropout_rng: PRNG key for dropout (not supported on TPU).
        dropout_prob: Dropout probability (not supported on TPU).
        sliding_window: Sliding window as int or (left, right) tuple.
        softmax_dtype: Dtype for softmax (ignored, always float32 on TPU).

    Returns:
        Attention output [batch, seq_len, q_heads, v_head_dim].

    Raises:
        EjkernelRuntimeError: If unsupported features are requested.
        ValueError: On shape mismatches or invalid parameters.
    """
    reasons: list[str] = []
    if cu_seqlens is not None:
        reasons.append("cu_seqlens is not supported")
    if softmax_aux is not None:
        reasons.append("softmax_aux is not supported")
    if dropout_prob != 0.0:
        reasons.append("dropout_prob is not supported")
    if dropout_rng is not None:
        reasons.append("dropout_rng is not supported")
    if reasons:
        raise EjkernelRuntimeError("flash_mla (platform=pallas/tpu): " + "; ".join(reasons))
    del cu_seqlens, softmax_aux, dropout_rng, dropout_prob, deterministic, softmax_dtype

    window_tuple: tuple[int, int] | None = None
    if sliding_window is not None:
        if isinstance(sliding_window, int):
            if sliding_window < 0:
                raise ValueError("sliding_window must be non-negative.")
            window_tuple = (int(sliding_window), int(sliding_window))
        else:
            wl, wr = sliding_window
            if wl < 0 or wr < 0:
                raise ValueError("sliding_window bounds must be non-negative.")
            window_tuple = (int(wl), int(wr))

    if logits_soft_cap is not None and logits_soft_cap <= 0.0:
        raise ValueError("logits_soft_cap must be > 0.0.")

    batch, seq_q, q_heads, q_head_dim = query.shape
    batch_kv, kv_len, kv_lora_rank = key_value.shape
    if batch_kv != batch:
        raise ValueError(f"batch mismatch: query batch={batch} key_value batch={batch_kv}.")
    if kv_len != seq_q:
        raise ValueError(f"seq_len mismatch: query={seq_q} key_value={kv_len}.")
    if w_kc.shape[0] != kv_lora_rank or w_vc.shape[0] != kv_lora_rank:
        raise ValueError(f"kv_lora_rank mismatch: kv={kv_lora_rank} w_kc[0]={w_kc.shape[0]} w_vc[0]={w_vc.shape[0]}.")
    kv_heads = int(w_kc.shape[1])
    if int(w_vc.shape[1]) != kv_heads:
        raise ValueError(f"kv_heads mismatch: w_kc={kv_heads} w_vc={w_vc.shape[1]}.")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}).")
    d_nope = int(w_kc.shape[2])

    if b_k is None:
        rope_mode = ROPE_NONE
        if q_head_dim != d_nope:
            raise ValueError(f"No RoPE: query head_dim ({q_head_dim}) must equal w_kc head_dim ({d_nope}).")
        effective_dim = d_nope
    elif b_q is None:
        rope_mode = ROPE_FUSED
        rope_dim = int(b_k.shape[2])
        if q_head_dim != d_nope + rope_dim:
            raise ValueError(
                f"Fused RoPE: query head_dim ({q_head_dim}) must equal d_nope+rope_dim ({d_nope + rope_dim})."
            )
        effective_dim = d_nope + rope_dim
    else:
        rope_mode = ROPE_DECOUPLED
        rope_dim = int(b_k.shape[2])
        if b_q.shape[2] != rope_dim:
            raise ValueError(f"b_q/b_k rope_dim mismatch: {b_q.shape[2]} vs {rope_dim}.")
        if q_head_dim != d_nope:
            raise ValueError(f"Decoupled RoPE: query head_dim ({q_head_dim}) must equal d_nope ({d_nope}).")
        effective_dim = d_nope + rope_dim

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(effective_dim)

    if attention_mask is not None and bias is None:
        mask = attention_mask
        if mask.dtype != jnp.bool_:
            mask = mask.astype(jnp.bool_)
        if mask.shape[1] == 1:
            mask = jnp.broadcast_to(mask, (batch, q_heads, seq_q, kv_len))
        bias = jnp.where(mask, 0.0, jnp.finfo(jnp.float32).min).astype(jnp.float32)

    if bias is not None and bias.shape[1] == 1:
        bias = jnp.broadcast_to(bias, (batch, q_heads, seq_q, kv_len))

    block_q = min(128, seq_q)
    block_k = min(128, kv_len)
    block_b = 1

    while seq_q % block_q != 0 and block_q > 64:
        block_q //= 2
    while kv_len % block_k != 0 and block_k > 64:
        block_k //= 2

    q_t = query.transpose(0, 2, 1, 3)

    w_kc_t = jnp.transpose(w_kc, (1, 0, 2))
    w_vc_t = jnp.transpose(w_vc, (1, 0, 2))

    o = flash_mla_impl(
        q_t,
        key_value,
        w_kc_t,
        w_vc_t,
        b_q,
        b_k,
        bias,
        rope_mode,
        causal,
        softmax_scale,
        window_tuple,
        logits_soft_cap,
        block_b,
        block_q,
        block_k,
    )
    return o.transpose(0, 2, 1, 3)


__all__ = ("flash_mla",)

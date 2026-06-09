# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoski).
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

"""Flash Multi-head Latent Attention (MLA) interface for XLA backend.

This module provides a pure JAX/XLA implementation of Flash Multi-head Latent
Attention (MLA), a memory-efficient attention mechanism used in DeepSeek-V2
and similar architectures.

MLA achieves memory efficiency by using a low-rank factorization of the KV cache:
instead of storing full key and value tensors, it stores a compressed latent
representation that is projected on-the-fly during attention computation.

Key Features:
1. Low-rank KV compression: Stores [batch, seq_len, kv_lora_rank] instead of
   full [batch, seq_len, num_heads, head_dim] for keys and values
2. On-the-fly projection: Reconstructs K and V using learned projection matrices
3. RoPE integration: Supports separate RoPE components via b_q/b_k tensors
4. GQA support: Handles grouped-query attention with kv_heads < q_heads
5. Attention masking: Boolean mask and additive bias support
6. Logits soft-capping: Numerical stability via tanh-based capping (Gemma2/3)
7. Attention sinks: Auxiliary sink logits for StreamingLLM-style patterns
8. Sliding-window: Local attention for efficient long-context processing
9. Dropout: Optional attention-weight dropout during training

The attention computation is:
    K = key_value @ w_kc  (project to per-head keys)
    V = key_value @ w_vc  (project to per-head values)
    If using RoPE:
        K_full = concat(K_nope, K_rope) where K_rope comes from b_k
        Q_full = concat(Q_nope, Q_rope) where Q_rope comes from b_q or query
    output = softmax(Q @ K^T / sqrt(d)) @ V

Memory savings come from kv_lora_rank << num_heads * head_dim, typically
achieving 2-4x reduction in KV cache size.

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.flash_mla import flash_mla
    >>>
    >>> batch, seq_len, q_heads, kv_heads = 2, 1024, 32, 8
    >>> head_dim, kv_lora_rank = 64, 512
    >>>
    >>> query = jnp.ones((batch, seq_len, q_heads, head_dim))
    >>> key_value = jnp.ones((batch, seq_len, kv_lora_rank))
    >>> w_kc = jnp.ones((kv_lora_rank, kv_heads, head_dim))
    >>> w_vc = jnp.ones((kv_lora_rank, kv_heads, head_dim))
    >>>
    >>> output = flash_mla(query, key_value, w_kc, w_vc, causal=True)
    >>> output.shape
    (2, 1024, 32, 64)

Reference:
    DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
    https://arxiv.org/abs/2405.04434
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray  # noqa: F401


def _repeat_kv_for_gqa(x: Array, q_heads: int) -> Array:
    """Repeat KV heads to match query head count for GQA/MQA support.

    In grouped-query attention (GQA) and multi-query attention (MQA), the
    number of key-value heads is less than query heads. This function
    replicates KV heads to match the query head count.

    Args:
        x: Input tensor with KV heads at axis 2.
            Shape: [batch, seq_len, kv_heads, head_dim]
        q_heads: Number of query heads to match.

    Returns:
        Tensor with repeated KV heads to match q_heads.
            Shape: [batch, seq_len, q_heads, head_dim]

    Raises:
        ValueError: If q_heads is not divisible by kv_heads.
    """
    kv_heads = int(x.shape[2])
    if kv_heads == q_heads:
        return x
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}).")
    reps = q_heads // kv_heads
    return jnp.repeat(x, reps, axis=2)


def _normalize_softmax_aux(
    softmax_aux: Array,
    *,
    num_heads: int,
    dtype: jnp.dtype,
) -> Array:
    """Normalize softmax_aux into per-head sink logits [num_heads, num_sinks].

    Attention sinks are auxiliary logits that participate in softmax
    normalization but don't contribute to the output, letting the model
    absorb probability mass (StreamingLLM-style).

    Args:
        softmax_aux: Sink logits in one of these formats:
            - 1D (num_sinks,): Shared across all heads
            - 1D (num_heads,): Per-head single sink
            - 2D (num_heads, num_sinks): Per-head multiple sinks
            - 2D (1, num_sinks): Broadcast across all heads
        num_heads: Number of attention heads after GQA expansion.
        dtype: Target dtype for the output tensor.

    Returns:
        Normalized sink logits with shape [num_heads, num_sinks].

    Raises:
        ValueError: If softmax_aux has incompatible shape or rank > 2.
    """
    aux = jnp.asarray(softmax_aux, dtype=dtype)

    if aux.ndim == 1:
        if aux.shape[0] == num_heads:
            return aux[:, None]
        return jnp.broadcast_to(aux[None, :], (num_heads, aux.shape[0]))

    if aux.ndim == 2:
        if aux.shape[0] == num_heads:
            return aux
        if aux.shape[0] == 1:
            return jnp.broadcast_to(aux, (num_heads, aux.shape[1]))
        raise ValueError(f"softmax_aux first dim must be 1 or num_heads ({num_heads}); got {aux.shape[0]}.")

    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}.")


def _flash_mla_xla(
    query: Array,
    key_value: Array,
    w_kc: Array,
    w_vc: Array,
    b_q: Array | None,
    b_k: Array | None,
    softmax_scale: float | None,
    causal: bool,
    attention_mask: Array | None,
    bias: Array | None,
    softmax_aux: Array | None,
    logits_soft_cap: float | None,
    deterministic: bool,
    dropout_rng: Array | None,
    dropout_prob: float,
    sliding_window: int | tuple[int, int] | None,
    softmax_dtype: jnp.dtype | None,
) -> Array:
    """Core XLA implementation of Flash MLA attention.

    Performs the MLA attention computation by:
    1. Projecting compressed key_value to full keys and values
    2. Optionally incorporating RoPE components from b_q/b_k
    3. Applying logits soft-capping, sliding window, causal and custom masks
    4. Softmax with optional attention sinks and dropout
    5. Computing weighted values

    Args:
        query: Query tensor [batch, seq_q, q_heads, q_head_dim]
        key_value: Compressed KV latent [batch, kv_len, kv_lora_rank]
        w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim]
        w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim]
        b_q: Optional query RoPE component [batch, seq_q, qk_rope_head_dim]
        b_k: Optional key RoPE component [batch, kv_len, qk_rope_head_dim]
        softmax_scale: Attention scale factor. If None, computed from dimensions.
        causal: Apply causal masking if True.
        attention_mask: Boolean mask [batch, heads_or_1, seq_q, kv_len].
        bias: Additive bias [batch, heads_or_1, seq_q, kv_len].
        softmax_aux: Attention sink logits (see _normalize_softmax_aux).
        logits_soft_cap: Tanh soft-cap value for logits (Gemma2/3).
        deterministic: Disable dropout when True.
        dropout_rng: JAX PRNG key for dropout.
        dropout_prob: Attention-weight dropout probability.
        sliding_window: Local attention window size or (left, right) tuple.
        softmax_dtype: Dtype for softmax accumulation; defaults to float32.

    Returns:
        Attention output [batch, seq_q, q_heads, v_head_dim]

    Raises:
        ValueError: On shape mismatches or invalid configurations.
    """
    if softmax_dtype is None:
        softmax_dtype = jnp.float32
    if not deterministic and dropout_prob > 0.0 and dropout_rng is None:
        raise ValueError("dropout_rng must be provided when deterministic=False and dropout_prob > 0.")

    if query.ndim != 4:
        raise ValueError("query must have shape (batch, seq_len, q_heads, head_dim).")
    if key_value.ndim != 3:
        raise ValueError("key_value must have shape (batch, seq_len, kv_lora_rank).")
    if w_kc.ndim != 3 or w_vc.ndim != 3:
        raise ValueError("w_kc/w_vc must have shape (kv_lora_rank, kv_heads, head_dim).")

    batch, seq_len_q, q_heads, q_dim = query.shape
    batch_kv, seq_len_k, kv_lora_rank = key_value.shape
    if batch_kv != batch:
        raise ValueError(f"batch mismatch: query batch={batch} key_value batch={batch_kv}.")
    if seq_len_k != seq_len_q:
        raise ValueError(f"seq_len mismatch: query seq_len={seq_len_q} key_value seq_len={seq_len_k}.")

    if w_kc.shape[0] != kv_lora_rank or w_vc.shape[0] != kv_lora_rank:
        raise ValueError(
            "kv_lora_rank mismatch: "
            f"key_value last dim={kv_lora_rank}, w_kc[0]={w_kc.shape[0]}, w_vc[0]={w_vc.shape[0]}."
        )

    kv_heads = int(w_kc.shape[1])
    if int(w_vc.shape[1]) != kv_heads:
        raise ValueError(f"kv_heads mismatch: w_kc[1]={kv_heads} w_vc[1]={w_vc.shape[1]}.")

    d_nope = int(w_kc.shape[2])
    if b_k is None:
        if q_dim != d_nope:
            raise ValueError(
                "When b_k is None, query head_dim must equal w_kc head_dim. "
                f"Got query head_dim={q_dim}, w_kc head_dim={d_nope}."
            )
    else:
        if b_k.ndim != 3 or b_k.shape[0] != batch or b_k.shape[1] != seq_len_k:
            raise ValueError(
                f"b_k must have shape (batch, seq_len, qk_rope_head_dim). Got b_k shape={getattr(b_k, 'shape', None)}."
            )
        rope_dim = int(b_k.shape[2])
        expected_q_dim = d_nope + rope_dim
        if b_q is None:
            if q_dim != expected_q_dim:
                raise ValueError(
                    "When b_k is provided and b_q is None, query head_dim must be "
                    f"w_kc head_dim + rope_dim ({expected_q_dim}). Got {q_dim}."
                )
        else:
            if b_q.ndim != 3 or b_q.shape[0] != batch or b_q.shape[1] != seq_len_q:
                raise ValueError(
                    "b_q must have shape (batch, seq_len, qk_rope_head_dim). "
                    f"Got b_q shape={getattr(b_q, 'shape', None)}."
                )
            if int(b_q.shape[2]) != rope_dim:
                raise ValueError(f"b_q/b_k rope_dim mismatch: b_q={b_q.shape[2]} b_k={rope_dim}.")
            if q_dim != d_nope:
                raise ValueError(
                    "When both b_q and b_k are provided, query is expected to contain only the "
                    "non-RoPE part (qk_nope). "
                    f"Got query head_dim={q_dim}, expected {d_nope}."
                )

    q_f = query.astype(jnp.float32)
    kv_f = key_value.astype(jnp.float32)
    w_kc_f = w_kc.astype(jnp.float32)
    w_vc_f = w_vc.astype(jnp.float32)

    k_nope = jnp.einsum("btr,rhd->bthd", kv_f, w_kc_f, optimize=True)
    v = jnp.einsum("btr,rhd->bthd", kv_f, w_vc_f, optimize=True)

    k_nope = _repeat_kv_for_gqa(k_nope, q_heads)
    v = _repeat_kv_for_gqa(v, q_heads)

    if b_k is None:
        logits = jnp.einsum("bqhd,bkhd->bhqk", q_f, k_nope, optimize=True)
        d_scale = float(d_nope)
    elif b_q is None:
        rope_dim = int(b_k.shape[2])
        q_nope = q_f[..., :d_nope]
        q_rope = q_f[..., d_nope : d_nope + rope_dim]
        logits_nope = jnp.einsum("bqhd,bkhd->bhqk", q_nope, k_nope, optimize=True)
        logits_rope = jnp.einsum("bqhd,bkd->bhqk", q_rope, b_k.astype(jnp.float32), optimize=True)
        logits = logits_nope + logits_rope
        d_scale = float(d_nope + rope_dim)
    else:
        logits_nope = jnp.einsum("bqhd,bkhd->bhqk", q_f, k_nope, optimize=True)
        logits_rope = jnp.einsum("bqd,bkd->bqk", b_q.astype(jnp.float32), b_k.astype(jnp.float32), optimize=True)
        logits = logits_nope + logits_rope[:, None, :, :]
        d_scale = float(d_nope + int(b_k.shape[2]))

    scale = float(1.0 / math.sqrt(d_scale)) if softmax_scale is None else float(softmax_scale)
    logits = logits * jnp.asarray(scale, dtype=jnp.float32)

    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    if sliding_window is not None:
        left_w, right_w = (sliding_window, sliding_window) if isinstance(sliding_window, int) else sliding_window
        q_pos = jnp.arange(seq_len_q)[:, None]
        k_pos = jnp.arange(seq_len_k)[None, :]
        win_mask = (k_pos >= q_pos - left_w) & (k_pos <= q_pos + right_w)
        logits = jnp.where(win_mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    if causal:
        q_idx = jnp.arange(seq_len_q)[:, None]
        k_idx = jnp.arange(seq_len_k)[None, :]
        causal_mask = k_idx <= q_idx
        logits = jnp.where(causal_mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    if bias is not None:
        logits = logits + bias.astype(jnp.float32)
    elif attention_mask is not None:
        if attention_mask.dtype != jnp.bool_:
            attention_mask = attention_mask.astype(jnp.bool_)
        logits = jnp.where(attention_mask, logits, jnp.finfo(logits.dtype).min)

    if softmax_aux is not None:
        aux = _normalize_softmax_aux(softmax_aux, num_heads=q_heads, dtype=jnp.float32)
        sinks = jnp.broadcast_to(aux[None, :, None, :], (batch, q_heads, seq_len_q, aux.shape[-1]))
        combined = jnp.concatenate([logits, sinks], axis=-1)
        combined = combined.astype(softmax_dtype)
        combined = combined - jnp.max(combined, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined, axis=-1).astype(jnp.float32)
        weights = probs[..., :seq_len_k]
    else:
        weights = jax.nn.softmax(logits.astype(softmax_dtype), axis=-1).astype(jnp.float32)

    if not deterministic and dropout_prob > 0.0:
        keep_prob = 1.0 - dropout_prob
        keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape[-2:])
        weights = weights * (keep.astype(jnp.float32) / float(keep_prob))

    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v, optimize=True)
    return out.astype(query.dtype)


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
    attention_mask: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
    logits_soft_cap: float | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    dropout_prob: float = 0.0,
    sliding_window: int | tuple[int, int] | None = None,
    softmax_dtype: DTypeLike | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Flash Multi-head Latent Attention (MLA) using XLA backend.

    Computes attention using a compressed key-value representation, achieving
    significant memory savings compared to standard multi-head attention.
    Keys and values are reconstructed on-the-fly from the low-rank latent
    using learned projection matrices.

    The computation flow is:
        K_nope = key_value @ w_kc  (project to keys, no positional encoding)
        V = key_value @ w_vc  (project to values)

        If b_k is provided (RoPE case):
            K = [K_nope; K_rope] where K_rope from b_k
            Q = [Q_nope; Q_rope] where Q_rope from b_q or query

        output = softmax((Q @ K^T) * scale) @ V

    Args:
        query: Query tensor. If b_k is None, expected shape is
            [batch, seq_len, q_heads, qk_nope_head_dim].
            If b_k is provided and b_q is None, expected shape is
            [batch, seq_len, q_heads, qk_nope_head_dim + qk_rope_head_dim].
            If both b_q and b_k are provided, expected shape is
            [batch, seq_len, q_heads, qk_nope_head_dim].
        key_value: Compressed key-value latent representation.
            Shape: [batch, seq_len, kv_lora_rank]
        w_kc: Projection matrix for keys (non-RoPE component).
            Shape: [kv_lora_rank, kv_heads, qk_nope_head_dim]
        w_vc: Projection matrix for values.
            Shape: [kv_lora_rank, kv_heads, v_head_dim]
        b_q: Optional pre-computed query RoPE component (shared across heads).
            Shape: [batch, seq_len, qk_rope_head_dim]
        b_k: Optional pre-computed key RoPE component (shared across heads).
            Shape: [batch, seq_len, qk_rope_head_dim]
        softmax_scale: Scaling factor for attention scores. If None, computed
            automatically based on the effective head dimension.
        causal: Whether to apply causal masking for autoregressive attention.
        cu_seqlens: Cumulative sequence lengths for packed variable-length
            sequences. Currently not supported in XLA implementation.
        attention_mask: Optional boolean attention mask.
            Shape: [batch, 1 or q_heads, seq_len, kv_len].
            True = attend, False = ignore.  Ignored when ``bias`` is provided.
        bias: Optional additive attention bias (e.g. ALiBi, RPE).
            Shape: [batch, 1 or q_heads, seq_len, kv_len].
            Takes precedence over ``attention_mask``.
        softmax_aux: Optional attention sink logits for StreamingLLM-style
            patterns.  Supported shapes:
            - [num_sinks]: Shared across all heads
            - [q_heads]: One sink per head
            - [q_heads, num_sinks]: Multiple sinks per head
            These logits participate in softmax but don't contribute to output.
        logits_soft_cap: Optional float for capping attention logits via tanh:
            ``cap * tanh(logits / cap)``.  Prevents score explosion (Gemma2/3).
        deterministic: If True, disables dropout (default).
        dropout_rng: JAX PRNG key for dropout.  Required when
            ``deterministic=False`` and ``dropout_prob > 0``.
        dropout_prob: Dropout probability applied to attention weights.
        sliding_window: Optional sliding window attention constraint.
            - int: Symmetric window of that radius
            - (left, right): Asymmetric window
        softmax_dtype: Dtype for softmax accumulation.  Defaults to float32
            for numerical stability.

    Returns:
        Attention output tensor.
            Shape: [batch, seq_len, q_heads, v_head_dim]

    Raises:
        ValueError: On shape mismatches between query, key_value, and projections.
        NotImplementedError: If cu_seqlens is provided.

    Notes:
        - The kv_lora_rank is typically much smaller than num_heads * head_dim,
          providing significant memory savings (e.g., 512 vs 2048).
        - GQA is supported: q_heads can be a multiple of kv_heads.
        - RoPE components (b_q, b_k) are shared across heads for efficiency.

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, seq_len, q_heads, kv_heads = 2, 1024, 32, 8
        >>> head_dim, kv_lora_rank = 64, 512
        >>>
        >>> # Without RoPE
        >>> query = jnp.ones((batch, seq_len, q_heads, head_dim))
        >>> key_value = jnp.ones((batch, seq_len, kv_lora_rank))
        >>> w_kc = jnp.ones((kv_lora_rank, kv_heads, head_dim))
        >>> w_vc = jnp.ones((kv_lora_rank, kv_heads, head_dim))
        >>>
        >>> output = flash_mla(query, key_value, w_kc, w_vc, causal=True)
        >>> output.shape
        (2, 1024, 32, 64)
        >>>
        >>> # With logits soft cap and sliding window
        >>> output = flash_mla(
        ...     query, key_value, w_kc, w_vc,
        ...     causal=True, logits_soft_cap=50.0, sliding_window=256,
        ... )
    """
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens is not supported for XLA flash_mla yet.")
    return _flash_mla_xla(
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=softmax_scale,
        causal=causal,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        logits_soft_cap=logits_soft_cap,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        dropout_prob=dropout_prob,
        sliding_window=sliding_window,
        softmax_dtype=softmax_dtype,
    )

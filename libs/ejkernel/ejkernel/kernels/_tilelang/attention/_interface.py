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

"""Tile-lang ``attention`` — generic multi-head attention with weights output."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ..flash_attention._impl import _dropout_seed_buffer, _normalize_window, flash_attention_tilelang
from ._kernel import (
    make_attention_weights_bwd_dk_prim_func,
    make_attention_weights_bwd_dq_prim_func,
    make_attention_weights_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_WEIGHTS_CACHE: dict[tuple, Callable] = {}
_WEIGHTS_BWD_DQ_CACHE: dict[tuple, Callable] = {}
_WEIGHTS_BWD_DK_CACHE: dict[tuple, Callable] = {}
_WEIGHTS_LOCK = threading.Lock()


def _to_bhnd(x: jax.Array) -> jax.Array:
    """Transpose ``(B, N, H, D)`` to ``(B, H, N, D)`` for kernel-internal layout."""
    return jnp.transpose(x, (0, 2, 1, 3))


def _as_bias_buffer(x: jax.Array | None, query: jax.Array) -> tuple[jax.Array, bool]:
    """Normalise an optional bias to rank-4 without materialising score space.

    Args:
        x: optional ``bias`` broadcastable to ``(B, H, Sq, Sk)``.
        query: query array, used only to obtain the placeholder dtype.

    Returns:
        ``(buffer, use_bias)`` where ``buffer`` is rank-4 and ``use_bias``
        indicates whether a real bias was provided.
    """
    if x is None:
        return jnp.empty((1, 1, 1, 1), dtype=query.dtype), False
    if x.ndim > 4:
        raise ValueError(f"bias must be broadcastable to (B,H,Sq,Sk); got rank {x.ndim}.")
    return jnp.reshape(x, (1,) * (4 - x.ndim) + x.shape), True


def _as_mask_buffer(x: jax.Array | None) -> tuple[jax.Array, bool]:
    """Normalise an optional boolean/int mask to rank-4.

    Handles ranks 2, 3 and 4 by prepending unit dimensions.

    Returns:
        ``(buffer, use_mask)`` where ``buffer`` is rank-4 bool/int.

    Raises:
        ValueError: if ``x`` has rank < 2 or rank > 4.
    """
    if x is None:
        return jnp.empty((1, 1, 1, 1), dtype=jnp.bool_), False
    if x.ndim == 4:
        return x, True
    if x.ndim == 3:
        return jnp.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2])), True
    if x.ndim == 2:
        return jnp.reshape(x, (x.shape[0], 1, 1, x.shape[1])), True
    raise ValueError(f"Unsupported attention_mask shape: {x.shape}")


def _as_aux_buffer(x: jax.Array | None, query: jax.Array, num_heads: int, num_kv_heads: int) -> tuple[jax.Array, bool]:
    """Normalise an optional attention-sink (``softmax_aux``) array to rank-2.

    The buffer shape ``(AH, NS)`` accepted by the kernel is inferred from:

    * ``(num_sinks,)`` → ``(1, num_sinks)``
    * ``(num_heads, num_sinks)`` or ``(num_kv_heads, num_sinks)`` → kept as-is
    * ``(num_sinks,)`` that does not match either head count → ``(1, num_sinks)``

    Args:
        x: optional softmax-aux logits.
        query: query array (dtype used for placeholder only).
        num_heads: number of query heads.
        num_kv_heads: number of KV heads.

    Returns:
        ``(buffer, use_softmax_aux)`` where ``buffer`` is rank-2.

    Raises:
        ValueError: if ``x.ndim > 2`` or the first dimension is not
            in ``{1, num_kv_heads, num_heads}``.
    """
    if x is None:
        return jnp.empty((1, 1), dtype=query.dtype), False
    if x.ndim == 1:
        if x.shape[0] == num_heads or x.shape[0] == num_kv_heads:
            return jnp.reshape(x, (x.shape[0], 1)), True
        return jnp.reshape(x, (1, x.shape[0])), True
    if x.ndim == 2:
        if x.shape[0] not in (1, num_heads, num_kv_heads):
            raise ValueError(
                f"softmax_aux first dim must be 1, num_kv_heads ({num_kv_heads}) or num_q_heads ({num_heads}); "
                f"got {x.shape[0]}"
            )
        return x, True
    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {x.shape}")


def _validate_feature_shapes(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    bias_shape: tuple[int, int, int, int],
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    use_mask: bool,
) -> None:
    """Validate that bias/mask shapes broadcast correctly to ``(B, H, Sq, Sk)``.

    Raises:
        NotImplementedError: if the bias batch, seq_q or seq_k dimension does
            not match ``1`` or the corresponding actual dimension.
        NotImplementedError: if the bias heads dimension is not in
            ``{1, num_kv_heads, num_heads}``.
        ValueError: if the mask dimensions do not broadcast.
    """
    if use_bias:
        bb, bh, bq, bk = bias_shape
        if bb not in (1, batch) or bq not in (1, seq_len_q) or bk not in (1, seq_len_k):
            raise NotImplementedError("bias shape must broadcast to (batch, heads, seq_len, kv_len).")
        if bh not in (1, num_heads, num_kv_heads):
            raise NotImplementedError("bias heads wont match!")
    if use_mask:
        mb, _mh, mq, mk = mask_shape
        if mb not in (1, batch) or mq not in (1, seq_len_q) or mk not in (1, seq_len_k):
            raise ValueError(f"Unsupported attention_mask shape after normalization: {mask_shape}")


def _get_attention_weights_ffi(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    dtype: jnp.dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    softmax_aux_shape: tuple[int, int],
    softmax_aux_dtype,
    use_softmax_aux: bool,
    window: tuple[int, int] | None,
    dropout_prob: float,
    logits_soft_cap: float | None,
    block_q: int,
    block_k: int,
):
    """Build and cache the native dense-weights FFI call.

    ``block_q`` / ``block_k`` are **required** — the caller (operation
    layer via the kernel interface's ``weights_block_q`` /
    ``weights_block_k`` keyword args) is responsible for choosing them.
    The kernel does not pick from shape.
    """
    block_q = int(block_q)
    block_k = int(block_k)
    key = (
        batch,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_q,
        block_k,
        round(float(softmax_scale), 8),
        bool(causal),
        str(jnp.dtype(dtype)),
        tuple(bias_shape),
        str(jnp.dtype(bias_dtype)),
        bool(use_bias),
        tuple(mask_shape),
        str(jnp.dtype(mask_dtype)),
        bool(use_mask),
        tuple(softmax_aux_shape),
        str(jnp.dtype(softmax_aux_dtype)),
        bool(use_softmax_aux),
        None if window is None else tuple(window),
        round(float(dropout_prob), 8),
        None if logits_soft_cap is None else round(float(logits_soft_cap), 8),
    )
    with _WEIGHTS_LOCK:
        cached = _WEIGHTS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_attention_weights_prim_func(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_q=block_q,
            block_k=block_k,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            dtype=dtype,
            bias_shape=tuple(bias_shape),
            bias_dtype=bias_dtype,
            use_bias=bool(use_bias),
            mask_shape=tuple(mask_shape),
            mask_dtype=mask_dtype,
            use_mask=bool(use_mask),
            softmax_aux_shape=tuple(softmax_aux_shape),
            softmax_aux_dtype=softmax_aux_dtype,
            use_softmax_aux=bool(use_softmax_aux),
            window=window,
            dropout_prob=float(dropout_prob),
            logits_soft_cap=logits_soft_cap,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, seq_len_k), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _WEIGHTS_CACHE[key] = ffi
        return ffi


def _get_attention_weights_bwd_dq_ffi(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    logits_soft_cap: float | None,
    dtype: jnp.dtype,
):
    """Build and cache native dQ for dense attention weights."""
    block_q = 8
    block_d = 16
    key = (
        batch,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_q,
        block_d,
        round(float(softmax_scale), 8),
        None if logits_soft_cap is None else round(float(logits_soft_cap), 8),
        str(jnp.dtype(dtype)),
    )
    with _WEIGHTS_LOCK:
        cached = _WEIGHTS_BWD_DQ_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_attention_weights_bwd_dq_prim_func(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_q=block_q,
            block_d=block_d,
            softmax_scale=float(softmax_scale),
            logits_soft_cap=logits_soft_cap,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _WEIGHTS_BWD_DQ_CACHE[key] = ffi
        return ffi


def _get_attention_weights_bwd_dk_ffi(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    softmax_scale: float,
    logits_soft_cap: float | None,
    dtype: jnp.dtype,
):
    """Build and cache native dK for dense attention weights."""
    block_k = 8
    block_d = 16
    key = (
        batch,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_k,
        block_d,
        round(float(softmax_scale), 8),
        None if logits_soft_cap is None else round(float(logits_soft_cap), 8),
        str(jnp.dtype(dtype)),
    )
    with _WEIGHTS_LOCK:
        cached = _WEIGHTS_BWD_DK_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_attention_weights_bwd_dk_prim_func(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            head_dim=head_dim,
            block_k=block_k,
            block_d=block_d,
            softmax_scale=float(softmax_scale),
            logits_soft_cap=logits_soft_cap,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, num_kv_heads, seq_len_k, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _WEIGHTS_BWD_DK_CACHE[key] = ffi
        return ffi


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(6, 23)))
def _attention_weights_core(
    query_bhnd,
    key_bhnd,
    bias_buf,
    mask_buf,
    aux_buf,
    seed_buf,
    softmax_scale,
    causal,
    num_kv_heads,
    bias_shape,
    bias_dtype,
    use_bias,
    mask_shape,
    mask_dtype,
    use_mask,
    softmax_aux_shape,
    softmax_aux_dtype,
    use_softmax_aux,
    window,
    dropout_prob,
    logits_soft_cap,
    block_q,
    block_k,
):
    """Differentiable native dense attention weights."""
    batch, num_heads, seq_len_q, head_dim = query_bhnd.shape
    _, _, seq_len_k, _ = key_bhnd.shape
    ffi = _get_attention_weights_ffi(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        dtype=query_bhnd.dtype,
        bias_shape=bias_shape,
        bias_dtype=bias_dtype,
        use_bias=use_bias,
        mask_shape=mask_shape,
        mask_dtype=mask_dtype,
        use_mask=use_mask,
        softmax_aux_shape=softmax_aux_shape,
        softmax_aux_dtype=softmax_aux_dtype,
        use_softmax_aux=use_softmax_aux,
        window=window,
        dropout_prob=float(dropout_prob),
        logits_soft_cap=logits_soft_cap,
        block_q=int(block_q),
        block_k=int(block_k),
    )
    return ffi(query_bhnd, key_bhnd, bias_buf, mask_buf, aux_buf, seed_buf)


def _attention_weights_core_fwd(
    query_bhnd,
    key_bhnd,
    bias_buf,
    mask_buf,
    aux_buf,
    seed_buf,
    softmax_scale,
    causal,
    num_kv_heads,
    bias_shape,
    bias_dtype,
    use_bias,
    mask_shape,
    mask_dtype,
    use_mask,
    softmax_aux_shape,
    softmax_aux_dtype,
    use_softmax_aux,
    window,
    dropout_prob,
    logits_soft_cap,
    block_q,
    block_k,
):
    batch, num_heads, seq_len_q, head_dim = query_bhnd.shape
    _, _, seq_len_k, _ = key_bhnd.shape
    ffi = _get_attention_weights_ffi(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        dtype=query_bhnd.dtype,
        bias_shape=bias_shape,
        bias_dtype=bias_dtype,
        use_bias=use_bias,
        mask_shape=mask_shape,
        mask_dtype=mask_dtype,
        use_mask=use_mask,
        softmax_aux_shape=softmax_aux_shape,
        softmax_aux_dtype=softmax_aux_dtype,
        use_softmax_aux=use_softmax_aux,
        window=window,
        dropout_prob=float(dropout_prob),
        logits_soft_cap=logits_soft_cap,
        block_q=int(block_q),
        block_k=int(block_k),
    )
    weights = ffi(query_bhnd, key_bhnd, bias_buf, mask_buf, aux_buf, seed_buf)
    return weights, (query_bhnd, key_bhnd, weights)


def _attention_weights_core_bwd(
    softmax_scale,
    causal,
    num_kv_heads,
    bias_shape,
    bias_dtype,
    use_bias,
    mask_shape,
    mask_dtype,
    use_mask,
    softmax_aux_shape,
    softmax_aux_dtype,
    use_softmax_aux,
    window,
    dropout_prob,
    logits_soft_cap,
    block_q,
    block_k,
    residual,
    dweights,
):
    _ = block_q, block_k
    query_bhnd, key_bhnd, weights = residual
    batch, num_heads, seq_len_q, head_dim = query_bhnd.shape
    _, _, seq_len_k, _ = key_bhnd.shape
    dq_ffi = _get_attention_weights_bwd_dq_ffi(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        logits_soft_cap=logits_soft_cap,
        dtype=query_bhnd.dtype,
    )
    dk_ffi = _get_attention_weights_bwd_dk_ffi(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        logits_soft_cap=logits_soft_cap,
        dtype=query_bhnd.dtype,
    )
    dq = dq_ffi(query_bhnd, key_bhnd, weights, dweights)
    dk = dk_ffi(query_bhnd, key_bhnd, weights, dweights)
    return dq, dk, None, None, None, None


_attention_weights_core.defvjp(_attention_weights_core_fwd, _attention_weights_core_bwd)


def _attention_weights_tilelang(
    query,
    key,
    softmax_scale,
    causal,
    bias,
    attention_mask,
    softmax_aux,
    sliding_window,
    logits_soft_cap,
    dropout_prob,
    dropout_rng,
    *,
    block_q: int,
    block_k: int,
):
    """Compute the dense ``(B, H, Sq, Sk)`` attention probability matrix.

    Recomputes logits from ``Q`` and ``K`` (public ``(B, N, H, D)`` layout),
    applies all score-space modifiers via the native kernel, and returns
    softmax probabilities.  Differentiable w.r.t. ``query`` and ``key``
    through :func:`_attention_weights_core`.

    Args:
        query: ``(batch, seq_len_q, num_heads, head_dim)``.
        key: ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        softmax_scale: ``QK^T`` multiplier (pre-computed by the caller).
        causal: apply upper-triangular causal mask.
        bias: optional additive logit bias.
        attention_mask: optional boolean/int keep-mask.
        softmax_aux: optional attention-sink logits.
        sliding_window: optional local-attention window ``(left, right)``.
        logits_soft_cap: optional ``cap * tanh(logits / cap)`` soft cap.
        dropout_prob: attention-weight dropout probability.
        dropout_rng: optional ``uint32[2]`` dropout seed buffer.

    Returns:
        ``(batch, num_heads, seq_len_q, seq_len_k)`` probability tensor.
    """
    batch, seq_len_q, num_heads, _head_dim = query.shape
    _, seq_len_k, num_kv_heads, _ = key.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads}).")
    bias_buf, use_bias = _as_bias_buffer(bias, query)
    mask_buf, use_mask = _as_mask_buffer(attention_mask)
    aux_buf, use_softmax_aux = _as_aux_buffer(softmax_aux, query, num_heads, num_kv_heads)
    window = _normalize_window(sliding_window)
    _validate_feature_shapes(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        bias_shape=bias_buf.shape,
        use_bias=use_bias,
        mask_shape=mask_buf.shape,
        use_mask=use_mask,
    )
    seed_buf = _dropout_seed_buffer(None, dropout_rng)
    return _attention_weights_core(
        _to_bhnd(query),
        _to_bhnd(key),
        bias_buf,
        mask_buf,
        aux_buf,
        seed_buf,
        float(softmax_scale),
        bool(causal),
        int(num_kv_heads),
        bias_buf.shape,
        bias_buf.dtype,
        bool(use_bias),
        mask_buf.shape,
        mask_buf.dtype,
        bool(use_mask),
        aux_buf.shape,
        aux_buf.dtype,
        bool(use_softmax_aux),
        window,
        float(dropout_prob),
        logits_soft_cap,
        int(block_q),
        int(block_k),
    )


@kernel_registry.register("attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads vhead_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    dtype: DTypeLike | None = jnp.bfloat16,
    softmax_dtype: DTypeLike | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    *,
    weights_block_q: int = 64,
    weights_block_k: int = 64,
) -> tuple[
    Float[Array, "batch seq_len num_q_heads vhead_dim"],
    Float[Array, "batch num_heads seq_len kv_len"],
]:
    """Multi-head attention returning both the output and the dense weight matrix.

    The attention output is computed by :func:`flash_attention_tilelang`
    (online-softmax FlashAttention-2).  The dense ``(B, H, Sq, Sk)``
    probability matrix is computed by a separate native kernel that
    re-materialises the logits and applies the same score-space modifiers.

    Both paths use native tile-lang kernels with VJP support (dQ, dK, dV
    for the output; dQ, dK for the weights).

    Args:
        query: ``(batch, seq_len, num_q_heads, head_dim)``.
        key: ``(batch, kv_len, num_kv_heads, head_dim)``.
        value: ``(batch, kv_len, num_kv_heads, vhead_dim)``.
            Must satisfy ``vhead_dim == head_dim`` for this backend.
        attention_mask: optional boolean/int keep-mask broadcastable to
            ``(batch, num_heads, seq_len, kv_len)``.
        bias: optional additive logit bias with the same broadcast shape.
        init_bias: optional callable that lazily constructs the bias when
            ``bias`` is not supplied; called at most once.
        deterministic: if False and ``dropout_rng`` is provided, attention
            dropout is applied with probability ``dropout_prob``.
        dropout_rng: optional ``uint32[2]`` dropout seed (or legacy PRNGKey).
        softmax_aux: attention-sink logits ``(num_sinks,)`` or
            ``(num_heads, num_sinks)``.
        softmax_scale: ``QK^T`` multiplier; defaults to ``1/sqrt(head_dim)``.
        logits_soft_cap: ``cap * tanh(logits / cap)`` soft cap.
        dtype: dtype to cast ``q/k/v`` to before the kernels (default bfloat16).
            ``None`` leaves dtypes unchanged.
        softmax_dtype: accepted but ignored — the kernel always accumulates
            softmax in float32.
        dropout_prob: dropout probability (used when ``deterministic=False``
            and ``dropout_rng`` is provided).
        causal: apply upper-triangular causal mask.
        sliding_window: local-attention window (symmetric int or
            ``(left, right)``).
        fwd_params: Optional TileLang FlashAttention forward tile hints.
        bwd_params: Optional TileLang FlashAttention backward tile hints.

    Returns:
        A tuple ``(output, weights)`` where:

        * ``output``: ``(batch, seq_len, num_q_heads, vhead_dim)`` attention
          output in ``dtype`` (or original dtype if ``dtype=None``).
        * ``weights``: ``(batch, num_heads, seq_len, kv_len)`` softmax
          probability matrix (same dtype as ``query`` after casting).

    Raises:
        EjkernelRuntimeError: if ``vhead_dim != head_dim``.
    """
    if value.shape[-1] != query.shape[-1]:
        raise EjkernelRuntimeError("tile-lang attention requires head_dim == vhead_dim.")
    if bias is None and init_bias is not None:
        bias = init_bias()
    if dtype is not None:
        query = query.astype(dtype)
        key = key.astype(dtype)
        value = value.astype(dtype)
    if softmax_dtype is not None:
        jnp.dtype(softmax_dtype)

    drop_p = float(dropout_prob) if (not deterministic and dropout_rng is not None) else 0.0
    output = flash_attention_tilelang(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=causal,
        bias=bias,
        attention_mask=attention_mask,
        softmax_aux=softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        dropout_prob=drop_p,
        dropout_key=dropout_rng,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(query.shape[-1])
    weights = _attention_weights_tilelang(
        query,
        key,
        scale,
        causal,
        bias,
        attention_mask,
        softmax_aux,
        sliding_window,
        logits_soft_cap,
        drop_p,
        dropout_rng,
        block_q=int(weights_block_q),
        block_k=int(weights_block_k),
    )
    return output, weights


__all__ = ["attention"]

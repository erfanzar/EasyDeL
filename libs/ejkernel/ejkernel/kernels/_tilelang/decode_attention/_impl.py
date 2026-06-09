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

"""JAX glue for tile-lang decode attention.

* Short KV (``L < 512``): reuse the FA forward with the ``seq_len_q == 1``
  tile-picker path.
* Long KV (``L >= 512``): use the FlashDecoding-style split-K kernels in
  :mod:`._split_kernel` which spread the KV axis across more CTAs.
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ..flash_attention._impl import _flash_attention_fwd_only
from ._split_kernel import make_combine_prim_func, make_split_decode_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_SPLIT_THRESHOLD = 16384

_SPLIT_FFI_CACHE: dict[tuple, callable] = {}
_COMBINE_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_split_ffi(B, H, L, D, scale, dtype, *, num_splits: int, block_k: int):
    """Build (cached) split-K decode FFI.

    ``num_splits`` and ``block_k`` are **required** — the caller
    (operation layer or interface) picks them; this kernel does not
    pick from shape.
    """
    num_splits = int(num_splits)
    block_k = int(block_k)
    key = (B, H, L, D, num_splits, block_k, round(float(scale), 8), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _SPLIT_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_split_decode_prim_func(
            batch=B,
            num_heads=H,
            seq_len_kv=L,
            head_dim=D,
            num_splits=num_splits,
            block_k=block_k,
            softmax_scale=float(scale),
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((num_splits, B, H, D), jnp.float32),
                jax.ShapeDtypeStruct((num_splits, B, H), jnp.float32),
                jax.ShapeDtypeStruct((num_splits, B, H), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _SPLIT_FFI_CACHE[key] = ffi
        return ffi


def _get_combine_ffi(B, H, D, num_splits, dtype):
    key = (B, H, D, num_splits, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _COMBINE_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_combine_prim_func(
            batch=B,
            num_heads=H,
            head_dim=D,
            num_splits=num_splits,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, H, D), dtype),
                jax.ShapeDtypeStruct((B, H), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _COMBINE_FFI_CACHE[key] = ffi
        return ffi


def _decode_split_path(query, key_buffer, value_buffer, *, softmax_scale, num_splits: int, block_k: int):
    B, H, D = query.shape
    total = key_buffer.shape[0]
    L = total // B
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    k = key_buffer.reshape(B, L, H, D)
    v = value_buffer.reshape(B, L, H, D)

    split_ffi = _get_split_ffi(B, H, L, D, scale, query.dtype, num_splits=num_splits, block_k=block_k)
    o_partial, m_partial, l_partial = split_ffi(query, k, v)

    combine_ffi = _get_combine_ffi(B, H, D, num_splits, query.dtype)
    out, lse = combine_ffi(o_partial, m_partial, l_partial)
    return out, lse


def _decode_fa_path(query, key_buffer, value_buffer, *, softmax_scale):
    B, H, D = query.shape
    total = key_buffer.shape[0]
    L = total // B
    k = key_buffer.reshape(B, L, H, D)
    v = value_buffer.reshape(B, L, H, D)
    q = query[:, None, :, :]
    out_bnhd, lse_bhn = _flash_attention_fwd_only(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
    )
    out = out_bnhd[:, 0, :, :]
    lse_nat = lse_bhn[..., 0] * jnp.log(2.0).astype(lse_bhn.dtype)
    return out, lse_nat


def decode_attention_tilelang(
    query: jax.Array,
    key_buffer: jax.Array,
    value_buffer: jax.Array,
    *,
    softmax_scale: float | None = None,
    num_splits: int = 8,
    block_k: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Single-Q decode attention (forward only) with adaptive split-K routing.

    Routes to one of two kernels based on the effective per-batch KV length
    ``L = total_tokens // batch``:

    * ``L < 16384``: ``_decode_fa_path`` — uses the lean FlashAttention
      forward with ``seq_q=1`` (lower kernel-launch overhead for short KV).
    * ``L >= 16384``: ``_decode_split_path`` — FlashDecoding split-K using
      :func:`make_split_decode_prim_func` / :func:`make_combine_prim_func`
      (better SM utilisation for very long KV).

    Args:
        query: ``(batch, num_heads, head_dim)``.
        key_buffer: flat KV store ``(total_tokens, num_heads, head_dim)``
            where ``total_tokens`` must be divisible by ``batch``.
        value_buffer: same shape as ``key_buffer``.
        softmax_scale: ``QK^T`` multiplier; defaults to ``1/sqrt(head_dim)``.

    Returns:
        ``(output, lse)`` where:

        * ``output``: ``(batch, num_heads, head_dim)`` attention output.
        * ``lse``: ``(batch, num_heads)`` float32 natural-log log-sum-exp.

    Raises:
        RuntimeError: if the tile-lang FFI is unavailable.
        ValueError: if ``total_tokens % batch != 0``.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang decode_attention requires `tilelang` + `jax_tvm_ffi`.")
    B, _H, _D = query.shape
    total = key_buffer.shape[0]
    if total % B != 0:
        raise ValueError("decode_attention v0 requires total_tokens divisible by batch (contiguous layout).")
    L = total // B
    if L >= _SPLIT_THRESHOLD:
        return _decode_split_path(
            query,
            key_buffer,
            value_buffer,
            softmax_scale=softmax_scale,
            num_splits=int(num_splits),
            block_k=int(block_k),
        )
    return _decode_fa_path(
        query,
        key_buffer,
        value_buffer,
        softmax_scale=softmax_scale,
    )

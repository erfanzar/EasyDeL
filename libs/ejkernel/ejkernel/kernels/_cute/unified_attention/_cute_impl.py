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

"""Unified attention execution helpers for the CuTe platform entry point.

The CuTe unified-attention registration must keep full behavioral parity with
the other backends. This implementation uses Triton's optimized kernel path
and raises an explicit error when that path is unavailable.
"""

from __future__ import annotations

import os

import jax

from ejkernel.errors import EjkernelRuntimeError

try:  # pragma: no cover - import availability is environment-dependent.
    from ejkernel.kernels._triton.unified_attention._triton_impl_fwd import unified_attention_triton

    _HAS_TRITON_UNIFIED = True
except Exception:  # pragma: no cover
    unified_attention_triton = None  # type: ignore[assignment]
    _HAS_TRITON_UNIFIED = False

_TRUTHY = {"1", "true", "yes", "on"}


def _device_platform(x: jax.Array) -> str | None:
    """Return the device platform string for *x* (e.g. ``"gpu"``), or ``None`` on failure."""
    device_attr = getattr(x, "device", None)
    if device_attr is None:
        return None
    try:
        dev = device_attr() if callable(device_attr) else device_attr
    except Exception:
        return None
    return getattr(dev, "platform", None)


def _should_prefer_triton(queries: jax.Array) -> bool:
    """Return ``True`` when the Triton fast path should be used for this call.

    Returns ``False`` when Triton unified attention is not imported, when
    the environment variable ``EJKERNEL_CUTE_UNIFIED_ATTENTION_DISABLE_TRITON``
    is set to a truthy value, or when the device platform is not GPU/CUDA.
    """
    if not _HAS_TRITON_UNIFIED:
        return False

    force_disable = os.getenv("EJKERNEL_CUTE_UNIFIED_ATTENTION_DISABLE_TRITON", "").strip().lower()
    if force_disable in _TRUTHY:
        return False

    platform = _device_platform(queries)
    if platform is not None:
        return platform in {"gpu", "cuda"}
    return jax.default_backend() in {"gpu", "cuda"}


def unified_attention_cute(
    *,
    queries: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    block_tables: jax.Array,
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    seq_threshold_3d: int | None,
    num_par_softmax_segments: int | None,
    alibi_slopes: jax.Array | None,
    qq_bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    num_warps: int | None,
    num_stages: int | None,
) -> jax.Array:
    """Run unified attention through the fastest available implementation.

    Args:
        queries: Packed query tensor with shape
            ``(total_tokens, num_q_heads, head_dim)``.
        key_cache: Paged key cache with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Paged value cache with the same shape as ``key_cache``.
        block_tables: Per-sequence page table with shape
            ``(num_seqs, max_blocks_per_seq)``.
        kv_lens: Per-sequence KV lengths with shape ``(num_seqs,)``.
        query_start_loc: Prefix offsets for packed queries with shape
            ``(num_seqs + 1,)``.
        softmax_scale: Optional scaling factor for QK scores. Defaults to
            ``1 / sqrt(head_dim)`` when ``None``.
        causal: Whether to apply causal masking. Must be ``True``.
        sliding_window: Optional local-attention window size.
        logits_soft_cap: Optional tanh soft-capping value for logits.
        seq_threshold_3d: Triton decode-kernel selection threshold.
        num_par_softmax_segments: Triton segmented-softmax factor.
        alibi_slopes: Optional ALiBi slopes per query head.
        qq_bias: Optional query-query bias matrix.
        softmax_aux: Optional sink logits per query head.
        num_warps: Optional Triton kernel override.
        num_stages: Optional Triton kernel override.

    Returns:
        Attention output tensor with the same shape and dtype as ``queries``.

    Raises:
        NotImplementedError: If ``causal`` is ``False``.
    """
    if not causal:
        raise NotImplementedError("unified_attention (platform=cute) only supports causal attention.")

    if _should_prefer_triton(queries):
        return unified_attention_triton(
            queries=queries,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            kv_lens=kv_lens,
            query_start_loc=query_start_loc,
            softmax_scale=softmax_scale,
            causal=causal,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            seq_threshold_3d=seq_threshold_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            alibi_slopes=alibi_slopes,
            qq_bias=qq_bias,
            softmax_aux=softmax_aux,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    del (
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        query_start_loc,
        softmax_scale,
        sliding_window,
        logits_soft_cap,
        seq_threshold_3d,
        num_par_softmax_segments,
        alibi_slopes,
        qq_bias,
        softmax_aux,
        num_warps,
        num_stages,
    )
    raise EjkernelRuntimeError(
        "unified_attention (platform=cute) requires Triton unified_attention; fallback paths are disabled."
    )

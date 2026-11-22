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

"""Tile-lang Native Sparse Attention (NSA).

The selected-block sparse primitive is native TileLang for forward and
backward. The higher-level compression/top-k/gated NSA wrapper remains
explicitly gated until those stages are implemented natively too.
"""

from __future__ import annotations

import math

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from .._gate_impl import head_gate_tilelang
from ._impl import apply_sparse_attention_tilelang


@kernel_registry.register("apply_native_sparse_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def apply_native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: (
        Int[Array, "batch seq_len num_kv_heads num_selected_blocks"]
        | Int[Array, "batch num_kv_heads num_blocks num_selected_blocks"]
    ),
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | Int[Array, "batch num_kv_heads num_blocks"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Apply native selected-block causal sparse attention (TileLang GPU backend).

    Runs the selected-block sparse forward pass (and VJP backward pass) using
    the TileLang kernels in ``_impl.py``.  Only the padded ``(B, T, HQ, D)``
    input layout is supported; packed/ragged inputs are not yet implemented.

    Args:
        query: ``[batch, seq_len, num_q_heads, head_dim]`` query tensor.
        key: ``[batch, seq_len, num_kv_heads, head_dim]`` key tensor.
        value: ``[batch, seq_len, num_kv_heads, head_dim]`` value tensor.
        block_indices: Selected KV block indices.  Accepted shapes:
            ``[batch, seq_len, num_kv_heads, num_selected_blocks]``
            (token layout) or
            ``[batch, num_kv_heads, num_blocks, num_selected_blocks]``
            (block layout).
        block_counts: Number of valid entries in *block_indices* per position.
            Accepted as:
            - ``[batch, seq_len, num_kv_heads]`` (token layout),
            - ``[batch, num_kv_heads, num_blocks]`` (block layout),
            - a plain ``int`` for a uniform static count.
            Defaults to 16.
        block_size: Tokens per KV block.  Defaults to 64.
        softmax_scale: Attention temperature.  Defaults to
            ``1 / sqrt(head_dim)``.
        cu_seqlens: Not yet supported; raises ``EjkernelRuntimeError`` if
            provided.
        token_indices: Not yet supported; raises ``EjkernelRuntimeError`` if
            provided.
        block_k: Accepted for API compatibility with Triton; ignored by TileLang.
        block_v: Accepted for API compatibility with Triton; ignored by TileLang.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``[batch, seq_len, num_q_heads, head_dim]`` attention output.

    Raises:
        EjkernelRuntimeError: If *token_indices* or *cu_seqlens* are provided.
    """
    if token_indices is not None:
        raise EjkernelRuntimeError("tile-lang apply_native_sparse_attention does not yet support token_indices.")
    if cu_seqlens is not None:
        raise EjkernelRuntimeError("tile-lang apply_native_sparse_attention does not yet support cu_seqlens.")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(query.shape[-1])
    return apply_sparse_attention_tilelang(
        query,
        key,
        value,
        block_indices,
        block_counts,
        block_size,
        scale,
    )


@kernel_registry.register("native_sparse_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
    g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Native selected-block sparse attention with optional gating (TileLang GPU backend).

    Higher-level wrapper around :func:`apply_native_sparse_attention`.
    When *g_slc* is provided, the output is element-wise multiplied by the
    gate via ``head_gate_tilelang``.  Compressed/top-k attention (controlled
    by *g_cmp*) is not yet implemented.

    Args:
        query: ``[batch, seq_len, num_q_heads, head_dim]``.
        key: ``[batch, seq_len, num_kv_heads, head_dim]``.
        value: ``[batch, seq_len, num_kv_heads, head_dim]``.
        g_cmp: Compressed/top-k gate; not yet supported — raises
            ``EjkernelRuntimeError`` if not ``None``.
        g_slc: Per-head output gate, shape
            ``[batch, seq_len, num_q_heads]``.  When provided, the
            attention output is multiplied by this gate after the softmax
            accumulation step.
        block_indices: Selected KV block indices (required; raises if ``None``
            until native top-k is added).
        block_counts: Per-position block count or scalar; defaults to 16.
        block_size: Tokens per KV block; defaults to 64.
        softmax_scale: Attention temperature; defaults to
            ``1 / sqrt(head_dim)``.
        cu_seqlens: Not yet supported.
        block_k: Accepted for API compatibility with Triton; ignored by TileLang.
        block_v: Accepted for API compatibility with Triton; ignored by TileLang.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``[batch, seq_len, num_q_heads, head_dim]`` attention output,
        optionally gated by *g_slc*.

    Raises:
        EjkernelRuntimeError: If *g_cmp* is provided, *cu_seqlens* is
            provided, or *block_indices* is ``None``.
    """
    if g_cmp is not None:
        raise EjkernelRuntimeError("tile-lang native_sparse_attention does not yet support compressed/top-k attention.")
    if cu_seqlens is not None:
        raise EjkernelRuntimeError("tile-lang native_sparse_attention does not yet support cu_seqlens.")
    if block_indices is not None:
        out = apply_native_sparse_attention(
            query,
            key,
            value,
            block_indices,
            block_counts,
            block_size,
            softmax_scale,
        )
        if g_slc is not None:
            out = head_gate_tilelang(out, g_slc, int(block_v))
        return out
    raise EjkernelRuntimeError("tile-lang native_sparse_attention requires block_indices until native top-k is added.")


__all__ = ["apply_native_sparse_attention", "native_sparse_attention"]

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

"""Tile-lang block-sparse attention.

Block-sparse attention is dense attention with a structured mask. The mask
implied by segment ids, ``attention_mask``, ``sliding_window`` and ``causal``
is applied natively by the feature-complete tile-lang FlashAttention
kernels — ``bias``, ``softmax_aux`` and ``logits_soft_cap`` likewise. The
opaque ``mask_builder`` / ``qkv_layouts`` sparse-layout callbacks and the
sequence-parallel mesh axis are gated with explicit errors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ..flash_attention._impl import flash_attention_tilelang

Mask = Any
SparseMask = Any


@kernel_registry.register("blocksparse_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None
    ) = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """Tile-lang block-sparse attention (forward + backward).

    Inputs are ``(B, H, S, D)``; transposed to ``(B, S, H, D)`` for the
    FlashAttention kernels. Segment ids, ``attention_mask``, ``bias``,
    ``sliding_window``, ``softmax_aux`` and ``logits_soft_cap`` are applied
    natively. ``chunk_size`` / ``fused_backward`` are scheduling hints (the
    kernels are already fused); ``q_positions`` / ``kv_positions`` participate
    via sequence-position causal/window masking.

    Raises:
        EjkernelRuntimeError: if ``vhead_dim != head_dim``; if
            ``sequence_parallelism_mesh_axis_name`` is set (multi-device);
            or if ``mask_builder`` / ``qkv_layouts`` (opaque sparse-layout
            callbacks) are supplied.
    """
    if value.shape[-1] != query.shape[-1]:
        raise EjkernelRuntimeError("tile-lang blocksparse_attention requires head_dim == vhead_dim.")
    if sequence_parallelism_mesh_axis_name is not None:
        raise EjkernelRuntimeError(
            "tile-lang blocksparse_attention: sequence-parallel mesh coordination is not "
            "implemented; run on a single device or use the XLA backend."
        )
    if mask_builder is not None or qkv_layouts is not None:
        raise EjkernelRuntimeError(
            "tile-lang blocksparse_attention: opaque mask_builder / qkv_layouts callbacks are not "
            "supported; pass explicit q_segment_ids / kv_segment_ids / attention_mask / bias."
        )
    _scheduling_hints = (chunk_size, fused_backward, q_positions, kv_positions)

    q_bnhd = jnp.transpose(query, (0, 2, 1, 3))
    k_bnhd = jnp.transpose(key, (0, 2, 1, 3))
    v_bnhd = jnp.transpose(value, (0, 2, 1, 3))
    out_bnhd = flash_attention_tilelang(
        q_bnhd,
        k_bnhd,
        v_bnhd,
        softmax_scale=softmax_scale,
        causal=causal,
        bias=bias,
        attention_mask=attention_mask,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_aux=softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    return jnp.transpose(out_bnhd, (0, 2, 1, 3))


__all__ = ["blocksparse_attention"]

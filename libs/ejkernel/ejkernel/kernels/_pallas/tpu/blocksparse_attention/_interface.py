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
"""Public interface and kernel-registry registration for TPU block-sparse attention.

Registers ``blocksparse_attention`` under ``Platform.PALLAS / Backend.TPU``
and delegates to ``_kernel.blocksparse_attention`` (the Splash Attention
implementation).

Runtime type-checking is applied via ``beartype`` + ``jaxtyping``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ...._registry import Backend, Platform, kernel_registry
from ._kernel import Array, Bool, BwdParams, Callable, Float, FwdParams, Int
from ._kernel import blocksparse_attention as _blocksparse_attention_impl


@kernel_registry.register("blocksparse_attention", Platform.PALLAS, Backend.TPU)
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
    """Pallas TPU block-sparse (Splash) attention.

    Computes multi-head attention with block-level sparsity using JAX Pallas
    kernels on TPU.  The sparsity pattern can be derived automatically from
    ``causal``, ``sliding_window``, and ``chunk_size``, or supplied explicitly
    via ``qkv_layouts`` / ``mask_builder``.

    Supports MHA and GQA (``kv_num_heads < num_heads``).

    Args:
        query: ``[batch, num_heads, seq_len, head_dim]`` — query tensor.
        key: ``[batch, kv_num_heads, kv_len, head_dim]`` — key tensor.
        value: ``[batch, kv_num_heads, kv_len, vhead_dim]`` — value tensor.
            ``vhead_dim`` may differ from ``head_dim``.
        q_segment_ids: Optional ``[batch, seq_len]`` integer segment IDs.
            Tokens with different IDs are prevented from attending to each
            other (cross-segment masking is ANDed with the sparsity mask).
        kv_segment_ids: Optional ``[batch, kv_len]`` integer segment IDs.
        q_positions: Optional ``[batch, seq_len]`` position indices.
            Accepted but currently unused on TPU.
        kv_positions: Optional ``[batch, kv_len]`` position indices.
            Accepted but currently unused on TPU.
        softmax_aux: Optional ``[num_sinks]`` float array of attention-sink
            log-values (StreamingLLM-style).  When provided, these values are
            incorporated into the running online-softmax max and denominator.
        bias: Optional additive bias of shape
            ``[batch, num_heads, seq_len, kv_len]``.  Added to attention
            logits before softmax.
        attention_mask: Optional boolean or integer mask of shape
            ``[batch, num_heads_or_1, seq_len, kv_len]``.
        sequence_parallelism_mesh_axis_name: Optional mesh axis name for
            sequence-parallel sharding of the attention computation.
        logits_soft_cap: Optional logit soft-cap scalar.  When not None,
            applies ``logits_soft_cap * tanh(logits / logits_soft_cap)``
            to attention scores before the mask (Gemma-2 style).  Gradients
            are computed with the correct tanh Jacobian.
        qkv_layouts: Optional pre-built ``SparseMask`` layouts (from the
            Triton blocksparse backend) used directly as the block-sparsity
            pattern.  When provided, ``mask_builder``, ``causal``,
            ``sliding_window``, and ``chunk_size`` are ignored.
        softmax_scale: Scaling factor for attention logits.  Defaults to
            ``1/sqrt(head_dim)`` when ``None``.
        fwd_params: Forward-kernel tiling parameters (:class:`FwdParams`).
            Relevant fields: ``blocksize_m`` (query block), ``blocksize_n``
            (KV block).  Defaults are chosen by the kernel when ``None``.
        bwd_params: Backward-kernel tiling parameters (:class:`BwdParams`).
            Used only when ``fused_backward=True`` or when computing
            gradients.
        mask_builder: Optional callable for generating the sparsity mask.
            Accepts two forms:
            - ``(batch, heads, seq_len, kv_len, head_dim) -> SparseMask``
            - ``() -> SparseMask``
            Takes precedence over ``causal`` / ``sliding_window`` /
            ``chunk_size`` when provided.
        sliding_window: Local attention window size.  Options:
            - ``int``: symmetric window (same left and right size).
            - ``tuple[int, int]``: ``(left_window, right_window)`` for
              asymmetric windows.
            - ``None``: disabled.
        chunk_size: Chunk size for chunked-causal attention (Llama4-style).
            When not None, uses a ``ChunkedCausalMask`` where each token
            attends causally within its chunk but not across chunks.
        causal: If ``True`` (default), applies a causal (lower-triangular)
            mask.  Combined with ``sliding_window`` / ``chunk_size`` when
            those are also specified.
        fused_backward: If ``True``, uses the fused dQ/dKV backward kernel
            (requires appropriate ``BlockSizes.use_fused_bwd_kernel``).

    Returns:
        Attention output of shape ``[batch, num_heads, seq_len, vhead_dim]``.
    """
    return _blocksparse_attention_impl(
        query,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        q_positions,
        kv_positions,
        softmax_aux,
        bias,
        attention_mask,
        sequence_parallelism_mesh_axis_name,
        logits_soft_cap,
        qkv_layouts,
        softmax_scale,
        fwd_params,
        bwd_params,
        mask_builder,
        sliding_window,
        chunk_size,
        causal,
        fused_backward,
    )


__all__ = ("blocksparse_attention",)

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


"""Lightning Attention module with automatic optimization.

This module implements Lightning Attention, a linear attention mechanism that
combines intra-block and inter-block attention computations. It adapts behavior
based on layer position in deep transformer networks.

Key features of Lightning Attention:
    - Linear complexity: O(T) time through chunk-based processing
    - Layer-aware optimization: Adapts computation based on layer position
    - Hybrid attention: Combines causal intra-chunk with cross-chunk attention
    - Recurrent state: Maintains cumulative KV states across chunks
    - Variable-length sequence support via cumulative sequence lengths

The algorithm combines two attention components:
    1. Intra-block attention (causal within each chunk):
        - Standard attention within fixed-size blocks
        - Computes local dependencies efficiently
        - Uses causal masking for autoregressive generation

    2. Inter-block attention (linear attention across chunks):
        - Maintains cumulative state: S_t = Σ_{i<t} k_i @ v_i^T
        - Output from accumulated state: o_inter = q @ S
        - Enables long-range dependencies with O(1) memory per position

    The final output combines both components:
        o_t = o_intra_t + o_inter_t

Mathematical formulation:
    For a sequence divided into chunks of size C:
        Intra-chunk: o_intra = softmax(Q_c @ K_c^T / √d) @ V_c  (causal)
        Inter-chunk: o_inter = Q_c @ S_{c-1}  where S_c = S_{c-1} + K_c^T @ V_c
        Output: o = o_intra + o_inter

    This achieves O(T·C + T/C·K·V) = O(T) complexity for appropriate chunk sizes.

Layer-aware optimization:
    The layer_idx and num_layers parameters enable:
    - Different attention patterns for early vs. late layers
    - Optimized computation based on layer characteristics
    - Potential for mixed precision across depths

Supports:
    - Variable sequence lengths via cu_seqlens (cumulative sequence lengths)
    - Reverse processing for bidirectional models
    - Initial state for continuation across chunks
    - Multi-head and grouped-query attention patterns
    - Custom softmax scaling

References:
    - Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths
      in Large Language Models (Qin et al., 2024) https://arxiv.org/abs/2401.04658
    - TransNormerLLM: https://github.com/OpenNLPLab/TransnormerLLM
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import LightningAttentionConfig


class LightningAttention(Kernel[LightningAttentionConfig, Array]):
    """Lightning Attention with custom optimization logic.

    Implements a layer-aware linear attention mechanism optimized for deep transformer
    architectures. The attention computation combines intra-block causal attention
    with inter-block linear attention, achieving O(N) complexity with O(KxV) memory
    per head.

    Features:
        - Hybrid attention: Causal intra-chunk + linear inter-chunk computation
        - Layer-specific optimization strategies based on layer position
        - Support for stateful computation with initial states
        - Bidirectional and reverse sequence processing
        - Variable-length sequence handling via cumulative lengths
        - Automatic platform selection (Triton/Pallas/XLA/CUDA)
        - Automatic configuration caching for consistent performance
        - Optional autotuning to find optimal implementation

    This is particularly useful for:
        - Very deep transformers where layer position matters
        - Models with recurrent-style attention patterns
        - Long-context processing with bounded memory
        - Streaming inference with state continuation

    Example:
        >>> from ejkernel.modules import LightningAttention, create_default_executor
        >>>
        >>> # Basic usage with layer information
        >>> executor = create_default_executor()
        >>> attn = LightningAttention()
        >>> output = executor(attn, query, key, value, layer_idx=5, num_layers=32)
        >>>
        >>> # Streaming inference with state
        >>> output, state = executor(
        ...     attn, query, key, value,
        ...     layer_idx=0, num_layers=24,
        ...     initial_state=prev_state,
        ...     return_state=True
        ... )
        >>>
        >>> # Variable-length sequences
        >>> output = executor(
        ...     attn, query, key, value,
        ...     layer_idx=10, num_layers=32,
        ...     cu_seqlens=cu_seqs
        ... )
    """

    def __init__(self):
        """Initialize Lightning Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="lightning_attn")

    def get_impl(self, cfg: LightningAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for lightning attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("lightning_attn", cfg.platform)
        return kernel_registry.get("lightning_attn", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads qk_head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
        layer_idx: int,
        num_layers: int,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: LightningAttentionConfig,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute lightning attention with layer-specific optimization.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            layer_idx: Index of current layer in the model (0-indexed)
            num_layers: Total number of layers in the model
            softmax_scale: Optional scaling factor for attention scores
            initial_state: Optional initial hidden state [batch, num_heads, head_dim, head_dim]
            reverse: If True, process sequence in reverse order
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            return_state: If True, return tuple (output, final_state) instead of just output
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            If return_state=False: Attention output [batch, seq_len, num_heads, head_dim]
            If return_state=True: Tuple of (output, final_state) where final_state
                is [batch, num_heads, head_dim, head_dim]

        Note:
            The layer_idx and num_layers parameters enable layer-specific
            optimizations that can improve performance in deep networks.
        """

        if platform is not None:
            cfg = LightningAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        resolved_platform = detect_platform("lightning_attn", cfg.platform)
        impl = kernel_registry.get("lightning_attn", platform=resolved_platform, backend=cfg.backend)
        impl_kwargs = {}
        if resolved_platform == Platform.TRITON:
            impl_kwargs = {
                "block_k": cfg.block_k,
                "block_v": cfg.block_d,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
        result = impl(
            query=query,
            key=key,
            value=value,
            layer_idx=layer_idx,
            num_layers=num_layers,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            **impl_kwargs,
        )

        if isinstance(result, tuple):
            if return_state:
                return result
            else:
                return result[0]
        return result

    def heuristic_cfg(self, inv: Invocation[LightningAttentionConfig, Array]) -> LightningAttentionConfig:
        """Provide default configuration with block sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical lightning attention workloads across various layer depths
        """
        return LightningAttentionConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[LightningAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Lightning attention's layer-aware design means performance may vary
            across layer depths. Candidates cover a range of block sizes.
        """
        block_configs = [
            (64, 64, 64, 4, 1),
            (128, 64, 64, 4, 2),
            (128, 128, 64, 8, 2),
        ]

        candidates = []
        for block_q, block_k, block_d, num_warps, num_stages in block_configs:
            candidates.append(
                LightningAttentionConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=block_d,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[LightningAttentionConfig, Array]):
        """Generate GPU candidates for Lightning Attention.

        Lightning attention is a linear-attention variant; ``block_d``
        (head_dim tile) drives most of the GPU performance. Sweep:

        * ``block_d`` ∈ {32, 64, 128}; 128 for wide heads.
        * ``block_q`` / ``block_k`` ∈ {64, 128}.
        * ``num_warps`` ∈ {4, 8}; 8 only when blocks are big.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        block_configs = [
            (64, 64, 32, 4, 1),
            (64, 64, 64, 4, 1),
            (128, 64, 64, 4, 2),
            (64, 128, 64, 4, 2),
            (128, 128, 64, 8, 2),
            (128, 128, 128, 8, 2),
        ]
        tilelang_configs = [
            (64, 64, 64, 4, 1),
            (128, 128, 64, 4, 2),
            (128, 128, 128, 8, 2),
        ]
        candidates: list[LightningAttentionConfig] = []
        if "triton" in platforms:
            for block_q, block_k, block_d, num_warps, num_stages in block_configs:
                candidates.append(
                    LightningAttentionConfig(
                        block_q=block_q,
                        block_k=block_k,
                        block_d=block_d,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="triton",
                        backend="gpu",
                    )
                )
        if "tilelang" in platforms:
            for block_q, block_k, block_d, num_warps, num_stages in tilelang_configs:
                candidates.append(
                    LightningAttentionConfig(
                        block_q=block_q,
                        block_k=block_k,
                        block_d=block_d,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(
                LightningAttentionConfig(
                    block_q=64,
                    block_k=64,
                    block_d=64,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[LightningAttentionConfig, Array]):
        """Generate TPU candidates for the XLA Lightning Attention path."""
        return [
            LightningAttentionConfig(
                block_q=64,
                block_k=64,
                block_d=64,
                num_warps=4,
                num_stages=1,
                platform="xla",
                backend="any",
            )
        ]


_lightning_executor: Executor[LightningAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("lightning-attention"),
    )
)


def lightning_attention(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    layer_idx: int,
    num_layers: int,
    softmax_scale: float | None = None,
    reverse: bool = False,
    return_state: bool = False,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: LightningAttentionConfig | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """Execute lightning attention with automatic optimization.

    Lightning attention is an efficient attention mechanism that uses
    layer-specific optimizations for improved performance.

    Args:
        query: Query tensor ``[batch, seq_len, num_heads, qk_head_dim]``.
        key: Key tensor ``[batch, seq_len, num_kv_heads, qk_head_dim]``.
        value: Value tensor ``[batch, seq_len, num_kv_heads, v_head_dim]``.
        initial_state: Optional initial recurrent state
            ``[..., num_heads, qk_head_dim, v_head_dim]``. Used for streaming
            inference or continuation across document boundaries.
        cu_seqlens: Cumulative sequence lengths for variable-length batches
            ``[num_seqs + 1]``. When ``None``, the full batch dimension is
            treated as a single sequence.
        layer_idx: Zero-indexed layer position in the model. Passed through to
            the backend for layer-specific optimisation strategies.
        num_layers: Total number of layers in the model. Together with
            ``layer_idx``, enables depth-aware tuning.
        softmax_scale: Scaling factor applied to intra-chunk attention scores.
            Defaults to ``1 / sqrt(qk_head_dim)`` inside the kernel.
        reverse: If ``True``, process the sequence in reverse order (for
            bidirectional models).
        return_state: If ``True``, return a ``(output, final_state)`` tuple
            instead of just the output tensor.
        platform: Override platform selection (``"triton"``, ``"pallas"``,
            ``"cuda"``, ``"xla"``, ``"auto"``).
        cfg: Optional :class:`~.configs.LightningAttentionConfig` override
            (block sizes, warp count, pipeline stages).

    Returns:
        If ``return_state=False``: Attention output
            ``[batch, seq_len, num_heads, v_head_dim]``.
        If ``return_state=True``: Tuple of
            ``(output, final_state)`` where ``final_state`` has shape
            ``[..., num_heads, qk_head_dim, v_head_dim]``.

    Example:
        >>> out = lightning_attention(query, key, value, layer_idx=5, num_layers=32)

        >>> out = lightning_attention(
        ...     query, key, value, layer_idx=0, num_layers=24, softmax_scale=0.125
        ... )

        >>> out = lightning_attention(
        ...     query, key, value, layer_idx=10, num_layers=32, cu_seqlens=cu_seqs
        ... )

        >>> out = lightning_attention(
        ...     query, key, value, layer_idx=0, num_layers=24, platform="pallas"
        ... )
    """
    return _lightning_executor(
        LightningAttention(),
        query=query,
        key=key,
        value=value,
        layer_idx=layer_idx,
        num_layers=num_layers,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )

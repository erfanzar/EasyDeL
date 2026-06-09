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


"""Native sparse attention module with automatic optimization.

This module implements native sparse attention using explicit block indices to
define sparsity patterns. Unlike block-sparse attention which uses mask builders,
this implementation directly specifies which blocks to attend to via index arrays.

Key Features:
    - Direct block index specification for maximum sparsity control
    - Configurable block sizes for compute/memory trade-offs
    - Support for Grouped Query Attention (GQA)
    - Gating mechanisms (g_cmp, g_slc) for hybrid attention patterns
    - Variable-length sequence support via cu_seqlens
    - Multiple platform support (Triton/Pallas/CUDA/XLA)

This approach is particularly efficient when:
    - The sparse pattern is known ahead of time
    - Block indices can be precomputed and reused
    - Fine-grained control over sparsity is needed
    - Long sequences require sub-quadratic complexity

The Sparsity Pattern:
    The sparse pattern is defined by block_indices and block_counts arrays:
        - block_indices[batch, num_kv_heads, q_block, k_block]: Which key blocks to attend
        - block_counts: Number of key blocks per query block (int or per-block array)

    Example patterns:
        - Local window: Attend to nearby blocks only
        - Strided: Attend to every N-th block for global context
        - Document-aware: Custom patterns based on document structure

Use Cases:
    - Long document processing (>32K tokens)
    - Efficient inference with structured sparsity
    - Hybrid local-global attention patterns
    - Memory-constrained deployment

Example:
    >>> from ejkernel.modules.operations import native_sparse_attention
    >>>
    >>> # Basic sparse attention with default block pattern
    >>> output = native_sparse_attention(query, key, value)
    >>>
    >>> # Custom sparse pattern with explicit indices
    >>> output = native_sparse_attention(
    ...     query, key, value,
    ...     block_indices=custom_indices,
    ...     block_counts=num_selected_blocks,
    ... )

References:
    - Longformer: https://arxiv.org/abs/2004.05150
    - BigBird: https://arxiv.org/abs/2007.14062
    - Native Sparse Attention (NSA): https://arxiv.org/abs/2502.11089
"""

from __future__ import annotations

import os
import typing

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
from .configs import NativeSparseAttentionConfig


class NativeSparseAttention(Kernel[NativeSparseAttentionConfig, Array]):
    """Native Sparse Attention with custom optimization logic.

    Implements sparse attention using explicit block index specification. This provides
    direct control over which blocks participate in attention computation, enabling
    efficient sparse patterns without runtime mask building.

    Features:
        - Direct block index specification for sparsity
        - Configurable block size and block counts
        - Support for variable-length sequences
        - Token-level sparse patterns via token_indices
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The sparsity is controlled by:
        - block_indices: Which blocks each query block attends to
        - block_counts: Number of key blocks per query block
        - token_indices: Fine-grained token-level sparsity (optional)
    """

    def __init__(self):
        """Initialize Native Sparse Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="native_sparse_attention")

    def get_impl(self, cfg: NativeSparseAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for native sparse attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("native_sparse_attention", cfg.platform)
        return kernel_registry.get("native_sparse_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
        g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
        block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
        block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
        softmax_scale: float | None = None,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: NativeSparseAttentionConfig,
    ) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
        """Execute native sparse attention with explicit block indices.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            g_cmp: Optional compression gate weights [batch, seq_len, num_q_heads].
                Used for gated sparse attention to modulate the compressed path.
                When provided, scales the attention output from compressed tokens.
            g_slc: Optional selection gate weights [batch, seq_len, num_q_heads].
                Used for gated sparse attention to modulate the selected path.
                When provided, scales the attention output from selected tokens.
            block_indices: Indices of key blocks to attend to for each query block.
                The declared jaxtyping shape is
                ``[batch, seq_len, num_kv_heads, num_selected_blocks]``, but
                the XLA and Triton backends interpret this as a block-indexed
                table (see Note below).
            block_counts: Number of key blocks per query block (can be int or array)
            softmax_scale: Optional scaling factor for attention scores
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Sparse attention output [batch, seq_len, num_heads, head_dim]

        Note:
            When ``block_indices`` is ``None``, a default pattern may be used
            depending on the implementation. Providing explicit indices gives
            full control over the sparsity pattern.

            The ``block_indices`` shape annotation ``[batch, seq_len,
            num_kv_heads, num_selected_blocks]`` uses ``seq_len`` as a
            positional dimension, but the underlying Triton kernel (NSA)
            treats the second dimension as the number of query blocks, not
            raw token positions. The XLA fallback is selected automatically
            when the GQA group size is not a multiple of 16 (the Triton
            constraint).
        """

        if (platform is None or platform == "auto") and cfg.platform in (None, "auto"):
            num_q_heads = int(query.shape[2])
            num_kv_heads = int(key.shape[2])
            group_size = num_q_heads // num_kv_heads if (num_kv_heads > 0 and num_q_heads % num_kv_heads == 0) else 0
            if group_size % 16 != 0:
                cfg = NativeSparseAttentionConfig(
                    block_q=cfg.block_q,
                    block_k=cfg.block_k,
                    block_d=cfg.block_d,
                    block_size=cfg.block_size,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                    platform="xla",
                    backend=Backend.ANY,
                )

        if platform is not None:
            cfg = NativeSparseAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d,
                block_size=cfg.block_size,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        resolved_platform = detect_platform("native_sparse_attention", cfg.platform)
        impl = kernel_registry.get("native_sparse_attention", platform=resolved_platform, backend=cfg.backend)
        impl_kwargs = {}
        if resolved_platform == Platform.TRITON:
            impl_kwargs = {
                "block_k": cfg.block_k,
                "block_v": cfg.block_d,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
        return impl(
            query=query,
            key=key,
            value=value,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=cfg.block_size,
            softmax_scale=softmax_scale,
            cu_seqlens=cu_seqlens,
            g_cmp=g_cmp,
            g_slc=g_slc,
            **impl_kwargs,
        )

    def heuristic_cfg(self, inv: Invocation[NativeSparseAttentionConfig, Array]) -> NativeSparseAttentionConfig:
        """Provide default configuration with block sizes.

        Selects balanced block sizes that work well for typical sparse patterns.
        The default configuration uses uniform block sizes for simplicity.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with block_size=64 for balanced performance
        """
        return NativeSparseAttentionConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            block_size=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates a basic set of candidates for platform-agnostic tuning.
        Sparse attention benefits from consistent block sizes across dimensions.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List with single configuration using block_size=64 as baseline
        """
        block_configs = [(64, 64, 64)]

        candidates = []
        for block_q, block_k, block_d in block_configs:
            candidates.append(
                NativeSparseAttentionConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=block_d,
                    block_size=64,
                    num_warps=4,
                    num_stages=1,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate GPU-optimized candidate configurations for NSA.

        NSA's selected-block sparse attention has four interacting tiles:

        * ``block_q`` / ``block_k`` — inner attention tile (within a
          selected KV block).
        * ``block_d`` — head_dim tile (clamped by actual head_dim below).
        * ``block_size`` — granularity of KV block selection; smaller
          values mean finer-grained sparsity but more bookkeeping.

        On H100:
        * ``block_size`` ∈ {32, 64, 128}; 32 for short prompts, 128
          amortises selection cost for long prompts.
        * ``num_warps`` ∈ {2, 4, 8}; 8 only when tile area is large.
        * ``num_stages`` ∈ {1, 2}; 2 helps memory-bound long-context.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        q = inv.kwargs.get("query")
        head_dim = int(q.shape[-1]) if getattr(q, "shape", None) else 64
        bd_choices = [d for d in (32, 64, 128) if d <= max(head_dim, 32)] or [64]
        configs: list[NativeSparseAttentionConfig] = []

        if "triton" in platforms:
            for block_size in (32, 64, 128):
                for bd in bd_choices:
                    for num_warps, num_stages in ((2, 1), (4, 1), (4, 2), (8, 2)):
                        configs.append(
                            NativeSparseAttentionConfig(
                                block_q=block_size,
                                block_k=block_size,
                                block_d=bd,
                                block_size=block_size,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                platform="triton",
                                backend="gpu",
                            )
                        )
        if "tilelang" in platforms:
            for block_size in (64, 128):
                for bk in (64, 128) if block_size == 128 else (64,):
                    for bv in (64, 128):
                        configs.append(
                            NativeSparseAttentionConfig(
                                block_q=block_size,
                                block_k=bk,
                                block_d=min(bv, max(head_dim, 32)),
                                block_size=block_size,
                                num_warps=8 if max(block_size, bk) >= 128 else 4,
                                num_stages=2,
                                platform="tilelang",
                                backend="gpu",
                            )
                        )
        if "xla" in platforms:
            configs.extend(self.candidate_cfgs_xla(inv))
        return configs

    def candidate_cfgs_tpu(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning.

        Creates configurations tailored for TPU execution with Pallas backend.
        TPUs prefer larger block sizes (64, 128) for better vectorization.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of TPU-specific configurations optimized for matrix units
        """
        return self.candidate_cfgs_xla(inv)

    def candidate_cfgs_xla(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate XLA-optimized candidate configurations for autotuning.

        Creates configurations for XLA compiler backend. XLA handles optimization
        internally, so we provide conservative block sizes that work well across
        different hardware targets.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of XLA-compatible configurations with standard block sizes
        """
        configs = []
        for block_size in [64, 128]:
            configs.append(
                NativeSparseAttentionConfig(
                    block_q=block_size,
                    block_k=block_size,
                    block_d=block_size,
                    block_size=block_size,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return configs

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_xla = candidate_cfgs_xla


_sparse_executor: Executor[NativeSparseAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("nsa"),
    )
)


def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
    g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: NativeSparseAttentionConfig | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Execute native sparse attention with automatic optimization.

    Sparse attention computes attention only on specified blocks or patterns,
    reducing computational cost for long sequences. This is particularly useful
    for models that need to process very long contexts efficiently.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        g_cmp: Optional compression gate weights [batch, seq_len, num_q_heads]
            for gated sparse attention patterns
        g_slc: Optional slice gate weights [batch, seq_len, num_q_heads]
            for controlling attention slice contributions
        block_indices: Indices of key blocks to attend to for each query block.
            Declared shape: ``[batch, seq_len, num_kv_heads, num_selected_blocks]``.
            The Triton backend treats the ``seq_len`` dimension as the number of
            query blocks. The XLA fallback accepts either interpretation.
        block_counts: Number of blocks per query block (default: 16)
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        softmax_scale: Scaling factor for attention (default: 1/sqrt(head_dim))
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional configuration override

    Returns:
        Attention output [batch, seq_len, num_q_heads, head_dim]

    Example:
        >>> # Basic sparse attention with default block pattern
        >>> out = native_sparse_attention(query, key, value)

        >>> # With custom block count
        >>> out = native_sparse_attention(query, key, value, block_counts=32)

        >>> # With explicit block indices for custom sparse pattern
        >>> out = native_sparse_attention(
        ...     query, key, value,
        ...     block_indices=custom_indices,
        ...     block_counts=num_selected,
        ... )

        >>> # Force specific platform
        >>> out = native_sparse_attention(query, key, value, platform="triton")

    Note:
        When block_indices is None, a default sparse pattern is used.
        Providing explicit indices gives full control over the sparsity pattern.
    """
    return _sparse_executor(
        NativeSparseAttention(),
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        g_cmp=g_cmp,
        g_slc=g_slc,
        platform=platform,
        _cfg=cfg,
    )

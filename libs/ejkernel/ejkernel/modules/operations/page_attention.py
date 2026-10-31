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


"""Page Attention module with automatic optimization.

This module implements Page Attention, a specialized attention mechanism designed
for efficient KV cache management in serving and inference workloads. Page Attention
organizes the KV cache in fixed-size blocks (pages), enabling:
    - Dynamic memory allocation without pre-allocating for max sequence length
    - Efficient memory sharing across sequences (e.g., for beam search or prefix caching)
    - Reduced memory fragmentation compared to contiguous allocation
    - Better GPU memory utilization through page-level management

Page Attention is particularly valuable for:
    - LLM serving with variable-length sequences
    - Batch inference with dynamic batching
    - Memory-constrained deployment scenarios
    - Systems requiring efficient KV cache sharing

Key Concepts:
    Pages: Fixed-size blocks holding a portion of KV cache (e.g., 16 or 32 tokens)
    Block Tables: Mapping from logical sequence positions to physical page indices
    Context Lengths: Actual sequence lengths (excluding padding)

The paged approach enables:
    - Near-zero memory waste (only last page per sequence may be partially filled)
    - Easy insertion/deletion of sequences without memory reshuffling
    - Natural support for prefix sharing in beam search

Mathematical Foundation:
    For query position i:
        output[i] = sum_{j in valid_pages} softmax(Q[i] @ K[pages[j]].T) @ V[pages[j]]

    Where valid_pages are determined by block_tables and context_lens.

Memory Layout:
    Instead of: [seq_len, num_heads, head_dim] (contiguous per sequence)
    Use: [num_pages, page_size, num_heads, head_dim] (page-based allocation)

References:
    Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"
    https://arxiv.org/abs/2309.06180 (vLLM paper)
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
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
from .configs import PageAttentionConfig


class PageAttention(Kernel[PageAttentionConfig, Array]):
    """Page Attention with custom optimization logic.

    Efficient attention over paged KV cache for serving workloads.
    Optimized for dynamic batching with variable context lengths.

    Features:
        - Paged KV cache management for memory efficiency
        - Support for variable context lengths per sequence
        - Automatic partitioning for long contexts
        - Multi-split attention for improved throughput
        - Optimized for inference and serving workloads
        - Logit soft capping for numerical stability
        - Configurable pages per compute block
        - TPU megacore mode support

    The paged layout provides:
        - O(1) insertion/deletion of sequences
        - Efficient prefix sharing for beam search
        - Minimal memory fragmentation
        - Better batch utilization through dynamic allocation
    """

    def __init__(self):
        """Initialize Page Attention module.

        Sets up the kernel for paged KV cache attention computation
        with automatic platform selection and optimization.
        """
        super().__init__(op_id="page_attention")

    def get_impl(self, cfg: PageAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for page attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("page_attention", cfg.platform)
        return kernel_registry.get("page_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "num_seqs num_heads head_dim"],
        key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
        value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
        context_lens: Int[Array, "num_seqs"],
        block_tables: Int[Array, "num_seqs max_blocks"],
        attn_scale: float | None = None,
        max_context_len: int | None = None,
        num_splits: int = 0,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: PageAttentionConfig,
        mask_value: float = -2.381976426469702e38,
        attn_logits_soft_cap: float | None = None,
        pages_per_compute_block: int | None = None,
        megacore_mode: str | None = None,
        inline_seq_dim: bool = True,
        sliding_window: int | None = None,
    ) -> Float[Array, "num_seqs num_heads head_dim"]:
        """Execute page attention over paged KV cache.

        Computes attention where the KV cache is organized in fixed-size pages,
        with each sequence's tokens potentially scattered across non-contiguous pages.

        Args:
            query: Query tensor [num_seqs, num_heads, head_dim] for current decode step
            key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
            value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
            context_lens: Actual context length per sequence [num_seqs]
            block_tables: Page index mapping [num_seqs, max_blocks] where block_tables[i, j]
                gives the physical page index for sequence i's jth logical block
            attn_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
            max_context_len: Maximum context length across all sequences
            num_splits: Number of splits for partitioned attention (0 = auto, 1 = no split)
            mask_value: Value used for masked positions (default: -inf)
            attn_logits_soft_cap: Optional soft cap for attention logits
            pages_per_compute_block: Number of pages to process per compute block
            megacore_mode: TPU-specific megacore execution mode
            inline_seq_dim: Whether to inline the sequence dimension
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [num_seqs, num_heads, head_dim]

        Note:
            Block tables define the mapping from logical to physical pages:
                logical_page_idx = position // block_size
                physical_page_idx = block_tables[seq_idx, logical_page_idx]

        Example:
            >>>
            >>>
            >>> block_tables = jnp.array([[3, 7, 0], [1, 5, 0]])
            >>> context_lens = jnp.array([32, 24])
            >>> out = page_attention(q, k_cache, v_cache, context_lens, block_tables)
        """

        if platform is not None:
            cfg = PageAttentionConfig(
                num_splits=cfg.num_splits,
                pages_per_compute_block=cfg.pages_per_compute_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        resolved_platform = detect_platform("page_attention", cfg.platform)
        impl = kernel_registry.get("page_attention", platform=resolved_platform, backend=cfg.backend)
        resolved_num_splits = num_splits if num_splits != 0 else cfg.num_splits
        resolved_pages_per_compute_block = (
            pages_per_compute_block if pages_per_compute_block is not None else cfg.pages_per_compute_block
        )
        impl_kwargs = {}
        if resolved_platform == Platform.TRITON:
            impl_kwargs = {
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
        return impl(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            attn_scale=attn_scale,
            max_context_len=max_context_len,
            num_splits=resolved_num_splits,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=resolved_pages_per_compute_block,
            megacore_mode=megacore_mode,
            inline_seq_dim=inline_seq_dim,
            sliding_window=sliding_window,
            **impl_kwargs,
        )

    def heuristic_cfg(self, inv: Invocation[PageAttentionConfig, Array]) -> PageAttentionConfig:
        """Provide default configuration optimized for paged attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default KernelConfig with block sizes suitable for typical
            serving workloads with variable context lengths
        """
        return PageAttentionConfig(
            num_splits=0,
            pages_per_compute_block=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[PageAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for different batch sizes and
        context lengths commonly seen in serving scenarios.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Page attention doesn't have tunable block sizes in the traditional sense.
            The num_splits and pages_per_compute_block are auto-determined.
        """
        return [
            PageAttentionConfig(
                num_splits=num_splits,
                pages_per_compute_block=None,
                num_warps=num_warps,
                num_stages=num_stages,
                platform="auto",
                backend="any",
            )
            for num_splits in (0, 1, 2, 4)
            for num_warps, num_stages in ((2, 4), (4, 3), (4, 4), (8, 2), (8, 3), (8, 4), (16, 2), (16, 3), (16, 4))
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[PageAttentionConfig, Array]):
        """Generate GPU candidates for paged attention across Triton, TileLang and XLA."""
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[PageAttentionConfig] = []
        if "triton" in platforms:
            for num_splits in (0, 1, 2, 4):
                for num_warps, num_stages in (
                    (2, 4),
                    (4, 3),
                    (4, 4),
                    (8, 2),
                    (8, 3),
                    (8, 4),
                    (16, 2),
                    (16, 3),
                    (16, 4),
                ):
                    candidates.append(
                        PageAttentionConfig(
                            num_splits=num_splits,
                            pages_per_compute_block=None,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            platform="triton",
                            backend="gpu",
                        )
                    )
        if "tilelang" in platforms:
            for pages_per_compute_block in (None, 1, 2, 4):
                candidates.append(
                    PageAttentionConfig(
                        num_splits=0,
                        pages_per_compute_block=pages_per_compute_block,
                        num_warps=4,
                        num_stages=3,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(
                PageAttentionConfig(
                    num_splits=0,
                    pages_per_compute_block=None,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[PageAttentionConfig, Array]):
        """Generate TPU candidates for Pallas and XLA page attention."""
        return [
            PageAttentionConfig(
                num_splits=num_splits,
                pages_per_compute_block=None,
                num_warps=4,
                num_stages=1,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for num_splits in (0, 1, 2)
        ]

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "num_seqs num_heads head_dim"],
        key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
        value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
        context_lens: Int[Array, "num_seqs"],
        block_tables: Int[Array, "num_seqs max_blocks"],
        attn_scale: float | None = None,
        max_context_len: int | None = None,
        num_splits: int = 0,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: PageAttentionConfig | None = None,
        mask_value: float = -2.381976426469702e38,
        attn_logits_soft_cap: float | None = None,
        pages_per_compute_block: int | None = None,
        megacore_mode: str | None = None,
        inline_seq_dim: bool = True,
        sliding_window: int | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed execution.

        Creates a wrapper function that applies shard_map to distribute the page attention
        computation across devices according to the provided sharding specifications.

        Args:
            query: Query tensor [num_seqs, num_heads, head_dim]
            key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
            value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
            context_lens: Context length per sequence [num_seqs]
            block_tables: Block mapping table [num_seqs, max_blocks]
            attn_scale: Attention scaling factor
            max_context_len: Maximum context length across all sequences
            num_splits: Number of splits for partitioned attention
            platform: Platform to use for execution
            cfg: Configuration for the kernel
            mask_value: Value for masked positions
            attn_logits_soft_cap: Soft cap value for attention logits
            pages_per_compute_block: Pages per compute block
            megacore_mode: Megacore execution mode
            inline_seq_dim: Whether to inline sequence dimension
            mesh: JAX mesh for distributed execution
            in_specs: Partition specifications for input tensors
            out_specs: Partition specifications for output tensor
            check_vma: Whether to check for valid memory access patterns

        Returns:
            Tuple of (shard_map_fn, call_args) where shard_map_fn is the wrapped
            function and call_args are the arguments to pass to it.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapper(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
        ):
            """Shard-map compatible wrapper that delegates to self.run with captured params."""
            return self.run(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                context_lens=context_lens,
                block_tables=block_tables,
                attn_scale=attn_scale,
                max_context_len=max_context_len,
                num_splits=num_splits,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
                mask_value=mask_value,
                attn_logits_soft_cap=attn_logits_soft_cap,
                pages_per_compute_block=pages_per_compute_block,
                megacore_mode=megacore_mode,
                inline_seq_dim=inline_seq_dim,
                sliding_window=sliding_window,
            )

        shard_map_fn = shard_map(
            _wrapper,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        call_args = (
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
        )

        return shard_map_fn, call_args


_page_attention_executor: Executor[PageAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("page-attention"),
    )
)


def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    /,
    *,
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    mask_value: float = -2.381976426469702e38,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    sliding_window: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: PageAttentionConfig | None = None,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Execute page attention with automatic optimization.

    Page attention performs efficient attention computation over paged KV cache
    for serving and inference workloads with dynamic batching.

    Args:
        query: Query tensor [num_seqs, num_heads, head_dim]
        key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
        context_lens: Context length per sequence [num_seqs]
        block_tables: Block mapping table [num_seqs, max_blocks]
        attn_scale: Attention scaling factor
        max_context_len: Maximum context length across all sequences
        num_splits: Number of splits for partitioned attention (0=auto)
        mask_value: Value for masked positions (default: -inf)
        attn_logits_soft_cap: Soft cap value for attention logits
        pages_per_compute_block: Pages per compute block
        megacore_mode: Megacore execution mode
        inline_seq_dim: Whether to inline sequence dimension
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional configuration override

    Returns:
        Attention output [num_seqs, num_heads, head_dim]

    Example:
        >>> out = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        >>> out = page_attention(
        ...     query, key_cache, value_cache, context_lens, block_tables,
        ...     num_splits=4, max_context_len=8192
        ... )

        >>> out = page_attention(
        ...     query, key_cache, value_cache, context_lens, block_tables,
        ...     attn_logits_soft_cap=50.0
        ... )

        >>> out = page_attention(query, key_cache, value_cache, context_lens,
        ...                      block_tables, platform="triton")
    """
    return _page_attention_executor(
        PageAttention(),
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=attn_scale,
        max_context_len=max_context_len,
        num_splits=num_splits,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        pages_per_compute_block=pages_per_compute_block,
        megacore_mode=megacore_mode,
        inline_seq_dim=inline_seq_dim,
        sliding_window=sliding_window,
        platform=platform,
        _cfg=cfg,
    )

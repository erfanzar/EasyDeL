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


"""Prefill Page Attention module with automatic optimization.

This module implements chunked prefill attention with paged KV cache, designed
for the prefill phase of LLM inference. It complements the decode-only PageAttention
by handling multiple query tokens with causal masking.

Key Features:
    - Processes multiple query tokens (chunk) during prefill
    - Causal masking for autoregressive generation
    - Sliding window attention support
    - Paged KV cache for memory efficiency
    - Grouped Query Attention (GQA) support

Use Cases:
    - Initial prompt processing in LLM serving
    - Chunked prefill for long contexts
    - Combined with PageAttention for full inference pipeline

Mathematical Foundation:
    For query position q_pos in chunk:
        output[q_pos] = sum_{kv_pos <= q_pos} softmax(Q[q_pos] @ K[kv_pos].T) @ V[kv_pos]

    With sliding window (window_size W):
        output[q_pos] = sum_{q_pos - W + 1 <= kv_pos <= q_pos} softmax(...) @ V[kv_pos]

References:
    - JetStream chunked prefill: https://github.com/AI-Hypercomputer/JetStream
    - PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
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
from .configs import PrefillPageAttentionConfig


class PrefillPageAttention(Kernel[PrefillPageAttentionConfig, Array]):
    """Prefill Page Attention with custom optimization logic.

    Efficient chunked prefill attention over paged KV cache for serving workloads.
    Designed for the prefill phase where multiple query tokens are processed.

    Features:
        - Chunked prefill with causal masking
        - Paged KV cache management for memory efficiency
        - Sliding window attention support
        - Grouped Query Attention (GQA) support
        - Logit soft capping for numerical stability
        - TPU optimized with async DMA prefetching
    """

    def __init__(self):
        """Initialize Prefill Page Attention module."""
        super().__init__(op_id="prefill_page_attention")

    def get_impl(self, cfg: PrefillPageAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for prefill page attention
        """
        platform = detect_platform("prefill_page_attention", cfg.platform)
        return kernel_registry.get("prefill_page_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "chunk_size num_heads head_dim"],
        key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
        value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
        context_len: Int[Array, "1"],
        page_indices: Int[Array, "num_pages"],
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: PrefillPageAttentionConfig,
        softmax_scale: float | None = None,
        mask_value: float = -2.381976426469702e38,
        attn_logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
    ) -> Float[Array, "chunk_size num_heads head_dim"]:
        """Execute prefill page attention over paged KV cache.

        Processes a chunk of query tokens with causal attention over a paged KV cache.
        Each query can only attend to itself and previous positions.

        Args:
            query: Query tensor [chunk_size, num_q_heads, head_dim] for prefill tokens
            key_cache: Paged key cache [num_kv_heads, total_num_pages, page_size, head_dim]
            value_cache: Paged value cache [num_kv_heads, total_num_pages, page_size, head_dim]
            context_len: Total context length including this chunk [1]
            page_indices: Page indices for this sequence [num_pages]
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object
            softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim))
            mask_value: Value used for masked positions (default: -inf)
            attn_logits_soft_cap: Optional soft cap for attention logits
            sliding_window: If set, only attend to the last `sliding_window` tokens

        Returns:
            Attention output [chunk_size, num_q_heads, head_dim]

        Example:
            >>> # Process a chunk of 128 tokens during prefill
            >>> output = prefill_page_attention(
            ...     query,  # [128, 32, 128]
            ...     key_cache, value_cache,
            ...     context_len=jnp.array([512]),
            ...     page_indices=page_indices,
            ... )
        """
        if platform is not None:
            cfg = PrefillPageAttentionConfig(
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_len=context_len,
            page_indices=page_indices,
            softmax_scale=softmax_scale,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            sliding_window=sliding_window,
            block_k=cfg.block_k,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    def heuristic_cfg(self, inv: Invocation[PrefillPageAttentionConfig, Array]) -> PrefillPageAttentionConfig:
        """Provide default configuration optimized for prefill page attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration suitable for prefill workloads
        """
        query = inv.kwargs["query"]
        head_dim = int(query.shape[-1])
        return PrefillPageAttentionConfig(
            block_k=128 if head_dim >= 64 else 64,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[PrefillPageAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning
        """
        base = self.heuristic_cfg(inv)
        return [
            PrefillPageAttentionConfig(
                block_k=base.block_k,
                num_warps=base.num_warps,
                num_stages=base.num_stages,
                platform="auto",
                backend="any",
            ),
            PrefillPageAttentionConfig(
                block_k=base.block_k,
                num_warps=base.num_warps,
                num_stages=base.num_stages,
                platform="xla",
                backend="any",
            ),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[PrefillPageAttentionConfig, Array]):
        """Generate GPU candidates for TileLang prefill paged attention."""
        query = inv.kwargs["query"]
        page_indices = inv.kwargs["page_indices"]
        requested = inv.kwargs.get("platform", None)
        head_dim = int(query.shape[-1])
        pages = int(page_indices.shape[0])
        block_choices = (64, 128, 256) if head_dim >= 64 else (32, 64, 128)
        if pages <= 2:
            block_choices = tuple(b for b in block_choices if b <= 128)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[PrefillPageAttentionConfig] = []
        if "tilelang" in platforms:
            for block_k in block_choices:
                for stages in (2, 3, 4):
                    candidates.append(
                        PrefillPageAttentionConfig(
                            block_k=block_k,
                            num_warps=4,
                            num_stages=stages,
                            platform="tilelang",
                            backend="gpu",
                        )
                    )
        if "xla" in platforms:
            base = self.heuristic_cfg(inv)
            candidates.append(
                PrefillPageAttentionConfig(
                    block_k=base.block_k,
                    num_warps=base.num_warps,
                    num_stages=base.num_stages,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[PrefillPageAttentionConfig, Array]):
        """Generate TPU candidates for Pallas and XLA prefill paged attention."""
        base = self.heuristic_cfg(inv)
        return [
            PrefillPageAttentionConfig(
                block_k=base.block_k,
                num_warps=4,
                num_stages=1,
                platform="pallas",
                backend="tpu",
            ),
            PrefillPageAttentionConfig(
                block_k=base.block_k,
                num_warps=base.num_warps,
                num_stages=base.num_stages,
                platform="xla",
                backend="any",
            ),
        ]

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "chunk_size num_heads head_dim"],
        key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
        value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
        context_len: Int[Array, "1"],
        page_indices: Int[Array, "num_pages"],
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: PrefillPageAttentionConfig | None = None,
        softmax_scale: float | None = None,
        mask_value: float = -2.381976426469702e38,
        attn_logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed execution.

        Args:
            query: Query tensor [chunk_size, num_q_heads, head_dim]
            key_cache: Paged key cache
            value_cache: Paged value cache
            context_len: Total context length
            page_indices: Page indices for sequence
            platform: Platform to use
            cfg: Configuration for the kernel
            softmax_scale: Attention scaling factor
            mask_value: Value for masked positions
            attn_logits_soft_cap: Soft cap value
            sliding_window: Sliding window size
            mesh: JAX mesh for distributed execution
            in_specs: Partition specifications for input tensors
            out_specs: Partition specifications for output tensor
            check_vma: Whether to check for valid memory access patterns

        Returns:
            Tuple of (shard_map_fn, call_args)
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapper(
            query,
            key_cache,
            value_cache,
            context_len,
            page_indices,
        ):
            """Shard-map compatible wrapper that delegates to self.run with captured params."""
            return self.run(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                context_len=context_len,
                page_indices=page_indices,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
                softmax_scale=softmax_scale,
                mask_value=mask_value,
                attn_logits_soft_cap=attn_logits_soft_cap,
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
            context_len,
            page_indices,
        )

        return shard_map_fn, call_args


_prefill_page_attention_executor: Executor[PrefillPageAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("prefill-page-attention"),
    )
)


def prefill_page_attention(
    query: Float[Array, "chunk_size num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_len: Int[Array, "1"],
    page_indices: Int[Array, "num_pages"],
    /,
    *,
    softmax_scale: float | None = None,
    mask_value: float = -2.381976426469702e38,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: PrefillPageAttentionConfig | None = None,
) -> Float[Array, "chunk_size num_heads head_dim"]:
    """Execute prefill page attention with automatic optimization.

    Chunked prefill attention with paged KV cache for the prefill phase of
    LLM inference. Supports causal masking and sliding window attention.

    Args:
        query: Query tensor [chunk_size, num_q_heads, head_dim]
        key_cache: Paged key cache [num_kv_heads, total_num_pages, page_size, head_dim]
        value_cache: Paged value cache [num_kv_heads, total_num_pages, page_size, head_dim]
        context_len: Total context length including this chunk [1]
        page_indices: Page indices for this sequence [num_pages]
        softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim))
        mask_value: Value for masked positions (default: -inf)
        attn_logits_soft_cap: Soft cap value for attention logits
        sliding_window: If set, only attend to the last `sliding_window` tokens
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional configuration override

    Returns:
        Attention output [chunk_size, num_q_heads, head_dim]

    Example:
        >>> # Basic prefill with 128 token chunk
        >>> output = prefill_page_attention(
        ...     query, key_cache, value_cache,
        ...     context_len=jnp.array([512]),
        ...     page_indices=page_indices,
        ... )

        >>> # With sliding window attention
        >>> output = prefill_page_attention(
        ...     query, key_cache, value_cache,
        ...     context_len=jnp.array([1024]),
        ...     page_indices=page_indices,
        ...     sliding_window=256,
        ... )
    """
    return _prefill_page_attention_executor(
        PrefillPageAttention(),
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_len=context_len,
        page_indices=page_indices,
        softmax_scale=softmax_scale,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window=sliding_window,
        platform=platform,
        _cfg=cfg,
    )

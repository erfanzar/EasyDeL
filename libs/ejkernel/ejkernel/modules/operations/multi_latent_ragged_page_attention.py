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

"""Multi-Latent Ragged Page Attention (MLRPA) operation module.

This module provides the MLRPA operation, a paged attention mechanism
designed for Multi-Latent Attention (MLA) architectures such as
DeepSeek-V2/V3. It handles ragged (variable-length) sequences with
a paged KV cache layout, supporting both prefill and decode phases
in a single fused kernel.

Key characteristics:
    - Ragged batching: Variable-length sequences packed into a single
      tensor, indexed by ``query_start_loc``.
    - Paged KV cache: KV entries stored in fixed-size pages addressed
      via ``block_tables``, enabling efficient memory management.
    - MLA decomposition: Separate non-positional (nope) and positional
      (pe) query/key components for latent attention.
    - In-place cache update: New KV entries are written into the cache
      during the forward pass, returning the updated cache alongside
      the attention output.

Algorithm:
    For each query token the kernel computes scaled dot-product attention
    over its corresponding KV history stored in the paged cache. Prefill
    tokens with long contexts can be chunked (``chunk_prefill_size``) to
    bound peak memory.

Features:
    - Automatic platform selection (Pallas preferred on TPU, XLA fallback)
    - Shape-aware heuristic for ``num_kv_pages_per_block`` based on
      pages-per-sequence ratio
    - Autotuning over ``num_queries_per_block`` candidates [16, 32]
    - Configuration caching via ``PersistentCache``

Example:
    >>> from ejkernel.modules.operations import multi_latent_ragged_page_attention
    >>>
    >>> outputs, updated_cache = multi_latent_ragged_page_attention(
    ...     queries_nope, queries_pe,
    ...     keys_values, keys_pe,
    ...     kv_cache, kv_lens, block_tables,
    ...     query_start_loc, distribution,
    ...     softmax_scale=1.0 / head_dim**0.5,
    ... )
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int32

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
from .configs import MultiLatentRaggedPageAttentionConfig


class MultiLatentRaggedPageAttention(Kernel[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]]):
    """Multi-Latent Ragged Page Attention (MLRPA) operation.

    A paged attention kernel for MLA architectures that fuses cache
    update and attention computation for ragged-batched sequences.
    Supports both prefill (long context) and decode (single token)
    workloads through a unified interface.

    The kernel reads KV history from a paged cache addressed by
    ``block_tables``, computes scaled dot-product attention with
    separate non-positional and positional query/key streams, and
    writes incoming KV entries back into the cache in-place.

    Attributes:
        op_id: Operation identifier for registry lookup
            (``"multi_latent_ragged_page_attention"``).
    """

    version = "1"

    def __init__(self):
        """Initialize MLRPA with its registry operation identifier."""
        super().__init__(op_id="multi_latent_ragged_page_attention")

    def get_impl(self, cfg: MultiLatentRaggedPageAttentionConfig):
        """Resolve the concrete kernel implementation from the registry.

        Prefers Pallas on TPU when available, falling back to XLA.

        Args:
            cfg: Configuration specifying platform and backend.

        Returns:
            Callable kernel implementation for MLRPA.

        Raises:
            ValueError: If no matching implementation is found.
        """
        platform = detect_platform(
            "multi_latent_ragged_page_attention",
            cfg.platform,
            prefer_pallas=True,
        )
        return kernel_registry.get(
            "multi_latent_ragged_page_attention",
            platform=platform,
            backend=cfg.backend,
        )

    def run(
        self,
        queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
        queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
        keys_values: Float[Array, "total_tokens kv_latent_dim"],
        keys_pe: Float[Array, "total_tokens qk_pe_dim"],
        kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        *,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        mask_value: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: MultiLatentRaggedPageAttentionConfig,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads kv_latent_dim"],
        Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    ]:
        """Execute multi-latent ragged paged attention.

        Bare keyword arguments (``chunk_prefill_size``,
        ``num_kv_pages_per_block``, etc.) override the corresponding
        ``cfg`` fields when provided, allowing per-call overrides on
        top of the autotuned configuration.

        Args:
            queries_nope: Non-positional query component.
                Shape: [total_tokens, num_q_heads, qk_nope_dim]
            queries_pe: Positional (RoPE) query component.
                Shape: [total_tokens, num_q_heads, qk_pe_dim]
            keys_values: Incoming compressed KV / value projection to
                be written into the cache and attended over.
                Shape: [total_tokens, qk_nope_dim]
            keys_pe: Incoming positional key component.
                Shape: [total_tokens, qk_pe_dim]
            kv_cache: Paged KV cache tensor. Updated in-place with new
                entries and returned alongside the attention output.
                Shape: [num_pages, page_size/kv_packing, kv_packing,
                kv_dim_padded]
            kv_lens: Number of valid KV tokens per sequence.
                Shape: [max_num_seqs]
            block_tables: Flattened page table mapping each sequence to
                its cache pages (row-major, ``pages_per_seq`` columns).
                Shape: [max_num_seqs * pages_per_seq]
            query_start_loc: Cumulative token offsets marking the start
                of each sequence in the ragged batch (length
                ``max_num_seqs + 1``).
                Shape: [max_num_seqs + 1]
            distribution: Three-element descriptor ``[num_prefill_seqs,
                num_prefill_tokens, num_decode_tokens]`` characterising
                the workload mix for kernel dispatch.
                Shape: [3]
            softmax_scale: Attention logit scaling factor. Defaults to
                ``1 / sqrt(qk_nope_dim + qk_pe_dim)`` inside the kernel
                when ``None``.
            sliding_window: If set, restricts causal attention to the
                last ``sliding_window`` KV tokens per query.
            logits_soft_cap: If set, applies ``tanh`` soft-capping to
                attention logits before softmax.
            mask_value: Value used for masked logit positions (default
                is a large negative chosen by the kernel).
            q_scale: Optional multiplicative scale applied to queries.
            k_scale: Optional multiplicative scale applied to keys.
            v_scale: Optional multiplicative scale applied to values.
            platform: Override automatic platform selection.
            cfg: Kernel configuration object (from autotuner or manual).

        Returns:
            Tuple of:
                - output: Attention output.
                    Shape: [total_tokens, num_q_heads, qk_nope_dim]
                - updated_kv_cache: KV cache with new entries written.
                    Same shape as the input ``kv_cache``.
        """
        if platform is not None:
            cfg = MultiLatentRaggedPageAttentionConfig(
                chunk_prefill_size=cfg.chunk_prefill_size,
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                vmem_limit_bytes=cfg.vmem_limit_bytes,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        return impl(
            queries_nope=queries_nope,
            queries_pe=queries_pe,
            keys_values=keys_values,
            keys_pe=keys_pe,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            distribution=distribution,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=cfg.chunk_prefill_size,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
            vmem_limit_bytes=cfg.vmem_limit_bytes,
        )

    def _estimate_kv_pages(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ) -> int:
        """Estimate ``num_kv_pages_per_block`` from input shapes.

        Uses the ratio ``block_tables.shape[0] / kv_lens.shape[0]`` as
        a proxy for pages-per-sequence, then bins into a reasonable
        block size that balances compute intensity against register
        pressure.

        Args:
            inv: Invocation object carrying the call's keyword arguments.

        Returns:
            Heuristic ``num_kv_pages_per_block`` value (2, 4, 8, or 16).
        """
        try:
            kv_lens = inv.kwargs["kv_lens"]
            block_tables = inv.kwargs["block_tables"]
            pages_per_seq = max(1, int(block_tables.shape[0]) // int(kv_lens.shape[0]))
        except Exception:
            pages_per_seq = 16

        if pages_per_seq >= 256:
            return 16
        elif pages_per_seq >= 128:
            return 8
        elif pages_per_seq >= 64:
            return 4
        else:
            return 2

    def heuristic_cfg(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ) -> MultiLatentRaggedPageAttentionConfig:
        """Provide a default configuration based on input shapes.

        Picks ``num_kv_pages_per_block`` via the pages-per-sequence
        heuristic and defaults ``num_queries_per_block`` to 32.

        Args:
            inv: Invocation object with call metadata.

        Returns:
            Configuration with shape-derived block sizes.
        """
        return MultiLatentRaggedPageAttentionConfig(
            chunk_prefill_size=None,
            num_kv_pages_per_block=self._estimate_kv_pages(inv),
            num_queries_per_block=32,
            vmem_limit_bytes=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ):
        """Generate candidate configurations for autotuning.

        Keeps ``num_kv_pages_per_block`` fixed at the heuristic value
        and sweeps ``num_queries_per_block`` over [16, 32].

        Args:
            inv: Invocation object with call metadata.

        Returns:
            List of ``MultiLatentRaggedPageAttentionConfig`` candidates
            to benchmark.
        """
        kv_pages = self._estimate_kv_pages(inv)
        return [
            MultiLatentRaggedPageAttentionConfig(
                chunk_prefill_size=None,
                num_kv_pages_per_block=kv_pages,
                num_queries_per_block=q_block,
                vmem_limit_bytes=None,
                num_warps=4,
                num_stages=1,
                platform="auto",
                backend="any",
            )
            for q_block in (16, 32)
        ]

    def candidate_cfgs_gpu(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ):
        """Generate GPU candidates for TileLang and XLA MLRPA.

        Sweep the two main MLRPA knobs:

        * ``num_kv_pages_per_block`` ∈ {estimated, 2×estimated, 4} —
          bracket the per-page estimate.
        * ``num_queries_per_block`` ∈ {8, 16, 32, 64} — covers narrow
          (decode-style) and wide (prefill-style) batches.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        kv_pages = self._estimate_kv_pages(inv)
        kv_choices = sorted({max(1, kv_pages // 2), kv_pages, kv_pages * 2})
        candidates: list[MultiLatentRaggedPageAttentionConfig] = []
        for platform, backend in (("tilelang", "gpu"), ("xla", "any")):
            if platform not in platforms:
                continue
            candidates.extend(
                MultiLatentRaggedPageAttentionConfig(
                    chunk_prefill_size=None,
                    num_kv_pages_per_block=kv_block,
                    num_queries_per_block=q_block,
                    vmem_limit_bytes=None,
                    num_warps=4,
                    num_stages=1,
                    platform=platform,
                    backend=backend,
                )
                for kv_block in kv_choices
                for q_block in (8, 16, 32, 64)
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ):
        """Generate TPU candidates for Pallas and XLA MLRPA."""
        kv_pages = self._estimate_kv_pages(inv)
        return [
            MultiLatentRaggedPageAttentionConfig(
                chunk_prefill_size=None,
                num_kv_pages_per_block=kv_pages,
                num_queries_per_block=q_block,
                vmem_limit_bytes=None,
                num_warps=4,
                num_stages=1,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for q_block in (16, 32)
        ]


_mlrpa_executor: Executor[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=3, iters=40),
        persistent=PersistentCache("multi_latent_ragged_page_attention"),
    )
)


def multi_latent_ragged_page_attention(
    queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
    keys_values: Float[Array, "total_tokens kv_latent_dim"],
    keys_pe: Float[Array, "total_tokens qk_pe_dim"],
    kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    /,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: MultiLatentRaggedPageAttentionConfig | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Execute Multi-Latent Ragged Page Attention with automatic optimization.

    Fuses paged-cache update and MLA-style attention for ragged-batched
    sequences. Automatically selects the best platform (Pallas on TPU,
    XLA elsewhere) and autotunes ``num_queries_per_block``.

    Args:
        queries_nope: Non-positional query component.
            Shape: [total_tokens, num_q_heads, qk_nope_dim]
        queries_pe: Positional (RoPE) query component.
            Shape: [total_tokens, num_q_heads, qk_pe_dim]
        keys_values: Incoming compressed KV / value projection.
            Shape: [total_tokens, qk_nope_dim]
        keys_pe: Incoming positional key component.
            Shape: [total_tokens, qk_pe_dim]
        kv_cache: Paged KV cache tensor (updated in-place).
            Shape: [num_pages, page_size/kv_packing, kv_packing,
            kv_dim_padded]
        kv_lens: Number of valid KV tokens per sequence.
            Shape: [max_num_seqs]
        block_tables: Flattened page table (row-major).
            Shape: [max_num_seqs * pages_per_seq]
        query_start_loc: Cumulative token offsets per sequence.
            Shape: [max_num_seqs + 1]
        distribution: Workload descriptor ``[num_prefill_seqs,
            num_prefill_tokens, num_decode_tokens]``.
            Shape: [3]
        softmax_scale: Attention logit scale (default: auto).
        sliding_window: Causal sliding-window radius.
        logits_soft_cap: Tanh soft-cap for attention logits.
        mask_value: Value for masked logit positions.
        q_scale: Multiplicative query scale.
        k_scale: Multiplicative key scale.
        v_scale: Multiplicative value scale.
        platform: Override platform selection.
        cfg: Optional kernel configuration.

    Returns:
        Tuple of:
            - output: Attention output.
                Shape: [total_tokens, num_q_heads, qk_nope_dim]
            - updated_kv_cache: KV cache with new entries written.
                Same shape as ``kv_cache``.
    """
    return _mlrpa_executor(
        MultiLatentRaggedPageAttention(),
        queries_nope=queries_nope,
        queries_pe=queries_pe,
        keys_values=keys_values,
        keys_pe=keys_pe,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        platform=platform,
        _cfg=cfg,
    )


__all__ = (
    "MultiLatentRaggedPageAttention",
    "multi_latent_ragged_page_attention",
)

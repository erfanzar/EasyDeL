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


"""Ragged Page Attention module with automatic optimization.

This module implements ragged page attention, combining the benefits of both
ragged (variable-length) sequence processing and paged KV cache management.
This approach is particularly efficient for serving scenarios where sequences
have variable lengths and KV cache is organized in fixed-size pages.

Ragged page attention addresses key challenges in LLM inference:
    - Variable-length sequences without padding overhead
    - Efficient memory management through paged KV cache
    - Dynamic batching with different sequence lengths
    - Memory sharing for beam search and prefix caching

Key Concepts:
    Ragged Layout: Sequences are concatenated without padding, with start
        locations tracking where each sequence begins
    Pages: Fixed-size blocks holding portions of KV cache
    Block Tables: Mapping from logical sequence positions to physical pages

The combination provides:
    - Zero padding overhead (ragged layout)
    - Flexible memory allocation (paged cache)
    - Efficient batching of variable-length sequences
    - Support for dynamic sequence management

Memory Layout:
    Queries: [total_tokens, num_heads, head_dim] (ragged, no padding)
    KV Cache: [num_pages, page_size, num_heads, head_dim] (paged)

Mathematical Foundation:
    For token i in sequence s:
        start_idx = query_start_loc[s]
        end_idx = query_start_loc[s + 1]
        output[i] = attention(Q[start_idx:end_idx], K[pages[s]], V[pages[s]])

This is the most memory-efficient attention variant for serving workloads.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int

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
from .configs import RaggedPageAttentionv2Config


def _xla_block_candidates_v2(inv: Invocation[RaggedPageAttentionv2Config, Array]) -> list[RaggedPageAttentionv2Config]:
    """Generate power-of-2 XLA configurations with larger block sizes.

    Creates candidate configurations optimized for XLA backend execution by
    exploring power-of-2 block sizes for both KV pages and query tokens.
    These configurations are designed to reduce XLA loop overhead through
    larger tile sizes.

    Args:
        inv: Invocation object containing the attention arguments including
            queries tensor and block_tables for shape extraction.

    Returns:
        List of RaggedPageAttentionv2Config objects with XLA-optimized block
        sizes. Returns empty list if required arguments are missing or have
        invalid shapes.

    Note:
        KV page candidates: 16, 32, 64 (bounded by pages_per_seq)
        Query candidates: 64, 128, 256 (bounded by total_tokens)
    """
    try:
        queries = inv.kwargs["queries"]
        block_tables = inv.kwargs["block_tables"]
    except KeyError:
        return []

    total_tokens = int(getattr(queries, "shape", (0,))[0] or 0)
    pages_per_seq = int(getattr(block_tables, "shape", (0, 0))[1] or 0)
    if total_tokens <= 0 or pages_per_seq <= 0:
        return []

    kv_candidates = [k for k in (16, 32, 64) if k <= pages_per_seq]
    if not kv_candidates:
        kv_candidates = [min(16, pages_per_seq)]

    q_candidates = [q for q in (64, 128, 256) if q <= total_tokens]
    if not q_candidates:
        q_candidates = [min(64, total_tokens)]

    configs: list[RaggedPageAttentionv2Config] = []
    for kv in kv_candidates:
        for q in q_candidates:
            configs.append(
                RaggedPageAttentionv2Config(
                    num_kv_pages_per_block=kv,
                    num_queries_per_block=q,
                    num_warps=None,
                    num_stages=None,
                    platform="xla",
                    backend="any",
                )
            )
    return configs


class RaggedPageAttentionv2(Kernel[RaggedPageAttentionv2Config, Array]):
    """Ragged Page Attention with custom optimization logic.

    Combines ragged (variable-length) sequence processing with paged KV cache
    management for maximum memory efficiency in serving workloads.

    Features:
        - Zero padding overhead through ragged layout
        - Efficient paged KV cache management
        - Support for variable context lengths per sequence
        - Sliding window attention for long contexts
        - Logit soft capping for numerical stability
        - Attention sink mechanism for improved long-context performance
        - Configurable block sizes and memory limits
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    This implementation is particularly efficient for:
        - LLM serving with dynamic batching
        - Variable-length inference workloads
        - Memory-constrained deployment
        - Scenarios requiring efficient KV cache sharing

    The ragged layout eliminates padding overhead while paged cache
    enables flexible memory management and sharing.
    """

    def __init__(self):
        """Initialize Ragged Page Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="ragged_page_attention_v2")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
        context_lens: Int[Array, "num_seqs"],
        block_tables: Int[Array, "num_seqs pages_per_seq"],
        query_start_loc: Int[Array, "num_seqs_plus_one"],
        num_seqs: Array | int,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv2Config | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper specifically for ragged page attention.

        Ragged page attention handles variable-length sequences with paged KV cache,
        ideal for serving scenarios.

        Args:
            queries: Flattened queries [total_tokens, num_q_heads, head_dim]
            kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim]
            context_lens: Context lengths [num_seqs]
            block_tables: Block mapping [num_seqs, pages_per_seq]
            query_start_loc: Start locations [num_seqs + 1]
            num_seqs: Number of sequences
            All other args: Ragged page attention parameters to be fixed
            mesh: JAX device mesh
            in_specs: Input partition specs
                (for queries, kv_pages, context_lens, block_tables, query_start_loc, num_seqs, softmax_aux)
            out_specs: Output partition spec

        Returns:
            Tuple of (shard_map_fn, call_args)
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_ragged_page_attn(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
            context_lens: Int[Array, "num_seqs"],
            block_tables: Int[Array, "num_seqs pages_per_seq"],
            query_start_loc: Int[Array, "num_seqs_plus_one"],
            num_seqs: Array | int,
            softmax_aux: Float[Array, "num_sinks"] | None = None,
        ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
            """Shard-map compatible wrapper that delegates to self.run with captured params."""
            return self.run(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=num_seqs,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                compute_dtype=compute_dtype,
                optimized=optimized,
                sliding_window=sliding_window,
                softmax_aux=softmax_aux,
                mask_value=mask_value,
                vmem_limit_bytes=vmem_limit_bytes,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs,
            softmax_aux,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_ragged_page_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: RaggedPageAttentionv2Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ragged page attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ragged_page_attention_v2", cfg.platform)
        return kernel_registry.get("ragged_page_attention_v2", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
        context_lens: Int[Array, "num_seqs"],
        block_tables: Int[Array, "num_seqs pages_per_seq"],
        query_start_loc: Int[Array, "num_seqs_plus_one"],
        num_seqs: Array | int,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        *,
        cfg: RaggedPageAttentionv2Config,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute ragged page attention over variable-length sequences.

        Computes attention where queries are in ragged (concatenated) format
        and KV cache is organized in pages, providing maximum memory efficiency.

        Args:
            queries: Ragged query tensor [total_tokens, num_q_heads, head_dim]
                All sequences concatenated without padding
            kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim]
                Combined key-value cache in page format
            context_lens: Actual context length per sequence [num_seqs]
            block_tables: Page mapping [num_seqs, pages_per_seq] mapping logical
                pages to physical page indices
            query_start_loc: Start indices for each sequence in queries [num_seqs + 1]
                query_start_loc[i] to query_start_loc[i+1] defines sequence i
            num_seqs: Number of sequences in the batch
            softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            logits_soft_cap: Optional soft cap to bound attention logits
            compute_dtype: Data type for computation (default: bfloat16)
            optimized: Use optimized kernel implementation
            sliding_window: Window size for local attention (None for full attention)
            softmax_aux: Optional attention sink logits for long-context handling
            mask_value: Value to use for masked positions (default: -inf)
            vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific)
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object containing num_kv_pages_per_block and num_queries_per_block

        Returns:
            Attention output [total_tokens, num_q_heads, head_dim] in ragged format

        Note:
            The ragged format eliminates all padding overhead. Combined with paged
            KV cache, this provides the most memory-efficient attention implementation
            for serving workloads with variable-length sequences.

        Example:
            >>>
            >>> query_start_loc = jnp.array([0, 10, 25])
            >>> out = ragged_page_attention_v2(
            ...     queries, kv_pages, context_lens,
            ...     block_tables, query_start_loc, num_seqs=2
            ... )
        """

        if platform is not None:
            cfg = RaggedPageAttentionv2Config(
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            compute_dtype=compute_dtype,
            optimized=optimized,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            mask_value=mask_value,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    def heuristic_cfg(self, inv: Invocation[RaggedPageAttentionv2Config, Array]) -> RaggedPageAttentionv2Config:
        """Provide default configuration optimized for ragged page attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical ragged attention workloads with variable sequence lengths
        """
        return RaggedPageAttentionv2Config(
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RaggedPageAttentionv2Config, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for ragged attention scenarios with
        various batch sizes and sequence lengths.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Ragged attention performance depends on the distribution of sequence
            lengths and the page size. Candidates are chosen to work well across
            common serving scenarios.
        """

        block_configs = [
            (None, None, None, None),
            (1, 64, None, None),
            (2, 128, None, None),
        ]

        candidates = []
        for num_kv_pages, num_queries, num_warps, num_stages in block_configs:
            candidates.append(
                RaggedPageAttentionv2Config(
                    num_kv_pages_per_block=num_kv_pages,
                    num_queries_per_block=num_queries,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedPageAttentionv2Config, Array]):
        """Generate candidate configurations for autotuning on GPU (Triton).

        Produces a set of kernel configurations optimized for GPU execution with
        Triton backend. The heuristics balance memory efficiency and parallelism
        for ragged page attention workloads.

        Heuristics:
            - For small head_dim (<=64): BLOCK_M up to 128 is efficient
            - For medium head_dim (65-128): BLOCK_M 32-128 works well
            - For large head_dim (>128): Prefer smaller BLOCK_M (32-64)
            - More KV pages per block helps small page_size (<=32)
            - Constrain S_block = page_size * num_kv_pages_per_block <= 256
            - Higher warps (4-8) for larger blocks
            - Stages 2-4 for multi-page blocks

        Args:
            inv: Invocation object containing input tensors and metadata.
                Used to determine optimal block sizes based on workload shape.

        Returns:
            List of RaggedPageAttentionv2Config objects optimized for GPU.

        Note:
            The generated configs prioritize high-value combinations first
            (empirically good block sizes), then explore a broader grid.
        """
        q = inv.kwargs["queries"]
        kv = inv.kwargs["kv_pages"]
        block_tables = inv.kwargs["block_tables"]
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)

        _total_tokens, num_q_heads, head_dim = map(int, q.shape)
        page_size = int(kv.shape[1])
        pages_per_seq = int(block_tables.shape[1])
        combined_kv_heads = int(kv.shape[2])
        assert combined_kv_heads % 2 == 0
        num_kv_heads = combined_kv_heads // 2
        assert num_q_heads % num_kv_heads == 0

        if head_dim <= 64:
            m_opts = [32, 64, 128]
        elif head_dim <= 128:
            m_opts = [32, 64, 128]
        elif head_dim <= 192:
            m_opts = [32, 64]
        else:
            m_opts = [32, 64]

        if page_size <= 16:
            p_opts = [2, 4, 8]
        elif page_size <= 32:
            p_opts = [1, 2, 4]
        elif page_size <= 64:
            p_opts = [1, 2]
        else:
            p_opts = [1]

        max_S_block = 256
        p_opts = [p for p in p_opts if p * page_size <= max_S_block]

        p_opts = [p for p in p_opts if p <= pages_per_seq]
        if not p_opts:
            p_opts = [min(2, pages_per_seq)]

        def pick_warps_stages(block_m: int, npages: int) -> tuple[int, int]:
            """Select warp count and pipeline stages based on block size and head dimension."""
            if head_dim <= 64:
                warps = 2 if block_m <= 64 else 4
            elif head_dim <= 128:
                warps = 4 if block_m <= 64 else 8
            else:
                warps = 4 if block_m <= 64 else 8

            if npages >= 4:
                stages = 4
            elif npages == 2:
                stages = 3 if page_size <= 32 else 2
            else:
                stages = 2
            return warps, stages

        high_value: list[tuple[int, int, int | None, int | None]] = []

        hv_core = [(64, 2), (128, 2)]

        if page_size <= 32 and pages_per_seq >= 4:
            hv_core += [(64, 4)]
            if page_size <= 16 and pages_per_seq >= 8:
                hv_core += [(128, 4), (64, 8)]

        if page_size >= 64:
            hv_core += [(64, 1), (128, 1)]

        if head_dim >= 160:
            hv_core += [(32, 2), (64, 1)]

        seen_hv = set()
        for m, p in hv_core:
            if p in p_opts and m in m_opts and (m, p) not in seen_hv:
                w, s = pick_warps_stages(m, p)
                high_value.append((m, p, w, s))
                seen_hv.add((m, p))

        grid: list[tuple[int, int, int | None, int | None]] = []
        for m in m_opts:
            for p in p_opts:
                if (m, p) in seen_hv:
                    continue
                w, s = pick_warps_stages(m, p)
                grid.append((m, p, w, s))

        block_configs: list[tuple[int, int, int | None, int | None]] = []
        seen = set()
        for tup in high_value + grid:
            m, p, w, s = tup
            if (m, p) in seen:
                continue
            seen.add((m, p))
            block_configs.append((m, p, w, s))

        max_candidates = 18
        block_configs = block_configs[:max_candidates]

        candidates = (
            [
                RaggedPageAttentionv2Config(
                    num_kv_pages_per_block=npages,
                    num_queries_per_block=block_m,
                    num_warps=warps,
                    num_stages=stages,
                    platform="triton",
                    backend="gpu",
                )
                for (block_m, npages, warps, stages) in block_configs
            ]
            if "triton" in platforms
            else []
        )

        if "tilelang" in platforms:
            for stages in (2, 3, 4):
                candidates.append(
                    RaggedPageAttentionv2Config(
                        num_kv_pages_per_block=None,
                        num_queries_per_block=None,
                        num_warps=4,
                        num_stages=stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.extend(_xla_block_candidates_v2(inv))

        return candidates or [
            RaggedPageAttentionv2Config(
                num_kv_pages_per_block=None,
                num_queries_per_block=None,
                num_warps=None,
                num_stages=None,
                platform="xla",
                backend="any",
            )
        ]

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedPageAttentionv2Config, Array]):
        """Generate candidate configurations for autotuning on TPU (Pallas backend).

        Produces configurations optimized for TPU execution with Pallas backend.
        The heuristics balance memory efficiency and parallelism for ragged page
        attention workloads on TPU hardware.

        Heuristics:
            - For small head_dim, larger BLOCK_M is efficient (64-128)
            - For large head_dim (>=160), prefer smaller BLOCK_M (32-64)
            - More KV pages per block helps with small page_size (<=32)
            - Constrains S_block = page_size * num_kv_pages_per_block <= 256

        Args:
            inv: Invocation object containing input tensors and metadata.
                Used to determine optimal block sizes based on workload shape.

        Returns:
            List of RaggedPageAttentionv2Config objects optimized for TPU.
            Returns empty list if required arguments are missing or invalid.

        Note:
            KV page candidates: 16, 32, 64 (bounded by pages_per_seq)
            Query candidates: 4, 8, 16, 32, 64 (bounded by total_tokens)
        """

        try:
            queries = inv.kwargs["queries"]
            block_tables = inv.kwargs["block_tables"]
        except KeyError:
            return []

        total_tokens = int(getattr(queries, "shape", (0,))[0] or 0)
        pages_per_seq = int(getattr(block_tables, "shape", (0, 0))[1] or 0)
        if total_tokens <= 0 or pages_per_seq <= 0:
            return []

        kv_candidates = [k for k in (16, 32, 64) if k <= pages_per_seq]
        if not kv_candidates:
            kv_candidates = [min(16, pages_per_seq)]

        q_candidates = [q for q in (4, 8, 16, 32, 64) if q <= total_tokens]
        if not q_candidates:
            q_candidates = [min(4, total_tokens)]

        configs = []
        for kv in kv_candidates:
            for q in q_candidates:
                configs.append(
                    RaggedPageAttentionv2Config(
                        num_kv_pages_per_block=kv,
                        num_queries_per_block=q,
                        num_warps=None,
                        num_stages=None,
                        platform="pallas",
                        backend="tpu",
                    )
                )
        configs.extend(_xla_block_candidates_v2(inv))

        return configs

    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu


_ragged_page_attention_executor: Executor[RaggedPageAttentionv2Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ragged-page-attention"),
    )
)


def ragged_page_attention_v2(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    mask_value: float | None = None,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedPageAttentionv2Config | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Execute ragged page attention with automatic optimization.

    Ragged page attention efficiently handles variable-length sequences
    in a single batch using flattened token representation and page-based KV cache.

    Args:
        queries: Flattened query tensor [total_tokens, num_q_heads, head_dim]
        kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim]
        context_lens: Context length per sequence [num_seqs]
        block_tables: Block mapping table [num_seqs, pages_per_seq]
        query_start_loc: Start locations for each sequence [num_seqs + 1]
        num_seqs: Number of sequences in the batch
        softmax_scale: Softmax scaling factor
        logits_soft_cap: Soft capping value for logits
        compute_dtype: Computation dtype (default: bfloat16)
        optimized: Use optimized implementation
        sliding_window: Sliding window size for local attention
        softmax_aux: Attention sink logits
        mask_value: Value for masked positions
        vmem_limit_bytes: Memory limit in bytes
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional config override (num_kv_pages_per_block and num_queries_per_block are set via cfg)
        mesh: JAX device mesh for shard_map execution (optional)
        in_specs: Input partition specs for shard_map (optional)
        out_specs: Output partition spec for shard_map (optional)

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim]

    Example:
        >>>
        >>> out = ragged_page_attention_v2(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs
        ... )
        >>>
        >>>
        >>> out = ragged_page_attention_v2(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs, sliding_window=256
        ... )
        >>>
        >>>
        >>> out = ragged_page_attention_v2(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs, optimized=True, logits_soft_cap=50.0
        ... )
        >>>
        >>>
        >>> out = ragged_page_attention_v2(..., platform="triton")
    """

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _ragged_page_attention_executor(
        RaggedPageAttentionv2(),
        queries=queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        compute_dtype=compute_dtype,
        optimized=optimized,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        mask_value=mask_value,
        vmem_limit_bytes=vmem_limit_bytes,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

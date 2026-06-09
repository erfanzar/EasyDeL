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


"""Flash Attention module with automatic optimization.

This module implements Flash Attention, a memory-efficient attention mechanism
that uses tiling and recomputation to achieve O(N) memory complexity instead
of the standard O(N²) for sequence length N.

Key features of Flash Attention:
    - Memory-efficient: Uses tiling to process attention in blocks
    - IO-aware: Minimizes HBM (high bandwidth memory) accesses
    - Exact: Produces numerically identical results to standard attention
    - Fast: Often faster than standard attention despite recomputation

The algorithm works by:
    1. Splitting Q, K, V into blocks along sequence dimension
    2. Computing attention block-by-block with on-the-fly softmax
    3. Using online softmax correction for numerical stability
    4. Fusing operations to minimize memory transfers

Supports:
    - Causal and non-causal masking
    - Variable sequence lengths via cumulative sequence lengths
    - Dropout (during training)
    - Sliding window attention
    - Multi-query and grouped-query attention patterns
    - Attention biasing and soft capping

Mathematical formulation:
    Standard: Attention(Q,K,V) = softmax(QK^T/√d)V
    Flash: Same output, but computed in O(N) memory via tiling

Reference:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    (Dao et al., 2022) https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import math
import os
from typing import Literal

from jax import lax, shard_map
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    BwdParams,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    FwdParams,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache
from ejkernel.types.mask import MaskInfo

from ..base import detect_platform
from .configs import FlashAttentionConfig

PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]


class FlashAttention(Kernel[FlashAttentionConfig, Array]):
    """Flash Attention with custom optimization logic.

    Memory-efficient exact attention with O(N) memory complexity.
    Supports causal masking, dropout, sliding windows, and variable-length sequences.

    Features:
        - Automatic platform/backend selection (Triton/Pallas/XLA)
        - Configuration caching for consistent performance
        - Optional autotuning to find optimal implementation
        - Custom gradient support for efficient backpropagation
        - Support for variable-length sequences via cumulative sequence lengths
        - Sliding window attention for local attention patterns
        - Logits soft capping for numerical stability

    Example:
        >>> from ejkernel.modules import FlashAttention, create_default_executor
        >>>
        >>>
        >>> executor = create_default_executor()
        >>> attn = FlashAttention()
        >>>
        >>>
        >>> output = executor(attn, query, key, value, causal=True, softmax_scale=0.125)
        >>>
        >>>
        >>> output = executor(
        ...     attn, query, key, value,
        ...     cum_seqlens_q=cu_seqlens_q,
        ...     cum_seqlens_k=cu_seqlens_k
        ... )
        >>>
        >>>
        >>> output = executor(attn, query, key, value, sliding_window=(256, 256))
    """

    version = "2"

    def __init__(self):
        """Initialize Flash Attention module."""
        super().__init__(op_id="flash_attention")

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: DenseKV | PagedKV,
        value: DenseKV | PagedKV,
        attention_mask: (
            Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
            | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
            | None
        ) = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        softmax_scale: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        dropout_seed: int | None = None,
        cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
        cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
        block_tables: BlockTables | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        normalize_output: bool = True,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT,
        logits_dtype: DTypeLike = jnp.float32,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        cfg: FlashAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper specifically for flash attention.

        Args:
            query, key, value: Input tensors to be sharded
            mesh: JAX device mesh
            in_specs: Input partition specs (for q, k, v, and optionally mask/bias)
            out_specs: Output partition spec
            All other args: Flash attention parameters to be fixed via partial

        Returns:
            Tuple of (shard_map_fn, call_args)
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wraped_flash_attn(
            query: Float[Array, "batch num_heads seq_len head_dim"],
            key: Float[Array, "batch kv_num_heads kv_len head_dim"],
            value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
            bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
            softmax_aux: Float[Array, "num_sinks"] | None = None,
            cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
            cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
            attention_mask: Int[Array, "batch num_heads seq_len kv_len"] | None = None,
            block_tables: BlockTables | None = None,
            q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
            kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        ) -> Float[Array, "batch num_heads seq_len head_dim"]:
            """Shard-local flash attention forwarding to self.run."""
            return self.run(
                query=query,
                key=key,
                value=value,
                bias=bias,
                softmax_aux=softmax_aux,
                cum_seqlens_k=cum_seqlens_k,
                cum_seqlens_q=cum_seqlens_q,
                block_tables=block_tables,
                attention_mask=attention_mask,
                softmax_scale=softmax_scale,
                dropout_prob=dropout_prob,
                causal=causal,
                dropout_seed=dropout_seed,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                normalize_output=normalize_output,
                precision=precision,
                logits_dtype=logits_dtype,
                kv_segment_ids=kv_segment_ids,
                q_segment_ids=q_segment_ids,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            query,
            key,
            value,
            bias,
            softmax_aux,
            cum_seqlens_q,
            cum_seqlens_k,
            attention_mask,
            block_tables,
            q_segment_ids,
            kv_segment_ids,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wraped_flash_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: FlashAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation

        Raises:
            ValueError: If no matching implementation is found
        """
        return kernel_registry.get(
            algorithm="flash_attention",
            platform=detect_platform(
                "flash_attention",
                cfg.platform,
                prefer_cuda=True,
            ),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: DenseKV | PagedKV,
        value: DenseKV | PagedKV,
        attention_mask: (
            Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
            | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
            | None
        ) = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        softmax_scale: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        dropout_seed: int | None = None,
        cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
        cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
        block_tables: BlockTables | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        normalize_output: bool = True,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT,
        logits_dtype: DTypeLike = jnp.float32,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        cfg: FlashAttentionConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        """Execute flash attention with the given configuration.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch, seq_len_k, num_heads, head_dim]
            value: Value tensor [batch, seq_len_k, num_heads, head_dim]
            attention_mask: Optional attention mask (legacy, prefer bias)
            bias: Optional attention bias tensor
            softmax_scale: Scaling factor for attention scores
            dropout_prob: Dropout probability for attention weights
            causal: Whether to apply causal masking
            dropout_seed: Random seed for dropout
            cum_seqlens_q: Cumulative sequence lengths for variable-length queries
            cum_seqlens_k: Cumulative sequence lengths for variable-length keys
            block_tables: Optional paged-KV block table of shape
                ``(batch, max_blocks)``.
            sliding_window: Window size for local attention
            logits_soft_cap: Optional soft cap value for logits
            softmax_aux: Optional attention sink logits
            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
            cfg: Configuration object specifying platform/backend
            segment_ids: Segment IDs for grouped sequences (TPU-specific)
            block_sizes: Block sizes for kernel execution (TPU-specific)

        Returns:
            Attention output [batch, seq_len_q, num_heads, head_dim]
        """

        if platform is not None:
            cfg = FlashAttentionConfig(
                fwd_params=cfg.fwd_params,
                bwd_params=cfg.bwd_params,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        out = impl(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            causal=causal,
            dropout_seed=dropout_seed,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
            block_tables=block_tables,
            sliding_window=sliding_window,
            fwd_params=cfg.fwd_params,
            bwd_params=cfg.bwd_params,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=logits_dtype,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
        )
        if q_segment_ids is not None:
            q_valid = q_segment_ids >= 0
            out = jnp.where(q_valid[:, :, None, None], out, 0)
        return out

    def heuristic_cfg_gpu(self, inv: Invocation[FlashAttentionConfig, Array]) -> FlashAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        q = inv.kwargs["query"]
        head_dim = int(q.shape[-1])
        use_segments = (inv.kwargs.get("q_segment_ids") is not None) or (inv.kwargs.get("kv_segment_ids") is not None)

        kv_block = 32 if (use_segments or head_dim >= 128) else 128

        return FlashAttentionConfig(
            fwd_params=FwdParams(q_blocksize=64, kv_blocksize=kv_block, num_warps=4, num_stages=2),
            bwd_params=BwdParams(
                q_blocksize=32,
                kv_blocksize=32,
                num_warps=4,
                num_stages=2,
            ),
            platform="triton",
            backend="gpu",
        )

    def heuristic_cfg_tpu(self, inv: Invocation[FlashAttentionConfig, Array]) -> FlashAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return FlashAttentionConfig(
            fwd_params=FwdParams(
                q_blocksize=128,
                kv_blocksize=128,
                num_warps=None,
                num_stages=None,
            ),
            bwd_params=BwdParams(
                q_blocksize=128,
                kv_blocksize=128,
                num_warps=None,
                num_stages=None,
            ),
            platform="pallas",
            backend="tpu",
        )

    def heuristic_cfg(self, inv: Invocation[FlashAttentionConfig, Array]) -> FlashAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return FlashAttentionConfig(
            fwd_params=FwdParams(
                q_blocksize=128,
                kv_blocksize=128,
                num_warps=None,
                num_stages=None,
            ),
            bwd_params=BwdParams(
                q_blocksize=128,
                kv_blocksize=128,
                num_warps=None,
                num_stages=None,
            ),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates multiple block size configurations for benchmarking to find
        the optimal tiling parameters for the given input shapes.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of candidate configurations to test during autotuning

        Note:
            The autotuning system will benchmark each candidate and select
            the fastest one for the given input configuration.
        """

        block_configs = [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]

        candidates = []
        for chunk_q, chunk_k in block_configs:
            candidates.append(
                FlashAttentionConfig(
                    fwd_params=FwdParams(q_blocksize=chunk_q, kv_blocksize=chunk_k, num_warps=4, num_stages=2),
                    bwd_params=BwdParams(q_blocksize=chunk_q // 2, kv_blocksize=chunk_k // 2, num_warps=4, num_stages=2),
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate GPU-optimized candidate configurations for autotuning (Triton).

        Heuristics:
        - q/kv blocks adapt to head_dim and sequence lengths.
        - If sliding_window is set, kv blocks are capped near the window span.
        - num_warps: 2-8 based on head_dim and block sizes.
        - num_stages: 2-3 (kept low to reduce SMEM pressure).
        - Conservative shared-memory guard to avoid CUDA errors.
        - Backward blocks smaller to reduce register pressure.
        """

        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        head_dim = int(q.shape[-1])
        q_len = int(q.shape[1])
        k_len = int(k.shape[1])
        dtype = q.dtype

        sliding_window = inv.kwargs.get("sliding_window", None)
        causal = bool(inv.kwargs.get("causal", True))
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "cuda", "cute", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)

        def window_total(sw):
            """Compute total window span from a sliding window specification."""
            if sw is None:
                return None
            if isinstance(sw, int):
                right = 0 if causal else sw
                return sw + right + 1
            wl, wr = sw
            return wl + wr + 1

        win = window_total(sliding_window)

        smem_limit = int(os.getenv("EJKERNEL_TRITON_SMEM_LIMIT", str(99 * 1024)))

        def next_pow2_ge(x: int, min_val: int = 16) -> int:
            """Return the smallest power of two >= x, with a minimum of min_val."""
            return max(min_val, 1 << math.ceil(math.log2(max(1, x))))

        block_headdim = next_pow2_ge(head_dim, 16)
        elem_bytes = 2 if dtype in (jnp.float16, jnp.bfloat16) else 4

        def smem_est_bytes(qb: int, kb: int, num_stages: int) -> int:
            """Estimate shared memory usage in bytes for given block sizes and stages."""
            kv_bytes = 2 * kb * block_headdim * elem_bytes
            q_bytes = int(0.25 * qb * block_headdim * elem_bytes)
            base = kv_bytes + q_bytes
            stage_factor = 1.0 + 0.5 * max(0, num_stages - 2)
            fudge = 2.5
            return int(base * stage_factor * fudge)

        if head_dim <= 64:
            q_opts = [32, 64, 128]
        elif head_dim <= 128:
            q_opts = [32, 64, 128]
        elif head_dim <= 192:
            q_opts = [32, 64, 128]
        else:
            q_opts = [32, 64]

        base_kv = [32, 64, 128, 256]
        if win is not None:
            target = max(32, min(256, 1 << (int(math.log2(max(32, win))) if win > 0 else 5)))
            kv_opts = sorted(set([32, 64, min(128, target), min(256, target)]))
        else:
            kv_opts = base_kv

        if k_len < 128:
            kv_opts = [x for x in kv_opts if x <= 128] or [64, 128]
        if q_len < 128:
            q_opts = [x for x in q_opts if x <= 128] or [64, 128]

        def pick_warps_stages(qb: int, kb: int, dh: int) -> tuple[int, int]:
            """Select num_warps and num_stages based on block sizes and head dim."""
            if dh <= 64:
                warps = 2 if max(qb, kb) <= 64 else 4
            elif dh <= 128:
                warps = 4 if max(qb, kb) <= 128 else 8
            else:
                warps = 8 if max(qb, kb) >= 128 else 4

            stages = 3 if kb >= 128 else 2
            return warps, stages

        def bwd_block(x: int, cap: int = 128) -> int:
            """Compute backward block size from a forward block size."""
            return max(32, min(cap, x // 2 if x >= 64 else x))

        hv_pairs = []
        preferred = [(64, 64), (128, 64), (64, 128), (128, 128)]
        if win is not None:
            preferred.insert(0, (64, min(128, max(64, win))))
            preferred.insert(0, (32, min(128, max(64, win))))
        for qb, kb in preferred:
            if qb in q_opts and kb in kv_opts:
                hv_pairs.append((qb, kb))

        grid_pairs = []
        for qb in q_opts:
            for kb in kv_opts:
                if (qb, kb) not in hv_pairs:
                    grid_pairs.append((qb, kb))

        max_candidates = 18
        pairs = []
        seen = set()
        for qb, kb in hv_pairs + grid_pairs:
            if (qb, kb) in seen:
                continue
            w, s = pick_warps_stages(qb, kb, head_dim)
            if smem_est_bytes(qb, kb, s) <= smem_limit:
                seen.add((qb, kb))
                pairs.append((qb, kb, w, s))
                if len(pairs) >= max_candidates:
                    break

        if not pairs:
            qb, kb = 64, 64
            w, s = pick_warps_stages(qb, kb, head_dim)
            pairs = [(qb, kb, w, s)]

        configs: list[FlashAttentionConfig] = []
        if "triton" in platforms:
            for qb, kb, w, s in pairs:
                configs.append(
                    FlashAttentionConfig(
                        fwd_params=FwdParams(q_blocksize=qb, kv_blocksize=kb, num_warps=w, num_stages=s),
                        bwd_params=BwdParams(q_blocksize=bwd_block(qb), kv_blocksize=bwd_block(kb)),
                        platform="triton",
                        backend="gpu",
                    )
                )
        for platform, backend, max_platform_candidates in (
            ("cuda", "gpu", 4),
            ("cute", "gpu", 2),
            ("tilelang", "gpu", 4),
            ("xla", "any", 4),
        ):
            if platform not in platforms:
                continue
            for qb, kb, w, s in pairs[:max_platform_candidates]:
                configs.append(
                    FlashAttentionConfig(
                        fwd_params=FwdParams(q_blocksize=qb, kv_blocksize=kb, num_warps=w, num_stages=s),
                        bwd_params=BwdParams(q_blocksize=bwd_block(qb), kv_blocksize=bwd_block(kb)),
                        platform=platform,
                        backend=backend,
                    )
                )
        return configs

    def candidate_cfgs_tpu(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning (Pallas).

        Heuristics:
        - Favor moderate Q blocks (32-128) and KV blocks (64-256/512).
        - If sliding_window is set, prefer kv blocks ≲ window span.
        - Slightly smaller backward blocks to reduce VMEM/regs.
        - Keep the candidate list compact and ordered for fast convergence.
        """

        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        q_len = int(q.shape[1])
        k_len = int(k.shape[1])

        sliding_window = inv.kwargs.get("sliding_window", None)
        causal = bool(inv.kwargs.get("causal", True))

        def win_span(sw):
            """Compute total window span from a sliding window specification."""
            if sw is None:
                return None
            if isinstance(sw, int):
                right = 0 if causal else sw
                return sw + right + 1
            wl, wr = sw
            return wl + wr + 1

        def round128(x: int | float) -> int:
            """Round x up to the nearest multiple of 128."""
            return 128 * max(1, round(float(x) / 128.0))

        win = win_span(sliding_window)

        q_opts = [128, 256]
        kv_opts = [128, 256, 512]

        if win is not None:
            target = max(128, min(512, round128(win)))
            kv_opts = sorted(set([*kv_opts, target, min(512, 2 * target)]))

        if q_len < 256:
            q_opts = [x for x in q_opts if x <= 256] or [128]
        if k_len < 256:
            kv_opts = [x for x in kv_opts if x <= 256] or [128, 256]

        def bwd_tile(_x: int) -> int:
            """Return a fixed backward tile size of 128."""
            return 128

        hv_pairs: list[tuple[int, int]] = []
        if win is not None:
            t1 = max(128, min(512, round128(win)))
            hv_pairs += [(128, t1), (256, t1), (128, min(512, 2 * t1))]
        hv_pairs += [(128, 128), (128, 256), (256, 256), (256, 512)]

        selected: list[tuple[int, int]] = []
        seen = set()
        for qb, kb in hv_pairs:
            if qb in q_opts and kb in kv_opts and (qb, kb) not in seen:
                selected.append((qb, kb))
                seen.add((qb, kb))

        for qb in q_opts:
            for kb in kv_opts:
                if (qb, kb) not in seen:
                    selected.append((qb, kb))
                    seen.add((qb, kb))
                if len(selected) >= 16:
                    break
            if len(selected) >= 16:
                break

        configs: list[FlashAttentionConfig] = []
        for qb, kb in selected:
            configs.append(
                FlashAttentionConfig(
                    fwd_params=FwdParams(q_blocksize=qb, kv_blocksize=kb, num_warps=None, num_stages=None),
                    bwd_params=BwdParams(
                        q_blocksize=bwd_tile(qb), kv_blocksize=bwd_tile(kb), num_warps=None, num_stages=None
                    ),
                    platform="pallas",
                    backend="tpu",
                )
            )
        configs.extend(self.candidate_cfgs_xla(inv)[:4])
        return configs

    def candidate_cfgs_xla(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate XLA-optimized candidate configurations for autotuning.

        Heuristics:
        - Medium blocks (128-256) tend to be robust.
        - If sliding_window is set, keep kv blocks near window span.
        - Backward tiles are smaller.
        - Keep list small and ordered by likely winners.
        """

        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        q_len = int(q.shape[1])
        k_len = int(k.shape[1])

        sliding_window = inv.kwargs.get("sliding_window", None)
        causal = bool(inv.kwargs.get("causal", True))

        def win_span(sw):
            """Compute total window span from a sliding window specification."""
            if sw is None:
                return None
            if isinstance(sw, int):
                right = 0 if causal else sw
                return sw + right + 1
            wl, wr = sw
            return wl + wr + 1

        def round128(x: int | float) -> int:
            """Round x up to the nearest multiple of 128."""
            return 128 * max(1, round(float(x) / 128.0))

        win = win_span(sliding_window)

        q_opts = [128, 256]
        kv_opts = [128, 256, 512]

        if win is not None:
            target = max(128, min(512, round128(win)))
            kv_opts = sorted(set([*kv_opts, target, min(512, 2 * target)]))

        if q_len < 256:
            q_opts = [x for x in q_opts if x <= 256] or [128]
        if k_len < 256:
            kv_opts = [x for x in kv_opts if x <= 256] or [128, 256]

        def bwd_tile(_x: int) -> int:
            """Return a fixed backward tile size of 128."""
            return 128

        hv_pairs: list[tuple[int, int]] = []
        if win is not None:
            t1 = max(128, min(512, round128(win)))
            hv_pairs += [(128, t1), (256, t1)]
        hv_pairs += [(128, 128), (128, 256), (256, 256), (256, 128)]

        selected: list[tuple[int, int]] = []
        seen = set()
        for qb, kb in hv_pairs:
            if qb in q_opts and kb in kv_opts and (qb, kb) not in seen:
                selected.append((qb, kb))
                seen.add((qb, kb))

        for qb in q_opts:
            for kb in kv_opts:
                if (qb, kb) not in seen:
                    selected.append((qb, kb))
                    seen.add((qb, kb))
                if len(selected) >= 12:
                    break
            if len(selected) >= 12:
                break

        configs: list[FlashAttentionConfig] = []
        for qb, kb in selected:
            configs.append(
                FlashAttentionConfig(
                    fwd_params=FwdParams(q_blocksize=qb, kv_blocksize=kb, num_warps=None, num_stages=None),
                    bwd_params=BwdParams(
                        q_blocksize=bwd_tile(qb), kv_blocksize=bwd_tile(kb), num_warps=None, num_stages=None
                    ),
                    platform="xla",
                    backend="any",
                )
            )
        return configs

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_xla = candidate_cfgs_xla


_flash_executor: Executor[FlashAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("flash-attn", cfg_type=FlashAttentionConfig),
    )
)


def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: DenseKV | PagedKV,
    value: DenseKV | PagedKV,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    block_tables: BlockTables | None = None,
    /,
    *,
    mask_info: MaskInfo | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: FlashAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Execute flash attention with automatic optimization.

    Convenience function that uses a default executor and flash attention module.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_heads, head_dim]
        mask_info: Optional MaskInfo containing attention mask and/or segment IDs
        bias: Optional attention bias tensor
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights
        causal: Whether to apply causal masking
        dropout_seed: Random seed for dropout
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
        sliding_window: Window size for local attention (int or (left, right) tuple)
        logits_soft_cap: Optional soft cap value for logits
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When provided, *key* and *value* are
            interpreted as paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional configuration override
        mesh: JAX device mesh for shard_map execution (optional)
        in_specs: Input partition specs for shard_map (optional)
        out_specs: Output partition spec for shard_map (optional)

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = flash_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, cum_seqlens_q=cu_q, cum_seqlens_k=cu_k)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, platform="triton")
    """

    attention_mask = None
    q_segment_ids = None
    kv_segment_ids = None

    if mask_info is not None:
        attention_mask = mask_info._attention_mask
        if mask_info._q_segment_ids is not None or mask_info._kv_segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask_info.get_or_compute_segment_ids(per_head=False)
        elif attention_mask is None:
            attention_mask = mask_info.get_or_compute_attention_mask()

    if block_tables is not None and platform in (None, "auto"):
        platform = "cute"

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
        if mask_info is None:
            in_specs = (*in_specs, None, None, None, None)
        else:
            shardings = mask_info.get_shardings(False, mesh=mesh)
            in_specs = (
                *in_specs,
                shardings.attention_mask if attention_mask is not None else None,
                None,
                shardings.q_segment_ids if q_segment_ids is not None else None,
                shardings.kv_segment_ids if kv_segment_ids is not None else None,
            )

    return _flash_executor(
        FlashAttention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        block_tables=block_tables,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

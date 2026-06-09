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


"""Block-sparse attention module with automatic optimization.

This module implements block-sparse attention, which applies attention only to
predefined blocks of the attention matrix, significantly reducing computational
cost for long sequences while maintaining important attention patterns.

The block-sparse pattern is defined by a mask builder function that determines
which blocks should be computed. This is particularly useful for document-level
attention, local attention patterns, and sparse attention architectures.

Key Features:
    - Sparse block patterns via customizable mask builders
    - Computational complexity reduction from O(N^2) to O(N * B)
    - Support for causal masking and sliding windows
    - Automatic platform selection (Triton/Pallas/XLA)
    - Gradient support with custom VJP for efficient training
    - Logit soft capping for numerical stability (Gemma-2 style)

Use Cases:
    - Long document processing where full attention is prohibitive
    - Architectures with specific attention patterns (e.g., Longformer, BigBird)
    - Custom sparsity patterns based on document structure
    - Efficient attention for sequences with known locality properties

Mathematical Foundation:
    Standard attention computes the full N x N attention matrix:
        A_ij = softmax(Q_i @ K_j^T / sqrt(d)) @ V_j for all i, j

    Block-sparse attention only computes blocks where mask_builder returns True:
        A_ij = softmax(Q_i @ K_j^T / sqrt(d)) @ V_j for (i,j) in sparse_pattern

    This reduces computation from O(N^2 * d) to O(B * block_size^2 * d)
    where B is the number of active blocks.

Sparse Patterns Supported:
    - Local attention: each token attends to nearby tokens
    - Global + local: special tokens attend globally, others locally
    - Strided patterns: attend to every nth position
    - Document structure: based on paragraph/section boundaries

References:
    - Longformer: https://arxiv.org/abs/2004.05150
    - BigBird: https://arxiv.org/abs/2007.14062
    - Splash Attention (TPU): https://arxiv.org/abs/2309.08630
"""

from __future__ import annotations

import math
import os
import typing

from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Float, Int

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
from .configs import BlockSparseAttentionConfig

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask


class BlockSparseAttention(Kernel[BlockSparseAttentionConfig, Array]):
    """Block-sparse attention kernel with custom optimization logic.

    Implements attention computation over sparse block patterns, computing attention
    only for specified blocks rather than the full attention matrix. This reduces
    computational complexity from O(N^2) to O(N * B) where B is the average number
    of blocks per row.

    Features:
        - Configurable sparse block patterns via mask builder
        - Support for causal masking and sliding windows
        - Automatic platform/backend selection
        - Optional autotuning for optimal block sizes
        - Gradient support for training with custom VJP
        - Logit soft capping with tanh activation for numerical stability (Gemma-2 style)
        - Separate forward/backward block sizes for performance tuning

    The mask builder function defines which blocks to compute, enabling patterns like:
        - Local attention (nearby tokens only)
        - Global + local (attending to special tokens + local context)
        - Strided patterns (every nth block)
        - Custom patterns based on document structure

    Example:
        >>> from ejkernel.modules.operations import BlockSparseAttention
        >>> from ejkernel.modules import create_default_executor
        >>>
        >>> executor = create_default_executor()
        >>> attn = BlockSparseAttention()
        >>>
        >>>
        >>> def local_mask(q_idx, k_idx, q_size, k_size, window):
        ...
        ...     pass
        >>>
        >>> output = executor(
        ...     attn,
        ...     query, key, value,
        ...     mask_builder=local_mask,
        ...     chunk_size=128
        ... )
    """

    def __init__(self):
        """Initialize BlockSparseAttention module."""
        super().__init__(op_id="blocksparse_attention")

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        q_segment_ids: Int[Array, "batch seq_len"] | None = None,
        kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
        q_positions: Int[Array, "batch seq_len"] | None = None,
        kv_positions: Int[Array, "batch kv_len"] | None = None,
        sequence_parallelism_mesh_axis_name: str | None = None,
        logits_soft_cap: float | None = None,
        qkv_layouts: tuple["SparseMask"] | None = None,
        softmax_scale: float | None = None,
        mask_builder: (
            typing.Callable[[int, int, int, int, int], "Mask"] | typing.Callable[[], "SparseMask"] | None
        ) = None,
        sliding_window: int | tuple[int, int] | None = None,
        chunk_size: int | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: BlockSparseAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper specifically for blocksparse attention.

        Args:
            mesh: JAX device mesh
            in_specs: Input partition specs (must match length of tensor args)
            out_specs: Output partition spec
            query, key, value: Input tensors to be sharded
            All other args: Blocksparse attention parameters

        Returns:
            Tuple of (shard_map_fn, call_args)
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_blocksparse_attn(
            query: Float[Array, "batch num_heads seq_len head_dim"],
            key: Float[Array, "batch kv_num_heads kv_len head_dim"],
            value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
            softmax_aux: Float[Array, "num_sinks"] | None,
            bias: Float[Array, "batch num_heads seq_len kv_len"] | None,
            q_segment_ids: Int[Array, "batch seq_len"] | None,
            kv_segment_ids: Int[Array, "batch kv_len"] | None,
            q_positions: Int[Array, "batch seq_len"] | None,
            kv_positions: Int[Array, "batch kv_len"] | None,
        ) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
            """Shard-local blocksparse attention forwarding to self.run."""
            return self.run(
                query=query,
                key=key,
                value=value,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                q_positions=q_positions,
                kv_positions=kv_positions,
                softmax_aux=softmax_aux,
                bias=bias,
                sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
                logits_soft_cap=logits_soft_cap,
                qkv_layouts=qkv_layouts,
                softmax_scale=softmax_scale,
                mask_builder=mask_builder,
                sliding_window=sliding_window,
                chunk_size=chunk_size,
                causal=causal,
                fused_backward=fused_backward,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
            )

        call_args = (
            query,
            key,
            value,
            softmax_aux,
            bias,
            q_segment_ids,
            kv_segment_ids,
            q_positions,
            kv_positions,
        )

        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        shard_map_fn = shard_map(
            _wrapped_blocksparse_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: BlockSparseAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for block-sparse attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        return kernel_registry.get(
            algorithm="blocksparse_attention",
            platform=detect_platform("blocksparse_attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        q_segment_ids: Int[Array, "batch seq_len"] | None = None,
        kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
        q_positions: Int[Array, "batch seq_len"] | None = None,
        kv_positions: Int[Array, "batch kv_len"] | None = None,
        sequence_parallelism_mesh_axis_name: str | None = None,
        logits_soft_cap: float | None = None,
        qkv_layouts: tuple["SparseMask"] | None = None,
        softmax_scale: float | None = None,
        mask_builder: (
            typing.Callable[[int, int, int, int, int], "Mask"] | typing.Callable[[], "SparseMask"] | None
        ) = None,
        sliding_window: int | tuple[int, int] | None = None,
        chunk_size: int | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        attention_mask: (
            Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None
        ) = None,
        cfg: BlockSparseAttentionConfig,
    ) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
        """Execute block-sparse attention with the given configuration.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, kv_num_heads, kv_len, head_dim]
            value: Value tensor [batch, kv_num_heads, kv_len, vhead_dim]
            q_segment_ids: Segment IDs for queries to handle multiple sequences [batch, seq_len]
            kv_segment_ids: Segment IDs for keys/values [batch, kv_len]
            softmax_aux: Auxiliary values added to attention scores (e.g., for attention sinks)
            logits_soft_cap: Optional soft cap value to bound attention logits
            softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            mask_builder: Function that builds the sparse mask pattern. Takes (q_idx, k_idx,
                q_size, k_size, window_size) and returns a Mask object
            sliding_window: Window size for local attention, int for symmetric or (left, right) tuple
            chunk_size: Overall chunk size (alternative to separate query/key chunk sizes)
            causal: Whether to apply causal masking (default: True)
            fused_backward: Use fused backward pass for improved gradient computation
            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
            cfg: Configuration object specifying platform/backend and kernel parameters

        Returns:
            Attention output tensor [batch, seq_len_q, num_heads, head_dim]

        Note:
            The mask_builder function is critical for defining sparsity patterns.
            It should return a mask indicating which blocks to compute.
        """
        if platform is not None:
            cfg = BlockSparseAttentionConfig(
                fwd_params=cfg.fwd_params,
                bwd_params=cfg.bwd_params,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key=key,
            value=value,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            softmax_aux=softmax_aux,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            attention_mask=attention_mask,
            sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
            qkv_layouts=qkv_layouts,
            softmax_scale=softmax_scale,
            fwd_params=cfg.fwd_params,
            bwd_params=cfg.bwd_params,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            fused_backward=fused_backward,
        )

    def heuristic_cfg_gpu(self, inv: Invocation[BlockSparseAttentionConfig, Array]) -> BlockSparseAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return BlockSparseAttentionConfig(
            fwd_params=FwdParams(
                q_blocksize=64,
                kv_blocksize=64,
                num_warps=4,
                num_stages=2,
            ),
            bwd_params=BwdParams(
                q_blocksize=32,
                kv_blocksize=32,
                num_warps=4,
                num_stages=2,
            ),
            platform="triton",
            backend="gpu",
        )

    def heuristic_cfg_tpu(self, inv: Invocation[BlockSparseAttentionConfig, Array]) -> BlockSparseAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return BlockSparseAttentionConfig(
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

    def heuristic_cfg(self, inv: Invocation[BlockSparseAttentionConfig, Array]) -> BlockSparseAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return BlockSparseAttentionConfig(
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

    def candidate_cfgs(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
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

        block_configs = [(256, 256), (512, 512)]

        candidates = []
        for q_block, kv_block in block_configs:
            candidates.append(
                BlockSparseAttentionConfig(
                    fwd_params=FwdParams(
                        q_blocksize=q_block,
                        kv_blocksize=kv_block,
                        num_warps=4,
                        num_stages=2,
                    ),
                    bwd_params=BwdParams(
                        q_blocksize=q_block * 2,
                        kv_blocksize=kv_block * 2,
                        num_warps=4,
                        num_stages=2,
                    ),
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
        """Generate GPU-optimized candidate configurations for autotuning (Triton).

        Heuristics:
        - q/kv blocks in {32, 64, 128, 256} depending on head_dim
        - If sliding_window is set, favor kv blocks ≲ window size (rounded)
        - num_warps: 2-8 based on head_dim and block sizes
        - num_stages: 2-4 (bigger when kv block is large)
        - Backward block sizes smaller to reduce register pressure
        """
        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        head_dim = int(q.shape[-1])
        q_len = int(q.shape[-2])
        k_len = int(k.shape[-2])
        dtype = q.dtype

        sliding_window = inv.kwargs.get("sliding_window", None)
        causal = bool(inv.kwargs.get("causal", True))
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "cuda", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)

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

        block_headdim = max(1 << max(4, math.ceil(math.log2(max(1, head_dim)))), 16)
        elem_bytes = 2 if dtype in (jnp.float16, jnp.bfloat16) else 4

        def smem_est_bytes(qb: int, kb: int, num_stages: int) -> int:
            """Estimate shared memory usage in bytes for given block sizes and stages."""
            kv_bytes = 2 * kb * block_headdim * elem_bytes

            q_bytes = int(0.5 * qb * block_headdim * elem_bytes)
            base = kv_bytes + q_bytes

            stage_factor = 1.0 + 0.5 * max(0, num_stages - 2)

            fudge = 3.0
            return int(base * stage_factor * fudge)

        if head_dim <= 64:
            q_opts = [32, 64, 128]
        elif head_dim <= 128:
            q_opts = [32, 64, 128]
        elif head_dim <= 192:
            q_opts = [32, 64, 128]
        else:
            q_opts = [32, 64, 128]

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

            if kb >= 256:
                stages = 3
            elif kb >= 128:
                stages = 2
            else:
                stages = 2
            return warps, stages

        def bwd_block(x: int, cap: int = 128) -> int:
            """Compute backward block size from a forward block size."""
            return max(32, min(cap, x // 2 if x >= 64 else x))

        hv_pairs = []
        preferred = [(32, 64), (64, 64), (64, 128), (128, 64)]
        if win is not None:
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
            qb, kb = 32, 64
            w, s = pick_warps_stages(qb, kb, head_dim)
            pairs = [(qb, kb, w, s)]

        configs: list[BlockSparseAttentionConfig] = []
        if "triton" in platforms:
            for qb, kb, w, s in pairs:
                configs.append(
                    BlockSparseAttentionConfig(
                        fwd_params=FwdParams(
                            q_blocksize=qb,
                            kv_blocksize=kb,
                            num_warps=w,
                            num_stages=s,
                        ),
                        bwd_params=BwdParams(
                            q_blocksize=bwd_block(qb),
                            kv_blocksize=bwd_block(kb),
                            num_warps=w,
                            num_stages=max(2, s - 0),
                        ),
                        platform="triton",
                        backend="gpu",
                    )
                )
        for platform, backend, max_platform_candidates in (
            ("cuda", "gpu", 4),
            ("tilelang", "gpu", 4),
            ("xla", "any", 4),
        ):
            if platform not in platforms:
                continue
            for qb, kb, w, s in pairs[:max_platform_candidates]:
                configs.append(
                    BlockSparseAttentionConfig(
                        fwd_params=FwdParams(
                            q_blocksize=qb,
                            kv_blocksize=kb,
                            num_warps=w,
                            num_stages=s,
                        ),
                        bwd_params=BwdParams(
                            q_blocksize=bwd_block(qb),
                            kv_blocksize=bwd_block(kb),
                            num_warps=w,
                            num_stages=max(2, s),
                        ),
                        platform=platform,
                        backend=backend,
                    )
                )
        return configs

    def candidate_cfgs_tpu(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning (Pallas)."""
        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        q_len = int(q.shape[-2])
        k_len = int(k.shape[-2])

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

        def nearest_128_from_set(x: int, allowed=(128, 256, 512, 1024)) -> int:
            """Return the allowed value nearest to x, breaking ties by smallest."""
            return min(allowed, key=lambda v: (abs(v - x), v))

        allowed = (128, 256, 512, 1024)
        win = win_span(sliding_window)

        q_opts = [b for b in allowed if b <= max(128, q_len)] or [128]
        kv_opts = [b for b in allowed if b <= max(128, k_len)] or [128]

        if win is not None:
            t = nearest_128_from_set(max(128, min(1024, win)), allowed)
            kv_opts = sorted(set([*kv_opts, t, min(1024, 2 * t)]))

        q_opts = sorted(set(q_opts))
        kv_opts = sorted(set(kv_opts))

        def bwd_tile(x: int) -> int:
            """Compute backward tile size from a forward tile size."""
            return 128 if x <= 256 else 256

        hv_pairs: list[tuple[int, int]] = []
        if win is not None:
            t = nearest_128_from_set(max(128, min(1024, win)), allowed)
            for qb in (128, 256):
                if qb in q_opts and t in kv_opts:
                    hv_pairs.append((qb, t))
            if 2 * t <= 1024 and (128 in q_opts) and (2 * t in kv_opts):
                hv_pairs.append((128, 2 * t))
        hv_pairs += [(128, 128), (128, 256), (256, 256), (256, 512)]

        hv_pairs = [(qb, kb) for (qb, kb) in hv_pairs if qb in q_opts and kb in kv_opts]

        grid_pairs = []
        for qb in q_opts:
            for kb in kv_opts:
                if (qb, kb) not in hv_pairs:
                    grid_pairs.append((qb, kb))

        max_candidates = 16
        pairs: list[tuple[int, int]] = []
        seen = set()
        for qb, kb in hv_pairs + grid_pairs:
            if (qb, kb) in seen:
                continue
            seen.add((qb, kb))
            pairs.append((qb, kb))
            if len(pairs) >= max_candidates:
                break

        configs: list[BlockSparseAttentionConfig] = []
        for qb, kb in pairs:
            configs.append(
                BlockSparseAttentionConfig(
                    fwd_params=FwdParams(
                        q_blocksize=qb,
                        kv_blocksize=kb,
                        num_warps=None,
                        num_stages=None,
                    ),
                    bwd_params=BwdParams(
                        q_blocksize=bwd_tile(qb),
                        kv_blocksize=bwd_tile(kb),
                        num_warps=None,
                        num_stages=None,
                    ),
                    platform="pallas",
                    backend="tpu",
                )
            )
        configs.extend(self.candidate_cfgs_xla(inv)[:4])
        return configs

    def candidate_cfgs_xla(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
        """Generate XLA-optimized candidate configurations for autotuning.

        Produces block-size combinations in the range 128-1024 suitable for XLA
        compilation. Sliding window, sequence lengths, and causal mode are used
        to narrow the search space.

        Args:
            inv: Invocation object with arguments and metadata.

        Returns:
            List of BlockSparseAttentionConfig candidates for autotuning.
        """
        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        q_len = int(q.shape[-2])
        k_len = int(k.shape[-2])

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

        def nearest_128_from_set(x: int, allowed=(128, 256, 512, 1024)) -> int:
            """Return the allowed value nearest to x, breaking ties by smallest."""
            return min(allowed, key=lambda v: (abs(v - x), v))

        allowed = (128, 256, 512, 1024)
        win = win_span(sliding_window)

        q_opts = [b for b in allowed if b <= max(128, q_len)] or [128]
        kv_opts = [b for b in allowed if b <= max(128, k_len)] or [128]

        if win is not None:
            t = nearest_128_from_set(max(128, min(1024, win)), allowed)
            kv_opts = sorted(set([*kv_opts, t, min(1024, 2 * t)]))

        q_opts = sorted(set(q_opts))
        kv_opts = sorted(set(kv_opts))

        def bwd_tile(x: int) -> int:
            """Compute backward tile size from a forward tile size."""
            return 128 if x <= 256 else 256

        hv_pairs: list[tuple[int, int]] = []
        if win is not None:
            t = nearest_128_from_set(max(128, min(1024, win)), allowed)
            for qb in (128, 256):
                if qb in q_opts and t in kv_opts:
                    hv_pairs.append((qb, t))
        hv_pairs += [(128, 128), (128, 256), (256, 256), (256, 128)]
        hv_pairs = [(qb, kb) for (qb, kb) in hv_pairs if qb in q_opts and kb in kv_opts]

        grid_pairs = []
        for qb in q_opts:
            for kb in kv_opts:
                if (qb, kb) not in hv_pairs:
                    grid_pairs.append((qb, kb))

        max_candidates = 12
        pairs: list[tuple[int, int]] = []
        seen = set()
        for qb, kb in hv_pairs + grid_pairs:
            if (qb, kb) in seen:
                continue
            seen.add((qb, kb))
            pairs.append((qb, kb))
            if len(pairs) >= max_candidates:
                break

        configs: list[BlockSparseAttentionConfig] = []
        for qb, kb in pairs:
            configs.append(
                BlockSparseAttentionConfig(
                    fwd_params=FwdParams(
                        q_blocksize=qb,
                        kv_blocksize=kb,
                        num_warps=None,
                        num_stages=None,
                    ),
                    bwd_params=BwdParams(
                        q_blocksize=bwd_tile(qb),
                        kv_blocksize=bwd_tile(kb),
                        num_warps=None,
                        num_stages=None,
                    ),
                    platform="xla",
                    backend="any",
                )
            )
        return configs

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_xla = candidate_cfgs_xla


_executor: Executor[BlockSparseAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("blocksparse", cfg_type=BlockSparseAttentionConfig),
    ),
)


def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    *,
    mask_info: MaskInfo | None = None,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None
    ) = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    mask_builder: typing.Callable[[int, int, int, int, int], "Mask"] | typing.Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
    purify: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: BlockSparseAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """Execute block-sparse attention with automatic optimization.

    Performs efficient attention computation over sparse block patterns,
    significantly reducing memory and computation compared to dense attention
    while maintaining flexibility through custom mask builders.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, kv_num_heads, kv_len, head_dim].
        value: Value tensor [batch, kv_num_heads, kv_len, vhead_dim].
        softmax_aux: Optional attention-sink logits
            [num_sinks] added before softmax.
        bias: Optional additive bias [batch, num_heads, seq_len, kv_len].
        mask_info: Optional ``MaskInfo`` container; segment IDs and position
            arrays are extracted from it.  When ``mesh``/``in_specs``/
            ``out_specs`` are all provided, ``attention_mask`` is ignored
            (not supported in shard_map mode).
        attention_mask: Optional dense boolean/integer attention mask
            [batch, num_heads_or_1, seq_len, kv_len]. Not supported in
            shard_map mode; pass ``bias``/``mask_builder`` instead.
        sequence_parallelism_mesh_axis_name: Name of the sequence-parallel
            mesh axis for Splash (Pallas TPU) inter-device attention.
        logits_soft_cap: Soft cap applied to logits as
            ``cap * tanh(logits / cap)`` (Gemma-2 style).
        qkv_layouts: Pre-computed ``SparseMask`` layouts for the Triton
            backend.  When None the mask is built on-the-fly from
            ``mask_builder``.
        softmax_scale: Scale factor for attention logits
            (default: 1/sqrt(head_dim)).
        mask_builder: Callable that defines the sparse block pattern.
            On Triton: ``() -> SparseMask``.
            On Pallas TPU: ``(q_idx, k_idx, q_size, k_size, window) -> Mask``.
        sliding_window: Symmetric window (int) or asymmetric ``(left, right)``
            tuple for local attention.
        chunk_size: Unified query/key chunk size passed directly to the
            backend; overrides block sizes from ``cfg.fwd_params``.
        causal: Apply causal masking (default: True).
        fused_backward: Use fused VJP kernel (default: False).
        purify: Zero out output positions where ``q_segment_ids < 0``
            (i.e. padding positions).  Applied after the kernel returns.
        platform: Force a specific platform (``"triton"``, ``"pallas"``,
            ``"cuda"``, ``"xla"``, or ``"auto"``).
        cfg: Optional ``BlockSparseAttentionConfig`` override.
        mesh: JAX device mesh for ``shard_map`` distributed execution.
        in_specs: Input partition specs for ``shard_map``; must have the
            same length as the positional call arguments (see source for
            the exact order).
        out_specs: Output partition spec for ``shard_map``.

    Returns:
        Attention output [batch, num_heads, seq_len, vhead_dim].

    Example:
        >>> from ejkernel.modules.operations import blocksparse_attention
        >>>
        >>> # Causal attention with default dense fallback
        >>> output = blocksparse_attention(query, key, value, causal=True)
        >>>
        >>> # Sparse local + global pattern (Triton)
        >>> from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask
        >>> def build_mask():
        ...     return SparseMask(...)  # your pattern here
        >>> output = blocksparse_attention(
        ...     query, key, value,
        ...     mask_builder=build_mask,
        ...     sliding_window=256,
        ...     platform="triton",
        ... )

    Note:
        Block-sparse attention is particularly effective for long document
        processing where full attention is prohibitive and attention patterns
        have known locality structure (e.g., Longformer, BigBird).
    """

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    q_segment_ids = None
    kv_segment_ids = None
    q_positions = None
    kv_positions = None
    q_mask = None

    if mask_info is not None:
        q_segment_ids, kv_segment_ids = mask_info.get_or_compute_segment_ids()
        q_ids_for_mask = q_segment_ids
        if q_ids_for_mask is not None and q_ids_for_mask.ndim == 3:
            q_ids_for_mask = q_ids_for_mask[:, 0, :]
        if q_ids_for_mask is not None:
            q_mask = q_ids_for_mask >= 0

        if q_positions is None or kv_positions is None:
            mask_q_pos, mask_kv_pos = mask_info.get_or_compute_positions()
            if q_positions is None:
                q_positions = mask_q_pos
            if kv_positions is None:
                kv_positions = mask_kv_pos

        if method is None and attention_mask is None:
            attention_mask = mask_info.get_or_compute_attention_mask()

    if method == "shard_map":
        if attention_mask is not None:
            raise NotImplementedError(
                "`attention_mask` is not supported for shard_map blocksparse_attention; "
                "pass `bias`/`mask_builder` instead."
            )
        if mask_info is None:
            in_specs = (*in_specs, None, None, None, None)
        else:
            shardings = mask_info.get_shardings(False, mesh=mesh)
            in_specs = (
                *in_specs,
                shardings.q_segment_ids,
                shardings.kv_segment_ids,
                shardings.q_positions,
                shardings.kv_positions,
            )

    run_kwargs = dict(
        query=query,
        key=key,
        value=value,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_positions=q_positions,
        kv_positions=kv_positions,
        softmax_aux=softmax_aux,
        logits_soft_cap=logits_soft_cap,
        bias=bias,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        mask_builder=mask_builder,
        sliding_window=sliding_window,
        chunk_size=chunk_size,
        causal=causal,
        fused_backward=fused_backward,
        platform=platform,
    )
    if method is None:
        run_kwargs["attention_mask"] = attention_mask

    out = _executor(
        BlockSparseAttention(),
        **run_kwargs,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

    if q_mask is not None and purify:
        out = out * q_mask[:, None, :, None].astype(out.dtype)

    return out

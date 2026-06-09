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


"""Ring Attention module with automatic optimization.

This module implements Ring Attention, a distributed attention mechanism that enables
efficient processing of extremely long sequences by distributing computation across
multiple devices in a ring topology. Unlike standard attention which requires all KV
pairs to fit in a single device's memory, Ring Attention overlaps communication and
computation through pipelining.

Ring Attention is particularly valuable for:
    - Ultra-long sequence processing (100K+ tokens)
    - Training large language models with long contexts
    - Distributed inference scenarios
    - Memory-constrained environments requiring sequence parallelism

Key Innovation:
    Ring Attention partitions the KV pairs across devices and uses a ring-based
    communication pattern to stream KV blocks through each device. Each device:
    1. Computes attention with its local KV block
    2. Passes the KV block to the next device in the ring
    3. Receives the next KV block from the previous device
    4. Continues until all KV blocks have been processed

    This achieves O(N) memory per device while maintaining O(N^2) computation.

Mathematical Foundation:
    For a sequence of length N split across D devices:
    - Each device holds N/D query tokens
    - KV pairs are rotated through the ring
    - Attention is computed incrementally: softmax_i = exp(QK_i^T) / sum_j(exp(QK_j^T))
    - Running statistics (max, sum) are maintained for numerical stability

Communication Pattern:
    Device 0: KV_0 -> KV_1 -> ... -> KV_{D-1}
    Device 1: KV_1 -> KV_2 -> ... -> KV_0
    Device i: KV_i -> KV_{i+1} -> ... -> KV_{i-1} (mod D)

Performance Characteristics:
    - Memory: O(N/D) per device vs O(N) for standard attention
    - Computation: O(N^2/D) per device (same asymptotic cost)
    - Communication: O(N) per device (bandwidth-efficient with overlap)

References:
    Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context"
    https://arxiv.org/abs/2310.01889
"""

from __future__ import annotations

import os
import typing
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

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
from .configs import RingAttentionConfig

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


class RingAttention(Kernel[RingAttentionConfig, Array]):
    """Ring Attention with custom optimization logic.

    Implements distributed attention using ring communication topology for
    processing ultra-long sequences across multiple devices with memory efficiency.

    Features:
        - Distributed KV processing via ring communication
        - Overlapped computation and communication for efficiency
        - Causal and non-causal attention support
        - Sliding window attention for local patterns
        - Attention sink mechanism for long-context stability
        - Configurable chunk sizes for memory-computation tradeoffs
        - Gradient checkpointing support for training
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The implementation maintains numerical stability through:
        - Online softmax with running max/sum statistics
        - Logit soft capping to prevent overflow
        - Float32 logit accumulation (configurable)

    Typical Usage Patterns:
        - Multi-GPU training with sequence parallelism
        - Long-context inference on multiple devices
        - Blockwise transformer architectures
    """

    version = "1"

    def __init__(self):
        """Initialize Ring Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and distributed execution management.
        """
        super().__init__(op_id="ring_attention")

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        chunk_size: int | None = None,
        causal: bool = False,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        axis_name: str | None = None,
        fused_backward: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RingAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper specifically for ring attention.

        Ring attention naturally works with distributed execution, using
        collective communication across devices.

        Args:
            query: Query tensor to be sharded
            key: Key tensor to be sharded
            value: Value tensor to be sharded
            q_segment_ids: Optional query segment IDs
            kv_segment_ids: Optional KV segment IDs
            softmax_aux: Optional attention sink logits
            bias: Optional bias tensor
            mask_builder: Optional custom mask builder function
            sliding_window: Window size for local attention
            chunk_size: Chunk size for chunked causal attention
            causal: Whether to use causal masking
            logits_soft_cap: Soft cap value for attention logits
            softmax_scale: Scaling factor for attention scores
            axis_name: Axis name for ring communication
            fused_backward: Whether to use fused backward kernel
            platform: Target platform
            cfg: Kernel configuration object
            mesh: JAX device mesh
            in_specs: Input partition specs
            out_specs: Output partition spec
            check_vma: Check for virtual memory access

        Returns:
            Tuple of (shard_map_fn, call_args)
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_ring_attn(
            query: Float[Array, "batch seq_len_q num_heads head_dim"],
            key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
            value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
            softmax_aux: Float[Array, "num_sinks"] | None = None,
            bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
            q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
            kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
            q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
            kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
        ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
            """Shard-map compatible wrapper that delegates to self.run with captured params."""
            return self.run(
                query=query,
                key=key,
                value=value,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                q_position_ids=q_position_ids,
                kv_position_ids=kv_position_ids,
                softmax_aux=softmax_aux,
                bias=bias,
                mask_builder=mask_builder,
                sliding_window=sliding_window,
                chunk_size=chunk_size,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                softmax_scale=softmax_scale,
                axis_name=axis_name,
                fused_backward=fused_backward,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            query,
            key,
            value,
            softmax_aux,
            bias,
            q_segment_ids,
            kv_segment_ids,
            q_position_ids,
            kv_position_ids,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_ring_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: RingAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ring attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ring_attention", cfg.platform)
        return kernel_registry.get("ring_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        chunk_size: int | None = None,
        causal: bool = False,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        axis_name: str | None = None,
        fused_backward: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: RingAttentionConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        """Execute ring attention with distributed KV processing.

        Computes attention across devices using ring communication pattern,
        enabling efficient processing of sequences that don't fit in single device memory.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim] (distributed)
            value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim] (distributed)
            q_segment_ids: Optional query segment IDs [batch, seq_len_q]
            kv_segment_ids: Optional KV segment IDs [batch, seq_len_k]
            softmax_aux: Optional attention sink logits for long-context stability
            bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k]
            mask_builder: Custom mask builder function(q_len, kv_len, num_heads, head_idx, num_reps) -> Mask
            sliding_window: Window size for local attention (int or (left, right) tuple)
            chunk_size: Chunk size for chunked causal attention (Llama4 style)
            causal: Whether to use causal masking
            logits_soft_cap: Soft cap value to bound attention logits
            softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            axis_name: Name of the axis for collective operations (required for multi-device)
            fused_backward: Whether to use fused backward kernel
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [batch, seq_len_q, num_heads, head_dim]

        Note:
            Ring attention requires proper device mesh setup with the specified axis_name.
            Each device processes a slice of the sequence and communicates KV pairs
            through the ring topology.

        Example:
            >>>
            >>> mesh = jax.sharding.Mesh(devices, axis_names=['sp'])
            >>>
            >>>
            >>> with mesh:
            ...     out = ring_attention(q, k, v, axis_name='sp')
        """

        if platform is not None:
            cfg = RingAttentionConfig(
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
            q_position_ids=q_position_ids,
            kv_position_ids=kv_position_ids,
            softmax_aux=softmax_aux,
            bias=bias,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            axis_name=axis_name,
            fwd_params=cfg.fwd_params,
            bwd_params=cfg.bwd_params,
            fused_backward=fused_backward,
        )

    def heuristic_cfg(self, inv: Invocation[RingAttentionConfig, Array]) -> RingAttentionConfig:
        """Provide default configuration optimized for ring attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default RingAttentionConfig with block sizes balanced for communication
            and computation overlap in distributed settings
        """
        try:
            q_len = int(inv.kwargs["query"].shape[1])
            kv_len = int(inv.kwargs["key"].shape[1])
        except Exception:
            q_len = 512
            kv_len = 512

        def _largest_pow2_divisor(n: int, *, max_block: int = 512) -> int:
            """Return the largest power-of-2 divisor of n that is <= max_block."""
            n = max(1, int(n))
            for b in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
                if b <= max_block and b <= n and n % b == 0:
                    return b
            return 1

        block_q = _largest_pow2_divisor(q_len)
        block_kv = _largest_pow2_divisor(kv_len)
        return RingAttentionConfig(
            fwd_params=FwdParams(q_blocksize=block_q, kv_blocksize=block_kv, num_stages=2, num_warps=4),
            bwd_params=BwdParams(q_blocksize=block_q, kv_blocksize=block_kv, num_stages=2, num_warps=4),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RingAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for different sequence lengths and
        device counts, balancing chunk size with communication overhead.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Ring attention performance is sensitive to chunk sizes relative
            to sequence length per device and communication bandwidth.
        """
        candidates = []
        for block_q, block_k in [(128, 128), (256, 256), (512, 512)]:
            candidates.append(
                RingAttentionConfig(
                    fwd_params=FwdParams(q_blocksize=block_q, kv_blocksize=block_k, num_stages=2, num_warps=4),
                    bwd_params=BwdParams(q_blocksize=block_q, kv_blocksize=block_k, num_stages=2, num_warps=4),
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[RingAttentionConfig, Array]):
        """Generate GPU candidates for Triton, TileLang, and XLA ring attention.

        Ring attention shards the KV axis across devices; per-device tiles
        see ``seq_len_k / ring_size`` effectively. The candidates below
        cover both small per-device shards (``<=4K``) and large
        ones (``>=16K``). Backward uses smaller tiles (extra GEMMs per
        iteration → higher register pressure).
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        block_configs = [
            (64, 64, 4, 2),
            (128, 128, 4, 2),
            (128, 128, 4, 3),
            (256, 128, 8, 2),
            (128, 256, 8, 2),
            (256, 256, 8, 2),
            (512, 512, 8, 2),
        ]
        candidates: list[RingAttentionConfig] = []
        for platform, backend in (("triton", "gpu"), ("tilelang", "gpu"), ("xla", "any")):
            if platform not in platforms:
                continue
            for block_q, block_k, num_warps, num_stages in block_configs:
                candidates.append(
                    RingAttentionConfig(
                        fwd_params=FwdParams(
                            q_blocksize=block_q,
                            kv_blocksize=block_k,
                            num_stages=num_stages,
                            num_warps=num_warps,
                        ),
                        bwd_params=BwdParams(
                            q_blocksize=max(32, block_q // 2),
                            kv_blocksize=max(32, block_k // 2),
                            num_stages=num_stages,
                            num_warps=num_warps,
                        ),
                        platform=platform,
                        backend=backend,
                    )
                )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[RingAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning.

        TPU/Pallas kernels benefit from larger blocks for ring attention.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of TPU-optimized candidate configurations
        """
        block_configs = [
            (128, 128, 4, 1),
            (256, 256, 8, 2),
            (512, 512, 8, 2),
        ]

        candidates = []
        for platform, backend in (("pallas", "tpu"), ("xla", "any")):
            for block_q, block_k, num_warps, num_stages in block_configs:
                fwd = FwdParams(q_blocksize=block_q, kv_blocksize=block_k, num_stages=num_stages, num_warps=num_warps)
                bwd = BwdParams(q_blocksize=block_q, kv_blocksize=block_k, num_stages=num_stages, num_warps=num_warps)
                candidates.append(
                    RingAttentionConfig(
                        fwd_params=fwd,
                        bwd_params=bwd,
                        platform=platform,
                        backend=backend,
                    )
                )

        return candidates

    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_ring_executor: Executor[RingAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ring-attention", cfg_type=RingAttentionConfig),
    )
)


def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    /,
    *,
    mask_info: MaskInfo | None = None,
    mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = False,
    logits_soft_cap: float | None = None,
    softmax_scale: float | None = None,
    axis_name: str | None = None,
    fused_backward: bool = False,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RingAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Execute ring attention with automatic optimization.

    Ring attention distributes attention computation across devices in a ring topology,
    enabling efficient processing of very long sequences through communication-efficient
    parallelization.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        softmax_aux: Optional attention sink logits for long-context stability
        bias: Optional attention bias tensor
        mask_info: Optional MaskInfo containing attention mask and/or segment IDs
        mask_builder: Custom mask builder function(q_len, kv_len, num_heads, head_idx, num_reps) -> Mask
        sliding_window: Window size for local attention (int or (left, right) tuple)
        chunk_size: Chunk size for chunked causal attention (Llama4 style)
        causal: Whether to use causal masking
        logits_soft_cap: Soft capping value for logits
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        axis_name: Name of the axis for collective operations
        fused_backward: Whether to use fused backward kernel
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional configuration override
        mesh: JAX device mesh for shard_map execution (optional)
        in_specs: Input partition specs for shard_map (optional)
        out_specs: Output partition spec for shard_map (optional)

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = ring_attention(query, key, value, causal=True, axis_name="sp")
        >>>
        >>>
        >>> out = ring_attention(
        ...     query, key, value,
        ...     causal=True,
        ...     sliding_window=1024,
        ...     axis_name="sp",
        ... )
        >>>
        >>>
        >>> out = ring_attention(..., platform="triton")
    """

    q_segment_ids = None
    kv_segment_ids = None
    q_positions = None
    kv_positions = None
    if mask_info is not None:
        q_segment_ids, kv_segment_ids = mask_info.get_or_compute_segment_ids()
        q_positions, kv_positions = mask_info.get_or_compute_positions()

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
        if mask_info is None:
            in_specs = (*in_specs, None, None, None, None)
        else:
            shardings = mask_info.get_shardings(True, mesh=mesh)
            in_specs = (
                *in_specs,
                shardings.q_segment_ids,
                shardings.kv_segment_ids,
                shardings.q_positions,
                shardings.kv_positions,
            )
            assert mask_info.sequence_axis_name == axis_name, "mismatch between two sequence axis names!"

    return _ring_executor(
        RingAttention(),
        query=query,
        key=key,
        value=value,
        softmax_aux=softmax_aux,
        bias=bias,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_position_ids=q_positions,
        kv_position_ids=kv_positions,
        mask_builder=mask_builder,
        sliding_window=sliding_window,
        chunk_size=chunk_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=softmax_scale,
        axis_name=axis_name,
        fused_backward=fused_backward,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

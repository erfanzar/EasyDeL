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

"""Chunked prefill + paged decode module with automatic platform selection.

This module implements a fused operation that combines KV cache updates with
causal paged attention in a single pass, optimized for LLM serving workloads.
It updates a block-tabled KV cache with packed keys/values and computes
attention outputs simultaneously.

Key Features:
    - Fused KV cache update and attention computation
    - Block-tabled (paged) memory for efficient KV cache management
    - Support for variable-length sequences via packed tensor format
    - Causal masking for autoregressive generation
    - Sliding window attention for long contexts
    - Optional ALiBi positional bias support
    - Logit soft capping for numerical stability

Use Cases:
    - LLM inference serving with dynamic batching
    - Combined prefill and decode phases in a single operation
    - Memory-efficient serving with paged KV cache
    - Variable-length sequence processing without padding

Memory Layout:
    Queries/Keys/Values: Packed format [total_tokens, num_heads, head_dim]
    KV Cache: Block-tabled format [num_blocks, block_size, num_heads, head_dim]
    Block Tables: Logical to physical mapping [num_seqs, max_blocks_per_seq]

The operation flow:
    1. Insert new keys/values into the block-tabled cache at appropriate positions
    2. Compute causal attention over the updated cache for each query
    3. Return attention outputs along with the updated cache

References:
    - PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
    - FlashAttention: https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
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
from .configs import ChunkedPrefillPagedDecodeConfig


class ChunkedPrefillPagedDecode(Kernel[ChunkedPrefillPagedDecodeConfig, tuple[Array, Array, Array]]):
    """Chunked prefill + paged decode attention with KV-cache update.

    This kernel combines two operations in a single pass:
    1. Updates a block-tabled KV cache with packed keys/values
    2. Computes causal attention outputs for packed queries against the updated cache

    The operation is designed for efficient serving of LLMs where sequences have
    varying context lengths and are stored in a paged memory structure.
    """

    def __init__(self):
        """Initialize the chunked prefill + paged decode kernel."""
        super().__init__(op_id="chunked_prefill_paged_decode")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        kv_lens: Int32[Array, "num_seqs"],
        block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
        query_start_loc: Int32[Array, "num_seqs_plus_1"],
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        seq_threshold_3d: int | None = None,
        num_par_softmax_segments: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: ChunkedPrefillPagedDecodeConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, PartitionSpec, PartitionSpec] | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed execution.

        Args:
            queries: Packed query vectors [total_tokens, num_q_heads, head_dim]
            keys: Packed key vectors [total_tokens, num_kv_heads, head_dim]
            values: Packed value vectors [total_tokens, num_kv_heads, head_dim]
            key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
            kv_lens: Sequence KV lengths [num_seqs]
            block_tables: Physical block indices [num_seqs, max_blocks_per_seq]
            query_start_loc: Cumulative query positions [num_seqs + 1]
            alibi_slopes: Optional ALiBi attention bias slopes [num_q_heads]
            softmax_aux: Optional auxiliary softmax parameters [num_q_heads]
            softmax_scale: Softmax scaling factor (default: 1/√d)
            causal: Whether to apply causal masking (default: True)
            sliding_window: Optional sliding window size for local attention
            logits_soft_cap: Optional soft capping value for attention logits
            seq_threshold_3d: Sequence count threshold for 3D kernel selection
            num_par_softmax_segments: Number of parallel softmax segments (Triton only)
            platform: Target platform (triton/pallas/cuda/xla/auto)
            cfg: Optional kernel configuration
            mesh: JAX mesh for distributed execution
            in_specs: Input partition specifications
            out_specs: Output partition specifications
            check_vma: Whether to check virtual memory address validity

        Returns:
            Tuple of (shard_map_fn, call_args) for distributed execution.

        Raises:
            AssertionError: If mesh, in_specs, or out_specs are not provided, or if
                           in_specs length doesn't match the number of call arguments.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_chunked_prefill_paged_decode(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            keys: Float[Array, "total_tokens num_kv_heads head_dim"],
            values: Float[Array, "total_tokens num_kv_heads head_dim"],
            key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
            value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
            kv_lens: Int32[Array, "num_seqs"],
            block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
            query_start_loc: Int32[Array, "num_seqs_plus_1"],
            alibi_slopes: Float[Array, "num_q_heads"] | None,
            softmax_aux: Float[Array, "num_q_heads"] | None,
        ) -> tuple[
            Float[Array, "total_tokens num_q_heads head_dim"],
            Float[Array, "num_blocks block_size num_kv_heads head_dim"],
            Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        ]:
            """Shard-local chunked prefill + paged decode forwarding to self.run."""
            return self.run(
                queries=queries,
                keys=keys,
                values=values,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_lens=kv_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                alibi_slopes=alibi_slopes,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                causal=causal,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                seq_threshold_3d=seq_threshold_3d,
                num_par_softmax_segments=num_par_softmax_segments,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
            )

        call_args = (
            queries,
            keys,
            values,
            key_cache,
            value_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            alibi_slopes,
            softmax_aux,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_chunked_prefill_paged_decode,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def get_impl(self, cfg: ChunkedPrefillPagedDecodeConfig):
        """Get the platform-specific kernel implementation.

        Args:
            cfg: Kernel configuration specifying platform and backend.

        Returns:
            Platform-specific kernel implementation callable.
        """
        platform = detect_platform("chunked_prefill_paged_decode", cfg.platform)
        return kernel_registry.get("chunked_prefill_paged_decode", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        kv_lens: Int32[Array, "num_seqs"],
        block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
        query_start_loc: Int32[Array, "num_seqs_plus_1"],
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        seq_threshold_3d: int | None = None,
        num_par_softmax_segments: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: ChunkedPrefillPagedDecodeConfig,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads head_dim"],
        Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    ]:
        """Execute chunked prefill + paged decode attention.

        Args:
            queries: Packed query vectors [total_tokens, num_q_heads, head_dim]
            keys: Packed key vectors to insert into cache [total_tokens, num_kv_heads, head_dim]
            values: Packed value vectors to insert into cache [total_tokens, num_kv_heads, head_dim]
            key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
            kv_lens: Current KV lengths per sequence [num_seqs]
            block_tables: Logical-to-physical block mapping [num_seqs, max_blocks_per_seq]
            query_start_loc: Cumulative query token indices [num_seqs + 1]
            alibi_slopes: Optional ALiBi slopes for positional bias [num_q_heads]
            softmax_aux: Optional auxiliary attention parameters [num_q_heads]
            softmax_scale: Attention scaling factor (default: 1/√head_dim)
            causal: Apply causal masking (default: True)
            sliding_window: Optional local attention window size
            logits_soft_cap: Optional logit capping value for numerical stability
            seq_threshold_3d: Sequence count threshold for 3D kernel variant
            num_par_softmax_segments: Parallel softmax segments (Triton-specific)
            platform: Override platform selection (triton/pallas/cuda/xla/auto)
            cfg: Kernel configuration object

        Returns:
            Tuple of (attention_output, updated_key_cache, updated_value_cache)
                - attention_output: [total_tokens, num_q_heads, head_dim]
                - updated_key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
                - updated_value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        """
        if platform is not None:
            cfg = ChunkedPrefillPagedDecodeConfig(
                seq_threshold_3d=cfg.seq_threshold_3d,
                num_par_softmax_segments=cfg.num_par_softmax_segments,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        if seq_threshold_3d is None:
            seq_threshold_3d = cfg.seq_threshold_3d
        if num_par_softmax_segments is None:
            num_par_softmax_segments = cfg.num_par_softmax_segments

        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            keys=keys,
            values=values,
            key_cache=key_cache,
            value_cache=value_cache,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            alibi_slopes=alibi_slopes,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            causal=causal,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            seq_threshold_3d=seq_threshold_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    def heuristic_cfg(self, inv: Invocation[ChunkedPrefillPagedDecodeConfig, tuple[Array, Array, Array]]):
        """Generate default heuristic configuration.

        Args:
            inv: Invocation context.

        Returns:
            Default configuration with automatic platform selection.
        """
        queries = inv.kwargs["queries"]
        block_tables = inv.kwargs["block_tables"]
        num_seqs = int(block_tables.shape[0])
        head_dim = int(queries.shape[-1])
        seq_threshold_3d = 16 if num_seqs >= 16 else 32
        num_par_softmax_segments = 4 if num_seqs >= 16 else 2
        return ChunkedPrefillPagedDecodeConfig(
            seq_threshold_3d=seq_threshold_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            num_warps=8 if head_dim >= 128 else 4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[ChunkedPrefillPagedDecodeConfig, tuple[Array, Array, Array]]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation context.

        Returns:
            Portable fallback candidates for non-GPU execution.
        """
        base = self.heuristic_cfg(inv)
        return [
            ChunkedPrefillPagedDecodeConfig(
                seq_threshold_3d=base.seq_threshold_3d,
                num_par_softmax_segments=base.num_par_softmax_segments,
                num_warps=base.num_warps,
                num_stages=base.num_stages,
                platform="auto",
                backend="any",
            ),
            ChunkedPrefillPagedDecodeConfig(
                seq_threshold_3d=base.seq_threshold_3d,
                num_par_softmax_segments=base.num_par_softmax_segments,
                num_warps=base.num_warps,
                num_stages=base.num_stages,
                platform="xla",
                backend="any",
            ),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[ChunkedPrefillPagedDecodeConfig, tuple[Array, Array, Array]]):
        """Generate GPU candidates for fused prefill/update + paged decode."""
        queries = inv.kwargs["queries"]
        block_tables = inv.kwargs["block_tables"]
        requested = inv.kwargs.get("platform", None)
        num_seqs = int(block_tables.shape[0])
        head_dim = int(queries.shape[-1])
        threshold_choices = tuple(sorted({8, 16, 32, max(1, num_seqs)}))
        segment_choices = (1, 2, 4, 8) if num_seqs >= 16 else (1, 2, 4)
        warp_choices = (4, 8) if head_dim >= 128 else (2, 4)
        platforms = ("triton", "cute", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)

        candidates: list[ChunkedPrefillPagedDecodeConfig] = []
        if "triton" in platforms or "cute" in platforms:
            for platform_name in ("triton", "cute"):
                if platform_name not in platforms:
                    continue
                for threshold in threshold_choices[:3]:
                    for segments in segment_choices[:2]:
                        for warps in warp_choices:
                            candidates.append(
                                ChunkedPrefillPagedDecodeConfig(
                                    seq_threshold_3d=threshold,
                                    num_par_softmax_segments=segments,
                                    num_warps=warps,
                                    num_stages=2,
                                    platform=platform_name,
                                    backend="gpu",
                                )
                            )
        if "tilelang" in platforms:
            for stages in (2, 3, 4):
                candidates.append(
                    ChunkedPrefillPagedDecodeConfig(
                        seq_threshold_3d=None,
                        num_par_softmax_segments=None,
                        num_warps=4,
                        num_stages=stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            base = self.heuristic_cfg(inv)
            candidates.append(
                ChunkedPrefillPagedDecodeConfig(
                    seq_threshold_3d=base.seq_threshold_3d,
                    num_par_softmax_segments=base.num_par_softmax_segments,
                    num_warps=base.num_warps,
                    num_stages=base.num_stages,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[ChunkedPrefillPagedDecodeConfig, tuple[Array, Array, Array]]):
        """Return TPU candidates for the XLA fallback path."""
        base = self.heuristic_cfg(inv)
        return [
            ChunkedPrefillPagedDecodeConfig(
                seq_threshold_3d=base.seq_threshold_3d,
                num_par_softmax_segments=base.num_par_softmax_segments,
                num_warps=base.num_warps,
                num_stages=base.num_stages,
                platform="xla",
                backend="any",
            )
        ]


_chunked_prefill_paged_decode_executor: Executor[ChunkedPrefillPagedDecodeConfig, tuple[Array, Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("chunked-prefill-paged-decode"),
    )
)


def chunked_prefill_paged_decode(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, PartitionSpec, PartitionSpec] | None = None,
    cfg: ChunkedPrefillPagedDecodeConfig | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
]:
    """Execute chunked prefill + paged decode attention with KV cache update.

    This function combines KV cache update and attention computation in a single pass,
    designed for efficient LLM serving with variable-length sequences in paged memory.

    Args:
        queries: Packed query vectors [total_tokens, num_q_heads, head_dim]
        keys: Packed key vectors to insert into cache [total_tokens, num_kv_heads, head_dim]
        values: Packed value vectors to insert into cache [total_tokens, num_kv_heads, head_dim]
        key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
        kv_lens: Current KV lengths per sequence [num_seqs]
        block_tables: Logical-to-physical block mapping [num_seqs, max_blocks_per_seq]
        query_start_loc: Cumulative query token indices [num_seqs + 1]
        alibi_slopes: Optional ALiBi slopes for positional bias [num_q_heads]
        softmax_aux: Optional auxiliary attention parameters [num_q_heads]
        softmax_scale: Attention scaling factor (default: 1/√head_dim)
        causal: Apply causal masking (default: True)
        sliding_window: Optional local attention window size
        logits_soft_cap: Optional logit capping value for numerical stability
        seq_threshold_3d: Sequence count threshold for 3D kernel variant (Triton)
        num_par_softmax_segments: Parallel softmax segments (Triton-specific)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        cfg: Optional kernel configuration
        mesh: Optional JAX mesh for distributed execution
        in_specs: Optional input partition specifications for distributed execution
        out_specs: Optional output partition specifications for distributed execution

    Returns:
        Tuple of (attention_output, updated_key_cache, updated_value_cache):
            - attention_output: [total_tokens, num_q_heads, head_dim]
            - updated_key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            - updated_value_cache: [num_blocks, block_size, num_kv_heads, head_dim]

    Example:
        ```python
        outputs, new_k_cache, new_v_cache = chunked_prefill_paged_decode(
            queries=q,
            keys=k,
            values=v,
            key_cache=k_cache,
            value_cache=v_cache,
            kv_lens=kv_lengths,
            block_tables=block_mapping,
            query_start_loc=query_offsets,
        )
        ```
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _chunked_prefill_paged_decode_executor(
        ChunkedPrefillPagedDecode(),
        queries=queries,
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

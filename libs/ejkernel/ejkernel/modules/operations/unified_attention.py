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

"""Unified (paged) attention module with automatic platform selection.

This module implements unified paged attention that handles both prefill and decode
phases with a single kernel. It supports ragged query packing (variable-length
sequences without padding) and paged KV caches for memory-efficient inference.

This operation wraps the vLLM-style unified attention kernel implemented in
`ejkernel.kernels` and provides a high-level API consistent with other
`ejkernel.modules.operations` entry points.

Key Features:
    - Unified prefill and decode in a single kernel
    - Ragged query packing (no padding required for variable-length sequences)
    - Paged KV cache with block tables for memory efficiency
    - Automatic 2D/3D grid selection based on sequence characteristics
    - Grouped Query Attention (GQA) and Multi-Query Attention (MQA) support
    - Optional ALiBi positional biases
    - Optional attention sinks for streaming inference
    - Sliding window attention support
    - Logit soft capping (Gemma-2 style)

Use Cases:
    - High-throughput LLM serving with continuous batching
    - Mixed prefill + decode batches (chunked prefill)
    - Variable-length sequence processing without padding overhead
    - Streaming inference with attention sinks

Memory Layout:
    Queries: Packed tensor [total_tokens, num_q_heads, head_dim]
        - All sequences concatenated without padding
        - Boundaries defined by query_start_loc

    KV Cache: Paged layout [num_blocks, block_size, num_kv_heads, head_dim]
        - Fixed-size blocks that can be allocated/freed dynamically
        - Block tables map logical positions to physical blocks

    Block Tables: [num_seqs, max_blocks_per_seq]
        - Maps each sequence's logical block indices to physical block IDs

Mathematical Foundation:
    For each query token at position i in sequence s:
        context_start = query_start_loc[s]
        context_end = query_start_loc[s+1]

        For causal:
            output[i] = Attention(Q[i], K[:kv_lens[s]], V[:kv_lens[s]])

        Where K, V are gathered from paged cache via block_tables[s]

Grid Selection Heuristics:
    - Short sequences (< threshold): 2D grid for better parallelism
    - Long sequences (>= threshold): 3D grid with parallel softmax reduction
    - Threshold adapts based on num_kv_heads to maintain occupancy

Performance Characteristics:
    - No padding overhead: Processes exactly the required tokens
    - Memory efficient: Paged KV cache with dynamic allocation
    - Continuous batching friendly: Supports mixed prefill/decode

References:
    - vLLM: https://arxiv.org/abs/2309.06180
    - Continuous Batching: https://www.anyscale.com/blog/continuous-batching
"""

from __future__ import annotations

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
from .configs import UnifiedAttentionConfig

MIN_LAUNCH_GRID_SIZE_2D = 128
NUM_PAR_SOFTMAX_SEGMENTS = 16

_ARGUMENT_ORDER = (
    "queries",
    "key_cache",
    "value_cache",
    "kv_lens",
    "block_tables",
    "query_start_loc",
)
_ARGUMENT_INDEX = {name: idx for idx, name in enumerate(_ARGUMENT_ORDER)}


def _resolve_inv_arg(inv: Invocation, name: str):
    """Resolve an argument from an Invocation by name.

    Looks up the argument first in kwargs, then falls back to positional args
    based on the canonical argument ordering defined in _ARGUMENT_ORDER.

    Args:
        inv: The Invocation object containing args and kwargs.
        name: The argument name to resolve.

    Returns:
        The resolved argument value.

    Raises:
        KeyError: If the argument is not found in kwargs and the positional
            index is out of bounds.
    """
    if name in inv.kwargs:
        return inv.kwargs[name]
    idx = _ARGUMENT_INDEX.get(name)
    if idx is None or idx >= len(inv.args):
        raise KeyError(name)
    return inv.args[idx]


class UnifiedAttention(Kernel[UnifiedAttentionConfig, Array]):
    """vLLM-style unified attention over a paged KV cache (inference-only).

    This kernel implements unified paged attention for inference serving workloads,
    supporting both prefill and decode phases with a single kernel. It is designed
    to work with vLLM-style paged KV caches where key-value tensors are stored in
    fixed-size blocks that can be dynamically allocated and mapped per sequence.

    The unified attention kernel automatically selects between different execution
    strategies based on sequence lengths:
    - For short sequences: Uses a 2D grid launch for better parallelism
    - For long sequences: Uses a 3D grid launch with parallel softmax reduction

    Features:
        - Paged KV cache support with block tables for memory efficiency
        - Ragged query packing (variable-length sequences without padding)
        - Automatic 2D/3D grid selection based on sequence characteristics
        - Support for GQA/MQA (grouped/multi-query attention)
        - Optional sliding window attention
        - Optional logits soft capping (Gemma-2 style)
        - Optional ALiBi position biases
        - Optional attention sinks for streaming inference

    Example:
        >>> kernel = UnifiedAttention()
        >>> output = kernel.run(
        ...     queries=packed_queries,      # [total_tokens, num_q_heads, head_dim]
        ...     key_cache=key_cache,         # [num_blocks, block_size, num_kv_heads, head_dim]
        ...     value_cache=value_cache,     # [num_blocks, block_size, num_kv_heads, head_dim]
        ...     kv_lens=context_lengths,     # [num_seqs]
        ...     block_tables=block_tables,   # [num_seqs, max_blocks_per_seq]
        ...     query_start_loc=cu_seqlens,  # [num_seqs + 1]
        ...     cfg=UnifiedAttentionConfig(),
        ... )
    """

    def __init__(self):
        """Initialize the UnifiedAttention kernel."""
        super().__init__(op_id="unified_attention")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        kv_lens: Int32[Array, "num_seqs"],
        block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
        query_start_loc: Int32[Array, "num_seqs_plus_1"],
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,
        qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        softmax_scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        seq_threshold_3d: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: UnifiedAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed unified attention.

        Args:
            queries: Packed query tensor [total_tokens, num_q_heads, head_dim].
            key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
            value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
            kv_lens: Context lengths [num_seqs].
            block_tables: Block table mapping [num_seqs, max_blocks_per_seq].
            query_start_loc: Query start indices [num_seqs + 1].
            alibi_slopes: Optional ALiBi slopes [num_q_heads].
            qq_bias: Optional query-query bias [num_query_tokens, num_query_tokens].
            softmax_aux: Optional attention sink values [num_q_heads].
            All other args: Unified attention parameters to be fixed via partial.
            mesh: JAX device mesh for shard_map execution.
            in_specs: Tuple of PartitionSpec objects defining input tensor sharding.
                (for queries, key_cache, value_cache, kv_lens, block_tables,
                query_start_loc, alibi_slopes, qq_bias, softmax_aux)
            out_specs: PartitionSpec defining output tensor sharding.
            check_vma: Whether to check virtual memory alignment in shard_map.

        Returns:
            Tuple of (shard_map_fn, call_args).
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_unified_attention(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
            value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
            kv_lens: Int32[Array, "num_seqs"],
            block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
            query_start_loc: Int32[Array, "num_seqs_plus_1"],
            alibi_slopes: Float[Array, "num_q_heads"] | None,
            qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None,
            softmax_aux: Float[Array, "num_q_heads"] | None,
        ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
            """Shard-map compatible wrapper that delegates to self.run with captured params."""
            return self.run(
                queries=queries,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_lens=kv_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                alibi_slopes=alibi_slopes,
                qq_bias=qq_bias,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                causal=causal,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                seq_threshold_3d=seq_threshold_3d,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            queries,
            key_cache,
            value_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            alibi_slopes,
            qq_bias,
            softmax_aux,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_unified_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: UnifiedAttentionConfig):
        """Get the platform-specific implementation.

        Args:
            cfg: Configuration specifying platform and backend preferences.

        Returns:
            Callable implementation function from the kernel registry.
        """
        platform = detect_platform("unified_attention", cfg.platform, prefer_triton=True)
        return kernel_registry.get("unified_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
        kv_lens: Int32[Array, "num_seqs"],
        block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
        query_start_loc: Int32[Array, "num_seqs_plus_1"],
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,
        qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        *,
        softmax_scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        seq_threshold_3d: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: UnifiedAttentionConfig,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute unified paged attention.

        Args:
            queries: Packed query tensor of shape [total_tokens, num_q_heads, head_dim].
                Contains all query tokens from all sequences concatenated together.
            key_cache: Paged key cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
                Pre-allocated blocks storing key vectors for all sequences.
            value_cache: Paged value cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
                Pre-allocated blocks storing value vectors for all sequences.
            kv_lens: Context lengths per sequence of shape [num_seqs].
                Number of valid KV tokens for each sequence.
            block_tables: Block index mapping of shape [num_seqs, max_blocks_per_seq].
                Maps logical block indices to physical block indices in the cache.
            query_start_loc: Cumulative query token counts of shape [num_seqs + 1].
                query_start_loc[i] gives the starting token index for sequence i.
            softmax_scale: Scaling factor for attention scores. If None, uses 1/sqrt(head_dim).
            causal: Whether to apply causal masking. Default True.
            sliding_window: Optional sliding window size for local attention.
                If provided, each query only attends to the last `sliding_window` KV positions.
            logits_soft_cap: Optional soft cap for attention logits (Gemma-2 style).
                Applies tanh-based capping: logits = soft_cap * tanh(logits / soft_cap).
            alibi_slopes: Optional ALiBi slopes per head of shape [num_q_heads].
                Adds position-dependent bias: bias[i,j] = slope * (j - i).
            qq_bias: Optional query-query bias of shape [num_query_tokens, num_query_tokens].
                Added directly to attention logits between query positions.
            softmax_aux: Optional attention sink values per head of shape [num_q_heads].
                Adds constant attention to the first token for streaming inference stability.
            platform: Override platform selection. One of "triton", "pallas", "cuda", "xla", "auto".
            cfg: Kernel configuration with tuning parameters.

        Returns:
            Output tensor of shape [total_tokens, num_q_heads, head_dim] with attention results.
        """
        if platform is not None:
            cfg = UnifiedAttentionConfig(
                seq_threshold_3d=cfg.seq_threshold_3d,
                num_par_softmax_segments=cfg.num_par_softmax_segments,
                block_dim=cfg.block_dim,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        if seq_threshold_3d is None:
            seq_threshold_3d = cfg.seq_threshold_3d

        impl = self.get_impl(cfg)
        impl_kwargs = {
            "queries": queries,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "kv_lens": kv_lens,
            "block_tables": block_tables,
            "query_start_loc": query_start_loc,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "sliding_window": sliding_window,
            "logits_soft_cap": logits_soft_cap,
            "seq_threshold_3d": seq_threshold_3d,
            "num_par_softmax_segments": cfg.num_par_softmax_segments,
            "alibi_slopes": alibi_slopes,
            "qq_bias": qq_bias,
            "softmax_aux": softmax_aux,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
        }
        if cfg.platform == "cuda":
            impl_kwargs["block_dim"] = cfg.block_dim
        return impl(
            **impl_kwargs,
        )

    def heuristic_cfg(self, inv: Invocation[UnifiedAttentionConfig, Array]) -> UnifiedAttentionConfig:
        """Generate default configuration based on input characteristics.

        Follows vLLM's decode kernel selection heuristic to determine the
        sequence length threshold for switching between 2D and 3D grid launches.

        Args:
            inv: Invocation containing the input arguments and metadata.

        Returns:
            UnifiedAttentionConfig with heuristically determined parameters:
            - seq_threshold_3d: Sequence length above which 3D grid is used
            - num_par_softmax_segments: Number of parallel softmax reduction segments
        """
        key_cache = _resolve_inv_arg(inv, "key_cache")
        num_kv_heads = int(key_cache.shape[2])
        seq_threshold_3d = MIN_LAUNCH_GRID_SIZE_2D // max(1, num_kv_heads)

        return UnifiedAttentionConfig(
            seq_threshold_3d=int(seq_threshold_3d),
            num_par_softmax_segments=int(NUM_PAR_SOFTMAX_SEGMENTS),
            block_dim=128,
            num_warps=None,
            num_stages=None,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[UnifiedAttentionConfig, Array]):
        """Return candidate configurations for autotuning.

        This operation exposes the main tuning knobs directly via the config.

        Args:
            inv: Invocation containing the input arguments and metadata.

        Returns:
            CUDA candidates when CUDA is explicitly requested; otherwise a
            single heuristic configuration.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[UnifiedAttentionConfig, Array]):
        """Return GPU candidates for every registered unified-attention backend.

        The three knobs:

        * ``seq_threshold_3d`` — sequence count below which the 2D grid
          launches (lower decode overhead); above, the 3D segmented
          softmax kernel kicks in (better parallelism on long contexts).
          Heuristic value comes from vLLM; we also try the heuristic ÷ 2
          and × 2 to bracket it.
        * ``num_par_softmax_segments`` — segments for the 3D kernel's
          parallel softmax reduction; powers of two near the heuristic.
        * ``block_dim`` — CUDA thread-block dim; {64, 128, 256}, with 128
          a strong default for most head_dims.
        """
        base = self.heuristic_cfg(inv)
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "cuda", "cute", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        thr = max(1, int(base.seq_threshold_3d or 1))
        thr_choices = sorted({max(1, thr // 2), thr, thr * 2})
        seg = max(1, int(base.num_par_softmax_segments or 1))
        seg_choices = sorted({max(1, seg // 2), seg, min(seg * 2, 64)})
        candidates: list[UnifiedAttentionConfig] = []
        if "triton" in platforms:
            for t in thr_choices:
                for s in seg_choices:
                    candidates.append(
                        UnifiedAttentionConfig(
                            seq_threshold_3d=t,
                            num_par_softmax_segments=s,
                            block_dim=base.block_dim,
                            num_warps=None,
                            num_stages=None,
                            platform="triton",
                            backend="gpu",
                        )
                    )
        if "cuda" in platforms:
            candidates.extend(
                UnifiedAttentionConfig(
                    seq_threshold_3d=base.seq_threshold_3d,
                    num_par_softmax_segments=base.num_par_softmax_segments,
                    block_dim=block_dim,
                    num_warps=None,
                    num_stages=None,
                    platform="cuda",
                    backend="gpu",
                )
                for block_dim in (64, 128, 256)
            )
        if "cute" in platforms:
            candidates.append(
                UnifiedAttentionConfig(
                    seq_threshold_3d=base.seq_threshold_3d,
                    num_par_softmax_segments=base.num_par_softmax_segments,
                    block_dim=base.block_dim,
                    num_warps=None,
                    num_stages=None,
                    platform="cute",
                    backend="gpu",
                )
            )
        if "tilelang" in platforms:
            candidates.append(
                UnifiedAttentionConfig(
                    seq_threshold_3d=base.seq_threshold_3d,
                    num_par_softmax_segments=base.num_par_softmax_segments,
                    block_dim=base.block_dim,
                    num_warps=None,
                    num_stages=None,
                    platform="tilelang",
                    backend="gpu",
                )
            )
        if "xla" in platforms:
            candidates.append(
                UnifiedAttentionConfig(
                    seq_threshold_3d=base.seq_threshold_3d,
                    num_par_softmax_segments=base.num_par_softmax_segments,
                    block_dim=base.block_dim,
                    num_warps=None,
                    num_stages=None,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or [base]

    def candidate_cfgs_tpu(self, inv: Invocation[UnifiedAttentionConfig, Array]):
        """Return the XLA TPU candidate for unified attention."""
        base = self.heuristic_cfg(inv)
        return [
            UnifiedAttentionConfig(
                seq_threshold_3d=base.seq_threshold_3d,
                num_par_softmax_segments=base.num_par_softmax_segments,
                block_dim=base.block_dim,
                num_warps=None,
                num_stages=None,
                platform="xla",
                backend="any",
            )
        ]


_unified_attention_executor: Executor[UnifiedAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=False),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("unified-attention"),
    )
)


def unified_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
    cfg: UnifiedAttentionConfig | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Execute unified paged attention with automatic platform selection.

    This is the main entry point for vLLM-style unified attention, suitable for
    inference serving workloads with paged KV caches. It handles both prefill
    and decode phases efficiently using a single unified kernel.

    The function automatically selects the optimal platform (Triton, Pallas, XLA)
    based on available hardware and applies heuristic-based configuration tuning.

    Args:
        queries: Packed query tensor of shape [total_tokens, num_q_heads, head_dim].
            All query tokens from all sequences are concatenated together without padding.
        key_cache: Paged key cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
            Pre-allocated memory blocks storing key vectors. Blocks are shared across sequences
            and mapped via block_tables.
        value_cache: Paged value cache of shape [num_blocks, block_size, num_kv_heads, head_dim].
            Pre-allocated memory blocks storing value vectors, with same layout as key_cache.
        kv_lens: Context lengths per sequence of shape [num_seqs].
            Specifies how many KV tokens are valid for each sequence.
        block_tables: Block index mapping of shape [num_seqs, max_blocks_per_seq].
            Maps each sequence's logical block indices to physical block indices in the cache.
            For sequence i, block_tables[i, j] gives the physical block index for logical block j.
        query_start_loc: Cumulative query token counts of shape [num_seqs + 1].
            Defines the boundaries of each sequence in the packed queries tensor.
            Sequence i's queries span indices [query_start_loc[i], query_start_loc[i+1]).
        softmax_scale: Scaling factor for attention scores. Default: 1/sqrt(head_dim).
        causal: Whether to apply causal masking. Default: True.
            When True, each query can only attend to KV positions at or before its position.
        sliding_window: Optional sliding window size for local attention.
            If provided, each query only attends to the most recent `sliding_window` KV positions.
        logits_soft_cap: Optional soft cap for attention logits (Gemma-2 style).
            Applies: logits = soft_cap * tanh(logits / soft_cap) to prevent extreme values.
        alibi_slopes: Optional ALiBi position bias slopes per head [num_q_heads].
            Adds linear position-dependent bias to attention scores.
        qq_bias: Optional query-query attention bias [num_query_tokens, num_query_tokens].
            Directly added to attention logits for query-to-query interactions.
        softmax_aux: Optional attention sink values per head [num_q_heads].
            Provides stable attention to the first token for streaming/infinite context.
        platform: Override automatic platform selection. One of:
            - "triton": Force Triton (GPU)
            - "pallas": Force Pallas (TPU/GPU)
            - "xla": Force XLA fallback
            - "cuda": Force CUDA
            - "auto": Automatic selection (default)
            - None: Same as "auto"
        cfg: Optional configuration override. If None, uses heuristic-based defaults.

    Returns:
        Output tensor of shape [total_tokens, num_q_heads, head_dim] containing the
        attention results for all sequences, packed in the same order as queries.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.modules import unified_attention
        >>>
        >>> # Setup for 2 sequences with different lengths
        >>> total_tokens = 5  # seq1 has 2 tokens, seq2 has 3 tokens
        >>> queries = jnp.ones((total_tokens, 8, 64))  # 8 heads, dim 64
        >>> key_cache = jnp.ones((100, 16, 2, 64))     # 100 blocks, block_size=16, 2 KV heads
        >>> value_cache = jnp.ones((100, 16, 2, 64))
        >>> kv_lens = jnp.array([32, 48])              # context lengths
        >>> block_tables = jnp.array([[0, 1], [2, 3, 4]])  # block mappings
        >>> query_start_loc = jnp.array([0, 2, 5])     # [0, 2) for seq1, [2, 5) for seq2
        >>>
        >>> output = unified_attention(
        ...     queries, key_cache, value_cache,
        ...     kv_lens, block_tables, query_start_loc,
        ...     causal=True,
        ... )

    Note:
        This kernel is optimized for inference and does not support backward passes.
        For training, use `flash_attention` instead.
    """

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _unified_attention_executor(
        UnifiedAttention(),
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        softmax_aux=softmax_aux,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

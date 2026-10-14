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


"""Scaled Dot Product Attention module with automatic optimization.

This module implements the standard scaled dot-product attention mechanism,
which is the fundamental building block of transformer architectures. It computes:

    Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

where Q, K, V are the query, key, and value matrices, and d_k is the key dimension.

This implementation is designed for general-purpose attention computation with
automatic platform selection and optimization, serving as a simpler alternative
to FlashAttention when memory efficiency is not the primary concern.

Key Features:
    - Automatic platform selection (XLA, Triton, Pallas, CUDA)
    - Causal and bidirectional attention patterns
    - Sliding window attention for local context
    - Variable-length sequence handling via cumulative sequence lengths
    - Distributed execution support via shard_map
    - Attention biasing and masking
    - Lazy bias initialization for memory efficiency
    - Grouped Query Attention (GQA) support

Use Cases:
    - General transformer computations
    - Research and experimentation with attention mechanisms
    - Scenarios where XLA optimization provides sufficient performance
    - Simple attention patterns without complex memory requirements
    - Cross-attention between different modalities

Mathematical Foundation:
    Scaled dot-product attention computes:
        scores = (Q @ K^T) / sqrt(d_k)         # [batch, heads, seq_len, kv_len]
        weights = softmax(scores + bias)       # Apply masking via bias
        output = weights @ V                   # [batch, heads, seq_len, head_dim]

    With optional features:
        - Causal mask: Only attend to previous positions (j <= i)
        - Sliding window: Attend within a fixed window around each position
        - Bias: Add learnable or computed bias to attention scores
        - Soft cap: scores = soft_cap * tanh(scores / soft_cap) for stability

Performance Characteristics:
    - Memory: O(N^2) standard attention complexity
    - Compute: O(N^2 * d) where N is sequence length, d is head dimension
    - Platform: Uses XLA's built-in attention primitive when available
    - Best for: Moderate sequence lengths, cross-attention, and general use

Note:
    For memory-efficient attention on long sequences, prefer FlashAttention.
    For inference-only paged KV cache workloads, prefer PageAttention.

References:
    - Attention Is All You Need (Vaswani et al., 2017)
      https://arxiv.org/abs/1706.03762
    - JAX Pallas: https://jax.readthedocs.io/en/latest/pallas/
"""

from __future__ import annotations

import typing
from collections.abc import Callable

import jax
from jax import shard_map
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
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
from .configs import ScaledDotProductAttentionConfig


class ScaledDotProductAttention(Kernel[ScaledDotProductAttentionConfig, Array]):
    """Standard scaled dot-product attention kernel wrapper.

    Dispatches to XLA's built-in attention primitive or a platform-specific
    implementation.  Supports causal masking, sliding windows, and
    variable-length sequences, but does **not** support dropout.

    Note:
        Block sizes are not tunable for this operation — the XLA backend
        optimises tiling internally.  For block-size autotuning or fused
        dropout, use :class:`FlashAttention` instead.

    Features:
        - Automatic platform/backend selection (primarily XLA)
        - Configuration caching for consistent performance
        - Custom gradient support for efficient backpropagation
        - Support for variable-length sequences via cumulative sequence lengths
        - Sliding window attention for local attention patterns
        - Lazy bias initialisation via ``init_bias`` callable

    Example:
        >>> from ejkernel.modules import ScaledDotProductAttention, create_default_executor
        >>>
        >>>
        >>> executor = create_default_executor()
        >>> attn = ScaledDotProductAttention()
        >>>
        >>>
        >>> output = executor(attn, query, key, value, causal=True, softmax_scale=0.125)
        >>>
        >>>
        >>> output = executor(
        ...     attn, query, key, value,...
        ... )
        >>>
        >>>
        >>> output = executor(attn, query, key, value, sliding_window=(256, 256))
    """

    def __init__(self):
        """Initialize ScaledDotProductAttention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="scaled_dot_product_attention")

    def get_impl(self, cfg: ScaledDotProductAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation

        Raises:
            ValueError: If no matching implementation is found
        """
        return kernel_registry.get(
            algorithm="scaled_dot_product_attention",
            platform=detect_platform("scaled_dot_product_attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,
        cum_seqlens_k: Int[Array, "batch"] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: ScaledDotProductAttentionConfig,
    ) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
        """Execute scaled dot-product attention with the given configuration.

        Args:
            query: Query tensor [batch, seq_len_q, num_q_heads, head_dim].
            key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
            value: Value tensor [batch, kv_len, num_kv_heads, head_dim].
            attention_mask: Optional boolean/integer mask
                [batch, num_heads_or_1, seq_len_q, kv_len]. ``True`` (or 1)
                means *attend*, 0/``False`` means *mask out*. Prefer ``bias``
                for additive masking.
            bias: Optional additive attention bias
                [batch, num_heads, seq_len_q, kv_len]. Added directly to
                pre-softmax logits.
            init_bias: Optional callable that lazily creates the bias on-device.
                Called only when ``bias`` is ``None``; the returned tensor must
                match the expected bias shape.
            softmax_scale: Multiplicative scale applied to QK^T logits before
                softmax. Defaults to ``1 / sqrt(head_dim)`` when ``None``.
            causal: If ``True``, apply a causal (lower-triangular) mask so
                each query position attends only to earlier key positions.
            sliding_window: If set, restrict attention to a window of this many
                key positions around each query (int = symmetric window; tuple
                ``(left, right)`` for asymmetric).
            cum_seqlens_q: Cumulative sequence lengths for variable-length
                queries [batch].
            cum_seqlens_k: Cumulative sequence lengths for variable-length
                keys [batch].
            platform: Override platform selection ("triton", "pallas", "cuda",
                "xla", "auto").
            cfg: Configuration object specifying platform and backend.

        Returns:
            Attention output [batch, seq_len_q, num_q_heads, head_dim].
        """

        if platform is not None:
            cfg = ScaledDotProductAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        resolved_platform = detect_platform("scaled_dot_product_attention", cfg.platform)
        impl = self.get_impl(cfg)
        impl_kwargs = dict(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            init_bias=init_bias,
            sliding_window=sliding_window,
            causal=causal,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
        )
        if resolved_platform == Platform.TILELANG:
            impl_kwargs["fwd_params"] = FwdParams(
                q_blocksize=cfg.block_q,
                kv_blocksize=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
            impl_kwargs["bwd_params"] = BwdParams(
                q_blocksize=max(32, cfg.block_q // 2),
                kv_blocksize=max(32, cfg.block_k // 2),
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        return impl(**impl_kwargs)

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,
        cum_seqlens_k: Int[Array, "batch"] | None = None,
        *,
        mesh: jax.sharding.Mesh,
        in_specs: tuple[jax.sharding.PartitionSpec, ...],
        out_specs: jax.sharding.PartitionSpec,
        check_vma: bool = False,
        cfg: ScaledDotProductAttentionConfig,
        init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    ):
        """Create a shard_map wrapper for distributed ScaledDotProductAttention execution.

        Enables efficient distributed execution of attention across multiple devices
        using JAX's shard_map functionality. This is particularly useful for model
        parallelism and handling very large attention computations.

        Args:
            query: Query tensor [batch, seq_len, num_q_heads, head_dim]
            key: Key tensor [batch, kv_len, num_kv_heads, head_dim]
            value: Value tensor [batch, kv_len, num_kv_heads, head_dim]
            attention_mask: Optional attention mask [batch, 1, seq_len, kv_len]
            bias: Optional attention bias [batch, num_heads, seq_len, kv_len]
            cum_seqlens_q: Cumulative sequence lengths for queries [batch]
            cum_seqlens_k: Cumulative sequence lengths for keys [batch]
            mesh: JAX mesh defining device topology for distributed execution
            in_specs: Partition specifications for each input tensor
            out_specs: Partition specification for output tensor
            check_vma: Whether to check for virtual memory access patterns
            cfg: Configuration object specifying platform/backend
            init_bias: Optional callable to initialize bias on-device
            softmax_scale: Scaling factor for attention scores
            causal: Whether to apply causal masking
            sliding_window: Window size for local attention
            platform: Optional platform override

        Returns:
            Tuple of (shard_map function, call args) where:
                - shard_map function: Callable for distributed execution
                - call args: Tuple of arguments to pass to the shard_map function

        Note:
            The shard_map wrapper handles device placement and communication
            automatically based on the provided mesh and partition specs.
        """
        impl = self.get_impl(cfg)

        def _wrapped_sdpa(
            query,
            key,
            value,
            bias,
            cum_seqlens_q,
            cum_seqlens_k,
            attention_mask,
        ):
            """Shard-map compatible wrapper that delegates to impl with captured params."""
            return impl(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                bias=bias,
                cum_seqlens_q=cum_seqlens_q,
                cum_seqlens_k=cum_seqlens_k,
                init_bias=init_bias,
                softmax_scale=softmax_scale,
                causal=causal,
                sliding_window=sliding_window,
            )

        call_args = (
            query,
            key,
            value,
            bias,
            cum_seqlens_q,
            cum_seqlens_k,
            attention_mask,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_sdpa,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def heuristic_cfg(self, inv: Invocation[ScaledDotProductAttentionConfig, Array]) -> ScaledDotProductAttentionConfig:
        """Provide default configuration based on invocation context.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration for platform/backend selection
        """

        return ScaledDotProductAttentionConfig(
            block_q=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[ScaledDotProductAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        This operation uses XLA primitives directly without tunable block sizes,
        so autotuning provides no benefit. Returns empty list to skip autotuning.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Empty list - no candidates to autotune since XLA handles optimization

        Note:
            XLA's scaled_dot_product_attention primitive is not parameterized by
            block sizes, so there are no meaningful configurations to benchmark.
        """

        return [
            ScaledDotProductAttentionConfig(
                block_q=128,
                block_k=128,
                num_warps=4,
                num_stages=2,
                platform="auto",
                backend="any",
            ),
            ScaledDotProductAttentionConfig(
                block_q=128,
                block_k=128,
                num_warps=4,
                num_stages=2,
                platform="xla",
                backend="any",
            ),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[ScaledDotProductAttentionConfig, Array]):
        """Generate GPU candidates for SDPA.

        TileLang candidates sweep the FlashAttention-2 tile space:

        * ``block_q`` ∈ {32, 64, 128, 256} pruned by ``q_len``; 256 only
          when ``q_len >= 512`` (otherwise wasted CTAs).
        * ``block_k`` ∈ {32, 64, 128, 256}, capped at 128 when
          ``head_dim >= 128`` to keep SMEM in budget on H100.
        * ``num_stages`` ∈ {2, 3} — 3 helps memory-bound large-``S`` runs.

        Pallas/Triton route to the cuDNN-backed SDPA wrapper, which has
        its own internal autotuning; a single platform candidate suffices.
        """
        query = inv.kwargs["query"]
        key = inv.kwargs["key"]
        requested = inv.kwargs.get("platform", None)
        head_dim = int(query.shape[-1])
        q_len = int(query.shape[1])
        k_len = int(key.shape[1])
        q_opts = [32, 64, 128, 256] if q_len <= 256 else [64, 128, 256]
        k_opts = [32, 64, 128, 256] if k_len <= 256 else [64, 128, 256]
        if head_dim >= 128:
            k_opts = [k for k in k_opts if k <= 128] or [128]
            q_opts = [q for q in q_opts if q <= 128] or [128]
        if q_len < 512:
            q_opts = [q for q in q_opts if q < 256] or [128]
        if k_len < 512:
            k_opts = [k for k in k_opts if k < 256] or [128]
        platforms = ("tilelang", "pallas", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[ScaledDotProductAttentionConfig] = []
        if "tilelang" in platforms:
            for block_q in q_opts:
                for block_k in k_opts:
                    big = max(block_q, block_k) >= 128
                    warps = 8 if (head_dim >= 128 and big) else 4
                    for stages in (2, 3) if k_len >= 1024 else (2,):
                        candidates.append(
                            ScaledDotProductAttentionConfig(
                                block_q=block_q,
                                block_k=block_k,
                                num_warps=warps,
                                num_stages=stages,
                                platform="tilelang",
                                backend="gpu",
                            )
                        )
        if "pallas" in platforms or "triton" in platforms:
            platform_name = "triton" if "triton" in platforms else "pallas"
            candidates.append(
                ScaledDotProductAttentionConfig(
                    block_q=128,
                    block_k=128,
                    num_warps=4,
                    num_stages=2,
                    platform=platform_name,
                    backend="gpu",
                )
            )
        if "xla" in platforms:
            candidates.append(
                ScaledDotProductAttentionConfig(
                    block_q=128,
                    block_k=128,
                    num_warps=4,
                    num_stages=2,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[ScaledDotProductAttentionConfig, Array]):
        """Generate TPU candidates for XLA SDPA."""
        return [
            ScaledDotProductAttentionConfig(
                block_q=128,
                block_k=128,
                num_warps=4,
                num_stages=2,
                platform="xla",
                backend="any",
            )
        ]


_executor: Executor[ScaledDotProductAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=True),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("sdpa"),
    )
)


def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
    /,
    *,
    mask_info: MaskInfo | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None,
    out_specs: jax.sharding.PartitionSpec | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Execute scaled dot-product attention with automatic optimization.

    Standard ``Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`` computed
    via XLA or a platform-specific kernel.  Unlike :func:`flash_attention`,
    this function exposes no block-size tuning; the XLA backend optimises
    the computation internally.

    Note:
        This operation does **not** support dropout.  For dropout or
        memory-efficient training, use :func:`flash_attention` instead.

    Args:
        query: Query tensor [batch, seq_len_q, num_q_heads, head_dim].
        key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
        value: Value tensor [batch, kv_len, num_kv_heads, head_dim].
        bias: Optional additive bias [batch, num_heads, seq_len_q, kv_len].
            Added directly to pre-softmax logits.
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
            [batch].
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
            [batch].
        mask_info: Optional :class:`~ejkernel.types.mask.MaskInfo` providing
            an attention mask.  When supplied, the contained mask is extracted
            and passed as ``attention_mask`` to the backend.
        init_bias: Optional callable that lazily initialises the bias on-device
            (called only when ``bias`` is ``None``).
        softmax_scale: Scale factor for logits (default: ``1/sqrt(head_dim)``).
        causal: If ``True``, apply causal masking.
        sliding_window: Sliding window size for local attention (int or
            ``(left, right)`` tuple).
        platform: Override platform selection ("triton", "pallas", "cuda",
            "xla", "auto").
        mesh: JAX device mesh for ``shard_map`` distributed execution.
        in_specs: Input ``PartitionSpec`` tuple for ``shard_map``.
            Must contain 7 entries matching the shard_map call_args order
            (query, key, value, bias, cum_seqlens_q, cum_seqlens_k,
            attention_mask).
        out_specs: Output ``PartitionSpec`` for ``shard_map``.

    Returns:
        Attention output [batch, seq_len_q, num_q_heads, head_dim].

    Example:
        >>> out = scaled_dot_product_attention(query, key, value, causal=True)
        >>> out = scaled_dot_product_attention(
        ...     query, key, value, softmax_scale=0.125, causal=True
        ... )
        >>> out = scaled_dot_product_attention(
        ...     query, key, value,
        ...     cum_seqlens_q=cu_q, cum_seqlens_k=cu_k
        ... )
    """

    attention_mask = None

    if mask_info is not None:
        attention_mask = mask_info.get_or_compute_attention_mask()

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

        if mask_info is None:
            in_specs = (*in_specs, None)
        else:
            shardings = mask_info.get_shardings(False, mesh=mesh)
            in_specs = (*in_specs, shardings.attention_mask)
    return _executor(
        ScaledDotProductAttention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        init_bias=init_bias,
        sliding_window=sliding_window,
        causal=causal,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

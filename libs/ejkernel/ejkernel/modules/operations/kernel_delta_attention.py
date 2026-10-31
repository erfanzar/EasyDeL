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

"""Kernel Delta Attention (KDA) operation module with automatic optimization.

This module provides the KernelDeltaAttention operation, a linear attention
variant that uses delta rule updates to maintain a key-value memory matrix.
KDA achieves O(N) complexity while supporting efficient incremental inference.

Key characteristics of KDA:
    - Memory matrix: [num_heads, head_dim, value_dim] per batch
    - Delta updates: Only stores what's new in each value
    - Decay mechanism: Controls memory retention over time
    - Beta parameter: Per-token learning rate for updates

The core recurrence:
    h_t = exp(decay_t) * h_{t-1} + k_t ⊗ (beta_t * (v_t - h_{t-1} @ k_t))
    o_t = h_t @ q_t

Algorithm modes:
    - Chunked (default): Parallel within chunks for training efficiency
    - Recurrent: Pure sequential scan for memory efficiency
    - Single-step: Optimized path for autoregressive generation

Features:
    - Automatic platform selection (XLA primary)
    - Configuration caching for consistent performance
    - L2 normalization option for stability
    - State passthrough for incremental inference

Example:
    >>> from ejkernel.modules.operations import kernel_delta_attention
    >>>
    >>> # Training forward pass
    >>> output = kernel_delta_attention(
    ...     query, key, value, beta, decay,
    ...     chunk_size=64, use_chunked=True,
    ... )
    >>>
    >>> # Inference with state
    >>> output, state = kernel_delta_attention(
    ...     query, key, value, beta, decay,
    ...     return_state=True,
    ... )
    >>> # Next token
    >>> output_new, state_new = kernel_delta_attention(
    ...     q_new, k_new, v_new, beta_new, decay_new,
    ...     initial_state=state, return_state=True,
    ... )

References:
    - Delta Networks: https://arxiv.org/abs/1612.04859
    - Linear Transformers: https://arxiv.org/abs/2006.16236
"""

from __future__ import annotations

import typing
from typing import Literal

from jaxtyping import Array, Float

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
from .configs import KernelDeltaAttentionConfig


class KernelDeltaAttention(Kernel[KernelDeltaAttentionConfig, Array]):
    """Kernel Delta Attention (KDA) operation.

    A linear attention mechanism using delta rule updates to maintain
    an associative memory matrix. Supports both training (chunked) and
    inference (single-step) modes with O(N) complexity.

    The operation maintains a [head_dim, value_dim] memory matrix per head
    that stores key-value associations. The delta update mechanism ensures
    only novel information is added, improving memory efficiency.

    Attributes:
        op_id: Operation identifier for registry lookup ("kernel_delta_attention")
    """

    def __init__(self):
        """Initialize KernelDeltaAttention operation.

        Sets up the kernel with operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="kernel_delta_attention")

    def get_impl(self, cfg: KernelDeltaAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for KDA

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("kernel_delta_attention", cfg.platform)
        return kernel_registry.get("kernel_delta_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads qk_head_dim"],
        key: Float[Array, "batch seq_len num_heads qk_head_dim"],
        value: Float[Array, "batch seq_len num_heads v_head_dim"],
        beta: Float[Array, "batch seq_len num_heads"],
        decay: Float[Array, "batch seq_len num_heads"] | None = None,
        initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
        *,
        softmax_scale: float | None = None,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        use_chunked: bool = True,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: KernelDeltaAttentionConfig,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "batch num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute Kernel Delta Attention operation.

        Args:
            query: Query tensor for attention
                Shape: [batch, seq_len, num_heads, qk_head_dim]
            key: Key tensor for memory addressing
                Shape: [batch, seq_len, num_heads, qk_head_dim]
            value: Value tensor to store in memory
                Shape: [batch, seq_len, num_heads, v_head_dim]
            beta: Per-token learning rate for delta updates
                Shape: [batch, seq_len, num_heads]
            decay: Per-token decay for memory retention (should be <= 0)
                Shape: [batch, seq_len, num_heads]
            initial_state: Optional initial memory state
                Shape: [batch, num_heads, qk_head_dim, v_head_dim]
            softmax_scale: Scaling factor for queries (default: head_dim^-0.5)
            chunk_size: Block size for chunked algorithm
            use_qk_l2norm: Apply L2 normalization to queries and keys
            use_chunked: Use chunked (True) or recurrent (False) algorithm
            return_state: If True, return (output, state) tuple
            platform: Override platform selection
            cfg: Kernel configuration object

        Returns:
            If return_state is False:
                output: Attention output [batch, seq_len, num_heads, v_head_dim]
            If return_state is True:
                Tuple of (output, final_state) where final_state has shape
                [batch, num_heads, qk_head_dim, v_head_dim]
        """
        if platform is not None:
            cfg = KernelDeltaAttentionConfig(
                chunk_size=cfg.chunk_size,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        if not use_chunked:
            cfg = KernelDeltaAttentionConfig(chunk_size=cfg.chunk_size, platform="xla", backend="any")

        impl = self.get_impl(cfg)
        out, final_state = impl(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            softmax_scale=softmax_scale,
            chunk_size=int(cfg.chunk_size),
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
            use_chunked=use_chunked,
        )

        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[KernelDeltaAttentionConfig, Array]) -> KernelDeltaAttentionConfig:
        """Provide default configuration based on heuristics.

        Args:
            inv: Invocation object (unused, config is static)

        Returns:
            Default configuration with auto platform selection
        """
        return KernelDeltaAttentionConfig(
            chunk_size=int(inv.kwargs.get("chunk_size", 64)),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelDeltaAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object

        Returns:
            Chunk-size candidates for the XLA chunked path plus the automatic
            backend choice.
        """
        base_chunk = int(inv.kwargs.get("chunk_size", 64))
        chunks = sorted({max(16, base_chunk // 2), base_chunk, min(512, base_chunk * 2), 128, 256})
        return [KernelDeltaAttentionConfig(chunk_size=c, platform="auto", backend="any") for c in chunks]

    def candidate_cfgs_gpu(self, inv: Invocation[KernelDeltaAttentionConfig, Array]):
        """Generate GPU candidates for KDA.

        TileLang is a native recurrent scan; XLA exposes ``chunk_size`` as
        the meaningful tuning knob for chunked training workloads.
        """
        requested = inv.kwargs.get("platform", None)
        use_chunked = bool(inv.kwargs.get("use_chunked", True))
        base = int(inv.kwargs.get("chunk_size", 64))
        chunks = sorted({max(16, base // 2), base, min(512, base * 2), 128, 256})
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[KernelDeltaAttentionConfig] = []
        if "tilelang" in platforms and use_chunked:
            candidates.append(KernelDeltaAttentionConfig(chunk_size=base, platform="tilelang", backend="gpu"))
        if "xla" in platforms:
            candidates.extend(KernelDeltaAttentionConfig(chunk_size=c, platform="xla", backend="any") for c in chunks)
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[KernelDeltaAttentionConfig, Array]):
        """Generate TPU candidates for the XLA chunked KDA implementation."""
        base = int(inv.kwargs.get("chunk_size", 64))
        chunks = sorted({max(16, base // 2), base, min(512, base * 2), 128, 256})
        return [KernelDeltaAttentionConfig(chunk_size=c, platform="xla", backend="any") for c in chunks]


_executor: Executor[KernelDeltaAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=True),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("kernel_delta_attention"),
    )
)


def kernel_delta_attention(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: KernelDeltaAttentionConfig | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "batch num_heads qk_head_dim v_head_dim"],
    ]
):
    """Execute Kernel Delta Attention (KDA) with automatic optimization.

    KDA is a linear attention variant that uses delta rule updates to maintain
    an associative key-value memory matrix. It provides O(N) complexity while
    supporting efficient stateful incremental inference.

    The core recurrence:
        h_t = exp(decay_t) * h_{t-1} + k_t ⊗ (beta_t * (v_t - h_{t-1} @ k_t))
        o_t = h_t @ q_t

    Args:
        query: Query tensor for attention
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        key: Key tensor for memory addressing
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        value: Value tensor to store in memory
            Shape: [batch, seq_len, num_heads, v_head_dim]
        beta: Per-token learning rate for delta updates (typically in [0, 1])
            Shape: [batch, seq_len, num_heads]
        decay: Per-token decay for memory retention (should be <= 0)
            Shape: [batch, seq_len, num_heads]
            If None, defaults to zeros (no decay, full retention)
        initial_state: Optional initial memory state for incremental inference
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
        softmax_scale: Scaling factor for queries. If None, uses head_dim^-0.5
        chunk_size: Block size for chunked algorithm (default: 64)
        use_qk_l2norm: If True, apply L2 normalization to queries and keys
            before attention. Improves numerical stability (default: True)
        use_chunked: If True, use chunked algorithm (faster for training);
            else use recurrent scan (more memory efficient)
        return_state: If True, return (output, final_state) tuple
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional kernel configuration

    Returns:
        If return_state is False:
            output: Attention output [batch, seq_len, num_heads, v_head_dim]
        If return_state is True:
            Tuple of:
                - output: Attention output [batch, seq_len, num_heads, v_head_dim]
                - final_state: Final memory state for incremental inference
                    [batch, num_heads, qk_head_dim, v_head_dim]

    Example:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> # Training forward pass
        >>> batch, seq_len, num_heads, head_dim = 2, 64, 8, 32
        >>> key = random.PRNGKey(0)
        >>> q = random.normal(random.fold_in(key, 0), (batch, seq_len, num_heads, head_dim))
        >>> k = random.normal(random.fold_in(key, 1), (batch, seq_len, num_heads, head_dim))
        >>> v = random.normal(random.fold_in(key, 2), (batch, seq_len, num_heads, head_dim))
        >>> beta = jax.nn.sigmoid(random.normal(random.fold_in(key, 3), (batch, seq_len, num_heads)))
        >>> decay = random.normal(random.fold_in(key, 4), (batch, seq_len, num_heads)) * -0.01
        >>>
        >>> output = kernel_delta_attention(q, k, v, beta, decay)
        >>> output.shape
        (2, 64, 8, 32)
        >>>
        >>> # Inference with state
        >>> output, state = kernel_delta_attention(q, k, v, beta, decay, return_state=True)
        >>>
        >>> # Next token generation
        >>> q_new = random.normal(random.fold_in(key, 5), (batch, 1, num_heads, head_dim))
        >>> k_new = random.normal(random.fold_in(key, 6), (batch, 1, num_heads, head_dim))
        >>> v_new = random.normal(random.fold_in(key, 7), (batch, 1, num_heads, head_dim))
        >>> beta_new = jax.nn.sigmoid(random.normal(random.fold_in(key, 8), (batch, 1, num_heads)))
        >>> decay_new = random.normal(random.fold_in(key, 9), (batch, 1, num_heads)) * -0.01
        >>>
        >>> output_new, state_new = kernel_delta_attention(
        ...     q_new, k_new, v_new, beta_new, decay_new,
        ...     initial_state=state, return_state=True,
        ... )
    """
    return _executor(
        KernelDeltaAttention(),
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        softmax_scale=softmax_scale,
        chunk_size=chunk_size,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        use_chunked=use_chunked,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )


kda_attention = kernel_delta_attention

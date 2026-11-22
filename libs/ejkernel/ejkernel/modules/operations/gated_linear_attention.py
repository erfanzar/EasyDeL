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


"""GLA (Gated Linear Attention) module with automatic optimization.

This module implements Gated Linear Attention (GLA), an efficient attention mechanism
that uses gating to control information flow. GLA combines linear attention
properties with learned gates to achieve both efficiency and expressiveness.

Key features of Gated Linear Attention:
    - Linear complexity: O(T) time and O(KxV) memory per head
    - Dual gating mechanism: Token-level (g) and layer-level (g_gamma) gates
    - Recurrent formulation with explicit state management
    - Support for variable-length sequences and bidirectional processing
    - Seamless integration with transformer architectures

The algorithm implements a gated recurrent attention mechanism:
    At each timestep t:
        kv_t = k_t^T @ v_t                    (outer product)
        g_t = sigmoid(g) * g_gamma            (combined gating)
        h_t = g_t * h_{t-1} + (1 - g_t) * kv_t  (gated state update)
        o_t = q_t^T @ h_t                     (output computation)

    where:
        - h is the state matrix [num_heads, K, V] per batch
        - g controls how much of the previous state to retain
        - The gating allows selective memory of past information

The gating mechanism provides fine-grained control:
    - g (token-level gates): Applied to query representations, controlling
      information flow at each position [batch, seq_len, num_heads, head_dim]
    - g_gamma (layer-level gates): Scalar gates controlling overall attention
      strength per head [batch, num_heads]

Mathematical formulation:
    GLA can be viewed as a gated variant of linear attention:
        o_t = Σ_{i≤t} α_i(t) · (k_i^T · v_i) · q_t

    where α_i(t) are data-dependent decay factors computed from the gates,
    enabling the model to learn optimal retention patterns.

Supports:
    - Variable sequence lengths via cu_seqlens (cumulative sequence lengths)
    - Reverse processing for bidirectional models
    - Initial state for continuation across chunks
    - Multi-head and grouped-query attention patterns
    - Custom softmax scaling

References:
    - Gated Linear Attention Transformers with Hardware-Efficient Training
      (Yang et al., 2023) https://arxiv.org/abs/2312.06635
    - Flash-Linear-Attention: https://github.com/sustcsonglin/flash-linear-attention
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
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
from .configs import GLAttentionConfig


class GLAttention(Kernel[GLAttentionConfig, Array]):
    """Gated Linear Attention with custom optimization logic.

    Implements gated linear attention combining the efficiency of linear attention
    with learnable gating mechanisms for better expressiveness. The gating controls
    information flow at both the query-key interaction and the state update levels,
    achieving O(N) complexity with O(KxV) memory per head.

    Features:
        - Gated attention computation with g (query gates) and g_gamma (layer-wise gates)
        - Support for initial hidden states for chunked/streaming processing
        - Bidirectional and reverse sequence processing
        - Variable-length sequence handling via cumulative lengths
        - Multiple platform support (Triton/Pallas/CUDA/XLA)
        - Automatic configuration caching for consistent performance
        - Optional autotuning to find optimal implementation

    The dual gating mechanism (g and g_gamma) allows fine-grained control:
        - g: Token-level gates applied to query representations
        - g_gamma: Layer-level gates controlling overall attention strength

    Example:
        >>> from ejkernel.modules import GLAttention, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> gla = GLAttention()
        >>> output = executor(gla, query, key, value)
        >>>
        >>> # With gating for selective memory
        >>> output = executor(gla, query, key, value, g=gates, g_gamma=decay)
        >>>
        >>> # Streaming inference with state
        >>> output, state = executor(
        ...     gla, query, key, value,
        ...     initial_state=prev_state,
        ...     return_state=True
        ... )
    """

    def __init__(self):
        """Initialize GLA module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="gla")

    def get_impl(self, cfg: GLAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for gated linear attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("gla", cfg.platform)
        return kernel_registry.get("gla", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads qk_head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
        g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
        g_gamma: Float[Array, "... num_heads"] | None = None,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: GLAttentionConfig,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute gated linear attention computation.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            g: Token-level gating tensor [batch, seq_len, num_heads, head_dim]
            g_gamma: Layer-level gating parameter [batch, num_heads]
            softmax_scale: Optional scaling factor for attention scores
            initial_state: Initial hidden state [batch, num_heads, head_dim, head_dim]
            reverse: If True, process sequence in reverse order
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            return_state: If True, return tuple (output, final_state) instead of just output
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            If return_state=False: Gated attention output [batch, seq_len, num_heads, head_dim]
            If return_state=True: Tuple of (output, final_state) where final_state
                is [batch, num_heads, head_dim, head_dim]

        Note:
            Both g and g_gamma are optional. When provided, they enable more
            expressive attention patterns through learned gating.
        """

        if platform is not None:
            cfg = GLAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        resolved_platform = detect_platform("gla", cfg.platform)
        impl = kernel_registry.get("gla", platform=resolved_platform, backend=cfg.backend)
        impl_kwargs = {}
        if resolved_platform == Platform.TRITON:
            impl_kwargs = {
                "block_k": cfg.block_k,
                "block_v": cfg.block_d,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }

        result = impl(
            query=query,
            key=key,
            value=value,
            g=g,
            g_gamma=g_gamma,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            **impl_kwargs,
        )

        if isinstance(result, tuple):
            if return_state:
                return result
            else:
                return result[0]
        return result

    def heuristic_cfg(self, inv: Invocation[GLAttentionConfig, Array]) -> GLAttentionConfig:
        """Provide default configuration with block sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical gated linear attention workloads
        """
        return GLAttentionConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[GLAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            GLA performance depends on the gating mechanism effectiveness and
            sequence length. Candidates are chosen for typical configurations.
        """
        block_configs = [(64, 64, 64, 4, 1)]

        candidates = []
        for block_q, block_k, block_d, num_warps, num_stages in block_configs:
            candidates.append(
                GLAttentionConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=block_d,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[GLAttentionConfig, Array]):
        """Generate GPU candidates for GLA.

        Gated linear attention has multiplicative gating but the same
        underlying linear-attention tile structure as lightning_attention
        and recurrent — sweep the same (block_q, block_k, block_d) space.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        block_configs = [
            (64, 64, 32, 4, 1),
            (64, 64, 64, 4, 1),
            (128, 64, 64, 4, 2),
            (64, 128, 64, 4, 2),
            (128, 128, 64, 8, 2),
            (128, 128, 128, 8, 2),
        ]
        tilelang_configs = [
            (64, 64, 64, 4, 1),
            (128, 128, 64, 4, 2),
            (128, 128, 128, 8, 2),
        ]
        candidates: list[GLAttentionConfig] = []
        if "triton" in platforms:
            for block_q, block_k, block_d, num_warps, num_stages in block_configs:
                candidates.append(
                    GLAttentionConfig(
                        block_q=block_q,
                        block_k=block_k,
                        block_d=block_d,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="triton",
                        backend="gpu",
                    )
                )
        if "tilelang" in platforms:
            for block_q, block_k, block_d, num_warps, num_stages in tilelang_configs:
                candidates.append(
                    GLAttentionConfig(
                        block_q=block_q,
                        block_k=block_k,
                        block_d=block_d,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(
                GLAttentionConfig(
                    block_q=64,
                    block_k=64,
                    block_d=64,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[GLAttentionConfig, Array]):
        """Generate TPU candidates for the XLA GLA path."""
        return [
            GLAttentionConfig(
                block_q=64,
                block_k=64,
                block_d=64,
                num_warps=4,
                num_stages=1,
                platform="xla",
                backend="any",
            )
        ]


_gla_executor: Executor[GLAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("gla"),
    )
)


def gla_attention(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    reverse: bool = False,
    return_state: bool = False,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: GLAttentionConfig | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """Execute gated linear attention with automatic optimization.

    Convenience function that uses a default executor and GLA module.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        g: Gating tensor [batch, seq_len, num_heads, head_dim]
        g_gamma: Gating gamma [batch, num_heads]
        softmax_scale: Scaling factor for attention
        initial_state: Initial state for recurrent computation
        reverse: Whether to process sequence in reverse
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        return_state: If True, return tuple (output, final_state) instead of just output
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        If return_state=False: Attention output with same shape as query
        If return_state=True: Tuple of (output, final_state)

    Example:
        >>>
        >>> out = gla_attention(query, key, value)
        >>>
        >>>
        >>> out = gla_attention(query, key, value, g=gates, g_gamma=gamma)
        >>>
        >>>
        >>> out = gla_attention(query, key, value, cu_seqlens=cu_seqs)
        >>>
        >>>
        >>> out = gla_attention(..., platform="triton")
    """
    return _gla_executor(
        GLAttention(),
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )

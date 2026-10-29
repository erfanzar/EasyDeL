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


"""Recurrent Attention module with automatic optimization.

This module implements recurrent-style attention mechanisms that maintain and update
hidden states across sequence positions. Unlike standard attention which computes
all positions independently, recurrent attention processes sequences sequentially
with stateful computation, achieving O(N) complexity.

Key features of Recurrent Attention:
    - Stateful attention with initial_state support for chunked processing
    - Separate gating for queries (g), keys (gk), and values (gv)
    - Layer-wise gating control via g_gamma
    - Bidirectional processing support (forward and reverse)
    - Variable-length sequence handling via cumulative lengths

The algorithm implements a gated recurrent attention mechanism:
    At each timestep t:
        kv_t = k_t^T @ v_t           (outer product, shape [K, V])
        g_t = sigmoid(g) if g else 1  (optional gating)
        h_t = h_{t-1} * g_gamma + kv_t * g_t  (state update)
        o_t = q_t^T @ h_t             (output computation)

    where:
        - h is the state matrix [num_heads, qk_head_dim, v_head_dim] per batch
        - g, gk, gv provide fine-grained gating control
        - g_gamma controls layer-level decay of the hidden state

Mathematical formulation:
    The recurrence computes gated attention:
        o_t = Σ_{i≤t} g_gamma^(t-i) · (g_i · k_i^T · v_i) · q_t

    This achieves O(T) complexity while maintaining expressive attention patterns.

Supports:
    - Variable sequence lengths via cu_seqlens (cumulative sequence lengths)
    - Reverse processing for bidirectional models
    - Initial state for continuation across chunks
    - Multi-gating mechanisms for expressive control
    - Custom softmax scaling

This is particularly useful for:
    - Linear-time attention mechanisms
    - Models requiring sequential dependency modeling
    - Architectures with explicit state propagation
    - Efficient inference with incremental state updates
    - Long-context processing with bounded memory

Reference:
    Based on linear attention and gated recurrent mechanisms from:
    - Linear Transformers Are Secretly Fast Weight Programmers
      (Schlag et al., 2021) https://arxiv.org/abs/2102.11174
    - Flash-Linear-Attention implementation
      https://github.com/sustcsonglin/flash-linear-attention
"""

from __future__ import annotations

import os
import typing
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
from .configs import RecurrentAttentionConfig


class RecurrentAttention(Kernel[RecurrentAttentionConfig, Array]):
    """Recurrent Attention with custom optimization logic.

    Implements attention with recurrent state updates, enabling linear-time complexity
    for certain attention patterns. The mechanism maintains a hidden state that is
    updated at each sequence position, providing O(N) complexity with O(d²) memory
    per head.

    Features:
        - Stateful computation with hidden state propagation
        - Multiple gating mechanisms (g, gk, gv, g_gamma)
        - Forward and reverse processing modes
        - Support for initial states for chunked/streaming processing
        - Variable-length sequence handling via cu_seqlens
        - Multiple platform support (Triton/Pallas/CUDA/XLA)
        - Automatic configuration caching for consistent performance
        - Optional autotuning to find optimal implementation

    The gating mechanisms provide fine-grained control:
        - g: Query-level gates controlling input relevance
        - gk: Key-level gates modulating key contributions
        - gv: Value-level gates filtering value information
        - g_gamma: Layer-level decay controlling state persistence

    Example:
        >>> from ejkernel.modules import RecurrentAttention, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> attn = RecurrentAttention()
        >>> output = executor(attn, query, key, value)
        >>>
        >>> # With gating for more expressive attention
        >>> output = executor(attn, query, key, value, g=gates, g_gamma=decay)
        >>>
        >>> # Streaming inference with state
        >>> output, state = executor(
        ...     attn, query, key, value,
        ...     initial_state=prev_state,
        ...     return_state=True
        ... )
    """

    def __init__(self):
        """Initialize Recurrent Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="recurrent")

    def get_impl(self, cfg: RecurrentAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for recurrent attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("recurrent", cfg.platform)
        return kernel_registry.get("recurrent", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads qk_head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
        g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
        g_gamma: Float[Array, "... num_heads"] | None = None,
        gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
        gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: RecurrentAttentionConfig,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute recurrent attention with stateful computation.

        Args:
            query: Query tensor [batch, seq_len, num_heads, qk_head_dim].
            key: Key tensor [batch, seq_len, num_kv_heads, qk_head_dim].
            value: Value tensor [batch, seq_len, num_kv_heads, v_head_dim].
            g: Optional query-level gate tensor
                [batch, seq_len, num_heads, qk_head_dim].  Applied after
                sigmoid; if ``None``, no query gating is performed.
            g_gamma: Optional per-layer decay parameter broadcast to heads
                ``[..., num_heads]``.  Controls how fast the state forgets.
            gk: Optional key-level gate tensor
                [batch, seq_len, num_heads, qk_head_dim].
            gv: Optional value-level gate tensor
                [batch, seq_len, num_heads, v_head_dim].
            softmax_scale: Optional scaling factor for attention scores.
                Defaults to ``1 / sqrt(qk_head_dim)`` when ``None``.
            initial_state: Optional initial hidden state
                ``[..., num_heads, qk_head_dim, v_head_dim]``.  Used to
                continue processing from a previous chunk.
            reverse: If ``True``, process sequence in reverse temporal order.
            cu_seqlens: Cumulative sequence lengths for variable-length
                sequences [num_seqs + 1].  Enables FlashAttention-style
                packed sequences.
            return_state: If ``True``, return a tuple ``(output, final_state)``
                instead of just the output.
            platform: Optional platform override ("triton", "pallas", "cuda",
                "xla").
            cfg: Kernel configuration object (``block_q``, ``block_k``,
                ``block_d``, ``num_warps``, ``num_stages``).

        Returns:
            If ``return_state=False``:
                Attention output [batch, seq_len, num_heads, v_head_dim].
            If ``return_state=True``:
                Tuple of (output, final_state) where final_state has shape
                ``[..., num_heads, qk_head_dim, v_head_dim]``.

        Note:
            When the backend implementation returns a tuple, ``return_state``
            controls whether the state is included in the return value.  If
            the implementation always returns a tuple (output, state), passing
            ``return_state=False`` will discard the state silently.

            All gating parameters (``g``, ``gk``, ``gv``, ``g_gamma``) are
            optional.  When provided, they enable more sophisticated gated
            recurrent mechanisms.
        """

        if platform is not None:
            cfg = RecurrentAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d if hasattr(cfg, "block_d") else None,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        resolved_platform = detect_platform("recurrent", cfg.platform)
        impl = kernel_registry.get("recurrent", platform=resolved_platform, backend=cfg.backend)
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
            gk=gk,
            gv=gv,
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

    def heuristic_cfg(self, inv: Invocation[RecurrentAttentionConfig, Array]) -> RecurrentAttentionConfig:
        """Provide default configuration with block sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical recurrent attention workloads with stateful computation
        """
        return RecurrentAttentionConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RecurrentAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Recurrent attention performance is sensitive to state update patterns.
            Candidates are chosen to balance sequential processing efficiency.
        """
        block_configs = [
            (64, 64, 64, 4, 1),
            (128, 64, 64, 4, 2),
            (128, 128, 64, 8, 2),
        ]

        candidates = []
        for block_q, block_k, block_d, num_warps, num_stages in block_configs:
            candidates.append(
                RecurrentAttentionConfig(
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

    def candidate_cfgs_gpu(self, inv: Invocation[RecurrentAttentionConfig, Array]):
        """Generate GPU candidates for recurrent attention across Triton, TileLang and XLA.

        Recurrent attention has a small per-step state (one Q vector at
        a time); block_d (head_dim tile) is the dominant knob. On H100:

        * ``block_d`` ∈ {32, 64, 128}; 128 is best for head_dim>=128.
        * ``block_q`` / ``block_k`` mostly affect KV processing within a
          chunk; sweep {64, 128} for both.
        * ``num_warps`` ∈ {4, 8}; 8 for wide head_dim.
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
        candidates: list[RecurrentAttentionConfig] = []
        if "triton" in platforms:
            for block_q, block_k, block_d, num_warps, num_stages in block_configs:
                candidates.append(
                    RecurrentAttentionConfig(
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
                    RecurrentAttentionConfig(
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
                RecurrentAttentionConfig(
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

    def candidate_cfgs_tpu(self, inv: Invocation[RecurrentAttentionConfig, Array]):
        """Generate TPU candidates for the XLA recurrent-attention path."""
        return [
            RecurrentAttentionConfig(
                block_q=64,
                block_k=64,
                block_d=64,
                num_warps=4,
                num_stages=1,
                platform="xla",
                backend="any",
            )
        ]


_recurrent_executor: Executor[RecurrentAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("recurrent"),
    )
)


def recurrent_attention(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    reverse: bool = False,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RecurrentAttentionConfig | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """Execute recurrent attention with automatic optimization.

    Recurrent attention processes sequences with stateful computation,
    maintaining hidden states across timesteps for O(T) complexity.

    Args:
        query: Query tensor [batch, seq_len, num_heads, qk_head_dim].
        key: Key tensor [batch, seq_len, num_kv_heads, qk_head_dim].
        value: Value tensor [batch, seq_len, num_kv_heads, v_head_dim].
        g: Optional query-level gate tensor
            [batch, seq_len, num_heads, qk_head_dim].
        g_gamma: Optional per-layer decay parameter ``[..., num_heads]``.
        gk: Optional key-level gate tensor
            [batch, seq_len, num_heads, qk_head_dim].
        gv: Optional value-level gate tensor
            [batch, seq_len, num_heads, v_head_dim].
        initial_state: Optional initial hidden state
            ``[..., num_heads, qk_head_dim, v_head_dim]``.
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1] for packed
            variable-length sequences.
        softmax_scale: Scaling factor for attention scores.
        reverse: If ``True``, process sequence in reverse temporal order.
        return_state: If ``True``, return a tuple ``(output, final_state)``.
        platform: Platform override ("triton", "pallas", "cuda", "xla").
        cfg: Optional :class:`~.configs.RecurrentAttentionConfig` override.

    Returns:
        If ``return_state=False``:
            Attention output [batch, seq_len, num_heads, v_head_dim].
        If ``return_state=True``:
            Tuple of (output, final_state).

    Example:
        >>> out = recurrent_attention(query, key, value)
        >>> out = recurrent_attention(query, key, value, g=gates, gk=key_gates, gv=value_gates)
        >>> out = recurrent_attention(query, key, value, platform="xla")
    """
    return _recurrent_executor(
        RecurrentAttention(),
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )

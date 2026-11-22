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

"""RWKV-6 recurrence operation module.

This module implements the RWKV-6 linear attention recurrence mechanism, an evolution
of the RWKV architecture with multi-head support and time-varying decay. RWKV-6
combines the efficiency of linear RNNs with the expressiveness of attention.

Key features of RWKV-6:
    - Multi-head attention with per-head state matrices
    - Time-varying decay through learned w tensor
    - Per-head bonus tensor u for current-timestep weighting
    - Variable-length sequence packing support (FlashAttention-style)
    - Bidirectional processing with reverse option

The algorithm implements a linear attention recurrence:
    At each timestep t:
        kv_t = k_t^T @ v_t           (outer product, shape [K, V])
        o_t = r_t^T @ (h + kv_t * u)  (output with bonus)
        h_{t+1} = h * exp(w_t) + kv_t  (state update with decay)

    where:
        - h is the state matrix [num_heads, K, V] per batch
        - r (receptance) acts as the query
        - w controls time-varying exponential decay
        - u provides bonus weight for current timestep

Mathematical formulation:
    The recurrence computes exponentially-decayed attention:
        o_t = Σ_{i≤t} exp(Σ_{j=i+1}^{t} w_j) · (k_i^T · v_i) · r_t + u · (k_t^T · v_t) · r_t

    This is equivalent to softmax attention in the limit but computed in O(T)
    time and O(KxV) memory per head.

Supports:
    - Variable sequence lengths via cu_seqlens (cumulative sequence lengths)
    - Reverse processing for bidirectional models
    - Initial state for continuation across chunks
    - Custom softmax scaling

References:
    - RWKV-6: https://github.com/BlinkDL/RWKV-LM (RWKV-6 architecture)
    - Linear Attention Transformers: https://arxiv.org/abs/2006.16236
"""

from __future__ import annotations

import os
import typing
from typing import Literal

from jaxtyping import Array, Float, Int

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
from .configs import RWKV6Config


class RWKV6(Kernel[RWKV6Config, Array]):
    """RWKV-6 recurrence kernel wrapper.

    Implements the RWKV-6 linear attention recurrence with multi-head support
    and optional variable-length sequence packing, achieving O(T) complexity
    with O(KxV) memory per head.

    Features:
        - Multi-head attention with per-head state matrices
        - Time-varying decay via learned w tensor
        - Per-head bonus weight u for current timestep
        - FlashAttention-style variable-length packing support
        - Bidirectional processing with reverse option
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The recurrence computes:
        kv_t = k_t^T @ v_t
        o_t = r_t^T @ (h + kv_t * u)
        h_{t+1} = h * exp(w_t) + kv_t

    where h is the state matrix of shape [H, K, V] per batch element.

    Example:
        >>> from ejkernel.modules import RWKV6, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> rwkv = RWKV6()
        >>> output = executor(rwkv, r, k, v, w, u)
        >>>
        >>> # Streaming inference with state
        >>> output, state = executor(
        ...     rwkv, r, k, v, w, u,
        ...     initial_state=prev_state,
        ...     return_state=True
        ... )
        >>>
        >>> # Variable-length sequences
        >>> output = executor(rwkv, r, k, v, w, u, cu_seqlens=cu_seqs)

    Attributes:
        op_id: Operation identifier ("rwkv6").
    """

    def __init__(self) -> None:
        """Initialize RWKV-6 kernel module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="rwkv6")

    def get_impl(self, cfg: RWKV6Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for RWKV-6 recurrence

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("rwkv6", cfg.platform)
        return kernel_registry.get("rwkv6", platform=platform, backend=cfg.backend)

    def run(
        self,
        r: Float[Array, "batch seq_len num_heads qk_head_dim"],
        k: Float[Array, "batch seq_len num_heads qk_head_dim"],
        v: Float[Array, "batch seq_len num_heads v_head_dim"],
        w: Float[Array, "batch seq_len num_heads qk_head_dim"],
        u: Float[Array, "num_heads qk_head_dim"],
        *,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RWKV6Config,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute RWKV-6 linear attention recurrence.

        Computes the RWKV-6 recurrence over a sequence with multi-head
        support and time-varying decay.

        Args:
            r: Receptance (query) tensor [batch, seq_len, num_heads, qk_head_dim].
                Acts as the query for output computation.
            k: Key tensor [batch, seq_len, num_heads, qk_head_dim].
                Combined with values to form state updates.
            v: Value tensor [batch, seq_len, num_heads, v_head_dim].
                Values to be aggregated over time.
            w: Log decay tensor [batch, seq_len, num_heads, qk_head_dim].
                Time-varying exponential decay rates (typically negative).
            u: Bonus tensor [num_heads, qk_head_dim].
                Per-head bonus weight for current timestep.
            softmax_scale: Optional scale for receptance; defaults to K^(-0.5).
            initial_state: Optional initial state [batch, num_heads, K, V].
                For continuing from previous chunk. In packed mode: [N, H, K, V].
            reverse: If True, process sequence in reverse order.
            cu_seqlens: Optional cumulative sequence lengths [num_seqs + 1].
                Enables FlashAttention-style variable-length packing.
            return_state: If True, also return the final state.
            platform: Optional platform override ("triton", "pallas", "cuda", "xla").
            cfg: Kernel configuration object.

        Returns:
            If return_state=False: Output tensor [batch, seq_len, num_heads, v_head_dim].
            If return_state=True: Tuple of (output, final_state) where
                final_state is [batch, num_heads, qk_head_dim, v_head_dim] in float32.

        Note:
            The state matrix h has shape [num_heads, K, V] per batch element,
            enabling efficient O(T) sequential processing.
        """
        if platform is not None:
            cfg = RWKV6Config(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)

        impl = self.get_impl(cfg)
        out, final_state = impl(
            r=r,
            k=k,
            v=v,
            w=w,
            u=u,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV6Config, Array]) -> RWKV6Config:
        """Provide default configuration.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with auto platform selection
        """
        return RWKV6Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV6Config, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Empty list as RWKV-6 has minimal configuration options

        Note:
            RWKV-6's recurrence pattern relies primarily on platform selection
            for optimization.
        """
        return [
            RWKV6Config(platform="auto", backend="any"),
            RWKV6Config(platform="xla", backend="any"),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[RWKV6Config, Array]):
        """Generate GPU platform candidates for RWKV-6."""
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "triton", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[RWKV6Config] = []
        if "tilelang" in platforms:
            candidates.append(RWKV6Config(platform="tilelang", backend="gpu"))
        if "triton" in platforms:
            candidates.append(RWKV6Config(platform="triton", backend="gpu"))
        if "xla" in platforms:
            candidates.append(RWKV6Config(platform="xla", backend="any"))
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[RWKV6Config, Array]):
        """Return TPU candidates for the XLA RWKV-6 path."""
        return [RWKV6Config(platform="xla", backend="any")]


_executor: Executor[RWKV6Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv6"),
    )
)


def rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    /,
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RWKV6Config | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """RWKV-6 recurrence with automatic backend selection.

    Computes the RWKV-6 linear attention recurrence over a sequence with
    multi-head support and optional variable-length packing.

    Args:
        r: Receptance (query) tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        w: Log decay tensor `[B, T, H, K]`.
        u: Bonus tensor (per-head) `[H, K]`.
        softmax_scale: Optional scale for receptance; defaults to `K**-0.5`.
        initial_state: Optional initial state `[B, H, K, V]` (or `[N, H, K, V]`
            in packed mode).
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences
            (FlashAttention-style), shape `[N+1]`.
        return_state: If True, also return the final state.
        platform: Backend platform override.
        cfg: Optional configuration object.

    Returns:
        Output tensor `[B, T, H, V]` (dtype matches `v`), or tuple of
        (output, final_state) if `return_state=True`. Final state is float32.
    """
    return _executor(
        RWKV6(),
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )

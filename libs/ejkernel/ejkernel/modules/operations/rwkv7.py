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

"""RWKV-7 recurrence operation modules.

This module implements the RWKV-7 architecture, featuring a Diagonal + Low-Rank (DPLR)
state update mechanism. RWKV-7 enhances expressiveness through low-rank state
corrections while maintaining linear complexity.

Key features of RWKV-7:
    - Diagonal + Low-Rank (DPLR) state update
    - Two parameterizations: (a, b) direct and (kk, a) multiplicative
    - Multi-head attention with per-head state matrices
    - Time-varying decay through learned w tensor
    - Variable-length sequence packing support (FlashAttention-style)
    - Bidirectional processing with reverse option

The DPLR algorithm implements an enhanced state update:
    At each timestep t:
        h_t = diag(exp(w_t)) @ h_{t-1} + a_t @ (b_t^T @ h_{t-1}) + k_t @ v_t^T
        o_t = r_t^T @ h_t

    where:
        - h is the state matrix [num_heads, K, V] per batch
        - diag(exp(w_t)) provides diagonal (element-wise) decay
        - a_t @ (b_t^T @ h_{t-1}) is the low-rank correction term
        - k_t @ v_t^T is the standard outer product update

    The low-rank term enables richer state dynamics compared to RWKV-6's
    purely diagonal decay, allowing the model to selectively mix information
    across the key dimension.

Two parameterizations are provided:
    1. RWKV7 (a, b): Direct parameterization where a and b are explicit inputs
    2. RWKV7Mul (kk, a): Multiplicative form where:
        a' = kk * a
        b' = -kk
       This is often used by optimized kernel implementations.

Supports:
    - Variable sequence lengths via cu_seqlens (cumulative sequence lengths)
    - Reverse processing for bidirectional models
    - Initial state for continuation across chunks
    - Custom softmax scaling

References:
    - RWKV-7: https://github.com/BlinkDL/RWKV-LM (RWKV-7 architecture)
    - Eagle and Finch paper: https://arxiv.org/abs/2404.05892
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
from .configs import RWKV7Config, RWKV7MulConfig


class RWKV7(Kernel[RWKV7Config, Array]):
    """RWKV-7 (a,b) DPLR recurrence wrapper.

    Implements the RWKV-7 Diagonal + Low-Rank (DPLR) state update, achieving
    O(T) complexity with O(KxV) memory per head. The DPLR mechanism provides
    richer state dynamics than purely diagonal decay.

    State update equation:
        h_t = diag(exp(w_t)) @ h_{t-1} + a_t (b_t^T h_{t-1}) + k_t v_t^T
        o_t = r_t^T @ h_t

    where h is the state matrix of shape [H, K, V] per batch element.
    This is the (a, b) parameterization where a and b are explicit inputs.

    Features:
        - DPLR state update combining diagonal decay with low-rank correction
        - Multi-head attention with per-head state matrices
        - Variable-length sequence handling via cumulative lengths
        - Bidirectional processing with reverse option
        - Multiple platform support (Triton/Pallas/CUDA/XLA)
        - Automatic platform selection for optimal performance

    The low-rank term a_t (b_t^T h_{t-1}) enables selective mixing of
    information across the key dimension, providing richer dynamics than
    purely diagonal decay.

    Example:
        >>> from ejkernel.modules import RWKV7, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> rwkv = RWKV7()
        >>> output = executor(rwkv, r, w, k, v, a, b)
        >>>
        >>> # Streaming inference with state
        >>> output, state = executor(
        ...     rwkv, r, w, k, v, a, b,
        ...     initial_state=prev_state,
        ...     return_state=True
        ... )
        >>>
        >>> # Variable-length sequences
        >>> output = executor(rwkv, r, w, k, v, a, b, cu_seqlens=cu_seqs)

    Attributes:
        op_id: Operation identifier ("rwkv7").
    """

    def __init__(self) -> None:
        """Initialize RWKV-7 kernel module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="rwkv7")

    def get_impl(self, cfg: RWKV7Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for RWKV-7 DPLR recurrence

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("rwkv7", cfg.platform)
        return kernel_registry.get("rwkv7", platform=platform, backend=cfg.backend)

    def run(
        self,
        r: Float[Array, "batch seq_len num_heads qk_head_dim"],
        w: Float[Array, "batch seq_len num_heads qk_head_dim"],
        k: Float[Array, "batch seq_len num_heads qk_head_dim"],
        v: Float[Array, "batch seq_len num_heads v_head_dim"],
        a: Float[Array, "batch seq_len num_heads qk_head_dim"],
        b: Float[Array, "batch seq_len num_heads qk_head_dim"],
        *,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RWKV7Config,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute RWKV-7 DPLR recurrence with (a, b) parameterization.

        Computes the Diagonal + Low-Rank state update over a sequence,
        combining exponential decay with low-rank corrections.

        Args:
            r: Receptance (query) tensor [batch, seq_len, num_heads, qk_head_dim].
                Acts as the query for output computation.
            w: Log decay tensor [batch, seq_len, num_heads, qk_head_dim].
                Controls diagonal exponential decay.
            k: Key tensor [batch, seq_len, num_heads, qk_head_dim].
                Combined with values for outer product update.
            v: Value tensor [batch, seq_len, num_heads, v_head_dim].
                Values to be aggregated over time.
            a: Low-rank update vector [batch, seq_len, num_heads, qk_head_dim].
                First component of the low-rank correction.
            b: Low-rank projection vector [batch, seq_len, num_heads, qk_head_dim].
                Second component of the low-rank correction.
            softmax_scale: Optional scale for receptance; defaults to K^(-0.5).
            initial_state: Optional initial state [batch, num_heads, K, V].
                For continuing from previous chunk.
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
            The DPLR update combines three terms:
            1. Diagonal decay: diag(exp(w)) @ h
            2. Low-rank correction: a @ (b^T @ h)
            3. Outer product: k @ v^T
        """
        block_v = getattr(cfg, "block_v", 32)
        num_warps = getattr(cfg, "num_warps", 4)
        num_stages = getattr(cfg, "num_stages", 3)
        cfg_backend = getattr(cfg, "backend", Backend.ANY)

        if platform is not None:
            cfg = RWKV7Config(
                block_v=block_v,
                num_warps=num_warps,
                num_stages=num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )

        resolved_platform = detect_platform("rwkv7", cfg.platform)
        impl = kernel_registry.get("rwkv7", platform=resolved_platform, backend=cfg.backend)
        impl_kwargs = {}
        if resolved_platform == Platform.TRITON:
            impl_kwargs = {
                "block_v": block_v,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        out, final_state = impl(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            **impl_kwargs,
        )
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV7Config, Array]) -> RWKV7Config:
        """Provide default configuration.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with auto platform selection
        """
        return RWKV7Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV7Config, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Empty list as RWKV-7 has minimal configuration options

        Note:
            RWKV-7's recurrence pattern relies primarily on platform selection
            for optimization.
        """
        return [
            RWKV7Config(block_v=block_v, num_warps=num_warps, num_stages=3, platform="auto", backend="any")
            for block_v in (16, 32, 64)
            for num_warps in (4, 8)
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[RWKV7Config, Array]):
        """Generate GPU candidates for RWKV-7.

        ``block_v`` is the head-dim tile for the Triton fused-recurrent
        kernel. Low-rank (a, b) updates are autotuned internally; this
        sweep covers the value-dim tile plus warps/stages:

        * ``block_v`` ∈ {16, 32, 64, 128} — 16/32 for small heads,
          64/128 for typical RWKV-7 widths.
        * ``num_warps`` ∈ {4, 8}.
        * ``num_stages`` ∈ {2, 3, 4} — deeper pipelines help memory-bound
          long sequences.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[RWKV7Config] = []
        if "triton" in platforms:
            candidates.extend(
                RWKV7Config(
                    block_v=block_v,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="triton",
                    backend="gpu",
                )
                for block_v in (16, 32, 64, 128)
                for num_warps in (4, 8)
                for num_stages in (2, 3)
            )
        if "tilelang" in platforms:
            for block_v in (32, 64, 128):
                candidates.append(
                    RWKV7Config(
                        block_v=block_v,
                        num_warps=4,
                        num_stages=3,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(RWKV7Config(block_v=64, num_warps=4, num_stages=3, platform="xla", backend="any"))
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[RWKV7Config, Array]):
        """Generate TPU candidates for the XLA RWKV-7 path."""
        return [RWKV7Config(block_v=64, num_warps=4, num_stages=3, platform="xla", backend="any")]


class RWKV7Mul(Kernel[RWKV7MulConfig, Array]):
    """RWKV-7 multiplicative (kk, a) parameterization wrapper.

    A reparameterization of RWKV-7 using (kk, a) inputs instead of (a, b),
    achieving O(T) complexity with O(KxV) memory per head. Internally converts
    to the standard DPLR form via:

        a' = kk * a
        b' = -kk

    This is often used by optimized kernel implementations that benefit from
    this particular parameterization for numerical stability or efficiency.

    Features:
        - Alternative parameterization of DPLR state update
        - Automatic conversion to standard (a, b) form internally
        - Same multi-head and variable-length support as RWKV7
        - Multiple platform support (Triton/Pallas/CUDA/XLA)
        - Automatic platform selection for optimal performance

    The multiplicative form can be more numerically stable in certain
    scenarios and is the preferred interface for some kernel implementations.

    Example:
        >>> from ejkernel.modules import RWKV7Mul, create_default_executor
        >>>
        >>> # Basic usage with multiplicative parameterization
        >>> executor = create_default_executor()
        >>> rwkv = RWKV7Mul()
        >>> output = executor(rwkv, r, w, k, v, kk, a)
        >>>
        >>> # Streaming inference with state
        >>> output, state = executor(
        ...     rwkv, r, w, k, v, kk, a,
        ...     initial_state=prev_state,
        ...     return_state=True
        ... )

    Attributes:
        op_id: Operation identifier ("rwkv7_mul").
    """

    def __init__(self) -> None:
        """Initialize RWKV-7 Mul kernel module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="rwkv7_mul")

    def get_impl(self, cfg: RWKV7MulConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for RWKV-7 Mul recurrence

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("rwkv7_mul", cfg.platform)
        return kernel_registry.get("rwkv7_mul", platform=platform, backend=cfg.backend)

    def run(
        self,
        r: Float[Array, "batch seq_len num_heads qk_head_dim"],
        w: Float[Array, "batch seq_len num_heads qk_head_dim"],
        k: Float[Array, "batch seq_len num_heads qk_head_dim"],
        v: Float[Array, "batch seq_len num_heads v_head_dim"],
        kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
        a: Float[Array, "batch seq_len num_heads qk_head_dim"],
        *,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RWKV7MulConfig,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        """Execute RWKV-7 DPLR recurrence with (kk, a) multiplicative parameterization.

        Computes the Diagonal + Low-Rank state update using the alternative
        multiplicative parameterization, converting internally to (a', b') form.

        Args:
            r: Receptance (query) tensor [batch, seq_len, num_heads, qk_head_dim].
                Acts as the query for output computation.
            w: Log decay tensor [batch, seq_len, num_heads, qk_head_dim].
                Controls diagonal exponential decay.
            k: Key tensor [batch, seq_len, num_heads, qk_head_dim].
                Combined with values for outer product update.
            v: Value tensor [batch, seq_len, num_heads, v_head_dim].
                Values to be aggregated over time.
            kk: Multiplicative factor [batch, seq_len, num_heads, qk_head_dim].
                Used to compute a' = kk * a and b' = -kk.
            a: Low-rank update base [batch, seq_len, num_heads, qk_head_dim].
                Scaled by kk to get the actual low-rank update vector.
            softmax_scale: Optional scale for receptance; defaults to K^(-0.5).
            initial_state: Optional initial state [batch, num_heads, K, V].
                For continuing from previous chunk.
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
            The conversion from (kk, a) to (a', b') is:
                a' = kk * a
                b' = -kk
            This parameterization can be more numerically stable for certain
            model configurations.
        """
        block_v = getattr(cfg, "block_v", 32)
        num_warps = getattr(cfg, "num_warps", 4)
        num_stages = getattr(cfg, "num_stages", 3)
        cfg_backend = getattr(cfg, "backend", Backend.ANY)

        if platform is not None:
            cfg = RWKV7MulConfig(
                block_v=block_v,
                num_warps=num_warps,
                num_stages=num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )

        resolved_platform = detect_platform("rwkv7_mul", cfg.platform)
        impl = kernel_registry.get("rwkv7_mul", platform=resolved_platform, backend=cfg.backend)
        impl_kwargs = {}
        if resolved_platform == Platform.TRITON:
            impl_kwargs = {
                "block_v": block_v,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        out, final_state = impl(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            **impl_kwargs,
        )
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV7MulConfig, Array]) -> RWKV7MulConfig:
        """Provide default configuration.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with auto platform selection
        """
        return RWKV7MulConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV7MulConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Empty list as RWKV-7 Mul has minimal configuration options

        Note:
            RWKV-7 Mul's recurrence pattern relies primarily on platform
            selection for optimization.
        """
        return [
            RWKV7MulConfig(block_v=block_v, num_warps=num_warps, num_stages=3, platform="auto", backend="any")
            for block_v in (16, 32, 64)
            for num_warps in (4, 8)
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[RWKV7MulConfig, Array]):
        """Generate GPU candidates for RWKV-7 Mul (same tile space as RWKV-7)."""
        requested = inv.kwargs.get("platform", None)
        platforms = ("triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[RWKV7MulConfig] = []
        if "triton" in platforms:
            candidates.extend(
                RWKV7MulConfig(
                    block_v=block_v,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="triton",
                    backend="gpu",
                )
                for block_v in (16, 32, 64, 128)
                for num_warps in (4, 8)
                for num_stages in (2, 3)
            )
        if "tilelang" in platforms:
            for block_v in (32, 64, 128):
                candidates.append(
                    RWKV7MulConfig(
                        block_v=block_v,
                        num_warps=4,
                        num_stages=3,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(RWKV7MulConfig(block_v=64, num_warps=4, num_stages=3, platform="xla", backend="any"))
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[RWKV7MulConfig, Array]):
        """Generate TPU candidates for the XLA RWKV-7 Mul path."""
        return [RWKV7MulConfig(block_v=64, num_warps=4, num_stages=3, platform="xla", backend="any")]


_executor_rwkv7: Executor[RWKV7Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv7"),
    )
)

_executor_rwkv7_mul: Executor[RWKV7MulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv7_mul"),
    )
)


def rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    /,
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RWKV7Config | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """RWKV-7 DPLR recurrence (a,b) with automatic backend selection.

    Computes the RWKV-7 Diagonal + Low-Rank recurrence over a sequence.

    Args:
        r: Receptance (query) tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        a: Low-rank update vector `[B, T, H, K]`.
        b: Low-rank projection vector `[B, T, H, K]`.
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
    return _executor_rwkv7(
        RWKV7(),
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )


def rwkv7_mul(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    /,
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RWKV7MulConfig | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """RWKV-7 recurrence (kk,a) multiplicative parameterization.

    Alternative parameterization of RWKV-7 that converts internally:
        a' = kk * a
        b' = -kk

    This form is commonly used by optimized kernel implementations.

    Args:
        r: Receptance (query) tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        kk: Multiplicative factor `[B, T, H, K]`.
        a: Low-rank update base `[B, T, H, K]`.
        softmax_scale: Optional scale for receptance; defaults to `K**-0.5`.
        initial_state: Optional initial state `[B, H, K, V]` (or `[N, H, K, V]`
            in packed mode).
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
        return_state: If True, also return the final state.
        platform: Backend platform override.
        cfg: Optional configuration object.

    Returns:
        Output tensor `[B, T, H, V]` (dtype matches `v`), or tuple of
        (output, final_state) if `return_state=True`. Final state is float32.
    """
    return _executor_rwkv7_mul(
        RWKV7Mul(),
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )

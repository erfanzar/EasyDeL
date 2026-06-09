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


"""SSM2 (Mamba2-style) Selective State Space operation module.

This module provides the StateSpaceV2 operation, implementing the Mamba2
selective state space model architecture used by Mamba2 and FalconH1.

Key characteristics of SSM2:
- 1D A vector: [num_heads] (per-head scalar)
- SSM state shape: [batch, num_heads, head_dim, ssm_state_size]
- B, C with n_groups grouping
- Output gating via gated RMSNorm or simple multiplication

The algorithm:
    Discretization:
        dA = exp(A * dt)  where A is per-head
        dB = dt * B

    Recurrence (per head):
        dBx = dt * B * x (outer product form)
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D

Features:
    - O(N) complexity through sequential processing
    - Per-head scalar decay (1D A vector)
    - n_groups support for B, C grouping
    - Gated RMSNorm output normalization
    - Conv state passthrough for caching

References:
    - Mamba2: https://arxiv.org/abs/2405.21060
    - FalconH1: https://huggingface.co/tiiuae/Falcon-H1-1B-Base
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable
from typing import Literal

import jax
from jax import lax
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
from .configs import StateSpaceV2Config


class StateSpaceV2(Kernel[StateSpaceV2Config, Array]):
    """SSM2 (Mamba2-style) Selective State Space operation.

    Implements the Mamba2 architecture with O(N) complexity, where N is the
    sequence length. Processes tokens sequentially with per-head scalar decay
    for improved efficiency and expressiveness.

    Features:
        - 1D A vector [num_heads] (per-head scalar)
        - n_groups for B, C grouping (broadcast to num_heads)
        - Hidden state shape [batch, num_heads, head_dim, ssm_state_size]
        - Gated RMSNorm output normalization option
        - Conv state passthrough for caching during inference
        - Multiple platform support (XLA primary)
        - Automatic platform selection for optimal performance

    The state update mechanism:
        dA = exp(A * dt)  where A is per-head scalar
        dBx = dt * B * x (outer product form)
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D

    Example:
        >>> from ejkernel.modules import StateSpaceV2, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> ssm = StateSpaceV2()
        >>> output, state, _ = executor(ssm, x, A, B, C, D, dt, n_groups=1)
        >>>
        >>> # With gated RMSNorm
        >>> output, state, _ = executor(
        ...     ssm, x, A, B, C, D, dt,
        ...     gate=gate, n_groups=1,
        ...     use_gated_rmsnorm=True, act_fn=jax.nn.silu
        ... )
        >>>
        >>> # Streaming inference with state continuation
        >>> output, state, conv_state = executor(
        ...     ssm, x[:, :1], A, B[:, :1], C[:, :1], D, dt[:, :1],
        ...     initial_state=prev_state, conv_state=prev_conv_state,
        ...     n_groups=1
        ... )
    """

    def __init__(self):
        """Initialize StateSpaceV2 module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="state_space_v2")

    def get_impl(self, cfg: StateSpaceV2Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for SSM2

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("state_space_v2", cfg.platform)
        return kernel_registry.get("state_space_v2", platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "batch seq_len num_heads head_dim"],
        A: Float[Array, "num_heads"],
        B: Float[Array, "batch seq_len n_groups ssm_state_size"],
        C: Float[Array, "batch seq_len n_groups ssm_state_size"],
        D: Float[Array, "num_heads"],
        dt: Float[Array, "batch seq_len num_heads"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
        conv_state: Float[Array, "batch conv_dim d_conv"] | None = None,
        n_groups: int = 1,
        act_fn: Callable[[jax.Array], jax.Array] | None = None,
        use_gated_rmsnorm: bool = False,
        rmsnorm_eps: float = 1e-5,
        precision: lax.Precision | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: StateSpaceV2Config,
    ) -> tuple[
        Float[Array, "batch seq_len intermediate_size"],
        Float[Array, "batch num_heads head_dim ssm_state_size"],
        Float[Array, "batch conv_dim d_conv"] | None,
    ]:
        """Execute SSM2 selective state space operation.

        Args:
            x: Input tensor
                Shape: [batch, seq_len, num_heads, head_dim]
            A: A vector in real form (typically negative for stability)
                Shape: [num_heads]
            B: B parameter (with n_groups grouping)
                Shape: [batch, seq_len, n_groups, ssm_state_size]
            C: C parameter (with n_groups grouping)
                Shape: [batch, seq_len, n_groups, ssm_state_size]
            D: Skip connection parameter
                Shape: [num_heads]
            dt: Time step after softplus activation
                Shape: [batch, seq_len, num_heads]
            gate: Optional gating tensor for output modulation
                Shape: [batch, seq_len, intermediate_size]
            initial_state: Optional initial SSM state for continuation
                Shape: [batch, num_heads, head_dim, ssm_state_size]
            conv_state: Optional convolution state for caching (passed through)
                Shape: [batch, conv_dim, d_conv]
            n_groups: Number of groups for B, C (B/C are repeated to num_heads)
            act_fn: Optional activation function for gating (e.g., jax.nn.silu)
            use_gated_rmsnorm: If True, apply RMSNorm before gating
            rmsnorm_eps: Epsilon for RMSNorm stability
            precision: JAX precision for matmul operations
            platform: Optional platform override
            cfg: Kernel configuration object

        Returns:
            Tuple of:
                - output: SSM output [batch, seq_len, intermediate_size]
                - ssm_state: Final hidden state [batch, num_heads, head_dim, ssm_state_size]
                - conv_state: Passed through conv_state (for caching)
        """
        cfg_block_e = int(getattr(cfg, "block_e", 128))
        cfg_backend = getattr(cfg, "backend", "any")

        if platform is not None:
            cfg = StateSpaceV2Config(
                n_groups=n_groups,
                use_gated_rmsnorm=use_gated_rmsnorm,
                rmsnorm_eps=rmsnorm_eps,
                block_e=cfg_block_e,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
            cfg_block_e = cfg.block_e

        impl = self.get_impl(cfg)
        return impl(
            x=x,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            gate=gate,
            initial_state=initial_state,
            conv_state=conv_state,
            n_groups=n_groups,
            act_fn=act_fn,
            use_gated_rmsnorm=use_gated_rmsnorm,
            rmsnorm_eps=rmsnorm_eps,
            precision=precision,
            block_e=cfg_block_e,
        )

    def heuristic_cfg(self, inv: Invocation[StateSpaceV2Config, Array]) -> StateSpaceV2Config:
        """Provide default configuration.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration for SSM2 operation
        """
        return StateSpaceV2Config(
            n_groups=int(inv.kwargs.get("n_groups", 1)),
            use_gated_rmsnorm=bool(inv.kwargs.get("use_gated_rmsnorm", False)),
            rmsnorm_eps=float(inv.kwargs.get("rmsnorm_eps", 1e-5)),
            block_e=128,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[StateSpaceV2Config, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            SSM2 uses XLA implementation primarily, so candidates are minimal.
        """
        return [
            self.heuristic_cfg(inv),
            StateSpaceV2Config(
                n_groups=int(inv.kwargs.get("n_groups", 1)),
                use_gated_rmsnorm=bool(inv.kwargs.get("use_gated_rmsnorm", False)),
                rmsnorm_eps=float(inv.kwargs.get("rmsnorm_eps", 1e-5)),
                block_e=128,
                platform="xla",
                backend="any",
            ),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[StateSpaceV2Config, Array]):
        """Generate GPU candidates for TileLang and XLA SSM2.

        Tunable knob: ``block_e`` — tile size along the head_dim axis
        for the silu_gate / rmsnorm_silu_gate helper applied after the
        scan. Sweep {64, 128, 256}; 64 helps small heads (head_dim<=64),
        256 amortises fixed cost on wide heads (>=128).
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        n_groups = int(inv.kwargs.get("n_groups", 1))
        use_gated_rmsnorm = bool(inv.kwargs.get("use_gated_rmsnorm", False))
        rmsnorm_eps = float(inv.kwargs.get("rmsnorm_eps", 1e-5))
        candidates: list[StateSpaceV2Config] = []
        if "tilelang" in platforms:
            for be in (64, 128, 256):
                candidates.append(
                    StateSpaceV2Config(
                        n_groups=n_groups,
                        use_gated_rmsnorm=use_gated_rmsnorm,
                        rmsnorm_eps=rmsnorm_eps,
                        block_e=be,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            candidates.append(
                StateSpaceV2Config(
                    n_groups=n_groups,
                    use_gated_rmsnorm=use_gated_rmsnorm,
                    rmsnorm_eps=rmsnorm_eps,
                    block_e=128,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[StateSpaceV2Config, Array]):
        """Generate TPU candidates for the XLA SSM2 path."""
        return [
            StateSpaceV2Config(
                n_groups=int(inv.kwargs.get("n_groups", 1)),
                use_gated_rmsnorm=bool(inv.kwargs.get("use_gated_rmsnorm", False)),
                rmsnorm_eps=float(inv.kwargs.get("rmsnorm_eps", 1e-5)),
                block_e=128,
                platform="xla",
                backend="any",
            )
        ]


_state_space_v2_executor: Executor[StateSpaceV2Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("state_space_v2"),
    )
)


def state_space_v2(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len n_groups ssm_state_size"],
    C: Float[Array, "batch seq_len n_groups ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    /,
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch conv_dim d_conv"] | None = None,
    *,
    n_groups: int = 1,
    act_fn: Callable[[jax.Array], jax.Array] | None = None,
    use_gated_rmsnorm: bool = False,
    rmsnorm_eps: float = 1e-5,
    precision: lax.Precision | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: StateSpaceV2Config | None = None,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
    Float[Array, "batch conv_dim d_conv"] | None,
]:
    """Execute SSM2 (Mamba2-style) selective state space with automatic optimization.

    SSM2 processes sequences with stateful computation using per-head scalar decay,
    maintaining hidden states across timesteps for O(N) complexity.

    Args:
        x: Input tensor
            Shape: [batch, seq_len, num_heads, head_dim]
        A: A vector in real form (typically negative for stability)
            Shape: [num_heads]
        B: B parameter (with n_groups grouping)
            Shape: [batch, seq_len, n_groups, ssm_state_size]
        C: C parameter (with n_groups grouping)
            Shape: [batch, seq_len, n_groups, ssm_state_size]
        D: Skip connection parameter
            Shape: [num_heads]
        dt: Time step after softplus activation
            Shape: [batch, seq_len, num_heads]
        gate: Optional gating tensor for output modulation
            Shape: [batch, seq_len, intermediate_size]
        initial_state: Optional initial SSM state for continuation
            Shape: [batch, num_heads, head_dim, ssm_state_size]
        conv_state: Optional convolution state for caching (passed through)
            Shape: [batch, conv_dim, d_conv]
        n_groups: Number of groups for B, C (B/C are repeated to num_heads)
        act_fn: Optional activation function for gating (e.g., jax.nn.silu).
            If gate is provided but act_fn is None, defaults to jax.nn.silu.
        use_gated_rmsnorm: If True, apply RMSNorm before gating
        rmsnorm_eps: Epsilon for RMSNorm stability
        precision: JAX precision for matmul operations
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional kernel configuration

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, intermediate_size]
            - ssm_state: Final hidden state [batch, num_heads, head_dim, ssm_state_size]
            - conv_state: Passed through conv_state (for caching)

    Example:
        >>> # Basic usage
        >>> output, ssm_state, _ = state_space_v2(x, A, B, C, D, dt, n_groups=1)
        >>>
        >>> # With gated RMSNorm
        >>> output, ssm_state, _ = state_space_v2(
        ...     x, A, B, C, D, dt,
        ...     gate=gate, n_groups=1,
        ...     use_gated_rmsnorm=True, act_fn=jax.nn.silu,
        ... )
        >>>
        >>> # Inference with cached state
        >>> output, new_state, conv_state = state_space_v2(
        ...     x[:, :1, :, :],
        ...     A, B[:, :1, :, :], C[:, :1, :, :], D, dt[:, :1, :],
        ...     initial_state=ssm_state, conv_state=conv_state,
        ...     n_groups=1,
        ... )
    """
    return _state_space_v2_executor(
        StateSpaceV2(),
        x=x,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        gate=gate,
        initial_state=initial_state,
        conv_state=conv_state,
        n_groups=n_groups,
        act_fn=act_fn,
        use_gated_rmsnorm=use_gated_rmsnorm,
        rmsnorm_eps=rmsnorm_eps,
        precision=precision,
        platform=platform,
        _cfg=cfg,
    )

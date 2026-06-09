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


"""SSM1 (Mamba1-style) Selective State Space operation module.

This module provides the StateSpaceV1 operation, implementing the original Mamba
selective state space model architecture used by Mamba and FalconMamba.

Key characteristics of SSM1:
- 2D A matrix: [intermediate_size, ssm_state_size]
- SSM state shape: [batch, intermediate_size, ssm_state_size]
- Separate dt_proj projection for time step
- Output gating: y * activation(gate)

The algorithm:
    Discretization:
        dA = exp(A * dt)
        dB = dt * B

    Recurrence:
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t

Features:
    - O(N) complexity through sequential processing
    - Stateful computation with hidden state propagation
    - Optional gating with configurable activation
    - Conv state passthrough for caching

References:
    - Mamba: https://arxiv.org/abs/2312.00752
    - FalconMamba: https://huggingface.co/tiiuae/falcon-mamba-7b
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable
from typing import Literal

import jax
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
from .configs import StateSpaceV1Config


class StateSpaceV1(Kernel[StateSpaceV1Config, Array]):
    """SSM1 (Mamba1-style) Selective State Space operation.

    Implements the original Mamba architecture with O(N) complexity, where N is
    the sequence length. Processes tokens sequentially, maintaining a hidden state
    that accumulates information through discretized state transitions.

    Features:
        - 2D A matrix [intermediate_size, ssm_state_size]
        - Hidden state propagation across timesteps
        - Optional gating with configurable activation function
        - Conv state passthrough for caching during inference
        - Multiple platform support (XLA primary)
        - Automatic platform selection for optimal performance

    The state update mechanism:
        dA = exp(A * dt)
        dB = dt * B
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t

    Example:
        >>> from ejkernel.modules import StateSpaceV1, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> ssm = StateSpaceV1()
        >>> output, state, _ = executor(ssm, hidden_states, A, B, C, D, dt)
        >>>
        >>> # With gating
        >>> output, state, _ = executor(
        ...     ssm, hidden_states, A, B, C, D, dt,
        ...     gate=gate, act_fn=jax.nn.silu
        ... )
        >>>
        >>> # Streaming inference with state continuation
        >>> output, state, conv_state = executor(
        ...     ssm, hidden_states[:, :1], A, B[:, :1], C[:, :1], D, dt[:, :1],
        ...     initial_state=prev_state, conv_state=prev_conv_state
        ... )
    """

    def __init__(self):
        """Initialize StateSpaceV1 module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="state_space_v1")

    def get_impl(self, cfg: StateSpaceV1Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for SSM1

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("state_space_v1", cfg.platform)
        return kernel_registry.get("state_space_v1", platform=platform, backend=cfg.backend)

    def run(
        self,
        hidden_states: Float[Array, "batch seq_len intermediate_size"],
        A: Float[Array, "intermediate_size ssm_state_size"],
        B: Float[Array, "batch seq_len ssm_state_size"],
        C: Float[Array, "batch seq_len ssm_state_size"],
        D: Float[Array, "intermediate_size"],
        dt: Float[Array, "batch seq_len intermediate_size"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
        conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
        act_fn: Callable[[jax.Array], jax.Array] | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: StateSpaceV1Config,
    ) -> tuple[
        Float[Array, "batch seq_len intermediate_size"],
        Float[Array, "batch intermediate_size ssm_state_size"],
        Float[Array, "batch intermediate_size d_conv"] | None,
    ]:
        """Execute SSM1 selective state space operation.

        Args:
            hidden_states: Input tensor after convolution and activation
                Shape: [batch, seq_len, intermediate_size]
            A: A matrix in real form (typically negative for stability)
                Shape: [intermediate_size, ssm_state_size]
            B: B parameter from input projection
                Shape: [batch, seq_len, ssm_state_size]
            C: C parameter from input projection
                Shape: [batch, seq_len, ssm_state_size]
            D: Skip connection parameter
                Shape: [intermediate_size]
            dt: Time step after softplus activation
                Shape: [batch, seq_len, intermediate_size]
            gate: Optional gating tensor for output modulation
                Shape: [batch, seq_len, intermediate_size]
            initial_state: Optional initial SSM state for continuation
                Shape: [batch, intermediate_size, ssm_state_size]
            conv_state: Optional convolution state for caching (passed through)
                Shape: [batch, intermediate_size, d_conv]
            act_fn: Optional activation function for gating (e.g., jax.nn.silu)
            platform: Optional platform override
            cfg: Kernel configuration object

        Returns:
            Tuple of:
                - output: SSM output [batch, seq_len, intermediate_size]
                - ssm_state: Final hidden state [batch, intermediate_size, ssm_state_size]
                - conv_state: Passed through conv_state (for caching)
        """
        cfg_block_d = int(
            getattr(cfg, "block_d", self._heuristic_block_d(int(hidden_states.shape[-1]), int(A.shape[-1])))
        )
        cfg_block_e = int(getattr(cfg, "block_e", 128))
        cfg_backend = getattr(cfg, "backend", "any")

        if platform is not None:
            cfg = StateSpaceV1Config(
                block_d=cfg_block_d,
                block_e=cfg_block_e,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
            cfg_block_d = cfg.block_d
            cfg_block_e = cfg.block_e

        impl = self.get_impl(cfg)
        return impl(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            gate=gate,
            initial_state=initial_state,
            conv_state=conv_state,
            act_fn=act_fn,
            block_d=cfg_block_d,
            block_e=cfg_block_e,
        )

    @staticmethod
    def _DN_from_inv(inv: Invocation[StateSpaceV1Config, Array]) -> tuple[int, int]:
        """Pull ``(intermediate_size, ssm_state_size)`` from invocation."""
        hidden = inv.kwargs.get("hidden_states")
        if hidden is None and inv.args:
            hidden = inv.args[0]
        A = inv.kwargs.get("A")
        if A is None and len(inv.args) >= 2:
            A = inv.args[1]
        D_size = int(hidden.shape[-1]) if getattr(hidden, "shape", None) else 0
        N_size = int(A.shape[-1]) if getattr(A, "shape", None) else 0
        return D_size, N_size

    @staticmethod
    def _heuristic_block_d(D: int, N: int) -> int:
        """Operation-side tile heuristic for the SSM-1 scan kernel.

        Mirrors the historical kernel-side ladder verbatim.
        """
        if D == 0:
            return 64
        if D <= 64:
            return D
        if N <= 16:
            return 64
        return 32

    def heuristic_cfg(self, inv: Invocation[StateSpaceV1Config, Array]) -> StateSpaceV1Config:
        """Cold-start configuration with shape-aware ``block_d``."""
        D, N = self._DN_from_inv(inv)
        return StateSpaceV1Config(
            block_d=self._heuristic_block_d(D, N),
            block_e=128,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[StateSpaceV1Config, Array]):
        """Generate candidate configurations for autotuning."""
        D, N = self._DN_from_inv(inv)
        return [
            self.heuristic_cfg(inv),
            StateSpaceV1Config(block_d=self._heuristic_block_d(D, N), block_e=128, platform="xla", backend="any"),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[StateSpaceV1Config, Array]):
        """Generate GPU candidates for TileLang and XLA SSM1."""
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        D, N = self._DN_from_inv(inv)
        candidates: list[StateSpaceV1Config] = []
        if "tilelang" in platforms:
            seen: set[int] = set()
            for bd in (self._heuristic_block_d(D, N), 32, 64, min(D, 128) if D > 0 else 128):
                if bd <= 0 or bd in seen:
                    continue
                seen.add(bd)
                candidates.append(StateSpaceV1Config(block_d=bd, block_e=128, platform="tilelang", backend="gpu"))
        if "xla" in platforms:
            candidates.append(StateSpaceV1Config(block_d=64, block_e=128, platform="xla", backend="any"))
        return candidates or [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[StateSpaceV1Config, Array]):
        """Generate TPU candidates for the XLA SSM1 path."""
        return [StateSpaceV1Config(block_d=64, block_e=128, platform="xla", backend="any")]


_state_space_v1_executor: Executor[StateSpaceV1Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("state_space_v1"),
    )
)


def state_space_v1(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    /,
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
    *,
    act_fn: Callable[[jax.Array], jax.Array] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: StateSpaceV1Config | None = None,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
    Float[Array, "batch intermediate_size d_conv"] | None,
]:
    """Execute SSM1 (Mamba1-style) selective state space with automatic optimization.

    SSM1 processes sequences with stateful computation, maintaining hidden states
    across timesteps for O(N) complexity selective state space modeling.

    Args:
        hidden_states: Input tensor after convolution and activation
            Shape: [batch, seq_len, intermediate_size]
        A: A matrix in real form (typically negative for stability)
            Shape: [intermediate_size, ssm_state_size]
        B: B parameter from input projection
            Shape: [batch, seq_len, ssm_state_size]
        C: C parameter from input projection
            Shape: [batch, seq_len, ssm_state_size]
        D: Skip connection parameter
            Shape: [intermediate_size]
        dt: Time step after softplus activation
            Shape: [batch, seq_len, intermediate_size]
        gate: Optional gating tensor for output modulation
            Shape: [batch, seq_len, intermediate_size]
        initial_state: Optional initial SSM state for continuation
            Shape: [batch, intermediate_size, ssm_state_size]
        conv_state: Optional convolution state for caching (passed through)
            Shape: [batch, intermediate_size, d_conv]
        act_fn: Optional activation function for gating (e.g., jax.nn.silu).
            If gate is provided but act_fn is None, defaults to jax.nn.silu.
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional kernel configuration

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, intermediate_size]
            - ssm_state: Final hidden state [batch, intermediate_size, ssm_state_size]
            - conv_state: Passed through conv_state (for caching)

    Example:
        >>> # Basic usage
        >>> output, ssm_state, _ = state_space_v1(hidden_states, A, B, C, D, dt)
        >>>
        >>> # With gating
        >>> output, ssm_state, _ = state_space_v1(
        ...     hidden_states, A, B, C, D, dt,
        ...     gate=gate, act_fn=jax.nn.silu,
        ... )
        >>>
        >>> # Inference with cached state
        >>> output, new_state, conv_state = state_space_v1(
        ...     hidden_states[:, :1, :],
        ...     A, B[:, :1, :], C[:, :1, :], D, dt[:, :1, :],
        ...     initial_state=ssm_state, conv_state=conv_state,
        ... )
    """
    return _state_space_v1_executor(
        StateSpaceV1(),
        hidden_states=hidden_states,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        gate=gate,
        initial_state=initial_state,
        conv_state=conv_state,
        act_fn=act_fn,
        platform=platform,
        _cfg=cfg,
    )

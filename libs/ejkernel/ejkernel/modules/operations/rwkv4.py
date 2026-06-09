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

"""RWKV-4 time-mix recurrence operation module.

This module implements the RWKV-4 time-mixing recurrence mechanism, a core component
of the RWKV (Receptance Weighted Key Value) architecture. RWKV-4 achieves efficient
sequential processing with linear complexity while maintaining expressive power.

Key features of RWKV-4:
    - Numerically stable recurrence using log-space computations
    - Single-channel architecture (no multi-head attention)
    - Three-component state (alpha, beta, eps) for numerical stability
    - Time decay controlled by learned parameter w
    - Efficient sequential processing suitable for inference

The algorithm computes a numerically-stable WKV (Weighted Key Value) operation:
    State tracking:
        alpha_t: Accumulated weighted values (in log space)
        beta_t: Accumulated weights (in log space)
        eps_t: Maximum log-weight seen so far (for numerical stability)

    Recurrence at each timestep:
        tau = max(u + k_t, eps_{t-1})
        wkv_t = (exp(eps_{t-1} - tau) * alpha_{t-1} + exp(u + k_t - tau) * v_t) /
                (exp(eps_{t-1} - tau) * beta_{t-1} + exp(u + k_t - tau))
        alpha_t = exp(w + eps_{t-1} - eps_t) * alpha_{t-1} + exp(w + k_t - eps_t) * v_t
        beta_t = exp(w + eps_{t-1} - eps_t) * beta_{t-1} + exp(w + k_t - eps_t)
        eps_t = max(w + eps_{t-1}, w + k_t)

Mathematical formulation:
    The time-mix mechanism computes weighted attention over past tokens:
        wkv_t = Σ_{i<t} exp(w(t-i-1) + k_i + u·δ_{i,t-1}) · v_i /
                Σ_{i<t} exp(w(t-i-1) + k_i + u·δ_{i,t-1})

    where:
        - w < 0 controls exponential decay (learned per channel)
        - u is the time-mix bias (learned per channel)
        - k, v are the key and value from input projections

References:
    - RWKV: Reinventing RNNs for the Transformer Era
      (Peng et al., 2023) https://arxiv.org/abs/2305.13048
    - Flash-Linear-Attention implementation
      https://github.com/sustcsonglin/flash-linear-attention
"""

from __future__ import annotations

import os
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
from .configs import RWKV4Config


class RWKV4(Kernel[RWKV4Config, Array]):
    """RWKV-4 time-mix recurrence kernel wrapper.

    Implements the numerically-stable RWKV-4 recurrence as described in
    Flash-Linear-Attention's fused recurrent formulation. The state tracks
    three components (alpha, beta, eps) for numerical stability, achieving
    O(N) complexity with O(C) memory where C is the channel dimension.

    Features:
        - Numerically stable log-space computation
        - Three-component state tracking (alpha, beta, eps)
        - Support for chunked processing via state continuation
        - Multiple platform support (Triton/Pallas/CUDA/XLA)
        - Automatic platform selection for optimal performance

    The recurrence computes:
        tau = max(u + k_t, eps)
        wkv_t = (exp(eps - tau) * alpha + exp(u + k_t - tau) * v_t) /
                (exp(eps - tau) * beta + exp(u + k_t - tau))

    Example:
        >>> from ejkernel.modules import RWKV4, create_default_executor
        >>>
        >>> # Basic usage
        >>> executor = create_default_executor()
        >>> rwkv = RWKV4()
        >>> output = executor(rwkv, w, u, k, v)
        >>>
        >>> # Chunked processing with state
        >>> output, state = executor(rwkv, w, u, k, v, return_state=True)
        >>> output2, state = executor(rwkv, w, u, k2, v2, state=state, return_state=True)

    Attributes:
        op_id: Operation identifier ("rwkv4").
    """

    def __init__(self) -> None:
        """Initialize RWKV-4 kernel module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="rwkv4")

    def get_impl(self, cfg: RWKV4Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for RWKV-4 recurrence

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("rwkv4", cfg.platform)
        return kernel_registry.get("rwkv4", platform=platform, backend=cfg.backend)

    def run(
        self,
        w: Float[Array, "chans"],
        u: Float[Array, "chans"],
        k: Float[Array, "batch seq_len chans"],
        v: Float[Array, "batch seq_len chans"],
        state: Float[Array, "batch three chans"] | None = None,
        *,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RWKV4Config,
    ) -> (
        Float[Array, "batch seq_len chans"]
        | tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]
    ):
        """Execute RWKV-4 time-mix recurrence.

        Computes the numerically-stable WKV operation over a sequence,
        maintaining a three-component state for numerical stability.

        Args:
            w: Time-decay parameter in log space [channels].
                Controls exponential decay rate. Internally uses -exp(w).
            u: Time-mix bias [channels].
                Bonus weight for the current timestep's key-value pair.
            k: Key tensor [batch, seq_len, channels].
                Log-space keys from input projection.
            v: Value tensor [batch, seq_len, channels].
                Values to be aggregated.
            state: Optional initial state [batch, 3, channels].
                Contains (alpha, beta, eps) for continuing from previous chunk.
                If None, initializes with alpha=0, beta=0, eps=-1e30.
            return_state: If True, also return the final state.
            platform: Optional platform override ("triton", "pallas", "cuda", "xla").
            cfg: Kernel configuration object.

        Returns:
            If return_state=False: Output tensor [batch, seq_len, channels].
            If return_state=True: Tuple of (output, final_state) where
                final_state is [batch, 3, channels] containing (alpha, beta, eps).

        Note:
            RWKV-4 operates on single channels without multi-head structure.
            The state enables efficient chunked processing for long sequences.
        """
        if platform is not None:
            cfg = RWKV4Config(
                block_c=cfg.block_c,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        resolved = detect_platform("rwkv4", cfg.platform)
        if resolved == "tilelang":
            out, final_state = impl(w=w, u=u, k=k, v=v, state=state, block_c=int(cfg.block_c))
        else:
            out, final_state = impl(w=w, u=u, k=k, v=v, state=state)
        if return_state:
            return out, final_state
        return out

    @staticmethod
    def _channels_from_inv(inv: Invocation[RWKV4Config, Array]) -> int:
        """Pull channel count ``C`` from the invocation's ``k`` tensor."""
        k = inv.kwargs.get("k")
        if k is None and len(inv.args) >= 3:
            k = inv.args[2]
        shape = getattr(k, "shape", None)
        if shape is None or len(shape) < 1:
            return 0
        return int(shape[-1])

    @staticmethod
    def _heuristic_block_c(channels: int) -> int:
        """Operation-side tile heuristic — single source of truth.

        Mirrors the historical kernel-side ladder.
        """
        if channels == 0 or channels <= 64:
            return 64
        return 128

    def heuristic_cfg(self, inv: Invocation[RWKV4Config, Array]) -> RWKV4Config:
        """Cold-start configuration with shape-aware ``block_c``."""
        return RWKV4Config(
            block_c=self._heuristic_block_c(self._channels_from_inv(inv)),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RWKV4Config, Array]):
        """Generate candidate configurations for autotuning."""
        return [
            self.heuristic_cfg(inv),
            RWKV4Config(block_c=self._heuristic_block_c(self._channels_from_inv(inv)), platform="xla", backend="any"),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[RWKV4Config, Array]):
        """Generate GPU platform + block_c candidates for RWKV-4.

        The RWKV-4 fused-recurrence kernel parallelises over the channel
        axis ``C``. Each CTA processes a ``block_c``-wide slab. On H100:

        * ``block_c=32`` only useful for tiny ``C`` (<=64) — high CTA count.
        * ``block_c=64`` good for moderate ``C`` (128–256); balances
          occupancy with register pressure.
        * ``block_c=128`` best for wide-channel models (>=512); amortises
          the per-CTA fixed cost.
        * ``block_c=256`` only when ``C`` is large *and* the
          per-channel state fits in registers — register-pressure
          guard is applied below.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "triton", "xla") if requested in (None, "auto") else (str(requested),)
        channels = self._channels_from_inv(inv)
        bc_choices: list[int] = []
        for bc in (32, 64, 128, 256):
            if channels == 0 or bc <= max(channels, 32):
                bc_choices.append(bc)
        if not bc_choices:
            bc_choices = [64]
        candidates: list[RWKV4Config] = []
        if "tilelang" in platforms:
            for bc in bc_choices:
                candidates.append(RWKV4Config(block_c=bc, platform="tilelang", backend="gpu"))
        if "triton" in platforms:
            candidates.append(RWKV4Config(block_c=64, platform="triton", backend="gpu"))
        if "xla" in platforms:
            candidates.append(RWKV4Config(block_c=64, platform="xla", backend="any"))
        return candidates or [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[RWKV4Config, Array]):
        """Return TPU candidates for the XLA RWKV-4 path."""
        return [RWKV4Config(block_c=64, platform="xla", backend="any")]


_executor: Executor[RWKV4Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv4"),
    )
)


def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
    /,
    *,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RWKV4Config | None = None,
) -> Float[Array, "batch seq_len chans"] | tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 time-mix recurrence with automatic backend selection.

    Computes the RWKV-4 time-mixing recurrence over a sequence.

    Args:
        w: Time-decay parameter in log space `[C]`. Internally uses `-exp(w)`.
        u: Time-mix bias `[C]`.
        k: Key tensor `[B, T, C]`.
        v: Value tensor `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` containing (alpha, beta, eps).
            If None, initializes with alpha=0, beta=0, eps=-1e30.
        return_state: If True, also return the final state.
        platform: Backend platform override ("triton", "pallas", "cuda", "xla", "auto").
        cfg: Optional configuration object.

    Returns:
        Output tensor `[B, T, C]` (dtype matches `v`), or tuple of
        (output, final_state) if `return_state=True`. Final state is `[B, 3, C]`.
    """
    return _executor(
        RWKV4(),
        w=w,
        u=u,
        k=k,
        v=v,
        state=state,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )

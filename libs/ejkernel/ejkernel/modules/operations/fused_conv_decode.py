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

"""Public fused conv-state decode operation.

This module owns the eJKernel operation wrapper for the single-token
conv-state shift + depthwise convolution + activation used during GDR/GDN
decode inference. The XLA backend carries the canonical reference
implementation; the TPU Pallas backend provides a tiled alternative for
microbenchmarking.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

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
from .configs import FusedConvDecodeConfig


class FusedConvDecode(Kernel[FusedConvDecodeConfig, tuple[Array, Array]]):
    """Executor-driven wrapper for fused conv-state decode.

    Implements the :class:`~ejkernel.ops.Kernel` protocol for the single-token
    conv-state shift + depthwise convolution + activation step. The kernel
    parameterizes over :class:`FusedConvDecodeConfig` and produces a
    ``(updated_state, conv_output)`` tuple. ``run`` resolves the backend from the
    config and routes any non-SiLU activation away from the Pallas path (which
    hard-codes SiLU) to the XLA reference.

    Attributes:
        version: Cache-busting version tag for the persistent config cache.
    """

    version = "1"

    def __init__(self):
        """Initialize the fused conv decode kernel.

        Registers the operation under ``op_id="fused_conv_decode"`` so the
        registry and persistent config cache can look it up.
        """
        super().__init__(op_id="fused_conv_decode")

    def get_impl(self, cfg: FusedConvDecodeConfig):
        """Resolve the backend implementation for a config.

        Args:
            cfg: Operation config carrying the requested ``platform`` and
                ``backend``.

        Returns:
            The registered callable implementing ``fused_conv_decode`` for the
            detected platform and the config's backend.
        """
        platform = detect_platform("fused_conv_decode", cfg.platform)
        return kernel_registry.get("fused_conv_decode", platform=platform, backend=cfg.backend)

    def run(
        self,
        conv_state: jnp.ndarray,
        new_tokens: jnp.ndarray,
        kernel: jnp.ndarray,
        *,
        output_dtype: jnp.dtype,
        activation: Callable[[Array], Array] | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: FusedConvDecodeConfig,
        **_,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run the selected fused conv decode backend implementation.

        Resolves the effective activation, then dispatches to the backend named by
        ``cfg`` (after any ``platform`` override). The TPU Pallas kernel hard-codes
        SiLU, so any non-SiLU activation (including ``activation="none"``, which
        maps to ``activation_fn=None``) forces the XLA reference path even when the
        config requested ``"pallas"`` or ``"auto"``.

        Args:
            conv_state: Current rolling conv state, shape
                ``[num_slots, conv_dim, d_conv]``.
            new_tokens: New per-slot token embeddings, shape
                ``[num_slots, conv_dim]``.
            kernel: Depthwise convolution kernel, shape ``[conv_dim, d_conv]``.
            output_dtype: dtype requested for the convolution output.
            activation: Explicit activation callable, or ``None`` to derive one
                from ``cfg.activation`` (``"silu"``/``"swish"`` -> SiLU,
                ``"none"`` -> no activation, anything else -> SiLU). Defaults to
                ``None``.
            platform: Optional implementation-family override. When given, a new
                config is built with this platform and ``backend=ANY`` for
                ``"xla"`` (else the existing backend). Defaults to ``None``.
            cfg: Operation config selected by the executor.
            **_: Ignored extra keyword arguments forwarded by the executor.

        Returns:
            Tuple ``(updated_state, conv_output)``: the shifted/updated conv state
            and the (optionally activated) convolution output in ``output_dtype``.
        """
        if platform is not None:
            cfg = FusedConvDecodeConfig(
                d_conv=cfg.d_conv,
                activation=cfg.activation,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        activation_fn = activation
        if activation_fn is None:
            if cfg.activation in ("silu", "swish"):
                activation_fn = jax.nn.silu
            elif cfg.activation == "none":
                activation_fn = None
            else:
                activation_fn = jax.nn.silu

        # Pallas backend only supports SiLU; route non-SiLU requests to XLA.
        if activation_fn is not jax.nn.silu and cfg.platform in ("pallas", "auto"):
            cfg = FusedConvDecodeConfig(
                d_conv=cfg.d_conv,
                activation=cfg.activation,
                platform="xla",
                backend="any",
            )

        impl = self.get_impl(cfg)
        return impl(
            conv_state,
            new_tokens,
            kernel,
            output_dtype=output_dtype,
            activation=activation_fn,
        )

    def heuristic_cfg(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]]):
        """Return the default XLA config for an invocation.

        Reads ``d_conv`` (default ``4``) and ``activation`` (default ``"silu"``)
        from the invocation kwargs and pins the platform to XLA.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A :class:`FusedConvDecodeConfig` on ``platform="xla"`` with
            ``backend="any"``.
        """
        d_conv = int(inv.kwargs.get("d_conv", 4))
        activation = str(inv.kwargs.get("activation", "silu"))
        return FusedConvDecodeConfig(d_conv=d_conv, activation=activation, platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]]):
        """Return the default candidate config list (heuristic only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]]):
        """Return GPU autotuning candidates (heuristic XLA only).

        No GPU-specific backend exists for this op, so the GPU candidate set is
        the same single XLA config as the heuristic.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[FusedConvDecodeConfig, tuple[Array, Array]]):
        """Return TPU autotuning candidates (XLA reference plus Pallas).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A two-element list: the heuristic XLA config followed by a Pallas/TPU
            config that reuses the heuristic's ``d_conv`` and ``activation``.
        """
        cfg = self.heuristic_cfg(inv)
        return [
            cfg,
            FusedConvDecodeConfig(d_conv=cfg.d_conv, activation=cfg.activation, platform="pallas", backend="tpu"),
        ]

    candidate_cfgs_xla = candidate_cfgs
    candidate_cfgs_any = candidate_cfgs


_executor: Executor[FusedConvDecodeConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("fused_conv_decode"),
    )
)


def fused_conv_decode(
    conv_state: jnp.ndarray,
    new_tokens: jnp.ndarray,
    kernel: jnp.ndarray,
    *,
    output_dtype: jnp.dtype,
    activation: Callable[[Array], Array] | Literal["silu", "swish", "none"] | None = None,
    d_conv: int = 4,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: FusedConvDecodeConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Execute fused conv-state decode through the eJKernel operation stack.

    Performs one decode step of the depthwise causal convolution: each slot's
    rolling state is shifted to admit ``new_tokens``, convolved with ``kernel``,
    and optionally passed through an activation. Backend selection, config
    caching, and the SiLU-only Pallas constraint are handled by the kernel/
    executor.

    Args:
        conv_state: Current conv state, shape ``[num_slots, conv_dim, d_conv]``.
        new_tokens: New per-slot token embeddings, shape
            ``[num_slots, conv_dim]``.
        kernel: Depthwise conv kernel, shape ``[conv_dim, d_conv]``.
        output_dtype: Desired dtype for the convolution output.
        activation: Either an activation callable, one of the string names
            ``"silu"``/``"swish"``/``"none"``, or ``None``. A string selects the
            named activation and is stored in the config; a callable is passed
            through directly (the config records the placeholder ``"silu"``).
            Defaults to ``None`` (config default ``"silu"``).
        d_conv: Convolution window size. Defaults to ``4``.
        platform: Optional implementation-family override. Defaults to ``None``
            (uses ``cfg.platform`` or ``"auto"``).
        cfg: Optional pre-built operation config. When ``None``, one is built
            from ``d_conv``, the resolved activation name, and ``platform``.

    Returns:
        Tuple ``(updated_state, conv_output)``: the updated conv state and the
        (optionally activated) convolution output cast to ``output_dtype``.
    """
    activation_name = "silu"
    activation_fn: Callable[[Array], Array] | None = None
    if isinstance(activation, str):
        activation_name = activation
        activation_fn = None
    elif callable(activation):
        activation_fn = activation
        activation_name = "silu"  # placeholder; config stores the string

    if cfg is None:
        cfg = FusedConvDecodeConfig(
            d_conv=d_conv,
            activation=activation_name,
            platform=platform or "auto",
        )

    return _executor(
        FusedConvDecode(),
        conv_state,
        new_tokens,
        kernel,
        output_dtype=output_dtype,
        activation=activation_fn,
        d_conv=d_conv,
        platform=platform,
        _cfg=cfg,
    )


fused_conv_decode_op = fused_conv_decode

# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Reusable Loop Transformer layers (per-token Gated Progressive Update)."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float

from .linears import ColumnParallelLinear, ParallelLinear
from .norms import RMSNorm


class LoopGate(spx.Module):
    """Per-token Gated Progressive Update block for looped transformer execution.

    Computes a bounded convex interpolation between the hidden state entering a
    loop iteration (``x_prev``) and the candidate state produced by the shared
    loop block (``h_cand``)::

        delta = h_cand - x_prev
        beta  = sigmoid(proj([RMSNorm(x_prev), RMSNorm(delta)]))
        x_out = x_prev + beta * delta

    Zero-initializing ``proj.weight`` and setting ``proj.bias`` to ``init_bias``
    (default ``-5.0`` on early passes, ``+5.0`` on the final pass) makes step-0
    continued training from a non-looped checkpoint nearly identity-preserving
    while maintaining healthy gradients into every pass from the first step.

    Both ``x_norm`` and ``delta_norm`` default to zero-centered ``(1 + w)``
    float32-scaled :class:`RMSNorm` so parameter weights initialize to ``0.0``
    everywhere (matching ``Qwen3NextRMSNorm`` / Gemma RMSNorm and simplifying
    checkpoint upgrade materialization).
    """

    def __init__(
        self,
        config: tp.Any,
        init_bias: float,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        norm_cls: type[spx.Module] | None = None,
        rngs: spx.Rngs,
    ):
        """Initialize a per-token Loop Transformer progressive update gate.

        Args:
            config: Model configuration defining ``hidden_size``, ``loop_gate_type``,
                and optionally ``rms_norm_eps``.
            init_bias: Initial scalar bias applied to ``proj.bias`` (passed through
                sigmoid to set the step-0 blend weight).
            dtype: Activation compute dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision passed to ``proj``.
            norm_cls: Optional custom RMSNorm class taking
                ``(hidden_size, eps=..., dtype=..., param_dtype=..., rngs=...)``.
                Defaults to zero-centered :class:`RMSNorm` (``scale_offset=1.0``,
                ``fp32_scale=True``, ``kernel_init=zeros``).
            rngs: SpectraX PRNG container.
        """
        self.config = config
        self.init_bias = float(init_bias)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        if norm_cls is not None:
            self.x_norm = norm_cls(
                config.hidden_size,
                eps=eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.delta_norm = norm_cls(
                config.hidden_size,
                eps=eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.x_norm = RMSNorm(
                config.hidden_size,
                eps=eps,
                dtype=dtype,
                param_dtype=param_dtype,
                scale_offset=1.0,
                fp32_scale=True,
                kernel_init=jax.nn.initializers.zeros,
            )
            self.delta_norm = RMSNorm(
                config.hidden_size,
                eps=eps,
                dtype=dtype,
                param_dtype=param_dtype,
                scale_offset=1.0,
                fp32_scale=True,
                kernel_init=jax.nn.initializers.zeros,
            )

        gate_type = str(getattr(config, "loop_gate_type", "scalar"))
        out_dim = 1 if gate_type == "scalar" else config.hidden_size
        linear_cls = ParallelLinear if out_dim == 1 else ColumnParallelLinear
        self.proj = linear_cls(
            2 * config.hidden_size,
            out_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.constant(self.init_bias),
            rngs=rngs,
        )

    def compute_beta(
        self,
        x_prev: Float[Array, "batch seq_len hidden_dim"],
        h_cand: Float[Array, "batch seq_len hidden_dim"],
    ) -> Float[Array, "batch seq_len gate_dim"]:
        """Compute per-token sigmoid gate values ``beta`` in float32."""
        delta = h_cand - x_prev
        gate_input = jnp.concatenate([self.x_norm(x_prev), self.delta_norm(delta)], axis=-1)
        return jax.nn.sigmoid(self.proj(gate_input).astype(jnp.float32))

    def forward(
        self,
        x_prev: Float[Array, "batch seq_len hidden_dim"],
        h_cand: Float[Array, "batch seq_len hidden_dim"],
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Blend ``x_prev`` and ``h_cand`` via a per-token sigmoid gate."""
        delta = h_cand - x_prev
        beta = self.compute_beta(x_prev, h_cand)
        updated = x_prev.astype(jnp.float32) + beta * delta.astype(jnp.float32)
        return checkpoint_name(updated.astype(x_prev.dtype), "loop_gate_output")


__all__ = ["LoopGate"]

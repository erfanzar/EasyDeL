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
from eformer.optimizers import stochastic_cast, stochastic_cast_bf16_with_bits
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
        pass_index: int = 0,
        stochastic_rounding: bool | None = None,
        norm_cls: type[spx.Module] | None = None,
        rngs: spx.Rngs,
    ):
        """Initialize a per-token Loop Transformer progressive update gate.

        Args:
            config: Model configuration defining ``hidden_size``, ``loop_gate_type``,
                and optionally ``rms_norm_eps`` and ``loop_gate_stochastic_rounding``.
            init_bias: Initial scalar bias applied to ``proj.bias`` (passed through
                sigmoid to set the step-0 blend weight).
            dtype: Activation compute dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision passed to ``proj``.
            pass_index: Zero-based loop iteration index used to decorrelate PRNG keys
                when ``stochastic_rounding`` is enabled.
            stochastic_rounding: Optional override for stochastic rounding on the
                residual cast ``fp32 -> x_prev.dtype`` during training. When ``None``,
                reads ``getattr(config, "loop_gate_stochastic_rounding", False)``.
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
        self.pass_index = int(pass_index)
        self.stochastic_rounding = (
            bool(getattr(config, "loop_gate_stochastic_rounding", False))
            if stochastic_rounding is None
            else bool(stochastic_rounding)
        )

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
        *,
        deterministic: bool | None = None,
        rng_key: Array | None = None,
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Blend ``x_prev`` and ``h_cand`` via a per-token sigmoid gate.

        When ``self.stochastic_rounding`` is enabled and the module is in training mode
        (``not deterministic``), casts the float32 residual blend back to narrow
        ``x_prev.dtype`` (e.g. ``bfloat16``) via exact unbiased stochastic rounding with
        an identity STE backward so small early-pass refinements (``beta ~ 0.008``) are
        not truncated to zero by deterministic round-to-nearest.
        """
        delta = h_cand - x_prev
        beta = self.compute_beta(x_prev, h_cand)
        updated = x_prev.astype(jnp.float32) + beta * delta.astype(jnp.float32)
        use_det = (not self.training) if deterministic is None else bool(deterministic)
        if self.stochastic_rounding and not use_det and jnp.finfo(x_prev.dtype).bits < 32:
            if rng_key is not None:
                out = stochastic_cast(updated, x_prev.dtype, rng_key)
            elif x_prev.dtype == jnp.bfloat16:
                # Zero-comm, batch-varying per-token SplitMix32-mixed 16-bit dither:
                # decorrelated across passes & tokens with zero RNG state mutation.
                base_bits = jax.random.bits(jax.random.key(self.pass_index + 1), updated.shape, dtype=jnp.uint32)
                stride = max(1, x_prev.shape[-1] // 64)
                raw_u = jax.lax.bitcast_convert_type(
                    jax.lax.stop_gradient(x_prev[..., ::stride]).astype(jnp.float32),
                    jnp.uint32,
                )
                tok_salt = jnp.sum(raw_u, axis=-1, keepdims=True, dtype=jnp.uint32)
                z = (base_bits ^ tok_salt) + jnp.uint32(0x9E3779B9)
                z = (z ^ (z >> jnp.uint32(16))) * jnp.uint32(0x85EBCA6B)
                z = (z ^ (z >> jnp.uint32(13))) * jnp.uint32(0xC2B2AE35)
                rand16 = (z ^ (z >> jnp.uint32(16))) >> jnp.uint32(16)
                out = stochastic_cast_bf16_with_bits(updated, rand16)
            else:
                out = stochastic_cast(updated, x_prev.dtype, jax.random.key(self.pass_index + 1))
        else:
            out = updated.astype(x_prev.dtype)
        return checkpoint_name(out, "loop_gate_output")


__all__ = ["LoopGate"]

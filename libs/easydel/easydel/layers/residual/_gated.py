# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Qwen4-style gated hyper-connections.

The residual state is ``hc_count`` parallel streams kept flat as
``[batch, seq, hc_count * hidden]``. Each sub-layer site owns one
:class:`GatedResidual` module, which performs both sides of the exchange:

**Read.** The flattened streams are normalized (a grouped, zero-centred
RMSNorm: each ``hidden``-sized group separately, scale ``(1 + w)``), mapped
through a silu/sigmoid low-rank bottleneck (``hc_hidden -> hc_lowrank ->
hc_hidden``), and used as an *element-wise* gate on the normed streams. The
gated streams are then averaged, producing the sub-layer input
``[batch, seq, hidden]``.

**Write.** When ``use_combine=True`` the module additionally returns the raw
(normed) streams and ``hc_count`` per-branch scalar gates
(``2 * sigmoid(W inject / hc_count)``). The caller writes the sub-layer output
``y`` back with :func:`inject_streams`: ``streams + (w_i * y)`` per branch.

At the model boundary the same module with ``use_combine=False`` collapses the
streams back to a single hidden sequence -- it *is* the final norm of the
Qwen4-Exp architecture (the checkpoint carries no separate final norm).

Attribute names (``hc_norm``, ``input_mix_weight_down``,
``input_mix_weight_up``, ``block_inject_weight``) match the released
checkpoint exactly so state-dict conversion needs no renaming.
"""

from __future__ import annotations

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..linears import ColumnParallelLinear, RowParallelLinear
from ..norms import RMSNorm

__all__ = ("GatedResidual", "expand_streams", "inject_streams")


def expand_streams(
    hidden_states: Float[Array, "batch seq hidden"], hc_count: int
) -> Float[Array, "batch seq hc*hidden"]:
    """Widen a single hidden sequence into ``hc_count`` concatenated streams.

    Args:
        hidden_states: Hidden states ``[batch, seq, hidden]``.
        hc_count: Number of residual streams.

    Returns:
        ``hidden_states`` tiled ``hc_count`` times along the last axis (block
        order ``[x | x | ...]``, matching ``torch.Tensor.repeat(1, 1, hc)``),
        giving ``[batch, seq, hc_count * hidden]``.
    """
    reps = (1,) * (hidden_states.ndim - 1) + (hc_count,)
    return jnp.tile(hidden_states, reps)


def inject_streams(
    hyper_input: Float[Array, "batch seq hc*hidden"],
    sublayer_output: Float[Array, "batch seq hidden"],
    injection_weights: Float[Array, "batch seq hc"],
) -> Float[Array, "batch seq hc*hidden"]:
    """Write a sub-layer output back onto the residual streams.

    Args:
        hyper_input: The streams as returned by :class:`GatedResidual` (the
            *unnormalized* input), ``[batch, seq, hc * hidden]``.
        sublayer_output: Sub-layer output ``[batch, seq, hidden]``.
        injection_weights: Per-branch scalar gates ``[batch, seq, hc]``.

    Returns:
        Updated streams: ``hyper_input + flatten(w_i * sublayer_output)``.
    """
    injection = sublayer_output[..., None, :] * injection_weights[..., None]
    return hyper_input + injection.reshape(hyper_input.shape)


class GatedResidual(spx.Module):
    """Low-rank gated read + scalar-gated write over ``hc_count`` residual streams.

    Attributes:
        hc_norm: Grouped RMSNorm over the flattened streams (each ``hidden``
            group normalized separately, ``(1 + w)`` scale convention).
        input_mix_weight_down: Bottleneck projection ``hc*hidden -> hc_lowrank``.
        input_mix_weight_up: Bottleneck projection ``hc_lowrank -> hc*hidden``.
        block_inject_weight: Per-branch gate projection ``hc*hidden -> hc_count``,
            present only when ``use_combine=True``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        eps: float = 1e-6,
        *,
        use_combine: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: spx.Rngs,
    ) -> None:
        """Build the gated residual module.

        Args:
            hidden_size: Width of each residual stream.
            hc_count: Number of parallel residual streams.
            hc_lowrank: Bottleneck rank of the input mixer.
            eps: RMSNorm epsilon.
            use_combine: When ``False`` the module only computes the read side
                (no ``block_inject_weight``); this is the model-entry/exit
                mixer form.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        if hc_count <= 1:
            raise ValueError(f"GatedResidual requires hc_count > 1, got {hc_count}.")
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.use_combine = use_combine
        hc_hidden_size = hc_count * hidden_size
        self.hc_hidden_size = hc_hidden_size

        # The reference computes the norm in float32 and casts back; passing
        # dtype=float32 reproduces that exactly while keeping storage at
        # ``param_dtype``.
        self.hc_norm = RMSNorm(
            hc_hidden_size,
            eps=eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            scale_offset=1.0,
            group_size=hidden_size,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        kernel_init = jax.nn.initializers.normal(0.02)
        self.input_mix_weight_down = RowParallelLinear(
            hc_hidden_size,
            hc_lowrank,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.input_mix_weight_up = ColumnParallelLinear(
            hc_lowrank,
            hc_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.block_inject_weight = (
            RowParallelLinear(
                hc_hidden_size,
                hc_count,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=kernel_init,
                rngs=rngs,
            )
            if use_combine
            else None
        )

    def forward(
        self, hyper_input: Float[Array, "batch seq hc*hidden"]
    ) -> (
        Float[Array, "batch seq hidden"]
        | tuple[Float[Array, "batch seq hidden"], Float[Array, "batch seq hc*hidden"], Float[Array, "batch seq hc"]]
    ):
        """Run the read (and optionally write-gate) side of the exchange.

        Args:
            hyper_input: Flattened residual streams
                ``[batch, seq, hc_count * hidden]``.

        Returns:
            With ``use_combine=False``: just ``mixed_input``
            ``[batch, seq, hidden]``. Otherwise the triple ``(mixed_input,
            hyper_input, injection_weights)`` where ``hyper_input`` is the
            unmodified input (the residual base) and ``injection_weights`` is
            ``[batch, seq, hc_count]`` for :func:`inject_streams`.

        Raises:
            ValueError: If the trailing axis is not ``hc_count * hidden``.
        """
        if hyper_input.shape[-1] != self.hc_hidden_size:
            raise ValueError(f"Expected {self.hc_hidden_size} hyper-connection features, got {hyper_input.shape[-1]}.")
        hc, hidden = self.hc_count, self.hidden_size
        normed = self.hc_norm(hyper_input)
        mix = jax.nn.silu(self.input_mix_weight_down(normed) / hc)
        mix = jax.nn.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], hc, hidden)
        gated = mix * normed.reshape(*normed.shape[:-1], hc, hidden)
        mixed_input = jnp.mean(gated, axis=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2.0 * jax.nn.sigmoid(self.block_inject_weight(normed) / hc)
        return mixed_input, hyper_input, injection_weights

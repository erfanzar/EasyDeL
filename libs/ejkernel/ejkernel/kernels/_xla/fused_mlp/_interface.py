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

"""XLA reference implementation of the fused MLP block.

Computes ``down(act(gate(x)) * up(x))`` for the three weight formats the TPU
quantization work supports (bf16 dense, channelwise int8, channelwise int4)
behind one signature. This is simultaneously:

* the mandatory ``Platform.XLA`` fallback for the ``fused_mlp`` kernel;
* the numerical ground truth the Pallas kernels are tested against;
* the **training** path — quantized formats execute through the same
  fused-upcast/int-dot compositions the channelwise op uses, so forward cost
  tracks the measured quantized matmul numbers.

Layout policy: kernels only ever see gate and up as **separate** ``[K, I]``
arrays. Fused ``[K, 2I]`` checkpoint layouts (plain concat or EasyDeL's
TP-interleaved segments) are normalized once by :func:`split_gate_up` at
pack/convert time — supporting layouts at the boundary keeps every kernel
free of layout branches.
"""

from __future__ import annotations

import typing as tp

import jax
import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry

__all__ = ["ACTIVATIONS", "fused_mlp_xla", "split_gate_up"]

#: Supported elementwise activations for the gate branch.
ACTIVATIONS: dict[str, tp.Callable[[jax.Array], jax.Array]] = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "gelu_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
}


def split_gate_up(
    gate_up: jax.Array,
    *,
    layout: tp.Literal["concat", "interleaved"] = "concat",
    segments: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Split a fused ``[K, 2I]`` gate_up weight into ``(gate, up)``.

    Args:
        gate_up: Fused weight (or quantized codes / packed rows — anything
            whose last axis carries the fused output layout).
        layout: ``"concat"`` for ``[gate | up]``; ``"interleaved"`` for the
            TP-portable EasyDeL layout where the last axis is ``segments``
            repetitions of ``[gate_seg | up_seg]`` (one segment per TP rank).
        segments: Number of interleave segments (the TP degree the layout was
            built for). Ignored for ``"concat"``.

    Returns:
        ``(gate, up)`` each with last axis ``I``.

    Raises:
        ValueError: On an unknown layout or a non-divisible interleave.
    """
    two_i = gate_up.shape[-1]
    if layout == "concat":
        return gate_up[..., : two_i // 2], gate_up[..., two_i // 2 :]
    if layout == "interleaved":
        if two_i % (2 * segments):
            raise ValueError(f"last axis {two_i} not divisible into 2*{segments} interleave segments.")
        seg = gate_up.reshape(*gate_up.shape[:-1], segments, 2, two_i // (2 * segments))
        gate = seg[..., :, 0, :].reshape(*gate_up.shape[:-1], two_i // 2)
        up = seg[..., :, 1, :].reshape(*gate_up.shape[:-1], two_i // 2)
        return gate, up
    raise ValueError(f"Unknown gate_up layout {layout!r}; expected 'concat' or 'interleaved'.")


def _project(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array | None,
    *,
    quantize_activations: bool,
    prefill_threshold: int,
) -> jax.Array:
    """One projection in whatever format the weight is stored.

    Args:
        x: Activations ``[m, k]``.
        weight: ``[k, n]`` — floating (dense) or integer codes (channelwise).
        scale: Per-output-channel scale for integer weights; ``None`` for
            dense.
        quantize_activations: Enable the integer-dot path at prefill sizes.
        prefill_threshold: Token count where the integer dot takes over.

    Returns:
        ``[m, n]`` in ``x``'s dtype.

    Raises:
        ValueError: If an integer weight arrives without its scale.
    """
    if jnp.issubdtype(weight.dtype, jnp.integer):
        if scale is None:
            raise ValueError("Integer (quantized) weights require a channel scale.")
        from ..quantized_matmul import channelwise_quantized_matmul

        return channelwise_quantized_matmul(
            x,
            weight,
            scale,
            quantize_activations=quantize_activations,
            prefill_threshold=prefill_threshold,
        )
    return x @ weight.astype(x.dtype)


@kernel_registry.register("fused_mlp", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def fused_mlp_xla(
    x: Float[Array, "m k"],
    w_gate: Array,
    w_up: Array,
    w_down: Array,
    gate_scale: Array | None = None,
    up_scale: Array | None = None,
    down_scale: Array | None = None,
    *,
    activation: str = "silu",
    quantize_activations: bool = False,
    prefill_threshold: int = 256,
) -> Float[Array, "m k"]:
    """Reference ``down(act(gate(x)) * up(x))`` for any supported format.

    Args:
        x: Input activations ``[m, k]``, floating dtype.
        w_gate: Gate projection ``[k, i]`` — floating or integer codes.
        w_up: Up projection ``[k, i]`` — same format as ``w_gate``.
        w_down: Down projection ``[i, k]`` — same format family.
        gate_scale: Channel scale for integer ``w_gate`` (``None`` for dense).
        up_scale: Channel scale for integer ``w_up``.
        down_scale: Channel scale for integer ``w_down``.
        activation: Key into :data:`ACTIVATIONS` applied to the gate branch.
        quantize_activations: For integer formats, run prefill-sized calls on
            the native int MXU path (per-token dynamic activation quant).
        prefill_threshold: Token count where the integer-dot path engages.

    Returns:
        ``[m, k]`` in ``x``'s dtype.

    Raises:
        KeyError: On an unknown activation name.
    """
    act = ACTIVATIONS[activation]
    gate = _project(
        x, w_gate, gate_scale, quantize_activations=quantize_activations, prefill_threshold=prefill_threshold
    )
    up = _project(x, w_up, up_scale, quantize_activations=quantize_activations, prefill_threshold=prefill_threshold)
    hidden = (act(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(x.dtype)
    return _project(
        hidden, w_down, down_scale, quantize_activations=quantize_activations, prefill_threshold=prefill_threshold
    )

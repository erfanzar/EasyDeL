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
from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry

__all__ = [
    "ACTIVATIONS",
    "fused_mlp_bf16_xla",
    "fused_mlp_w4a4_xla",
    "fused_mlp_xla",
    "resolve_mlp_combine",
    "split_gate_up",
]

#: Supported elementwise activations for the gate branch.
ACTIVATIONS: dict[str, tp.Callable[[jax.Array], jax.Array]] = {
    "silu": jax.nn.silu,
    "gelu": partial(jax.nn.gelu, approximate=False),
    "gelu_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
}


def _clamped_swiglu(limit: float):
    def combine(gate: jax.Array, up: jax.Array) -> jax.Array:
        gate = jnp.minimum(gate, limit)
        up = jnp.clip(up, -limit, limit)
        return jax.nn.silu(gate) * up

    return combine


def _swiglu_oai(alpha: float, limit: float):
    def combine(gate: jax.Array, up: jax.Array) -> jax.Array:
        gate = jnp.minimum(gate, limit)
        up = jnp.clip(up, -limit, limit)
        return (up + 1.0) * (gate * jax.nn.sigmoid(gate * alpha))

    return combine


def _scaled_silu(gate_multiplier: float):
    def combine(gate: jax.Array, up: jax.Array) -> jax.Array:
        return jax.nn.silu(gate * gate_multiplier) * up

    return combine


def _situ(beta: float, linear_beta: float):
    def combine(gate: jax.Array, up: jax.Array) -> jax.Array:
        g32 = gate.astype(jnp.float32)
        u32 = up.astype(jnp.float32)
        g = beta * jnp.tanh(g32 / beta) * jax.nn.sigmoid(g32)
        u = linear_beta * jnp.tanh(u32 / linear_beta) if linear_beta else u32
        return (g * u).astype(gate.dtype)

    return combine


#: Parameterized two-branch combine factories: name -> (arity, factory).
_COMBINE_FACTORIES: dict[str, tuple[int, tp.Callable[..., tp.Callable[[jax.Array, jax.Array], jax.Array]]]] = {
    "clamped_swiglu": (1, _clamped_swiglu),
    "swiglu_oai": (2, _swiglu_oai),
    "scaled_silu": (1, _scaled_silu),
    "situ": (2, _situ),
}


def resolve_mlp_combine(
    activation: str,
    act_params: tuple[float, ...] = (),
) -> tp.Callable[[jax.Array, jax.Array], jax.Array]:
    """Resolve the two-branch gate/up combine for the fused MLP kernels.

    Unary names from :data:`ACTIVATIONS` compose as ``act(gate) * up`` in the
    operands' dtype. Parameterized names take static float ``act_params``:

    * ``"clamped_swiglu"`` ``(limit,)`` — ``silu(min(g, l)) * clip(u, -l, l)``
      (DeepSeek-V4).
    * ``"swiglu_oai"`` ``(alpha, limit)`` — GPT-OSS/MiniMax recipe
      ``(clip(u, -l, l) + 1) * min(g, l) * sigmoid(alpha * min(g, l))``.
    * ``"scaled_silu"`` ``(gate_multiplier,)`` — muP ``silu(g * gm) * u``
      (Falcon-H1).
    * ``"situ"`` ``(beta, linear_beta)`` — Kimi K3
      ``beta * tanh(g / beta) * sigmoid(g) * squash(u)`` computed in float32;
      ``linear_beta == 0`` disables the linear-branch squash.

    Args:
        activation: Combine name.
        act_params: Static parameters, arity-checked per name.

    Returns:
        ``combine(gate, up) -> hidden``.

    Raises:
        ValueError: Unknown name or wrong ``act_params`` arity.
    """
    if activation in ACTIVATIONS:
        if act_params:
            raise ValueError(f"activation {activation!r} takes no act_params, got {act_params}.")
        act = ACTIVATIONS[activation]
        return lambda gate, up: act(gate) * up
    entry = _COMBINE_FACTORIES.get(activation)
    if entry is None:
        raise ValueError(
            f"Unknown activation {activation!r}; supported: {sorted(ACTIVATIONS) + sorted(_COMBINE_FACTORIES)}."
        )
    arity, factory = entry
    if len(act_params) != arity:
        raise ValueError(f"activation {activation!r} takes {arity} act_params, got {act_params}.")
    return factory(*(float(v) for v in act_params))


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
    activation_bits: int,
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
            activation_bits=activation_bits,
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
    act_params: tuple[float, ...] = (),
    quantize_activations: bool = False,
    activation_bits: int = 8,
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
        activation: Combine name (see :func:`resolve_mlp_combine`).
        act_params: Static parameters for parameterized combines.
        quantize_activations: For integer formats, run prefill-sized calls on
            the native int MXU path (per-token dynamic activation quant).
        prefill_threshold: Token count where the integer-dot path engages.

    Returns:
        ``[m, k]`` in ``x``'s dtype.

    Raises:
        KeyError: On an unknown activation name.
    """
    combine = resolve_mlp_combine(activation, act_params)
    gate = _project(
        x,
        w_gate,
        gate_scale,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
    )
    up = _project(
        x,
        w_up,
        up_scale,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
    )
    hidden = combine(gate.astype(jnp.float32), up.astype(jnp.float32)).astype(x.dtype)
    return _project(
        hidden,
        w_down,
        down_scale,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        prefill_threshold=prefill_threshold,
    )


def _unpack_adjacent_codes(w_packed: jax.Array) -> jax.Array:
    """Decode adjacent-rows packed int4 weights to signed codes ``[K, N]``.

    Args:
        w_packed: ``uint8`` ``[K // 2, N]`` in ``pack_int4_adjacent`` layout.

    Returns:
        Signed codes ``[K, N]`` as float32 (exact — int4 magnitudes).
    """
    packed = w_packed.astype(jnp.int32)
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    low = jnp.where(low > 7, low - 16, low)
    high = jnp.where(high > 7, high - 16, high)
    return jnp.stack([low, high], axis=1).reshape(low.shape[0] * 2, low.shape[1]).astype(jnp.float32)


@kernel_registry.register("fused_mlp_w4a4", Platform.XLA, Backend.ANY)
def fused_mlp_w4a4_xla(
    x4: jax.Array,
    gate_packed: jax.Array,
    up_packed: jax.Array,
    down_packed: jax.Array,
    gate_scale: jax.Array,
    up_scale: jax.Array,
    down_scale: jax.Array,
    x_scale: jax.Array,
    *,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
    tile_i: int = 1024,
    interpret: bool = False,
) -> jax.Array:
    """Reference for the fused W4A4 MLP, replicating its quantized semantics.

    Tile-for-tile float replication of the Pallas kernel: int4 input codes,
    per-channel weight scales, and per-``(token, tile_i)`` re-quantization of
    the hidden state — so parity against this reference asserts
    kernel-exactness, not a loose accuracy bound. Pure ``jnp``; runs on any
    backend. ``tile_i`` here is *semantic* (it changes the hidden-quant block
    boundaries) and must match the Pallas call being compared.

    Args:
        x4: Activations ``[m, k]`` quantized to ``int4``.
        gate_packed: ``uint8`` ``[k // 2, i]`` adjacent-rows packed gate.
        up_packed: ``uint8`` ``[k // 2, i]`` packed up projection.
        down_packed: ``uint8`` ``[i // 2, k]`` packed down projection.
        gate_scale: Per-channel scale for gate, reshapeable to ``[1, i]``.
        up_scale: Per-channel scale for up.
        down_scale: Per-channel scale ``[1, k]`` for down.
        x_scale: Per-token activation scale ``[m, 1]``.
        activation: Gate-branch activation name.
        tile_i: Hidden-quantization block width along I.
        interpret: Ignored (signature parity with the Pallas kernel).

    Returns:
        ``[m, k]`` bf16 in real units.

    Raises:
        ValueError: If ``x4`` is not int4.
    """
    del interpret
    if x4.dtype != jnp.int4:
        raise ValueError(f"fused_mlp_w4a4 expects int4 activations, got {x4.dtype}.")
    combine = resolve_mlp_combine(activation, act_params)
    i_dim = gate_packed.shape[1]
    k_dim = down_packed.shape[1]

    x_real = x4.astype(jnp.float32) * x_scale.reshape(-1, 1).astype(jnp.float32)
    gate_codes = _unpack_adjacent_codes(gate_packed)
    up_codes = _unpack_adjacent_codes(up_packed)
    down_codes = _unpack_adjacent_codes(down_packed)
    gate_s = gate_scale.reshape(1, i_dim).astype(jnp.float32)
    up_s = up_scale.reshape(1, i_dim).astype(jnp.float32)

    acc = jnp.zeros((x4.shape[0], k_dim), jnp.float32)
    for start in range(0, i_dim, tile_i):
        end = start + tile_i
        gate = (x_real @ gate_codes[:, start:end]) * gate_s[:, start:end]
        up = (x_real @ up_codes[:, start:end]) * up_s[:, start:end]
        hidden = combine(gate, up)
        habs = jnp.max(jnp.abs(hidden), axis=1, keepdims=True)
        h_scale = habs / 7.0
        h4 = jnp.clip(jnp.round(hidden * jnp.where(h_scale == 0, 0.0, 1.0 / h_scale)), -7.0, 7.0)
        acc += (h4 @ down_codes[start:end, :]) * h_scale
    return (acc * down_scale.reshape(1, k_dim).astype(jnp.float32)).astype(jnp.bfloat16)


@kernel_registry.register("fused_mlp_bf16", Platform.XLA, Backend.ANY)
def fused_mlp_bf16_xla(
    x: jax.Array,
    gate_up: jax.Array,
    w_down: jax.Array,
    *,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
    tile_m: int = 512,
    tile_i: int = 1024,
    save_preacts: bool = False,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """Reference for the dense-bf16 fused MLP over the fused weight.

    Same schedule XLA compiles for the model path: one fused ``[m, 2i]``
    dot, activation combine, down projection. Tiling arguments are accepted
    for signature parity and ignored — XLA schedules its own tiling.

    Args:
        x: Activations ``[m, k]``, floating dtype.
        gate_up: Fused weight ``[k, 2i]``, local concat layout.
        w_down: Down projection ``[i, k]``.
        activation: Gate-branch activation name.
        tile_m: Ignored (signature parity with the Pallas kernel).
        tile_i: Ignored (signature parity).
        save_preacts: Also return bf16 ``(gate_pre, up_pre)`` residuals.
        interpret: Ignored (signature parity).

    Returns:
        ``[m, k]`` in ``x``'s dtype, or ``(out, gate_pre, up_pre)``.
    """
    del tile_m, tile_i, interpret
    combine = resolve_mlp_combine(activation, act_params)
    i_dim = gate_up.shape[1] // 2
    gu = x @ gate_up.astype(x.dtype)
    gate_pre, up_pre = gu[:, :i_dim], gu[:, i_dim:]
    hidden = combine(gate_pre.astype(jnp.float32), up_pre.astype(jnp.float32)).astype(x.dtype)
    out = hidden @ w_down.astype(x.dtype)
    if save_preacts:
        return out, gate_pre.astype(jnp.bfloat16), up_pre.astype(jnp.bfloat16)
    return out

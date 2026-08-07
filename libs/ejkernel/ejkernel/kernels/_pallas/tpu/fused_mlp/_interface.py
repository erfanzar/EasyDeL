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

"""Decode-specialized fused MLP streaming packed-int4 weights (W4A4).

One ``pallas_call`` computes ``down(act(gate(x)) * up(x))`` for all three
projections, built on the two facts the packed-gemv investigation measured:

* per-dispatch fixed cost (~57us in-graph) is a large fraction of a decode
  op, so three dispatches -> one is a direct win;
* the only weight path faster than the element-conversion bound feeds the
  int4 MXU via ``pltpu.bitcast`` — a free register reinterpret of packed
  bytes (920 TOPS/core, 4x bf16).

The hidden activations never touch HBM: each I-tile's ``act(gate)*up`` is
computed in registers, re-quantized to int4 **per (token, tile)** — finer
scales than the per-token-global quantization that failed the accuracy gate
for single matmuls — and immediately consumed by the down-projection partial
accumulated in VMEM.

Weights arrive as three separate packed arrays (gate, up, down) in
:func:`~ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv.pack_int4_adjacent`
layout; fused/interleaved checkpoint layouts are normalized at pack time by
``split_gate_up`` — kernels carry no layout branches.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...._registry import Backend, Platform, kernel_registry
from ..quantized_matmul._packed_gemv import supports_int4_mxu

__all__ = ["fused_mlp_w4a4_pallas"]

_KERNEL_ACTIVATIONS: dict[str, tp.Callable[[jax.Array], jax.Array]] = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "gelu_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
}


def _kernel(
    x_ref,
    x_scale_ref,
    gate_ref,
    up_ref,
    down_ref,
    gate_scale_ref,
    up_scale_ref,
    down_scale_ref,
    out_ref,
    acc_ref,
    *,
    activation: str,
):
    """One I-tile of the fused MLP.

    Args:
        x_ref: Quantized activations ``[m, K]`` int4, fully resident.
        x_scale_ref: Per-token activation scale ``[m, 1]`` float32.
        gate_ref: Packed gate tile ``[K // 2, tile_i]`` uint8.
        up_ref: Packed up tile ``[K // 2, tile_i]`` uint8.
        down_ref: Packed down tile ``[tile_i // 2, K]`` uint8.
        gate_scale_ref: Gate channel scale ``[1, tile_i]`` float32.
        up_scale_ref: Up channel scale ``[1, tile_i]`` float32.
        down_scale_ref: Down channel scale ``[1, K]`` float32.
        out_ref: Output ``[m, K]`` bf16, written on the last tile.
        acc_ref: VMEM accumulator ``[m, K]`` float32.
        activation: Key into the kernel activation table.
    """
    step = pl.program_id(0)
    total = pl.num_programs(0)

    x_scale = x_scale_ref[...]
    gate_w = pltpu.bitcast(gate_ref[...], jnp.int4)
    up_w = pltpu.bitcast(up_ref[...], jnp.int4)

    gate = jnp.matmul(x_ref[...], gate_w, preferred_element_type=jnp.int32).astype(jnp.float32)
    gate *= gate_scale_ref[...] * x_scale
    up = jnp.matmul(x_ref[...], up_w, preferred_element_type=jnp.int32).astype(jnp.float32)
    up *= up_scale_ref[...] * x_scale

    hidden = _KERNEL_ACTIVATIONS[activation](gate) * up  # [m, tile_i] f32, never leaves registers

    # Re-quantize per (token, tile): each tile carries its own row scales, so
    # the down matmul below is exact w.r.t. these codes and the accumulated
    # partials just carry the per-tile scale.
    habs = jnp.max(jnp.abs(hidden), axis=1, keepdims=True)
    h_scale = habs / 7.0
    h_scale_inv = jnp.where(h_scale == 0, 0.0, 1.0 / h_scale)
    h4 = jnp.clip(jnp.round(hidden * h_scale_inv), -7.0, 7.0).astype(jnp.int4)

    down_w = pltpu.bitcast(down_ref[...], jnp.int4)
    partial = jnp.matmul(h4, down_w, preferred_element_type=jnp.int32).astype(jnp.float32)
    partial *= h_scale  # [m, 1] per-token-per-tile

    @pl.when(step == 0)
    def _():
        acc_ref[...] = partial

    @pl.when(step > 0)
    def _():
        acc_ref[...] += partial

    @pl.when(step == total - 1)
    def _():
        out_ref[...] = (acc_ref[...] * down_scale_ref[...]).astype(out_ref.dtype)


@kernel_registry.register("fused_mlp", Platform.PALLAS, Backend.TPU)
@functools.partial(jax.jit, static_argnames=("activation", "tile_i", "interpret"))
def fused_mlp_w4a4_pallas(
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
    tile_i: int = 1024,
    interpret: bool = False,
) -> jax.Array:
    """Fused W4A4 MLP block in one dispatch.

    Args:
        x4: Activations ``[m, k]`` already quantized to ``int4`` (per token).
        gate_packed: ``uint8`` ``[k // 2, i]`` adjacent-rows packed gate.
        up_packed: ``uint8`` ``[k // 2, i]`` packed up projection.
        down_packed: ``uint8`` ``[i // 2, k]`` packed down projection.
        gate_scale: Per-channel scale ``[1, i]`` (or reshapeable) for gate.
        up_scale: Per-channel scale for up.
        down_scale: Per-channel scale ``[1, k]`` for down.
        x_scale: Per-token activation scale ``[m, 1]`` float32; the output is
            already multiplied by it.
        activation: Gate-branch activation name.
        tile_i: Intermediate-dimension tile width per grid step.
        interpret: Pallas interpret mode. Note the CPU backend rejects int4
            dot_generals, so interpret mode only works on TPU hosts.

    Returns:
        ``[m, k]`` bf16 — the finished MLP output in real units.

    Raises:
        ValueError: On dtype/shape mismatches.
    """
    if x4.dtype != jnp.int4:
        raise ValueError(f"fused_mlp_w4a4 expects int4 activations, got {x4.dtype}.")
    if not interpret and not supports_int4_mxu():
        raise NotImplementedError(
            "fused_mlp_w4a4 requires native int4 x int4 MXU support, which this TPU "
            "generation does not report. Use the XLA fused_mlp path instead."
        )
    m, k_dim = x4.shape
    packed_k, i_dim = gate_packed.shape
    if packed_k * 2 != k_dim:
        raise ValueError(f"gate_packed has {packed_k} packed rows; expected {k_dim // 2}.")
    if up_packed.shape != (packed_k, i_dim):
        raise ValueError(f"up_packed shape {up_packed.shape} != gate_packed shape {(packed_k, i_dim)}.")
    if down_packed.shape != (i_dim // 2, k_dim):
        raise ValueError(f"down_packed shape {down_packed.shape}; expected {(i_dim // 2, k_dim)}.")
    if i_dim % tile_i:
        raise ValueError(f"tile_i={tile_i} must divide I={i_dim}.")
    if activation not in _KERNEL_ACTIVATIONS:
        raise ValueError(f"Unknown activation {activation!r}; supported: {sorted(_KERNEL_ACTIVATIONS)}.")

    gate_scale = gate_scale.reshape(1, i_dim).astype(jnp.float32)
    up_scale = up_scale.reshape(1, i_dim).astype(jnp.float32)
    down_scale = down_scale.reshape(1, k_dim).astype(jnp.float32)
    x_scale = x_scale.reshape(m, 1).astype(jnp.float32)

    grid = (i_dim // tile_i,)
    return pl.pallas_call(
        functools.partial(_kernel, activation=activation),
        grid=grid,
        in_specs=[
            pl.BlockSpec((m, k_dim), lambda j: (0, 0)),
            pl.BlockSpec((m, 1), lambda j: (0, 0)),
            pl.BlockSpec((packed_k, tile_i), lambda j: (0, j)),
            pl.BlockSpec((packed_k, tile_i), lambda j: (0, j)),
            pl.BlockSpec((tile_i // 2, k_dim), lambda j: (j, 0)),
            pl.BlockSpec((1, tile_i), lambda j: (0, j)),
            pl.BlockSpec((1, tile_i), lambda j: (0, j)),
            pl.BlockSpec((1, k_dim), lambda j: (0, 0)),
        ],
        out_specs=pl.BlockSpec((m, k_dim), lambda j: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((m, k_dim), jnp.bfloat16),
        scratch_shapes=[pltpu.VMEM((m, k_dim), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            # The default scoped-VMEM limit (~16MB) caps tile_i at 512;
            # raising it allows 1024-2048. Measured effect is small (~2%)
            # because the kernel is ingest-bound, not step-bound — but the
            # cap should be a measured choice, not an inherited default.
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
        ),
        interpret=interpret,
    )(x4, x_scale, gate_packed, up_packed, down_packed, gate_scale, up_scale, down_scale)

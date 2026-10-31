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

"""Dense bf16 fused MLP: ``down(act(gate(x)) * up(x))`` in one dispatch.

Measured motivation (TPU v5p, Qwen3.5-4B shapes K=2560 I=9216, 2026-08-08):

* vs XLA's fused-weight composition the kernel runs at 94-96% of the bf16
  compute roofline (XLA: 90-92%) — 1.08-1.09x at m >= 4096, 1.01x at m=512;
* at decode m (<~256) XLA's composition sits at the DMA roofline and the
  extra ``pallas_call`` dispatch makes this kernel a 0.8x LOSER — callers
  must route small m to the XLA composition;
* consuming the **fused** ``[K, 2I]`` gate_up weight directly (the same
  array passed twice with column-offset index maps) is load-bearing: a
  split-weights variant materializes ~2x weight bytes per layer and loses
  its advantage at the model level.

The hidden activations never touch HBM. With ``save_preacts=True`` the
kernel additionally streams out the bf16 pre-activation gate/up tiles —
training residuals that make a recompute-free ``custom_vjp`` structurally
identical to XLA's saved-residual backward (a recompute backward measured
~2% slower at the 4B model level).

Layout: ``gate_up`` must be the **local concat** layout ``[gate | up]``.
This is exactly what each device holds under EasyDeL's TP-interleaved fused
layout (the interleave exists so every tp rank's shard is a plain concat
pair), so tensor-parallel column shards feed this kernel unmodified.
"""

from __future__ import annotations

import functools
import typing as tp
from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.fused_mlp import resolve_mlp_combine

__all__ = ["fused_mlp_bf16_pallas"]

_KERNEL_ACTIVATIONS: dict[str, tp.Callable[[jax.Array], jax.Array]] = {
    "silu": jax.nn.silu,
    "gelu": partial(jax.nn.gelu, approximate=False),
    "gelu_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
}


def _kernel(x_ref, gate_ref, up_ref, down_ref, out_ref, acc_ref, *, activation: str, act_params: tuple):
    """One (m-tile, I-tile) step; gate_ref/up_ref are two views of gate_up.

    Args:
        x_ref: ``[tile_m, K]`` bf16 activations.
        gate_ref: ``[K, tile_i]`` gate columns (fused-array view).
        up_ref: ``[K, tile_i]`` up columns (fused-array view).
        down_ref: ``[tile_i, K]`` down rows.
        out_ref: ``[tile_m, K]`` output, written on the last I-tile.
        acc_ref: VMEM ``[tile_m, K]`` float32 accumulator.
        activation: Kernel activation-table key.
    """
    i_step = pl.program_id(1)
    total_i = pl.num_programs(1)

    x = x_ref[...]
    gate = jnp.dot(x, gate_ref[...], preferred_element_type=jnp.float32)
    up = jnp.dot(x, up_ref[...], preferred_element_type=jnp.float32)
    hidden = resolve_mlp_combine(activation, act_params)(gate, up).astype(jnp.bfloat16)
    partial = jnp.dot(hidden, down_ref[...], preferred_element_type=jnp.float32)

    @pl.when(i_step == 0)
    def _():
        acc_ref[...] = partial

    @pl.when(i_step > 0)
    def _():
        acc_ref[...] += partial

    @pl.when(i_step == total_i - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def _kernel_save_preacts(
    x_ref, gate_ref, up_ref, down_ref, out_ref, g_out_ref, u_out_ref, acc_ref, *, activation: str, act_params: tuple
):
    """As :func:`_kernel`, additionally writing bf16 pre-activation tiles."""
    i_step = pl.program_id(1)
    total_i = pl.num_programs(1)

    x = x_ref[...]
    gate = jnp.dot(x, gate_ref[...], preferred_element_type=jnp.float32)
    up = jnp.dot(x, up_ref[...], preferred_element_type=jnp.float32)
    g_out_ref[...] = gate.astype(g_out_ref.dtype)
    u_out_ref[...] = up.astype(u_out_ref.dtype)
    hidden = resolve_mlp_combine(activation, act_params)(gate, up).astype(jnp.bfloat16)
    partial = jnp.dot(hidden, down_ref[...], preferred_element_type=jnp.float32)

    @pl.when(i_step == 0)
    def _():
        acc_ref[...] = partial

    @pl.when(i_step > 0)
    def _():
        acc_ref[...] += partial

    @pl.when(i_step == total_i - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@kernel_registry.register("fused_mlp_bf16", Platform.PALLAS, Backend.TPU)
@functools.partial(
    jax.jit, static_argnames=("activation", "act_params", "tile_m", "tile_i", "save_preacts", "interpret")
)
def fused_mlp_bf16_pallas(
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
    """Fused dense-bf16 MLP block in one dispatch.

    Args:
        x: Activations ``[m, k]``, floating dtype (computed in bf16 MXU dots
            with float32 accumulation, matching the XLA schedule).
        gate_up: Fused weight ``[k, 2i]`` in local concat layout
            ``[gate | up]`` (each TP rank's shard of the interleaved layout
            is exactly this).
        w_down: Down projection ``[i, k]``.
        activation: Gate-branch activation name.
        tile_m: Rows per grid step; must divide ``m`` (clamped to ``m``).
        tile_i: Intermediate columns per grid step; must divide ``i``.
            Semantically free (pure tiling) — unlike the W4A4 kernel.
        save_preacts: Also return ``(gate_pre, up_pre)`` bf16 ``[m, i]`` —
            the training residuals for a recompute-free backward.
        interpret: Pallas interpret mode.

    Returns:
        ``[m, k]`` in ``x``'s dtype, or ``(out, gate_pre, up_pre)`` when
        ``save_preacts``.

    Raises:
        ValueError: On indivisible tiling or unknown activation.
    """
    m, k_dim = x.shape
    i_dim = gate_up.shape[1] // 2
    tile_m = min(tile_m, m)
    if m % tile_m or i_dim % tile_i:
        raise ValueError(f"tile_m={tile_m} must divide m={m}; tile_i={tile_i} must divide I={i_dim}.")
    if w_down.shape != (i_dim, k_dim):
        raise ValueError(f"w_down shape {w_down.shape}; expected {(i_dim, k_dim)}.")
    resolve_mlp_combine(activation, act_params)  # validates name + arity

    i_tiles = i_dim // tile_i
    grid = (m // tile_m, i_tiles)
    x_spec = pl.BlockSpec((tile_m, k_dim), lambda im, ii: (im, 0))
    gate_spec = pl.BlockSpec((k_dim, tile_i), lambda im, ii: (0, ii))
    up_spec = pl.BlockSpec((k_dim, tile_i), lambda im, ii, i_tiles=i_tiles: (0, i_tiles + ii))
    down_spec = pl.BlockSpec((tile_i, k_dim), lambda im, ii: (ii, 0))
    out_spec = pl.BlockSpec((tile_m, k_dim), lambda im, ii: (im, 0))
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary"),
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
    )

    if not save_preacts:
        return pl.pallas_call(
            functools.partial(_kernel, activation=activation, act_params=act_params),
            grid=grid,
            in_specs=[x_spec, gate_spec, up_spec, down_spec],
            out_specs=out_spec,
            out_shape=jax.ShapeDtypeStruct((m, k_dim), x.dtype),
            scratch_shapes=[pltpu.VMEM((tile_m, k_dim), jnp.float32)],
            compiler_params=compiler_params,
            interpret=interpret,
        )(x, gate_up, gate_up, w_down)

    preact_spec = pl.BlockSpec((tile_m, tile_i), lambda im, ii: (im, ii))
    return pl.pallas_call(
        functools.partial(_kernel_save_preacts, activation=activation, act_params=act_params),
        grid=grid,
        in_specs=[x_spec, gate_spec, up_spec, down_spec],
        out_specs=[out_spec, preact_spec, preact_spec],
        out_shape=[
            jax.ShapeDtypeStruct((m, k_dim), x.dtype),
            jax.ShapeDtypeStruct((m, i_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((m, i_dim), jnp.bfloat16),
        ],
        scratch_shapes=[pltpu.VMEM((tile_m, k_dim), jnp.float32)],
        compiler_params=compiler_params,
        interpret=interpret,
    )(x, gate_up, gate_up, w_down)

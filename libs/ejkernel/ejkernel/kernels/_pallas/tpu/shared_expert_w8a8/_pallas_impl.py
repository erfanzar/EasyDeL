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


"""Pallas TPU W8A8 shared expert for a few decode rows (``shared_expert_w8a8``).

Two ``pallas_call`` stages, each over a ``(cores "parallel", steps "arbitrary")`` grid of output-channel tiles, so
both TensorCores of a megacore chip work and no partial result is combined:

* gate/up: per-row int8 quantization of ``rows`` (once per core, into VMEM scratch), int8 x int8 -> int32 dots with
  the gate and the up tile of the same channels, the epilogue ``(acc * row_scale) * channel_scale``, a cast to the
  rows' dtype, then the float32 ``combine`` -> float32 hidden ``[r, I]``.
* down: per-row int8 quantization of the hidden (row absmax of the float32 hidden, codes of the hidden rounded to the
  rows' dtype), int8 dot with the down tile, the same epilogue -> float32 ``[r, H]``.

The down quantization reproduces what XLA compiles for the reference composition in one program (measured on v5p: it
takes the row absmax before the rounding to the rows' dtype); every other step is elementwise, a row max, or an
integer dot, so the result is bitwise equal to the XLA reference. Tiles come from the VMEM capacity.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
_INT8_SUBLANES = 32
_LANES = 128


#: Row bound of the kernel (decode: ceil(tokens / ep) rows; larger inputs belong to the XLA path).
MAX_ROWS = 64


def _tpu_info() -> tuple[int, int]:
    """``(vmem bytes, TensorCores)`` of the local chip (conservative defaults when unknown)."""
    try:
        info = pltpu.get_tpu_info()
        return int(info.vmem_capacity_bytes), max(1, int(info.num_cores))
    except Exception:  # no TPU info on this backend
        return 16 << 20, 1


def _codes(x32: jax.Array, amax_src: jax.Array, bits: int = 8) -> tuple[jax.Array, jax.Array]:
    """ejkernel ``quantize_rows`` (nearest-even half ties, clip) in float32, row absmax from ``amax_src``."""
    bound = (1 << (bits - 1)) - 1
    amax = jnp.max(jnp.abs(amax_src), axis=1, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / bound)
    normalized = x32 / scale
    rounded = jnp.round(normalized)
    lower = jnp.floor(jnp.abs(normalized))
    is_half = (jnp.abs(x32) * (2 * bound)) == ((2 * lower + 1) * amax)
    parity = (lower.astype(jnp.int32) & 1).astype(F32)
    rounded = jnp.where(is_half, jnp.sign(x32) * (lower + parity), rounded)
    return jnp.clip(rounded, -bound, bound), scale


def _quantize_to_scratch(x32, amax_src, pad_ref, q_ref, s_ref):
    """Codes into an int8 scratch padded to the int8 sublane tile (pad rows 0), row scales into ``s_ref``."""
    r = x32.shape[0]
    codes, scale = _codes(x32, amax_src)
    if pad_ref.shape[0] != r:
        pad_ref[...] = jnp.zeros(pad_ref.shape, F32)
    pad_ref[pl.ds(0, r), :] = codes
    q_ref[...] = pad_ref[...].astype(jnp.int32).astype(jnp.int8)
    s_ref[...] = scale


def _project(q, s, w_ref, cs_ref, r):
    acc = jnp.dot(q, w_ref[...], preferred_element_type=jnp.int32)[:r]
    return (acc.astype(F32) * s) * cs_ref[...].astype(F32)


def _gate_up_kernel(x_ref, g_ref, u_ref, gs_ref, us_ref, o_ref, pad_ref, q_ref, s_ref, *, combine, rows_dtype):
    @pl.when(pl.program_id(1) == 0)
    def _():
        x32 = x_ref[...].astype(F32)
        _quantize_to_scratch(x32, x32, pad_ref, q_ref, s_ref)

    r = o_ref.shape[0]
    q, s = q_ref[...], s_ref[...]
    gate = _project(q, s, g_ref, gs_ref, r).astype(rows_dtype).astype(F32)
    up = _project(q, s, u_ref, us_ref, r).astype(rows_dtype).astype(F32)
    o_ref[...] = combine(gate, up).astype(F32)


def _down_kernel(h_ref, w_ref, cs_ref, o_ref, pad_ref, q_ref, s_ref, *, rows_dtype):
    @pl.when(pl.program_id(1) == 0)
    def _():
        h32 = h_ref[...]
        _quantize_to_scratch(h32.astype(rows_dtype).astype(F32), h32, pad_ref, q_ref, s_ref)

    o_ref[...] = _project(q_ref[...], s_ref[...], w_ref, cs_ref, o_ref.shape[0])


def _tile(n: int, bytes_per_col: int, cores: int, vmem: int, min_steps: int = 2) -> tuple[int, int, int]:
    """``(tile, cores, steps)``: the largest lane-aligned tile with ``min_steps`` steps per core whose double buffer
    fits a quarter of VMEM (fewer cores when ``n`` does not split)."""
    budget = vmem // 4
    for c in (cores, 1):
        ok = [t for t in range(_LANES, n // c + 1, _LANES) if n % (c * t) == 0 and 2 * bytes_per_col * t <= budget]
        if ok:
            deep = [t for t in ok if n // (c * t) >= min_steps]
            tn = max(deep) if deep else min(ok)
            return tn, c, n // (c * tn)
    return n, 1, 1


def _vmem_need(r: int, k: int, tn: int, w_cols: int, in_bytes: int) -> int:
    mp = -(-r // _INT8_SUBLANES) * _INT8_SUBLANES
    return 2 * (w_cols * k * tn + r * k * in_bytes + w_cols * tn * 4 + r * tn * 4) + mp * k * 5


def unsupported_reason(rows, gate_up_codes, gate_up_scales, down_codes, down_scales) -> str | None:
    """Why the Pallas kernel cannot take these operands (``None``: supported)."""
    if rows.ndim != 2 or not jnp.issubdtype(rows.dtype, jnp.floating):
        return "rows must be a floating [r, H] array"
    r, h = rows.shape
    if gate_up_codes.ndim != 2 or down_codes.ndim != 2:
        return "codes must be 2-D"
    inter = down_codes.shape[0]
    if gate_up_codes.shape != (h, 2 * inter) or down_codes.shape[1] != h:
        return f"expected gate_up [H, 2I] and down [I, H], got {gate_up_codes.shape} and {down_codes.shape}"
    if gate_up_codes.dtype != jnp.int8 or down_codes.dtype != jnp.int8:
        return "codes must be int8"
    if gate_up_scales.size != 2 * inter or down_scales.size != h:
        return "scales must hold one value per output channel"
    if h % _LANES or inter % _LANES:
        return f"H ({h}) and I ({inter}) must be multiples of {_LANES}"
    if r < 1 or r > MAX_ROWS:
        return f"{r} rows (supported: 1 to {MAX_ROWS})"
    vmem, cores = _tpu_info()
    tn, _, _ = _tile(inter, 2 * h, cores, vmem)
    need_a = _vmem_need(r, h, tn, 2, rows.dtype.itemsize)
    tn, _, _ = _tile(h, inter, cores, vmem)
    need_b = _vmem_need(r, inter, tn, 1, 4)
    if max(need_a, need_b) > vmem * 3 // 4:
        return f"{r} rows do not fit VMEM ({max(need_a, need_b)} of {vmem} bytes)"
    return None


def _params(need: int, vmem: int):
    return pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary"),
        vmem_limit_bytes=int(min(vmem * 3 // 4, max(32 << 20, 2 * need))),
    )


def shared_expert_w8a8_pallas(
    rows: jax.Array,
    gate_up_codes: jax.Array,
    gate_up_scales: jax.Array,
    down_codes: jax.Array,
    down_scales: jax.Array,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Run both stages; returns float32 ``[r, H]``. See the module docstring."""
    reason = unsupported_reason(rows, gate_up_codes, gate_up_scales, down_codes, down_scales)
    if reason is not None:
        raise ValueError(f"shared_expert_w8a8 (Pallas TPU): {reason}")
    r, h = rows.shape
    inter = down_codes.shape[0]
    vmem, cores = _tpu_info()
    mp = -(-r // _INT8_SUBLANES) * _INT8_SUBLANES
    gus = gate_up_scales.reshape(1, -1).astype(F32)
    dns = down_scales.reshape(1, -1).astype(F32)

    tn, c, s = _tile(inter, 2 * h, cores, vmem)
    na = inter // tn

    def col(off, s=s):
        return lambda i, j: (0, i * s + j + off)

    hidden = pl.pallas_call(
        functools.partial(_gate_up_kernel, combine=combine, rows_dtype=rows.dtype),
        grid=(c, s),
        in_specs=[
            pl.BlockSpec((r, h), lambda i, j: (0, 0)),
            pl.BlockSpec((h, tn), col(0)),
            pl.BlockSpec((h, tn), col(na)),
            pl.BlockSpec((1, tn), col(0)),
            pl.BlockSpec((1, tn), col(na)),
        ],
        out_specs=pl.BlockSpec((r, tn), col(0)),
        out_shape=jax.ShapeDtypeStruct((r, inter), F32),
        scratch_shapes=[pltpu.VMEM((mp, h), F32), pltpu.VMEM((mp, h), jnp.int8), pltpu.VMEM((r, 1), F32)],
        compiler_params=_params(_vmem_need(r, h, tn, 2, rows.dtype.itemsize), vmem),
        name="ejkernel_shared_expert_w8a8_gate_up",
    )(rows, gate_up_codes, gate_up_codes, gus, gus)

    tn, c, s = _tile(h, inter, cores, vmem)

    def col_b(i, j, s=s):
        return (0, i * s + j)

    return pl.pallas_call(
        functools.partial(_down_kernel, rows_dtype=rows.dtype),
        grid=(c, s),
        in_specs=[
            pl.BlockSpec((r, inter), lambda i, j: (0, 0)),
            pl.BlockSpec((inter, tn), col_b),
            pl.BlockSpec((1, tn), col_b),
        ],
        out_specs=pl.BlockSpec((r, tn), col_b),
        out_shape=jax.ShapeDtypeStruct((r, h), F32),
        scratch_shapes=[pltpu.VMEM((mp, inter), F32), pltpu.VMEM((mp, inter), jnp.int8), pltpu.VMEM((r, 1), F32)],
        compiler_params=_params(_vmem_need(r, inter, tn, 1, 4), vmem),
        name="ejkernel_shared_expert_w8a8_down",
    )(hidden, down_codes, dns)

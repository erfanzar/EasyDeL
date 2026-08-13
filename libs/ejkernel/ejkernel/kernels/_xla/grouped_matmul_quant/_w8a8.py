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

"""W8A8 grouped matmul on XLA's native int8 ragged_dot.

TPU's ragged_dot custom call accepts int8 operands with an int32 accumulator
and keeps its near-roofline expert-streaming pipeline — so the W8A8 expert
path needs no custom kernel: quantize activations per row, run the int8
ragged_dot, and apply the (row activation scale) x (per-expert per-channel
weight scale) epilogue. Weight HBM traffic halves versus bf16, which is the
dominant cost of decode-shaped MoE steps.
"""

import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, Int8

from ..._registry import Backend, Platform, kernel_registry
from ..grouped_matmul._xla_impl_fwd import grouped_matmul as _grouped_matmul_impl


@kernel_registry.register("grouped_matmul_w8a8", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmul_w8a8(
    lhs: Float[Array, "m k"],
    rhs_codes: Int8[Array, "num_groups k n"],
    rhs_scales: Float[Array, "num_groups 1 n"],
    group_sizes: Int[Array, "num_groups"],
    preferred_element_type: DTypeLike = jnp.bfloat16,
    tiling: tuple[int, int, int] | None = None,
) -> Float[Array, "m n"]:
    """Grouped matmul with int8 weights and runtime-int8 activations.

    For each group i over its row slice R_i:
    ``out[R_i] = (q(lhs[R_i]) @ codes[i]) * lhs_scale[R_i] * scales[i]``
    where ``q`` is per-row symmetric int8 quantization. The int8 x int8
    ragged_dot accumulates exactly in int32; the only approximation is the
    activation/weight quantization itself.

    Args:
        lhs: Activations ``[m, k]`` (bf16/f32), rows sorted by group.
        rhs_codes: Per-group weight codes ``[num_groups, k, n]`` int8.
        rhs_scales: Per-group per-output-channel scales ``[num_groups, 1, n]``
            float32.
        group_sizes: Rows per group ``[num_groups]`` int32, summing to ``m``.
        preferred_element_type: Output dtype.

    Returns:
        Output ``[m, n]`` in ``preferred_element_type``.
    """
    m = lhs.shape[0]
    x32 = lhs.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(x32), axis=1, keepdims=True)
    act_scales = jnp.where(absmax == 0, 1.0, absmax / 127.0)
    lhs_codes = jnp.clip(jnp.round(x32 / act_scales), -127, 127).astype(jnp.int8)

    acc = _grouped_matmul_impl(
        lhs_codes,
        rhs_codes,
        group_sizes,
        jnp.int32,
        tiling,
        None,
        None,
        False,
        False,
        lax.Precision.DEFAULT,
    )

    # Per-row weight-scale view: rows are sorted by group, so repeating each
    # group's scale row by its group size aligns scales with output rows.
    w_scales_rows = jnp.repeat(
        rhs_scales[:, 0, :],
        group_sizes,
        axis=0,
        total_repeat_length=m,
    )
    out = acc.astype(jnp.float32) * act_scales * w_scales_rows
    return out.astype(preferred_element_type)


__all__ = ("grouped_matmul_w8a8",)

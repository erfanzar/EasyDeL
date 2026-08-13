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

"""XLA reference/fallback for the quantized grouped matmul.

Dequantizes the int8 codes to the activation dtype and delegates to the XLA
``grouped_matmul`` (ragged_dot). Materializes the dequantized weights, so it
carries no bandwidth win — it exists for correctness on every backend and as
the parity reference for the Pallas TPU kernel.
"""

import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, Int8

from ..._registry import Backend, Platform, kernel_registry
from ..grouped_matmul._interface import grouped_matmul as _xla_grouped_matmul


@kernel_registry.register("grouped_matmul_quant", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmul_quant(
    lhs: Float[Array, "m k"],
    rhs_codes: Int8[Array, "num_groups k n"] | Int8[Array, "num_groups n k"],
    rhs_scales: Float[Array, "num_groups 1 n"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    preferred_element_type: DTypeLike = jnp.bfloat16,
    tiling: tuple[int, int, int] | None = (128, 128, 128),
    group_offset: Int[Array, "..."] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> Float[Array, "m n"]:
    """Dequantize-then-ragged_dot reference for ``grouped_matmul_quant``.

    Args:
        lhs: Activations ``[m, k]``.
        rhs_codes: Weight codes ``[num_groups, k, n]`` int8
            (``[num_groups, n, k]`` when ``transpose_rhs``).
        rhs_scales: Scales ``[num_groups, 1, n]`` float32.
        group_sizes: Rows per group ``[num_groups]`` int32.
        preferred_element_type: Output dtype.
        tiling: Forwarded to the XLA grouped matmul as a hint.
        group_offset: Starting group for sharded execution.
        transpose_rhs: Whether codes are ``[g, n, k]``.
        interpret: Ignored (API compatibility).

    Returns:
        Output ``[m, n]``.
    """
    scales = rhs_scales.astype(jnp.float32)
    if transpose_rhs:
        # codes [g, n, k] * scales [g, 1, n] -> broadcast over k on axis -1
        rhs = rhs_codes.astype(jnp.float32) * scales.transpose(0, 2, 1)
    else:
        rhs = rhs_codes.astype(jnp.float32) * scales
    rhs = rhs.astype(lhs.dtype)
    return _xla_grouped_matmul(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        group_offset=group_offset,
        existing_out=None,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )


__all__ = ("grouped_matmul_quant",)

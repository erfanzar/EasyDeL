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

"""Block-diagonal grouped linear projection."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jaxtyping import Array, Float
from spectrax.common_types import ColumnWise

from easydel.infra.sharding import sharding_for_layout

__all__ = ("GroupedLinear",)


class GroupedLinear(spx.Module):
    """Block-diagonal grouped linear: group ``g`` of the input maps through its own ``[in, out]`` block.

    Used for DeepSeek-V4 / V4.1 ``o_a_proj``. The HF weight has shape ``[n_groups * out_per_group, in_per_group]``;
    after EasyDeL's standard 2-D transpose on conversion the kernel is ``[in_per_group, n_groups * out_per_group]`` and
    group ``g`` owns columns ``[g * out, (g + 1) * out)``.
    """

    def __init__(
        self,
        in_features_per_group: int,
        out_features_per_group: int,
        n_groups: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        kernel_init: tp.Callable | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the grouped linear.

        Args:
            in_features_per_group: Per-group input width.
            out_features_per_group: Per-group output width.
            n_groups: Number of groups.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            kernel_init: Weight initializer; defaults to LeCun normal.
            rngs: Random number generators.
        """
        self.in_features_per_group = in_features_per_group
        self.out_features_per_group = out_features_per_group
        self.n_groups = n_groups
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        if kernel_init is None:
            kernel_init = jax.nn.initializers.lecun_normal()
        # Column-parallel like ColumnParallelLinear: group g owns the contiguous columns [g*out, (g+1)*out), so a
        # tp split of the column axis hands every shard whole groups, matching the tp-sharded heads feeding it. A
        # replicated weight made XLA slice and relayout each shard's groups every step (measured on TPU v5, V4.1
        # decode cc=8, 40 layers: 120 weight dynamic-slices + 40 copies, 1.63 -> 1.10 ms, weight 2.68 -> 0.67 GB/chip,
        # bit-identical).
        self.weight = spx.Parameter(
            kernel_init(rngs.param, (in_features_per_group, n_groups * out_features_per_group), param_dtype),
            sharding=sharding_for_layout(ColumnWise),
        )

    def forward(self, x: Float[Array, "batch seq groups in_per_group"]) -> Float[Array, "batch seq groups out"]:
        """Apply the per-group projection.

        Args:
            x: Grouped input ``[B, S, n_groups, in_per_group]``.

        Returns:
            Grouped output ``[B, S, n_groups, out_per_group]``.
        """
        kernel = self.weight.value.astype(self.dtype if self.dtype is not None else x.dtype)
        kernel = kernel.reshape(self.in_features_per_group, self.n_groups, self.out_features_per_group)
        x = x.astype(kernel.dtype)

        # This layer contracts its own parameter with an einsum instead of delegating to ParallelLinear, so nothing
        # intercepts the dot_general on its behalf: it looks its own quantization rule up, exactly as ParallelLinear
        # does. Without this the module is stamped with a plan and then silently ignores it. Contraction is ``i``:
        # axis 0 of the kernel, trailing axis of ``x``.
        quant_rule = spx.quantization.rule_for(self, "dot_general")
        if quant_rule is not None:
            kernel = spx.quantization.fake_quant(kernel, rule=quant_rule, contracting_axes=(0,), is_weight=True)
            x = spx.quantization.fake_quant(x, rule=quant_rule, contracting_axes=(x.ndim - 1,), is_weight=False)

        return jnp.einsum("bsgi,igr->bsgr", x, kernel, precision=self.precision)

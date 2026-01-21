# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from eformer.ops.quantization.quantization_functions import (
    nf4xf32_to_f32,
    quantize_and_pack_nf4,
    quantize_int8,
)
from flax import nnx as nn
from flax.nnx import rnglib
from flax.nnx.nn import initializers
from flax.typing import Dtype, Initializer, PrecisionLike

from ..quants._quants import QuantizationType, quantize
from ._linear import ColumnParallelLinear, RowParallelLinear
from ._utils import nf4_qmm_jax

if tp.TYPE_CHECKING:
    from ..quants._quants import QuantizationConfig


Array = jax.Array
Axis = int
Size = int

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


class ParallelLinearQuantized(nn.Module):
    _direction: tp.Literal["row", "column"] | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        config: QuantizationConfig,
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.config = config
        self.rngs = rngs

        kernel = kernel_init(rngs.params(), (in_features, out_features), param_dtype)
        quant_kernel, quant_scales = self._quantize_array(kernel)

        self.quant_kernel = nn.Param(quant_kernel)
        self.quant_scales = nn.Param(quant_scales)

        if use_bias:
            self.bias = nn.Param(bias_init(rngs.params(), (out_features,), param_dtype))

    def _quantize_array(self, array: jax.Array):
        if self.config.dtype == QuantizationType.INT8:
            wq, scale = quantize_int8(array, axis=0)
            return wq, scale
        elif self.config.dtype == QuantizationType.NF4:
            wq, scale = quantize_and_pack_nf4(array, self.config.block_size)
        elif self.config.dtype in {QuantizationType.MXFP4, QuantizationType.MXFP8, QuantizationType.NVFP8}:
            wq = quantize(array=array, config=self.config, simulate=False)
            scale = None
        else:
            raise ValueError("given dtype is not Supported!")
        return wq, scale

    def _quantize_runtime(self, array: jax.Array):
        if self.config.runtime_dtype is not None:
            return quantize(
                array=array,
                dtype=self.config.runtime_dtype,
                block_size=self.config.block_size,
                simulate=False,
            )
        return array

    def _dequantize_array(self, wq: jax.Array, scale: jax.Array):
        if self.config.dtype == QuantizationType.INT8:
            array = wq * scale
        elif self.config.dtype == QuantizationType.NF4:
            high = (wq >> 4) & 0xF
            low = wq & 0xF
            unpacked = jnp.stack([high, low], axis=-1)
            *batch_dims, num_blocks, _ = wq.shape
            unpacked = unpacked.reshape(*batch_dims, num_blocks, self.config.block_size)
            dequantized = nf4xf32_to_f32(unpacked)
            array = dequantized * scale[..., None]
            array = array.reshape(*batch_dims, num_blocks * self.config.block_size)
        elif self.config.dtype in {QuantizationType.MXFP4, QuantizationType.MXFP8, QuantizationType.NVFP8}:
            array = wq.astype(self.param_dtype)
        else:
            raise ValueError("given dtype is not Supported for dequantization!")
        return array

    def from_quantized(self, rngs: rnglib.Rngs | None = None) -> RowParallelLinear | ColumnParallelLinear:
        """Convert this Linear8bit module back to a regular Linear module."""
        if rngs is None:
            rngs = nn.Rngs(0)
        if self._direction == "row":
            linear_class = RowParallelLinear
        elif self._direction == "column":
            linear_class = RowParallelLinear
        else:
            raise ValueError(
                "unknown direction detected Try To use module with Known "
                "direction in ur impls to stop getting such errors."
            )
        linear = nn.eval_shape(
            lambda r: linear_class(
                in_features=self.in_features,
                out_features=self.out_features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                rngs=r,
            ),
            rngs,
        )

        dequantized_kernel = self._dequantize_array(self.quant_kernel, self.quant_scales)
        linear.kernel = nn.Param(dequantized_kernel)

        if self.use_bias:
            linear.bias = nn.Param(self.bias.value)

        return linear

    def restage(self, kernel: jax.Array, bias: jax.Array | None):
        wq, scale = self._quantize_array(kernel)
        self.quant_kernel.value = wq
        self.quant_scales.value = scale
        if bias is not None:
            self.bias.copy_from = bias
        return self

    @jax.named_scope("easydel-linear-int8-call")
    def __call__(self, inputs: Array, w: Array | None = None) -> Array:
        """Applies a quantized linear transformation to the inputs along the last dimension."""
        inputs_gt_one_dim: bool = inputs.ndim > 1
        subscript: str
        kws = dict(precision=self.precision, optimize=True)
        if inputs_gt_one_dim:
            subscript = "...ik,...kj->...ij"
        else:
            subscript = "...k,...kj->...j"

        def normal_flow(inputs):
            kernel = self._dequantize_array(self.quant_kernel, self.quant_scales)

            if self.dtype is not None:
                inputs = inputs.astype(self.dtype)
                kernel = kernel.astype(self.dtype)

            return jnp.einsum(subscript, inputs, kernel, **kws)

        if w is not None:
            kernel = w
            if self.dtype is not None:
                inputs = inputs.astype(self.dtype)
                kernel = kernel.astype(self.dtype)

            out = jnp.einsum(subscript, inputs, kernel, **kws)
        elif self.config.dtype == QuantizationType.NF4:
            out = nf4_qmm_jax(inputs, self.quant_kernel, self.quant_scales)
        elif self.config.dtype in {QuantizationType.MXFP4, QuantizationType.MXFP8, QuantizationType.NVFP8}:
            kernel = self.quant_kernel.value

            if self.config.runtime_dtype is None:
                kernel = kernel.astype(self.dtype)
            else:
                inputs = self._quantize_runtime(inputs)

            out = jnp.einsum(subscript, inputs, kernel, **kws)
            out = out.astype(self.dtype)

        else:
            out = normal_flow(inputs)

        if self.use_bias and self.bias.value is not None:
            bias = self.bias.value
            if self.dtype is not None:
                bias = bias.astype(self.dtype)
            out = out + jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))

        return out

    @property
    def wqdtype(self) -> QuantizationType:
        return self.config.dtype

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"wqdtype={self.wqdtype}"
            ")"
        )

    def __str__(self):
        return self.__repr__()


class RowParallelLinearQuantized(ParallelLinearQuantized):
    _direction: tp.Literal["row"] = "row"


class ColumnParallelLinearQuantized(ParallelLinearQuantized):
    _direction: tp.Literal["column"] = "column"

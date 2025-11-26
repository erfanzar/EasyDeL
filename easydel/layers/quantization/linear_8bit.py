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

import jax
import jax.numpy as jnp
from eformer.ops.quantization import Array8B
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import initializers
from flax.typing import DotGeneralT, Dtype, Initializer, PrecisionLike
from jax import lax

from .base_quant import QauntModule

Array = jax.Array
Axis = int
Size = int

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


class Linear8bit(QauntModule):
    """An 8-bit quantized version of the linear transformation applied over the last dimension of the input.

    Uses eformer's Array8B implicit array for efficient 8-bit quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        do_init: bool = False,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        dot_general: DotGeneralT = lax.dot_general,
        rngs: rnglib.Rngs,
    ):
        super().__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        if do_init:
            kernel_key = rngs.params()
            kernel = kernel_init(kernel_key, (in_features, out_features), param_dtype)
            quant_kernel, quant_scales = self._quantize_kernel(kernel)
        else:
            quant_kernel, quant_scales = None, None

        self.quant_kernel = nnx.Param(quant_kernel)
        self.quant_scales = nnx.Param(quant_scales)

        if use_bias and do_init:
            bias_key = rngs.params()
            self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
        else:
            self.bias = nnx.Param(None)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general

    @classmethod
    def from_linear(
        cls,
        linear: nnx.Linear,
        rngs: rnglib.Rngs | None = None,
        **kwargs,
    ) -> Linear8bit:
        """Create a Linear8bit module from a regular Linear module."""
        if rngs is None:
            rngs = nnx.Rngs(0)

        instance = nnx.eval_shape(
            lambda: cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                use_bias=linear.use_bias,
                dtype=linear.dtype,
                param_dtype=linear.param_dtype,
                precision=linear.precision,
                kernel_init=linear.kernel_init,
                bias_init=linear.bias_init,
                dot_general=linear.dot_general,
                rngs=rngs,
            )
        )

        quant_kernel, quant_scales = cls._quantize_kernel(linear.kernel.value)
        instance.quant_kernel = nnx.Param(quant_kernel)
        instance.quant_scales = nnx.Param(quant_scales)

        if linear.use_bias:
            instance.bias = nnx.Param(linear.bias.value)

        return instance

    def to_linear(self, rngs: rnglib.Rngs | None = None) -> nnx.Linear:
        """Convert this Linear8bit module back to a regular Linear module."""
        if rngs is None:
            rngs = nnx.Rngs(0)

        linear = nnx.eval_shape(
            lambda: nnx.Linear(
                in_features=self.in_features,
                out_features=self.out_features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dot_general=self.dot_general,
                rngs=rngs,
            )
        )

        dequantized_kernel = self._dequantize_kernel()
        linear.kernel = nnx.Param(dequantized_kernel)

        if self.use_bias:
            linear.bias = nnx.Param(self.bias.value)

        return linear

    @staticmethod
    def _quantize_kernel(kernel):
        """Quantize the kernel weights using eformer's Array8B."""
        if kernel is None or isinstance(kernel, jax.ShapeDtypeStruct):
            return None, None
        quantized = Array8B.quantize(kernel)
        return quantized.weight, quantized.scale

    def _dequantize_kernel(self):
        """Dequantize the kernel weights."""
        if self.quant_kernel.value is None and self.quant_scales.value is None:
            return None
        elif self.quant_scales.value is None:
            return self.quant_kernel.value

        quantized = Array8B(
            weight=self.quant_kernel.value,
            scale=self.quant_scales.value,
            shape=(self.in_features, self.out_features),
            dtype=self.param_dtype,
        )
        return quantized.materialize()

    @jax.named_scope("easydel-linear-int8-call")
    def __call__(self, inputs: Array) -> Array:
        """Forward pass using 8-bit quantized weights."""
        kernel = self._dequantize_kernel()

        assert kernel is not None, (
            "loaded and dequantized kernel is None, which means it has been loaded from another None Kernel Linear"
        )

        if self.dtype is not None:
            inputs = inputs.astype(self.dtype)
            kernel = kernel.astype(self.dtype)

        out = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias and self.bias.value is not None:
            bias = self.bias.value
            if self.dtype is not None:
                bias = bias.astype(self.dtype)
            out = out + jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))

        return out

    def get_kernel(self):
        """Get the dequantized kernel weights."""
        return self._dequantize_kernel()

    def get_quantized_kernel(self):
        """Get the quantized kernel weights and scales."""
        return self.quant_kernel.value, self.quant_scales.value

    @staticmethod
    def metadata():
        return {"quant_mode": "int8"}

    @staticmethod
    def quantization_mapping():
        return {"kernel": ["quant_kernel", "quant_scales"]}

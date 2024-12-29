# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import initializers
from flax.typing import (
	DotGeneralT,
	Dtype,
	Initializer,
	PrecisionLike,
)
from jax import lax

from .base_quant import QauntModule

Array = jax.Array
Axis = int
Size = int

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


def quantize_8bit(x):
	"""
	Quantize a row of float32 values to 8-bit integers with blockwise scaling.
	"""
	max_val = jnp.amax(jnp.abs(x.astype(jnp.float32)), axis=-1, keepdims=True)
	max_val = jnp.clip(max_val, min=1e-5)
	qscale = max_val / 127
	qweight = jnp.clip(
		jnp.round(x * (1.0 / qscale)),
		min=-128,
		max=127,
	).astype(jnp.int8)
	qscale = qscale.astype(x.dtype)
	return qweight, qscale


def dequantize_8bit(quants, scales):
	"""
	Dequantize 8-bit integers back to values using blockwise scaling.
	"""
	dequantized = quants * scales
	return dequantized


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def quantized_matmul(x, qweight, qscale, transpose_weight=False):
	"""
	Forward pass for 8-bit quantized matrix multiplication.
	"""
	# Dequantize weights
	dequantized = dequantize_8bit(qweight, qscale)

	if transpose_weight:
		dequantized = dequantized.T

	return jnp.matmul(x, dequantized)


def quantized_matmul_fwd(x, qweight, qscale, transpose_weight):
	"""Forward pass that saves required values for backward pass."""

	# Dequantize weights for forward computation
	dequantized = dequantize_8bit(qweight, qscale)

	if transpose_weight:
		dequantized = dequantized.T

	# Compute output
	out = jnp.matmul(x, dequantized)

	# Save values needed for backward pass
	saved = (x, qweight, qscale, dequantized, transpose_weight)
	return out, saved


def quantized_matmul_bwd(transpose_weight, res, grad_output):
	"""
	Backward pass computing gradients for quantized matrix multiplication.
	Args:
	    transpose_weight: Whether weight matrix was transposed
	    res: Saved values from forward pass
	    grad_output: Gradient of loss with respect to output
	"""
	x, qweight, qscale, dequantized, _ = res

	# Gradient with respect to input x
	if transpose_weight:
		grad_x = jnp.matmul(grad_output, dequantized.T)
	else:
		grad_x = jnp.matmul(grad_output, dequantized)

	# Gradient with respect to quantized weights and scales
	if transpose_weight:
		grad_dequant = jnp.matmul(grad_output.T, x)
	else:
		grad_dequant = jnp.matmul(x.T, grad_output)

		# Optimize gradient calculation for scales
	abs_qweight = jnp.abs(qweight)

	# Calculate scaling factors more efficiently
	scale_grad_factor = jnp.where(
		abs_qweight > 0, qweight.astype(grad_dequant.dtype) / (127.0 * abs_qweight), 0.0
	)

	# Compute gradients for scaling factors using einsum for better performance
	grad_qscale = jnp.einsum("ij,ij->i", grad_dequant, scale_grad_factor)[:, jnp.newaxis]

	# Compute gradients for quantized weights, avoiding clipping here
	grad_qweight = grad_dequant * qscale

	# Pack gradients for weight tuple
	grad_weight = (grad_qweight, grad_qscale)

	return grad_x, quantize_8bit(grad_weight)


# Register the custom VJP (Vector-Jacobian Product) rules
quantized_matmul.defvjp(quantized_matmul_fwd, quantized_matmul_bwd)


class Linear8bit(QauntModule):
	"""An 8-bit quantized version of the linear transformation applied over the last dimension of the input."""

	def __init__(
		self,
		in_features: int,
		out_features: int,
		*,
		use_bias: bool = True,
		dtype: tp.Optional[Dtype] = None,
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
			quantized_kernel, scales = self._quantize_kernel(kernel)
		else:
			quantized_kernel, scales = None, None
		# Quantize the kernel

		self.kernel = nnx.Param(quantized_kernel)
		self.scales = nnx.Param(scales)

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
		rngs: tp.Optional[rnglib.Rngs] = None,
		**kwargs,
	) -> "Linear8bit":
		"""
		Create a Linear8bit module from a regular Linear module.

		Args:
				linear: The source Linear module
				rngs: Random number generator state

		Returns:
				A new Linear8bit module with quantized weights
		"""
		if rngs is None:
			rngs = nnx.Rngs(0)
		# Create a new instance with minimal initialization
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

		# Quantize the kernel from the original linear layer
		quantized_kernel, scales = cls._quantize_kernel(linear.kernel.value)

		# Update the parameters
		instance.kernel = nnx.Param(quantized_kernel)
		instance.scales = nnx.Param(scales)

		# Copy the bias if it exists
		if linear.use_bias:
			instance.bias = nnx.Param(linear.bias.value)

		return instance

	def to_linear(self, rngs: tp.Optional[rnglib.Rngs] = None) -> nnx.Linear:
		"""
		Convert this Linear8bit module back to a regular Linear module.

		Args:
				rngs: Random number generator state

		Returns:
				A new Linear module with dequantized weights
		"""
		if rngs is None:
			rngs = nnx.Rngs(0)
		# Create a new Linear instance
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

		# Dequantize the kernel and update the linear layer
		dequantized_kernel = self._dequantize_kernel()
		linear.kernel = nnx.Param(dequantized_kernel)

		# Copy the bias if it exists
		if self.use_bias:
			linear.bias = nnx.Param(self.bias.value)

		return linear

	@staticmethod
	def _quantize_kernel(kernel):
		"""Quantize the kernel weights."""
		if kernel is None or isinstance(kernel, jax.ShapeDtypeStruct):
			return None, None
		quantized, scales = quantize_8bit(kernel)
		return quantized, scales

	def _dequantize_kernel(self):  # in case somebody using tie word embedding.
		"""Dequantize the kernel weights."""
		if self.kernel.value is None and self.scales.value is None:
			return None
		elif self.scales.value is None:
			return self.kernel
		return dequantize_8bit(
			self.kernel.value,
			self.scales.value,
		).astype(self.param_dtype)

	@jax.named_scope("easydel-linear-8bit-call")
	def __call__(self, inputs: Array) -> Array:
		"""Forward pass using custom gradient computation."""
		out = quantized_matmul(
			inputs,
			self.kernel.value,
			self.scales.value,
			transpose_weight=False,
		)
		if self.use_bias:
			out = out + self.bias.value
		return out

	def get_kernel(self):
		"""Get the dequantized kernel weights."""
		return self._dequantize_kernel()

	def get_quantized_kernel(self):
		"""Get the quantized kernel weights and scales."""
		return self.kernel.value, self.scales.value

	@staticmethod
	def metadata():
		return {"quant_mode": "8bit"}

	@staticmethod
	def quantization_mapping():
		return {"kernel": ["kernel", "scales"]}

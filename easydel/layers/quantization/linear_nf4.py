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
from flax.nnx.nn import dtypes, initializers
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
NF4_TABLE = jnp.array(
	[
		-1.0,
		-0.6961928009986877,
		-0.5250730514526367,
		-0.39491748809814453,
		-0.28444138169288635,
		-0.18477343022823334,
		-0.09105003625154495,
		0.0,
		0.07958029955625534,
		0.16093020141124725,
		0.24611230194568634,
		0.33791524171829224,
		0.44070982933044434,
		0.5626170039176941,
		0.7229568362236023,
		1.0,
	],
	dtype=jnp.float32,
)

NF4_BOUNDARIES = jnp.array(
	[
		-float("inf"),
		-0.8480964004993439,
		-0.6106329262256622,
		-0.4599952697753906,
		-0.33967943489551544,
		-0.23460740596055984,
		-0.13791173323988914,
		-0.045525018125772476,
		0.03979014977812767,
		0.1202552504837513,
		0.2035212516784668,
		0.2920137718319893,
		0.3893125355243683,
		0.5016634166240692,
		0.6427869200706482,
		0.8614784181118011,
	],
	dtype=jnp.float32,
)


@partial(jax.jit, static_argnames=["block_size"])
def single_quantize_and_pack_nf4(blocks, block_size=64):
	"""
	Combined quantization and packing for better performance.
	Handles normalization, quantization, and packing in a single operation.
	"""
	# Pad and reshape into blocks
	blocks = blocks.reshape(-1, block_size)

	# Compute absolute maximum for each block
	absmax = jnp.max(jnp.abs(blocks), axis=1)

	# Normalize blocks
	normalized = blocks / absmax[:, None]

	# Quantize using vectorized operations
	quantized = jnp.searchsorted(NF4_BOUNDARIES, normalized.reshape(-1)) - 1

	# Pack pairs efficiently using bit operations
	quantized = quantized.reshape(-1, 2)
	packed = (quantized[:, 0] << 4) | quantized[:, 1]

	return packed.astype(jnp.uint8), absmax


@partial(jax.jit, static_argnames=["block_size"])
def single_dequantize_nf4(packed_values, absmax, block_size):
	"""
	Optimized dequantization combining unpacking and scaling in fewer operations.
	"""
	high = (packed_values >> 4) & 0xF
	low = packed_values & 0xF
	unpacked = jnp.stack([high, low], axis=1).reshape(-1)

	dequantized = NF4_TABLE[unpacked]

	num_blocks = len(absmax)
	dequantized = dequantized.reshape(num_blocks, block_size)
	scaled = dequantized * absmax[:, None]
	return scaled


@partial(jax.jit, static_argnames=["block_size"])
def quantize_and_pack_nf4(blocks, block_size=64):
	if blocks.ndim > 2:
		return jax.vmap(quantize_and_pack_nf4, in_axes=(0, None), out_axes=(0, 0))(
			blocks, block_size
		)
	return single_quantize_and_pack_nf4(blocks, block_size)


@partial(jax.jit, static_argnames=["block_size"])
def dequantize_nf4(packed_values, absmax, block_size):
	if packed_values.ndim > 2:
		return jax.vmap(dequantize_nf4, in_axes=(0, 0, None), out_axes=(0,))(
			packed_values, absmax, block_size
		)
	return single_dequantize_nf4(packed_values, absmax, block_size)


class LinearNF4(QauntModule):
	"""A 4-bit quantized version of the linear transformation using NF4 quantization."""

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
		block_size: int = 64,
	):
		super().__init__(
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		# Initialize the kernel
		if do_init:
			kernel_key = rngs.params()
			kernel = kernel_init(kernel_key, (in_features, out_features), param_dtype)
			kernel, scales = self._quantize_kernel(kernel)
		else:
			kernel, scales = None, None

		self.kernel = nnx.Param(kernel)
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
		self.block_size = block_size

	@classmethod
	def from_linear(
		cls,
		linear: nnx.Linear,
		rngs: tp.Optional[rnglib.Rngs] = None,
		block_size: int = 128,
		**kwargs,
	) -> "LinearNF4":
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
				block_size=block_size,
				rngs=rngs,
			)
		)

		kernel, scales = cls._quantize_kernel(linear.kernel.value, block_size)
		instance.kernel = nnx.Param(kernel)
		instance.scales = nnx.Param(scales)

		if linear.use_bias:
			instance.bias = nnx.Param(linear.bias.value)

		return instance

	def to_linear(self, rngs: tp.Optional[rnglib.Rngs] = None) -> nnx.Linear:
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
	def _quantize_kernel(kernel, block_size):
		"""Quantize the kernel weights using NF4."""
		if kernel is None or isinstance(kernel, jax.ShapeDtypeStruct):
			return None, None
		return quantize_and_pack_nf4(kernel, block_size)

	def _dequantize_kernel(self):  # in case someone's using tie word embedding.
		"""Dequantize the kernel weights from NF4."""
		if (
			self.kernel.value is None
			and self.scales.value is None
			and self.block_size is None
		):
			return None
		elif self.scales.value is None and self.block_size is None:
			return self.kernel
		return dequantize_nf4(
			self.kernel.value,
			self.scales.value,
			self.block_size,
		).reshape(self.in_features, self.out_features)

	@jax.named_scope("easydel-linear-nf4-call")
	def __call__(self, inputs: Array) -> Array:
		"""Applies a quantized linear transformation to the inputs along the last dimension."""
		kernel = self._dequantize_kernel()

		assert (
			kernel is not None
		), "loaded and dequantized kernel is None, which means it have been loaded from another None Kernel Linear"

		bias = self.bias.value

		inputs, kernel, bias = dtypes.promote_dtype(
			(inputs, kernel, bias), dtype=self.dtype
		)

		y = self.dot_general(
			inputs,
			kernel,
			(((inputs.ndim - 1,), (0,)), ((), ())),
			precision=self.precision,
		)

		assert self.use_bias == (bias is not None)
		if bias is not None:
			y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
		return y

	def get_kernel(self):
		"""Get the dequantized kernel weights."""
		return self._dequantize_kernel()

	def get_quantized_kernel(self):
		"""Get the quantized kernel weights and scales."""
		return self.kernel.value, self.scales.value

	@staticmethod
	def metadata():
		return {"quant_mode": "nf4"}

	@staticmethod
	def quantization_mapping():
		return {"kernel": ["kernel", "scales", "block_size"]}

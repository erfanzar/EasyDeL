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

from functools import partial

import jax
from jax import numpy as jnp

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
	blocks = blocks.reshape(-1, block_size)
	absmax = jnp.max(jnp.abs(blocks), axis=1)
	normalized = blocks / absmax[:, None]
	quantized = jnp.searchsorted(NF4_BOUNDARIES, normalized.reshape(-1)) - 1
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
	dequantized = dequantized.reshape(len(absmax), block_size)
	scaled = dequantized * absmax[:, None]
	return scaled


@partial(jax.jit, static_argnames=["block_size"])
def quantize_and_pack_nf4(
	blocks: jax.Array,
	block_size: int = 64,
):
	if blocks.ndim > 2:
		return jax.vmap(
			quantize_and_pack_nf4,
			in_axes=(0, None),
			out_axes=(0, 0),
		)(blocks, block_size)
	return single_quantize_and_pack_nf4(blocks, block_size)


@partial(jax.jit, static_argnames=["block_size"])
def dequantize_nf4(
	packed_values: jax.Array,
	absmax: jax.Array,
	block_size: int,
):
	packed_values = packed_values.astype(jnp.uint8)
	if packed_values.ndim > 2:
		return jax.vmap(dequantize_nf4, in_axes=(0, 0, None), out_axes=(0,))(
			packed_values, absmax, block_size
		)
	return single_dequantize_nf4(packed_values, absmax, block_size)

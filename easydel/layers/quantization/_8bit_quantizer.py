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

import jax
from jax import numpy as jnp


@jax.jit
def quantize_row_q8_0(x: jax.Array):
	"""
	Quantize a row of float32 values to 8-bit integers with blockwise scaling.
	Args:
	    x: input array
	Returns:
	    tuple of (scales, quantized_values)
	    - scales: float16 array of shape (nb,)
	    - quantized_values: int8 array of shape (k,)
	"""
	n_bit = 8
	eps = 1e-5
	max_int = 2 ** (n_bit - 1) - 1
	min_int = -(2 ** (n_bit - 1))
	max_val = jnp.amax(jnp.abs(x), axis=-1, keepdims=True)
	max_val = jnp.clip(max_val, min=eps)
	qscale = max_val / max_int
	qweight = jnp.clip(jnp.round(x * (1.0 / qscale)), min_int, max_int).astype(jnp.int8)
	return qweight, qscale


@jax.jit
def dequantize_row_q8_0(quants, scales):
	"""
	Dequantize 8-bit integers back to float32 values using blockwise scaling.

	Args:
	    quants: int8 array of shape (k,) containing quantized values
	    scales: float16 array of shape (nb,) containing scaling factors
	Returns:
	    float32 array of shape (k,) containing dequantized values
	"""

	return quants * scales

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

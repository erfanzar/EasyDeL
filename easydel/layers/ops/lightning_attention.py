import jax
import jax.numpy as jnp
from functools import partial
import math


def get_mask(n, slope=1.0):
	mask = jnp.triu(jnp.full((n, n), float("-inf")), k=1)
	indices = jnp.arange(n)[:, None]
	sub_indices = jnp.arange(n)[None, :]
	valid_mask = sub_indices <= indices
	positions = jnp.arange(n)[None, :] * slope
	positions = jnp.where(valid_mask, -jnp.flip(positions, axis=1), float("-inf"))
	mask = jnp.where(valid_mask, positions, mask)
	return jnp.exp(mask)


def get_full_mask(n, slopes):
	if slopes is None:
		return jnp.tril(jnp.ones((n, n)))
	else:
		masks = jax.vmap(lambda slope: get_mask(n, slope))(slopes.reshape(-1))
		return masks


@partial(jax.jit, static_argnums=(4,))
def linear_attn(q, k, v, slopes, dtype=jnp.float32):
	b, h, n, d = q.shape
	mask = get_full_mask(n, slopes).astype(dtype)

	# Compute attention scores
	qk = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
	qk = (qk.astype(dtype) * mask).astype(q.dtype)
	o = jnp.matmul(qk, v)

	return o


def build_slope_tensor(n_attention_heads):
	def get_slopes(n):
		def get_slopes_power_of_2(n):
			start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
			ratio = start
			return [start * ratio**i for i in range(n)]

		if math.log2(n).is_integer():
			return get_slopes_power_of_2(n)
		else:
			closest_power_of_2 = 2 ** math.floor(math.log2(n))
			return (
				get_slopes_power_of_2(closest_power_of_2)
				+ get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
			)

	slopes = jnp.array(get_slopes(n_attention_heads)).reshape(n_attention_heads, 1, 1)
	return slopes

import jax
import jax.numpy as jnp
import math
import typing as tp


def lightning_attention(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	slope_rate: jax.Array,
	attn_mask: tp.Optional[jax.Array] = None,
	past_key_value: tp.Optional[jax.Array] = None,
	init_cache: bool = False,
	dtype: jnp.dtype = jnp.float32,
) -> tp.Tuple[jax.Array, jax.Array]:
	ratio = jnp.exp(-slope_rate)
	slope_rate = jnp.asarray(slope_rate).astype(dtype)
	b, h, n, d = q.shape
	positions = jnp.arange(n) + 1
	index = positions[:, None] - positions[None, :]
	s_index = jnp.expand_dims(slope_rate * index, 0)
	diag_decay = jnp.where(index >= 0, jnp.exp(-s_index), 0.0).astype(dtype)
	if attn_mask is not None:
		attn_mask = attn_mask[:, None, :, None]
		v = v * attn_mask
	qk = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
	qk_decay = qk * diag_decay
	qkv_diag = jnp.matmul(qk_decay, v)
	if past_key_value is None and not init_cache:
		return qkv_diag, None
	else:
		if past_key_value is None:
			past_key_value = jnp.zeros((b, h, d, v.shape[-1]), dtype=v.dtype)
		kv = past_key_value
		q_decay = jnp.exp(-slope_rate * positions).reshape(1, 1, n, 1)
		q_decay = q_decay.astype(dtype)
		qkv_none_diag = jnp.matmul(q * q_decay, kv)
		output = qkv_none_diag + qkv_diag
		k_decay = jnp.exp(-slope_rate * (n - positions)).reshape(1, 1, n, 1)
		k_decay = k_decay.astype(dtype)
		kv = ratio * kv + jnp.matmul((k * k_decay).transpose(0, 1, 3, 2), v)
		return output, kv


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

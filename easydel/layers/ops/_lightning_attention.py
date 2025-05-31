import math

import jax
import jax.numpy as jnp


def lightning_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    slope_rate: jax.Array,
    position_ids: jax.Array | None = None,
    attn_mask: jax.Array | None = None,
    past_key_value: jax.Array | None = None,
    init_cache: bool = False,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array | None]:
    ratio = jnp.exp(-slope_rate)
    slope_rate = jnp.asarray(slope_rate).astype(dtype)
    b, h, n, d = q.shape
    if position_ids is None:
        positions = jnp.arange(n) + 1
    else:
        position_ids += 1
    index = positions[:, None] - positions[None, :]
    s_index = jnp.expand_dims(slope_rate * index, 0)
    s_index = jnp.where(index >= 0, -s_index, float("-inf")).astype(dtype)
    diag_decay = jnp.exp(s_index)
    if attn_mask is not None:
        attn_mask = attn_mask[:, None, :n, None]
        v = v * attn_mask
    qk = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * diag_decay
    qkv_diag = jnp.matmul(qk, v)
    output = qkv_diag
    if past_key_value is not None:
        output = []
        for i in range(n):
            past_key_value = ratio * past_key_value + jnp.einsum(
                "... n d, ... n e -> ... d e",
                k[:, :, i : i + 1],
                v[:, :, i : i + 1],
            )
            output.append(
                jnp.einsum(
                    "... n e, ... e d -> ... n d",
                    q[:, :, i : i + 1],
                    past_key_value.astype(q.dtype),
                )
            )
        output = jnp.concatenate(output, axis=-2)
    elif init_cache:
        if past_key_value is None:
            past_key_value = jnp.zeros((b, h, d, v.shape[-1]), dtype=v.dtype)
        q_decay = jnp.exp(-slope_rate * positions).reshape(h, n, 1).astype(dtype)
        k_decay = jnp.exp(-slope_rate * (n - positions)).reshape(h, n, 1).astype(dtype)
        qkv_none_diag = jnp.matmul(q * q_decay, past_key_value)
        output = qkv_none_diag + qkv_diag
        past_key_value = ratio * past_key_value + jnp.matmul(
            (k * k_decay).transpose(0, 1, 3, 2),
            v,
        )

    return output, past_key_value


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

import math
from functools import partial
import jax
from einops import rearrange
from jax import numpy as jnp, lax


def _query_chunk_flash_attention(q, k, v, b, q_chunk_size, k_chunk_size, epsilon):
    q_len, batch, heads, dim, = q.shape
    k_len, v_dim = k.shape[0], v.shape[-1]
    scale = 1 / jnp.sqrt(dim)
    q_scaled = q * scale

    def chunk_scanner(carries, _):
        chunk_idx, out, row_sum, row_max = carries
        k_chunk_sizes = min(k_chunk_size, k_len)

        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, heads, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, heads, v_dim))
        b_chunk = lax.dynamic_slice(b, (chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, 1, q_len))

        attn_weights = jnp.einsum("q ... d, k ... d -> q ... k", q_scaled, k_chunk)
        attn_weights = jnp.add(b_chunk, attn_weights)

        block_row_max = jnp.max(attn_weights, axis=-1, keepdims=True)

        new_row_max = jnp.maximum(block_row_max, row_max)
        exp_weights = jnp.exp(attn_weights - new_row_max)

        exp_weights = jnp.where(b_chunk == 0, exp_weights, 0.)
        block_row_sum = jnp.sum(exp_weights, axis=-1, keepdims=True) + epsilon

        exp_values = jnp.einsum("i ... j, j ... d -> i ... d", exp_weights, v_chunk)

        exp_row_max_diff = jnp.exp(row_max - new_row_max)

        new_row_sum = exp_row_max_diff * row_sum + block_row_sum
        out = (row_sum / new_row_sum) * exp_row_max_diff * out + (1. / new_row_sum) * exp_values

        return (chunk_idx + k_chunk_sizes, out, new_row_sum, new_row_max), None

    out = jnp.zeros((q_len, batch, heads, dim))
    row_sum = jnp.zeros((q_len, batch, heads, 1))
    row_max = jnp.ones((q_len, batch, heads, 1)) * -1e6

    (_, out, row_sum, row_max), _ = lax.scan(
        chunk_scanner,
        init=(0, out, row_sum, row_max), xs=None,
        length=math.ceil(k_len / k_chunk_size)
    )

    row_sum = rearrange(row_sum, "n ... 1 -> n ...")
    row_max = rearrange(row_max, "n ... 1 -> n ...")

    lse = jnp.log(row_sum) + row_max

    return out, lse


def _flash_attention(q, k, v, b, q_chunk_size, k_chunk_size, epsilon):
    batch, heads, q_len, dim, = q.shape

    def chunk_scanner(chunk_idx, _):
        chunk_sizes = min(q_chunk_size, q_len)
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0, 0, 0), slice_sizes=(chunk_sizes, batch, heads, dim))
        return (
            chunk_idx + chunk_sizes,
            _query_chunk_flash_attention(
                q_chunk,
                k,
                v,
                b,
                q_chunk_size,
                k_chunk_size,
                epsilon
            )
        )

    q, k, v = map(lambda t: rearrange(t, "b h n d -> n b h d"), (q, k, v))

    b = rearrange(b, "b h q k -> k b h q")

    _, (out, lse) = lax.scan(chunk_scanner, init=0, xs=None, length=math.ceil(q_len / q_chunk_size))

    out = rearrange(out, "c n b h d -> b h (c n) d")
    lse = rearrange(lse, "c n b h -> b h (c n)")

    return out, lse


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def flash_attention(q, k, v, b, q_chunk_size, k_chunk_size, epsilon):
    out, _ = _flash_attention(q, k, v, b, q_chunk_size, k_chunk_size, epsilon)
    return out


def _flash_attention_fwd(q, k, v, b, q_chunk_size, k_chunk_size, epsilon):
    out, lse = _flash_attention(q, k, v, b, q_chunk_size, k_chunk_size, epsilon)
    return out, (q, k, v, b, q_chunk_size, k_chunk_size, epsilon, out, lse)


def _query_chunk_flash_attention_backward(q, k, v, b, q_chunk_size, k_chunk_size, epsilon, o, do, lse):
    q_len, batch, heads, dim = q.shape
    k_len, v_dim = v.shape[0], v.shape[-1]

    scale = 1 / jnp.sqrt(dim)
    q_scaled = q * scale

    def chunk_scanner(carries, _):
        chunk_idx, dq = carries
        k_chunk_sizes = min(k_chunk_size, k_len)

        k_chunk = lax.dynamic_slice(k, (chunk_idx, batch, heads, 0), slice_sizes=(k_chunk_sizes, batch, heads, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, batch, heads, 0), slice_sizes=(k_chunk_sizes, batch, heads, v_dim))
        bias_chunk = lax.dynamic_slice(b, (chunk_idx, batch, k_len, 0),
                                       slice_sizes=(k_chunk_sizes, batch, 1, q_len))  # TODO: fix here

        attn_weights = jnp.einsum("i ... d, j ... d -> i ... j", q_scaled, k_chunk)

        p = jnp.exp(attn_weights - lse)

        p = jnp.add(bias_chunk, p, )

        dv_chunk = jnp.einsum("i ... j, i ... d -> j ... d", p, do)
        dp = jnp.einsum("i ... d, j ... d -> i ... j", do, v_chunk)

        D = jnp.sum(do * o, axis=-1, keepdims=True)
        ds = p * scale * (dp - D)

        dq_chunk = jnp.einsum("i ... j, j ... d -> i ... d", ds, k_chunk)
        dk_chunk = jnp.einsum("i ... j, i ... d -> j ... d", ds, q)

        return (chunk_idx + k_chunk_sizes, dq + dq_chunk), (dk_chunk, dv_chunk)

    dq = jnp.zeros_like(q)

    (_, dq), (dk, dv) = lax.scan(chunk_scanner, init=(0, dq), xs=None, length=math.ceil(k_len / k_chunk_size))

    dk = rearrange(dk, "c n ... -> (c n) ...")
    dv = rearrange(dv, "c n ... -> (c n) ...")
    return dq, dk, dv


def _flash_attention_bwd(q_chunk_size, k_chunk_size, epsilon, res, do):
    q, k, v, key_mask, o, lse = res

    batch, heads, q_len, dim = q.shape

    lse = rearrange(lse, "b h n -> n b h 1")

    q, k, v, o, do = map(lambda t: rearrange(t, "b h n d -> n b h d"), (q, k, v, o, do))
    key_mask = rearrange(key_mask, "b j -> j b")

    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        chunk_sizes = min(q_chunk_size, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, batch, heads, 0),
                                    slice_sizes=(chunk_sizes, batch, heads, q.shape[-1]))
        lse_chunk = lax.dynamic_slice(lse, (chunk_idx, batch, heads, 0), slice_sizes=(chunk_sizes, batch, heads, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, batch, heads, 0),
                                    slice_sizes=(chunk_sizes, batch, heads, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, batch, heads, 0),
                                     slice_sizes=(chunk_sizes, batch, heads, do.shape[-1]))

        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(
            q_chunk,
            k,
            v,
            b,
            o_chunk,
            do_chunk,
            lse_chunk
        )
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk), dq_chunk

    (_, dk, dv), dq = lax.scan(chunk_scanner, init=(0, dk, dv), xs=None, length=math.ceil(q_len / q_chunk_size))

    dq = rearrange(dq, "c n b h d -> b h (c n) d")
    dk, dv = map(lambda t: rearrange(t, "n b h d -> b h n d"), (dk, dv))

    return dq, dk, dv, None


flash_attention.defvjp(_flash_attention_fwd, _flash_attention_bwd)

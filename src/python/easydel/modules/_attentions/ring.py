from functools import partial

import jax
from einops import rearrange
from jax import numpy as jnp, lax


def _chunk_attention_bias(
        query_chunk_size,
        key_chunk_size,
        bias,
        segment_ids,
        deterministic,
        attn_dropout,
        attn_pdrop,
        causal,
        dtype,
        query_chunk_idx,
        key_chunk_idx
):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, 0, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if segment_ids is not None:
        q_segment_ids = lax.dynamic_slice(
            segment_ids,
            start_indices=(0, query_offset),
            slice_sizes=(segment_ids.shape[0], query_chunk_size)
        )
        k_segment_ids = lax.dynamic_slice(
            segment_ids,
            start_indices=(0, key_offset),
            slice_sizes=(segment_ids.shape[0], key_chunk_size)
        )
        segment_ids_mask = q_segment_ids[:, :, None] != k_segment_ids[:, None, :]
        segment_ids_mask = segment_ids_mask[:, None]  # B1QK
        segment_ids_bias = segment_ids_mask * jnp.finfo(dtype).min
        chunk_bias += segment_ids_bias

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)


def _block_wise_attention_fwd(
        query_states,
        key_states,
        value_states,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        bias,
        segment_ids,
        causal,
        query_chunk_size,
        key_chunk_size,
        deterministic,
        dropout_rng,
        attn_pdrop,
        dtype,
        policy,
        precision,
        prevent_cse
):
    batch, q_len, num_heads, dim_per_head = query_states.shape
    batch, kv_len, num_heads, dim_per_head = value_states.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query_states = query_states.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key_states = key_states.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value_states = value_states.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    query_states, key_states, value_states = map(lambda x: jnp.moveaxis(x, 1, 0),
                                                 (query_states, key_states, value_states))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

    scale = jnp.sqrt(query_states.shape[-1])
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size,
        key_chunk_size,
        bias,
        segment_ids,
        deterministic,
        attn_dropout,
        attn_pdrop,
        causal,
        dtype
    )

    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(_carry, _scan):
            k_chunk, value_chunk, k_chunk_idx = _scan
            _numerator_chunk, _denominator_chunk, _prev_max_score_chunk = _carry
            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk

            max_score_chunk = jnp.maximum(_prev_max_score_chunk, jnp.max(attn_weights, axis=-1))
            max_score_chunk = lax.stop_gradient(max_score_chunk)
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None])
            exp_values = jnp.einsum("bhqk,bkhd->bqhd", exp_weights, value_chunk, precision=precision)
            correction = rearrange(
                jnp.exp(
                    _prev_max_score_chunk - max_score_chunk
                ), "b h query_states -> b query_states h"
            )[..., None]
            _numerator_chunk = _numerator_chunk * correction + exp_values
            _denominator_chunk = _denominator_chunk * jnp.exp(
                _prev_max_score_chunk - max_score_chunk) + exp_weights.sum(
                axis=-1)
            return (_numerator_chunk, _denominator_chunk, max_score_chunk), None

        def skip_upper_half(_carry, _args):
            key_chunk, value_chunk, k_chunk_idx = _args
            skip_block = jnp.array(False)
            if causal:
                skip_block = q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda _carry, args: (_carry, None),
                scan_kv_block,
                _carry,
                _args
            )

        (numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
            skip_upper_half, init=(numerator_chunk, denominator_chunk, max_score_chunk),
            xs=(key_states, value_states, jnp.arange(0, num_kv))
        )
        output_chunk = numerator_chunk / rearrange(denominator_chunk, "b h query_states -> b query_states h")[
            ..., None].astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)

    _, (_, numerator, denominator, max_score) = lax.scan(scan_attention, init=(), xs=(
        query_states, numerator, denominator, max_score, jnp.arange(0, num_q)))

    numerator = jnp.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(lambda x: rearrange(x, "n b h c -> b h n c"), (denominator, max_score))
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score


def _block_wise_attention_bwd(
        query_states,
        key_states,
        value_states,
        g,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        bias,
        segment_ids,
        causal,
        query_chunk_size,
        key_chunk_size,
        deterministic,
        dropout_rng,
        attn_pdrop,
        dtype,
        policy,
        precision,
        prevent_cse
):
    batch, q_len, num_heads, dim_per_head = query_states.shape
    batch, kv_len, num_heads, dim_per_head = key_states.shape
    batch, kv_len, num_heads, dim_per_head = value_states.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry

    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

    query_states = query_states.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key_states = key_states.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value_states = value_states.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    query_states, key_states, value_states = map(lambda x: jnp.moveaxis(x, 1, 0),
                                                 (query_states, key_states, value_states))

    scale = jnp.sqrt(query_states.shape[-1])
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, segment_ids, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)

    def scan_attention(carry, scan):
        dk, dv = carry
        q_chunk, dq_chunk, g_chunk, output_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan
        dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None]) / denominator_chunk[..., None]

            ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights
            dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / scale
            dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / scale
            dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)
            return dq_chunk, (dk_chunk, dv_chunk)

        def skip_upper_half(_carry, _args):
            key_chunk, value_chunk, k_chunk_idx = _args
            skip_block = jnp.array(False)
            if causal:
                skip_block = q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
            return lax.cond(
                skip_block,
                lambda _carry, _args: (
                    _carry, (
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                    )
                ),
                scan_kv_block,
                _carry,
                _args
            )

        dq_chunk, (dk_part, dv_part) = lax.scan(
            skip_upper_half, init=dq_chunk, xs=(key_states, value_states, jnp.arange(0, num_kv))
        )
        return (dk + dk_part, dv + dv_part), dq_chunk

    (dk, dv), dq = lax.scan(scan_attention, init=(dk, dv),
                            xs=(query_states, dq, g, output, denominator, max_score, jnp.arange(0, num_q)))

    dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
    dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
    dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
    dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

    return dq, dk, dv


def _ring_attention_standard_fwd(
        query,
        key,
        value,
        attn_bias,
        scale,
        axis_name,
        float32_logits
):
    if float32_logits:
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    batch, q_len, num_heads, _ = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(query.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(query.dtype)
    axis_size = lax.psum(1, axis_name)

    def scan_kv_block(carry, idx):
        p_max_score, _numerator, _denominator, _key, _value = carry

        bias = lax.dynamic_slice_in_dim(
            lax.dynamic_slice_in_dim(
                attn_bias, (lax.axis_index(axis_name) - idx) % axis_size * q_len, q_len, axis=-2
            ), (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        )
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, _key) / scale
        attn_weights = jnp.add(bias, attn_weights)
        _max_score = jnp.maximum(p_max_score, jnp.max(attn_weights, axis=-1))
        exp_weights = jnp.exp(attn_weights - _max_score[..., None])
        correction = rearrange(jnp.exp(p_max_score - _max_score), "b h query_states -> b query_states h")[..., None]
        _numerator = _numerator * correction + jnp.einsum("bhqk,bkhd->bqhd", exp_weights, _value)
        _denominator = _denominator * jnp.exp(p_max_score - _max_score) + jnp.sum(exp_weights, axis=-1)

        _key, _value = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (_key, _value)
        )
        return (_max_score, _numerator, _denominator, _key, _value), None

    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(query.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, key, value),
        xs=jnp.arange(0, axis_size)
    )
    output = numerator / rearrange(denominator, "b h query_states -> b query_states h")[..., None]
    return output.astype(value.dtype), (output, query, key, value, attn_bias, numerator, denominator, max_score)


def _ring_attention_standard_bwd(
        scale,
        axis_name,
        float32_logits,
        res,
        g
):
    del float32_logits
    axis_size = lax.psum(1, axis_name)
    output, query, key, value, attn_bias, numerator, denominator, max_score = res
    dq = jnp.zeros_like(query, dtype=jnp.float32)
    dk = jnp.zeros_like(key, dtype=jnp.float32)
    dv = jnp.zeros_like(value, dtype=jnp.float32)
    q_len = query.shape[1]
    batch, kv_len, num_heads, dim_per_head = key.shape

    def scan_kv_block(carry, idx):
        _dq, _dk, _dv, _key, _value = carry
        bias = lax.dynamic_slice_in_dim(
            lax.dynamic_slice_in_dim(
                attn_bias, (lax.axis_index(axis_name) - idx) % axis_size * q_len, q_len, axis=-2
            ), (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1
        )
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, _key) / scale
        attn_weights = jnp.add(bias, attn_weights)
        exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        ds = jnp.einsum("bqhd,bkhd->bhqk", g, _value)
        dl = (ds - jnp.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        _dq = _dq + jnp.einsum("bhqk,bkhd->bqhd", dl, _key) / scale
        _dk = _dk + jnp.einsum("bqhd,bhqk->bkhd", query, dl) / scale
        _dv = _dv + jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        _key, _value, _dk, _dv = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (_key, _value, _dk, _dv)
        )
        return (_dq, _dk, _dv, _key, _value), None

    (dq, dk, dv, key, value), _ = lax.scan(
        scan_kv_block, init=(dq, dk, dv, key, value), xs=jnp.arange(0, axis_size)
    )
    dq, dk, dv = dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(value.dtype)
    return dq, dk, dv, None


@partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6])
def ring_attention_standard(query, key, value, attn_bias, scale, axis_name, float32_logits=True):
    y, _ = _ring_attention_standard_fwd(
        query,
        key,
        value,
        attn_bias,
        scale,
        axis_name,
        float32_logits
    )
    return y


ring_attention_standard.defvjp(_ring_attention_standard_fwd, _ring_attention_standard_bwd)


def _wise_ring_attention_fwd(
        query_states,
        key_states,
        value_states,
        attn_bias,
        segment_ids,
        axis_name,
        float32_logits,
        block_wise_kwargs
):
    if float32_logits:
        query_states, key_states = query_states.astype(jnp.float32), key_states.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = query_states.shape
    batch, kv_len, num_heads, dim_per_head = key_states.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(query_states.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(query_states.dtype)
    axis_size = lax.psum(1, axis_name)
    q_block_size, kv_block_size = q_len, kv_len
    query_chunk_size = block_wise_kwargs["query_chunk_size"]
    key_chunk_size = block_wise_kwargs["key_chunk_size"]

    def scan_kv_block(carry, idx):
        _prev_max_score, _numerator, _denominator, _key_states, _value_states = carry
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        _numerator, _denominator, _max_score = _block_wise_attention_fwd(
            query_states,
            _key_states,
            _value_states,
            (_numerator, _denominator, _prev_max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=attn_bias,
            segment_ids=segment_ids,
            **block_wise_kwargs
        )
        _key_states, _value_states = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (_key_states, _value_states)
        )
        return (_max_score, _numerator, _denominator, _key_states, _value_states), None

    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(query_states.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, key_states, value_states),
        xs=jnp.arange(0, axis_size)
    )
    reshaped_denominator = rearrange(denominator, "b h query_states -> b query_states h")[..., None]
    output = numerator / reshaped_denominator
    return output.astype(value_states.dtype), (
        output,
        query_states,
        key_states,
        value_states,
        attn_bias,
        segment_ids,
        denominator,
        max_score
    )


def _wise_ring_attention_bwd(
        axis_name,
        float32_logits,
        block_wise_kwargs,
        res,
        g
):
    del float32_logits
    output, query_states, key_states, value_states, attn_bias, segment_ids, denominator, max_score = res
    batch, q_len, num_heads, dim_per_head = query_states.shape
    batch, kv_len, num_heads, dim_per_head = key_states.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(query_states, dtype=query_states.dtype)
    dk = jnp.zeros_like(key_states, dtype=key_states.dtype)
    dv = jnp.zeros_like(value_states, dtype=key_states.dtype)
    query_chunk_size = block_wise_kwargs["query_chunk_size"]
    key_chunk_size = block_wise_kwargs["key_chunk_size"]
    q_block_size, kv_block_size = q_len, kv_len  # assumes this function is pre-sharded inside shard_map

    def scan_kv_block(carry, idx):
        _dq, _dk, _dv, _key_states, _value_states = carry
        q_block_idx = lax.axis_index(axis_name)
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        _dq, _dk, _dv = _block_wise_attention_bwd(
            query_states,
            _key_states,
            _value_states,
            g,
            (_dq, _dk, _dv, output, denominator, max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=attn_bias,
            segment_ids=segment_ids,
            **block_wise_kwargs
        )
        _key_states, _value_states, _dk, _dv = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (_key_states, _value_states, _dk, _dv)
        )
        return (_dq, _dk, _dv, _key_states, _value_states), None

    (dq, dk, dv, key_states, value_states), _ = lax.scan(
        scan_kv_block,
        init=(dq, dk, dv, key_states, value_states),
        xs=jnp.arange(0, axis_size)
    )
    dq, dk, dv = dq.astype(query_states.dtype), dk.astype(key_states.dtype), dv.astype(key_states.dtype)
    return dq, dk, dv, None, None


@partial(jax.custom_vjp, nondiff_argnums=[5, 6, 7])
def wise_ring_attention(
        query_states,
        key_states,
        value_states,
        attn_bias,
        segment_ids,
        axis_name,
        float32_logits,
        block_wise_kwargs
):
    out, _ = _wise_ring_attention_fwd(
        query_states,
        key_states,
        value_states,
        attn_bias,
        segment_ids,
        axis_name,
        float32_logits,
        block_wise_kwargs
    )
    return out


wise_ring_attention.defvjp(_wise_ring_attention_fwd, _wise_ring_attention_bwd)

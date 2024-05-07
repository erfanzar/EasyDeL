import math
from functools import partial
import jax
from chex import Array
from fjformer import with_sharding_constraint
from flax.linen.dtypes import promote_dtype
from jax import numpy as jnp, lax, random
from jax.sharding import PartitionSpec
from typing import Optional, Union


@partial(
    jax.jit,
    static_argnames=[
        "deterministic",
        "dropout_rng",
        "shard_attention_computation",
        "dtype",
        "precision",
        "attention_dropout",
    ]
)
def vanilla_attention(
        query_states: Array,
        key_states: Array,
        value_states: Array,
        bias: Optional[Array] = None,
        deterministic: bool = False,
        dropout_rng: Optional[random.PRNGKey] = None,
        shard_attention_computation: bool = True,
        dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = None,
        attention_dropout: float = 0.0
):
    assert query_states.ndim == key_states.ndim, "q, k must have same rank."
    assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
    assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
    assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."
    query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)
    sp, tp = "sp", "tp"
    if query_states.shape[1] == 1:
        sp, tp = None, "sp"
    elif query_states.shape[1] != key_states.shape[1]:
        sp, tp = [None] * 2
    if shard_attention_computation:
        query_states = with_sharding_constraint(query_states, PartitionSpec(("dp", "fsdp"), sp, tp, None))
        key_states = with_sharding_constraint(key_states, PartitionSpec(("dp", "fsdp"), sp, tp, None))
        value_states = with_sharding_constraint(value_states, PartitionSpec(("dp", "fsdp"), sp, tp, None))

    depth = query_states.shape[-1]
    query_states = query_states / jnp.sqrt(depth).astype(dtype)
    attention_weight = jnp.einsum("...qhd,...khd->...hqk", query_states, key_states, precision=precision)
    if shard_attention_computation:
        attention_weight = with_sharding_constraint(attention_weight, PartitionSpec(("dp", "fsdp"), None, sp, None))
        if bias is not None:
            bias = with_sharding_constraint(bias, PartitionSpec(("dp", "fsdp"), None, sp, None))
    if bias is not None:
        attention_weight = attention_weight + bias
    attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
    if not deterministic and attention_dropout > 0.0:
        keep_prob = 1.0 - attention_dropout
        dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attention_weight = attention_weight * multiplier
    attention = jnp.einsum("...hqk,...khd->...qhd", attention_weight, value_states, precision=precision)
    if shard_attention_computation:
        attention = with_sharding_constraint(
            attention,
            PartitionSpec(("dp", "fsdp"), sp, tp, None)
        )
    return attention, attention_weight


def _chunk_bias(bias, axis_name, query_length, key_length):
    index = lax.axis_index(axis_name)
    return lax.dynamic_slice(
        bias, (0, 0, index * query_length, index * key_length), (bias.shape[0], 1, query_length, key_length)
    )


def softmax_grad(X, output):  # output is the result of the softmax function
    softmax_output = output  # Assuming output is provided from the softmax function
    diag_softmax = jnp.diag(softmax_output)
    outer_softmax = softmax_output[:, None] * softmax_output[None, :]
    grad = diag_softmax - outer_softmax
    return grad


# def _shard_vanilla_attention_bwd(
#         deterministic,
#         dropout_rng,
#         dtype,
#         precision,
#         attention_dropout,
#         res,
#         g
# ):
#     # FWD FUNC
#     # batch_size, query_length, num_heads, dim = query_states.shape
#     # key_length = key_states.shape[1]
#     # assert query_states.ndim == key_states.ndim, "q, k must have same rank."
#     # assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
#     # assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
#     # assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."
#     # query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)
#     #
#     # depth = query_states.shape[-1]
#     # query_states = query_states / jnp.sqrt(depth).astype(dtype)
#     # attention_weight = jnp.einsum("...qhd,...khd->...hqk", query_states, key_states, precision=precision)
#     # index = lax.axis_index("sp")
#     # if bias is not None:
#     #     chunk_b = lax.dynamic_slice(
#     #         bias, (0, 0, index * query_length, index * key_length), (batch_size, 1, query_length, key_length)
#     #     )
#     #     attention_weight = attention_weight + chunk_b
#     # attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
#     # if not deterministic and attention_dropout > 0.0:
#     #     keep_prob = 1.0 - attention_dropout
#     #     dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
#     #     keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
#     #
#     #     multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
#     #     attention_weight = attention_weight * multiplier
#     # attention = jnp.einsum("...hqk,...khd->...qhd", attention_weight, value_states, precision=precision)
#     #
#     # return attention, (attention, query_states, key_states, value_states, attention_weight, bias)
#     attention, query_states, key_states, value_states, attention_weight, bias = res
#     dL_dA = jnp.einsum("...qhd,...khd->...hqk", g, value_states, precision=lax.Precision.HIGH)
#     dL_dA = dL_dA * attention_weight * (1 - attention_weight)  # softmax derivative
#
#     # Query and key gradients
#     dL_dQ = jnp.einsum("...hqk,...khd->...qhd", dL_dA, key_states, precision=lax.Precision.HIGH)
#     dL_dK = jnp.einsum("...hqk,...qhd->...khd", dL_dA, query_states, precision=lax.Precision.HIGH)
#
#     # Value gradients
#     dL_dV = jnp.einsum("...hqk,...qhd->...khd", attention_weight, g, precision=lax.Precision.HIGH)
#
#     # Dropout
#     if attention_dropout > 0.0:
#         keep_prob = 1.0 - attention_dropout
#         dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
#         keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
#         multiplier = keep.astype(query_states.dtype) / jnp.asarray(keep_prob, query_states.dtype)
#         dL_dQ = dL_dQ * multiplier
#         dL_dK = dL_dK * multiplier
#         dL_dV = dL_dV * multiplier
#
#     # Bias gradients (if applicable)
#     # dL_dbias = None
#     # if bias is not None:
#     #     dL_dbias = jnp.einsum("...hqk->...", dL_dA, precision=lax.Precision.HIGH)
#
#     return dL_dQ, dL_dK, dL_dV, None
#
#
# def _shard_vanilla_attention_fwd(
#         query_states: Array,
#         key_states: Array,
#         value_states: Array,
#         bias: Optional[Array] = None,
#         deterministic: bool = False,
#         dropout_rng: Optional[random.PRNGKey] = None,
#         dtype: jnp.dtype = jnp.float32,
#         precision: Optional[jax.lax.Precision] = None,
#         attention_dropout: float = 0.0
# ):
#     batch_size, query_length, num_heads, dim = query_states.shape
#     key_length = key_states.shape[1]
#     assert query_states.ndim == key_states.ndim, "q, k must have same rank."
#     assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
#     assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
#     assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."
#     query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)
#
#     depth = query_states.shape[-1]
#     query_states = query_states / jnp.sqrt(depth).astype(dtype)
#     attention_weight = jnp.einsum("...qhd,...khd->...hqk", query_states, key_states, precision=precision)
#     index = lax.axis_index("sp")
#     if bias is not None:
#         chunk_b = lax.dynamic_slice(
#             bias, (
#                 0,
#                 0,
#                 index * query_length,
#                 index * key_length
#             ),
#             (
#                 batch_size,
#                 1,
#                 query_length,
#                 key_length
#             )
#         )
#         attention_weight = attention_weight + chunk_b
#     attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
#     if not deterministic and attention_dropout > 0.0:
#         keep_prob = 1.0 - attention_dropout
#         dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
#         keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
#
#         multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
#         attention_weight = attention_weight * multiplier
#     attention = jnp.einsum("...hqk,...khd->...qhd", attention_weight, value_states, precision=precision)
#
#     return attention, (attention, query_states, key_states, value_states, attention_weight, bias)
#
#
# @partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
# def shard_vanilla_attention(
#         query_states: Array,
#         key_states: Array,
#         value_states: Array,
#         bias: Optional[Array] = None,
#         deterministic: bool = True,
#         dropout_rng: Optional[random.PRNGKey] = None,
#         dtype: jnp.dtype = jnp.float32,
#         precision: Optional[jax.lax.Precision] = None,
#         attention_dropout: float = 0.0
# ):
#     out, _ = _shard_vanilla_attention_fwd(
#         query_states=query_states,
#         key_states=key_states,
#         value_states=value_states,
#         bias=bias,
#         dropout_rng=dropout_rng,
#         deterministic=deterministic,
#         dtype=dtype,
#         precision=precision,
#         attention_dropout=attention_dropout
#     )
#
#     return out


# shard_vanilla_attention.defvjp(_shard_vanilla_attention_fwd, _shard_vanilla_attention_bwd)

def shard_vanilla_attention(
        query_states: Array,
        key_states: Array,
        value_states: Array,
        bias: Optional[Array] = None,
        deterministic: bool = True,
        dropout_rng: Optional[random.PRNGKey] = None,
        dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = None,
        attention_dropout: float = 0.0
):
    batch_size, query_length, num_heads, dim = query_states.shape
    key_length = key_states.shape[1]
    assert query_states.ndim == key_states.ndim, "q, k must have same rank."
    assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
    assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
    assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."
    query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)

    depth = query_states.shape[-1]
    query_states = query_states / jnp.sqrt(depth).astype(dtype)
    attention_weight = jnp.einsum("...qhd,...khd->...hqk", query_states, key_states, precision=precision)
    axis_size = lax.psum(1, "sp")
    if bias is not None:
        bias = lax.dynamic_slice_in_dim(
            lax.dynamic_slice_in_dim(
                bias,
                lax.axis_index("sp") % axis_size * query_length,
                query_length, axis=-2
            ),
            lax.axis_index("sp") % axis_size * key_length,
            key_length,
            axis=-1
        )
        attention_weight = jnp.add(attention_weight, bias)

    attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
    if not deterministic and attention_dropout > 0.0:
        keep_prob = 1.0 - attention_dropout
        dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attention_weight = attention_weight * multiplier
    attention = jnp.einsum("...hqk,...khd->...qhd", attention_weight, value_states, precision=precision)

    return attention


def attention_production(
        query_states: jax.Array,
        key_states: jax.Array,
        value_states: jax.Array,
        attention_bias: jax.Array | None = None,
        deterministic: bool = True,
        dropout_rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
        dropout_rate: float = 0.0
):
    batch, q_sequence_length, q_num_head, head_dim = query_states.shape
    _, kv_sequence_length, kv_num_head, _ = key_states.shape
    assert q_num_head % kv_num_head == 0, (
        f"`query_states` {q_num_head} must be a multiple of `key_states` "
        f"and `value_states` heads {kv_num_head}"
    )
    query_states = jnp.reshape(query_states,
                               (batch, q_sequence_length, kv_num_head, q_num_head // kv_num_head, head_dim))
    attention_score = jnp.einsum(
        "...thHd,...Thd->...hHtT",
        query_states,
        key_states
    ).astype(
        jnp.float32
    )
    attention_score *= 1 / math.sqrt(head_dim)
    max_attention_value = jnp.array(30.0, dtype=attention_score.dtype)
    attention_score = max_attention_value * jnp.tanh(attention_score / max_attention_value)
    attention_score = attention_score + attention_bias[:, :, None, :, :]
    attention_weights = jax.nn.softmax(attention_score).astype(query_states.dtype)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weights.shape[-2:]
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        multiplier = keep.astype(query_states.dtype) / jnp.asarray(keep_prob, dtype=query_states.dtype)
        attention_weights = attention_weights * multiplier

    attention = jnp.einsum("...hHtT,...Thd->...thHd", attention_weights, value_states).reshape(
        batch, q_sequence_length, q_num_head, head_dim
    )
    return attention


def static_sharded_attention_production(
        query_states: jax.Array,
        key_states: jax.Array,
        value_states: jax.Array,
        attention_bias: jax.Array | None = None,
        deterministic: bool = True,
        dropout_rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
        dropout_rate: float = 0.0
):
    assert key_states.shape[1] == value_states.shape[1], "miss match on key_states and value_states sequence length"
    is_generating = query_states.shape[1] == 1 or query_states.shape[1] != key_states.shape[1]
    sp = None if is_generating else "sp"
    tp = "sp" if is_generating else "tp"
    query_states = with_sharding_constraint(
        query_states,
        PartitionSpec(
            ("dp", "fsdp"),
            sp,
            tp,
            None
        )
    )
    key_states = with_sharding_constraint(
        key_states,
        PartitionSpec(
            ("dp", "fsdp"),
            sp,
            tp,
            None
        )
    )
    value_states = with_sharding_constraint(
        value_states,
        PartitionSpec(
            ("dp", "fsdp"),
            sp,
            tp,
            None
        )
    )

    batch, q_sequence_length, q_num_head, head_dim = query_states.shape
    _, kv_sequence_length, kv_num_head, _ = key_states.shape

    assert q_num_head % kv_num_head == 0, (
        f"`query_states` {q_num_head} must be a multiple of"
        f" `key_states` and `value_states` heads {kv_num_head}"
    )

    query_states = jnp.reshape(
        query_states,
        (batch, q_sequence_length, kv_num_head, q_num_head // kv_num_head, head_dim)
    )

    query_states = with_sharding_constraint(
        query_states, PartitionSpec(
            ("dp", "fsdp"),
            sp,
            tp,
            None,
            None
        )
    )

    attention_score = jnp.einsum(
        "...thHd,...Thd->...hHtT",
        query_states, key_states
    ).astype(jnp.float32)

    attention_score *= 1 / math.sqrt(head_dim)

    max_attention_value = jnp.array(30.0, dtype=attention_score.dtype)
    attention_score = max_attention_value * jnp.tanh(attention_score / max_attention_value)
    attention_score = attention_score + attention_bias[:, :, None, :, :]

    attention_weights = jax.nn.softmax(attention_score).astype(query_states.dtype)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weights.shape[-2:]
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        multiplier = keep.astype(query_states.dtype) / jnp.asarray(keep_prob, dtype=query_states.dtype)
        attention_weights = attention_weights * multiplier

    attention = jnp.einsum("...hHtT,...Thd->...thHd", attention_weights, value_states).reshape(
        batch, q_sequence_length, q_num_head, head_dim
    )

    attention = with_sharding_constraint(
        attention,
        PartitionSpec(
            ("dp", "fsdp"),
            sp,
            tp,
            None,
        )
    )

    return attention


def static_sharded_dot_product_attention(
        query_states: Array,
        key_states: Array,
        value_states: Array,
        bias: Optional[Array] = None,
        mask: Optional[Array] = None,
        broadcast_dropout: bool = True,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        dtype: Optional[jnp.dtype] = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        shard_attention_computation: bool = True
):
    assert key_states.shape[1] == value_states.shape[1], "miss match on key_states and value_states sequence length"
    assert query_states.ndim == key_states.ndim, "q, k must have same rank."
    assert query_states.shape[:-3] == key_states.shape[:-3], "q, k batch dims must match."
    assert query_states.shape[-2] == key_states.shape[-2], "q, k num_heads must match."
    assert query_states.shape[-1] == key_states.shape[-1], "q, k depths must match."

    query_states, key_states, value_states = promote_dtype(query_states, key_states, value_states, dtype=dtype)

    if query_states.shape[1] == 1:
        sp = None
        tp = "sp"
    elif query_states.shape[1] != key_states.shape[1]:
        sp = None
        tp = None
    else:
        sp = "sp"
        tp = "tp"

    if shard_attention_computation:
        query_states = with_sharding_constraint(
            query_states, PartitionSpec(
                ("dp", "fsdp"),
                sp,
                tp,
                None
            )
        )

        key_states = with_sharding_constraint(
            key_states, PartitionSpec(
                ("dp", "fsdp"),
                sp,
                tp,
                None
            )
        )

        value_states = with_sharding_constraint(
            value_states, PartitionSpec(
                ("dp", "fsdp"),
                sp,
                tp,
                None
            )
        )

    depth = query_states.shape[-1]
    query_states = query_states / jnp.sqrt(depth).astype(dtype)
    attention_weight = jnp.einsum(
        "...qhd,...khd->...hqk",
        query_states, key_states, precision=precision
    )
    if shard_attention_computation:
        attention_weight = with_sharding_constraint(
            attention_weight, PartitionSpec(
                ("dp", "fsdp"),
                None,
                sp,
                None
            )
        )
        if bias is not None:
            bias = with_sharding_constraint(
                bias, PartitionSpec(
                    ("dp", "fsdp"),
                    None,
                    sp,
                    None
                )
            )
    if bias is not None:
        attention_weight = attention_weight + bias
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attention_weight = jnp.where(mask, attention_weight, big_neg)
    attention_weight = jax.nn.softmax(attention_weight).astype(dtype)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key_states.ndim - 2)) + attention_weight.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attention_weight.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attention_weight = attention_weight * multiplier
    attention = jnp.einsum(
        "...hqk,...khd->...qhd",
        attention_weight, value_states, precision=precision
    )
    if shard_attention_computation:
        attention = with_sharding_constraint(
            attention, PartitionSpec(
                ("dp", "fsdp"),
                sp,
                tp,
                None
            )
        )
    return attention

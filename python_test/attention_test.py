import math
import os

import fjformer
import flax.linen.attention

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax.random
from lib.python.EasyDel import MistralConfig
from lib.python.EasyDel.modules.attention_module import (
    AttentionModule
)
from fjformer import GenerateRNG
from jax import numpy as jnp, lax, random
from flax.linen.attention import dot_product_attention
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from functools import partial

rng = GenerateRNG()
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
HEAD_DIM = 8
NUM_HEADS = 32
IMG_LOSS = 0.5649852
CHUNK_WISE_RING = SEQUENCE_LENGTH // 8

config = MistralConfig(
    axis_dims=(1, 1, 1, -1),
    block_q=CHUNK_WISE_RING,
    block_k=CHUNK_WISE_RING
)


def value_and_grad_wrapper(fn, **kwargs):
    @partial(jax.value_and_grad, **kwargs)
    def inner(*args, **kwargs):
        return jnp.sum(fn(*args, **kwargs))

    return inner


def diff(t1, t2):
    return jnp.max(jnp.abs(t1 - t2))


@value_and_grad_wrapper
def call_vanilla(q, k, v, b):
    attention_pred = AttentionModule(
        attn_mechanism="vanilla",
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=q.shape[-1],
        num_attention_heads=q.shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    ).__call__(
        query_states=q,
        key_states=k,
        value_states=v,
        bias=b,
        query_sequence_length=q.shape[1],
        key_value_sequence_length=k.shape[1]
    ).attention_outputs
    return attention_pred


@value_and_grad_wrapper
def call_dot_product(q, k, v, b):
    attention_pred = dot_product_attention(q, k, v, b)
    return attention_pred


@value_and_grad_wrapper
def call_wise_ring(q, k, v, b):
    attention_pred = AttentionModule(
        attn_mechanism="wise_ring",
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=q.shape[-1],
        num_attention_heads=q.shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    ).__call__(
        query_states=q,
        key_states=k,
        value_states=v,
        bias=b,
        query_sequence_length=q.shape[1],
        key_value_sequence_length=k.shape[1]
    ).attention_outputs
    return attention_pred


@value_and_grad_wrapper
def call_blockwise(q, k, v, b):
    attention_pred = AttentionModule(
        attn_mechanism="blockwise",
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=q.shape[-1],
        num_attention_heads=q.shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    ).__call__(
        query_states=q,
        key_states=k,
        value_states=v,
        bias=b,
        query_sequence_length=q.shape[1],
        key_value_sequence_length=k.shape[1]
    ).attention_outputs
    return attention_pred


@value_and_grad_wrapper
def call_ring(q, k, v, b):
    attention_pred = AttentionModule(
        attn_mechanism="local_ring",
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=q.shape[-1],
        num_attention_heads=q.shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    ).__call__(
        query_states=q,
        key_states=k,
        value_states=v,
        bias=b,
        query_sequence_length=q.shape[1],
        key_value_sequence_length=k.shape[1]
    ).attention_outputs
    return attention_pred


@value_and_grad_wrapper
def call_sharded_vanilla(q, k, v, b):
    attention_pred = AttentionModule(
        attn_mechanism="sharded_vanilla",
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=q.shape[-1],
        num_attention_heads=q.shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    ).__call__(
        query_states=q,
        key_states=k,
        value_states=v,
        bias=b,
        query_sequence_length=q.shape[1],
        key_value_sequence_length=k.shape[1]
    ).attention_outputs
    return attention_pred


def make_inputs():
    q = jax.random.normal(rng.rng, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), dtype="float32")
    k = jax.random.normal(rng.rng, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), dtype="float32")
    v = jax.random.normal(rng.rng, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), dtype="float32")
    c = flax.linen.attention.make_causal_mask(
        jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH))
    )
    a = jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH))
    a.at[:, SEQUENCE_LENGTH // 2:].set(0)
    b = jnp.where(
        flax.linen.attention.combine_masks(
            jnp.expand_dims(jnp.expand_dims(a, 1), 1), c
        ),
        0,
        -jnp.inf
    )

    return q, k, v, b


def main():
    q, k, v, b = make_inputs()

    excepted_output, excepted_grads = call_dot_product(q, k, v, b)
    fns = {
        "local_ring": call_ring,
        "blockwise": call_blockwise,
        "vanilla": call_vanilla,
        "wise_ring": call_wise_ring,
        "sharded_vanilla": call_sharded_vanilla
    }
    outs_and_grads = {nm: fn(q, k, v, b) for nm, fn in fns.items()}

    for key, (out, grad) in outs_and_grads.items():
        output_diff = diff(excepted_output, out)
        g_diff = [diff(*args) for args in zip(excepted_grads, grad)]
        print(f"Comparing {key.upper()} and BASE ATTENTION")
        print(f"OUTPUT DIFF : {output_diff}")
        print(f"GRAD DIFF :\n\t", end="")
        for i in range(len(g_diff)):
            print(g_diff[i], end=" | ")
        print("\n", "*" * 50)


if __name__ == "__main__":
    main()

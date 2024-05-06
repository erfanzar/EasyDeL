%%writefile python_test/attention_test.py
import math
import os
import time

import fjformer
import flax.linen.attention
import pandas

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
SEQUENCE_LENGTH = 128 * 8 
HEAD_DIM = 128
NUM_HEADS = 32
IMG_LOSS = 0.5649852
CHUNK_WISE_RING = 128

config = MistralConfig(
    axis_dims=(1, -1, 1, 1),
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
def call_dot_product(q, k, v, b):
    attention_pred = dot_product_attention(q, k, v, b)
    return attention_pred


@value_and_grad_wrapper
def call_attention_module(q, k, v, b, attn_mechanism):
    attention_pred = AttentionModule(
        attn_mechanism=attn_mechanism,
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=q.shape[-1],
        num_attention_heads=q.shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    )(
        query_states=q,
        key_states=k,
        value_states=v,
        bias=b,
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
    test_attentions = [
        "local_ring",
        "blockwise",
        "vanilla",
        "wise_ring",
        "sharded_vanilla",
        "flash",
        "splash",
        "cudnn"
    ]
    fns = {
        k: partial(call_attention_module, attn_mechanism=k) for k in test_attentions
    }
    outs_and_grads = {}
    for nm, fn in fns.items():
        try:
            start = time.time()
            out = jax.block_until_ready(fn(q, k, v, b))
            end = time.time() - start
            outs_and_grads[nm] = out + (end,)
        except OSError:
            outs_and_grads[nm] = (None, None, None)
    frame_out = {}
    for key, (out, grad, time_took) in outs_and_grads.items():

        if out is None and grad is None:
            frame_out[key.upper()] = {
                "OUT DIFF": "NA",
                "GRADIENT DIFF SUM": "NA",
                "TEST PASSED": "NA",
                "TIME": "NA"
            }
        else:
            output_diff = diff(excepted_output, out)
            g_diff = [diff(*args) for args in zip(excepted_grads, grad)]
            sum_g = sum(g_diff)
            frame_out[key.upper()] = {
                "OUT DIFF": output_diff,
                "GRADIENT DIFF SUM": sum_g,
                "TEST PASSED": sum_g < 1 and output_diff < 1e-2,
                "TIME": time_took
            }

    result = pandas.DataFrame.from_dict(frame_out)
    result = result.transpose()
    result.to_csv("attention_output.csv")
    print(result.to_string())


if __name__ == "__main__":
    main()

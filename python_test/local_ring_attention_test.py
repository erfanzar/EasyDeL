import math

import jax.random
from lib.python.EasyDel import MistralConfig
from lib.python.EasyDel.modules.attention_module import ring_attention_standard
from fjformer import GenerateRNG
from jax import numpy as jnp, lax
from flax.linen.attention import dot_product_attention
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from functools import partial

rng = GenerateRNG()
config = MistralConfig()
BATCH_SIZE = 1
SEQUENCE_LENGTH = 512
HEAD_DIM = 8
NUM_HEADS = 32
IMG_LOSS = 0.5649852


def call_vanilla(q, k, v, b):
    attention_pred = dot_product_attention(q, k, v, b)
    return IMG_LOSS, (attention_pred,)


def call_ring(q, k, v, b):
    attention_pred = shard_map(
        partial(
            ring_attention_standard,
            axis_name="sp",
            scale=1 / math.sqrt(HEAD_DIM),
            float32_logits=True,
        ),
        mesh=config.jax_mesh(),
        in_specs=(
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), None, "sp", None)
        ),
        out_specs=(
            PartitionSpec(("dp", "fsdp"), None, "sp", None)
        ),
        check_rep=False
    )(
        q, k, v, b
    )
    return IMG_LOSS, (attention_pred,)


def make_inputs():
    q = jax.random.normal(rng.rng, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), dtype="float32")
    k = jax.random.normal(rng.rng, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), dtype="float32")
    v = jax.random.normal(rng.rng, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM), dtype="float32")
    b = jnp.where(
        jnp.tril(jnp.ones((BATCH_SIZE, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH))),
        0, -jnp.inf,
    )
    return q, k, v, b


def main():
    has_aux = True
    q, k, v, b = make_inputs()
    grad_fn_ring = jax.value_and_grad(call_ring, has_aux=True)
    grad_fn_vanilla = jax.value_and_grad(call_vanilla, has_aux=True)

    (_, (ring_attention_output,)), gradient_ring = grad_fn_ring(q, k, v, b)
    (_, (vanilla_attention_output,)), gradient_vanilla = grad_fn_vanilla(q, k, v, b)

    print(f"VANILLA    : {vanilla_attention_output[0, 0, 0, :5]}")
    print(f"LOCAL RING : {ring_attention_output[0, 0, 0, :5]}")


if __name__ == "__main__":
    main()

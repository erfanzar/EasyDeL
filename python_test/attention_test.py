import math
import os

import fjformer
import flax.linen.attention

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import jax.random
from lib.python.EasyDel import MistralConfig
from lib.python.EasyDel.modules.attention_module import (
    ring_attention_standard,
    flash_attention,
    vanilla_attention,
    wise_ring_attention,
    shard_vanilla_attention,
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


@partial(jax.value_and_grad, has_aux=True)
def call_vanilla(params):
    attention_pred, _ = vanilla_attention(params["q"], params["k"], params["v"], params["b"])
    return IMG_LOSS, (attention_pred,)


@partial(jax.value_and_grad, has_aux=True)
def call_sharded_vanilla(params):
    attention_pred = shard_map(
        partial(
            shard_vanilla_attention,
            deterministic=True,
            dropout_rng=jax.random.key(0),
            dtype=jnp.float32,
            precision=None,
            attention_dropout=0.0
        ),
        mesh=config.jax_mesh(),
        in_specs=(
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), None, None, None),
        ),
        out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
    )(params["q"], params["k"], params["v"], params["b"])
    return IMG_LOSS, (attention_pred,)


@partial(jax.value_and_grad, has_aux=True)
def call_dot_product(params):
    attention_pred = dot_product_attention(params["q"], params["k"], params["v"], params["b"])
    return IMG_LOSS, (attention_pred,)


@partial(jax.value_and_grad, has_aux=True)
def call_wise_ring(params: dict):
    query_states = params.get("q")
    query_sequence_length = query_states.shape[1]
    key_states = params.get("k")
    value_states = params.get("v")
    segment_ids = params.get("segment_ids", None)
    if segment_ids is None:
        segment_ids = jnp.zeros((query_states.shape[0], query_sequence_length), dtype="i4")

    attn_output = shard_map(
        partial(
            wise_ring_attention,
            axis_name="sp",
            float32_logits=True,
            block_wise_kwargs=dict(
                deterministic=True,
                dropout_rng=jax.random.key(0),
                attn_pdrop=0.0,
                causal=True,
                query_chunk_size=CHUNK_WISE_RING,
                key_chunk_size=CHUNK_WISE_RING,
                dtype=jnp.float32,
                policy=jax.checkpoint_policies.nothing_saveable,
                precision=lax.Precision("fastest"),
                prevent_cse=False,
            )
        ),
        mesh=config.jax_mesh(),
        in_specs=(
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            PartitionSpec(("dp", "fsdp"), None, None, None),
            PartitionSpec(("dp", "fsdp"), None),
        ),
        out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        check_rep=False
    )(query_states, key_states, value_states, params["b"], segment_ids)
    attn_output = fjformer.with_sharding_constraint(attn_output, PartitionSpec(("dp", "fsdp"), "sp", "tp", None))

    return IMG_LOSS, attn_output


@partial(jax.value_and_grad, has_aux=True)
def call_flash(params):
    attention_pred = flash_attention(
        params["q"].transpose(0, 2, 1, 3),
        params["k"].transpose(0, 2, 1, 3),
        params["v"].transpose(0, 2, 1, 3),
        params["b"],
        128,
        128,
        -1e10
    )
    return IMG_LOSS, (attention_pred,)


@partial(jax.value_and_grad, has_aux=True)
def call_wise_ring(params):
    attention_pred = AttentionModule(
        attn_mechanism="wise_ring",
        axis_name="sp",
        dtype=jnp.float32,
        mesh=config.jax_mesh(),
        head_dims=params["q"].shape[-1],
        num_attention_heads=params["q"].shape[-2],
        block_q=config.block_q,
        block_k=config.block_k
    ).__call__(
        query_states=params["q"],
        key_states=params["k"],
        value_states=params["v"],
        bias=params["b"],
        query_sequence_length=params["q"].shape[1],
        key_value_sequence_length=params["k"].shape[1]
    ).attention_outputs
    return IMG_LOSS, (attention_pred,)


@partial(jax.value_and_grad, has_aux=True)
def call_ring(params):
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
        params["q"], params["k"], params["v"], params["b"]
    )
    return IMG_LOSS, (attention_pred,)


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

    params = {
        "q": q,
        "k": k,
        "v": v,
        "b": b,
    }

    (_, (sharded_vanilla_attention_output,)), gradient_sharded_vanilla = call_sharded_vanilla(params)
    (_, (vanilla_attention_output,)), gradient_vanilla = call_vanilla(params)
    (_, (ring_attention_output,)), gradient_ring = call_ring(params)
    (_, (flash_attention_output,)), gradient_flash = call_flash(params)
    (_, (dot_p_attention_output,)), gradient_dot_p = call_dot_product(params)
    (_, (wise_ring_attention_output,)), gradient_wise_ring = call_wise_ring(params)

    print(f"SHARD VAN  : {sharded_vanilla_attention_output[0, 0, 0, :5]}")
    print(f"WISE RING  : {wise_ring_attention_output[0, 0, 0, :5]}")
    print(f"VANILLA    : {vanilla_attention_output[0, 0, 0, :5]}")
    print(f"LOCAL RING : {ring_attention_output[0, 0, 0, :5]}")
    print(f"DOT PROD F : {dot_p_attention_output[0, 0, 0, :5]}")
    print(f"FLASH      : {flash_attention_output[0, 0, 0, :5]}")


if __name__ == "__main__":
    main()

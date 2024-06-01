import math

import flax.linen.attention as flt
import jax
from fjformer import GenerateRNG
from jax import numpy as jnp, random, lax

from easydel import PartitionAxis
from easydel.modules.attention_module import AttentionModule
from easydel.modules.easydel_modelling_utils import EasyDeLPretrainedConfig

BATCH_SIZE = len(jax.devices())
NUM_ATTN_HEADS = 32
CONTEXT_LENGTH = 8192
HEAD_DIM = 256


def main():
    rng_gen = GenerateRNG(seed=42)
    config = EasyDeLPretrainedConfig(
        axis_dims=(1, -1, 1, 1),
        axis_names=("dp", "fsdp", "tp", "sp"),
        block_q=512,
        block_k=512
    )

    def make_fake_input_data(
            batch_size: int,
            num_attention_head: int,
            context_length: int,
            head_dim: int,
    ):
        q = random.normal(next(rng_gen), (batch_size, context_length, num_attention_head, head_dim), dtype=jnp.float32)
        k = random.normal(next(rng_gen), (batch_size, context_length, num_attention_head, head_dim), dtype=jnp.float32)
        v = random.normal(next(rng_gen), (batch_size, context_length, num_attention_head, head_dim), dtype=jnp.float32)

        attention_mask = jnp.ones((batch_size, context_length))
        causal_mask = flt.make_causal_mask(attention_mask)

        cm_ = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        at_ = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), cm_.shape)
        at_ = flt.combine_masks(at_, cm_)

        attention_bias = lax.select(
            at_ > 0,
            jnp.full(at_.shape, 0.0).astype(jnp.float32),
            jnp.full(at_.shape, jnp.finfo(jnp.float32).min).astype(jnp.float32),
        )

        return (
            q, k, v, attention_mask, causal_mask, attention_bias
        )

    q, k, v, attention_mask, causal_mask, attention_bias = make_fake_input_data(
        BATCH_SIZE,
        NUM_ATTN_HEADS,
        CONTEXT_LENGTH,
        HEAD_DIM
    )

    flash_attention = AttentionModule(

        block_k_major=config.block_k_major,
        block_b=config.block_b,
        block_q=config.block_q,
        block_k=config.block_k,
        block_q_major_dkv=config.block_q_major_dkv,
        block_k_major_dkv=config.block_k_major_dkv,
        block_k_major_dq=config.block_k_major_dq,
        block_k_dkv=config.block_k_dkv,
        block_q_dkv=config.block_q_dkv,
        block_q_dq=config.block_q_dq,
        block_k_dq=config.block_k_dq,
        num_attention_heads=NUM_ATTN_HEADS,
        attention_dropout=0.0,
        head_dims=HEAD_DIM,
        shard_attention_computation=config.shard_attention_computation,
        precision=lax.Precision("fastest"),
        force_float32_tpu=True,
        attn_mechanism="flash",
        dtype=jnp.float32,
        partition_axis=PartitionAxis(
            batch_axis=("dp", "fsdp"),
            query_sequence_axis="sp",
            key_sequence_axis="sp",
            head_axis="tp",
            attention_dim_axis=None
        ),
        scan_ring_attention=config.scan_ring_attention,
        mesh=config.get_mesh(),
        sm_scale=1 / math.sqrt(q.shape[-1]),
        axis_name=config.attention_axis_name
    )

    normal_attention = AttentionModule(

        block_k_major=config.block_k_major,
        block_b=config.block_b,
        block_q=config.block_q,
        block_k=config.block_k,
        block_q_major_dkv=config.block_q_major_dkv,
        block_k_major_dkv=config.block_k_major_dkv,
        block_k_major_dq=config.block_k_major_dq,
        block_k_dkv=config.block_k_dkv,
        block_q_dkv=config.block_q_dkv,
        block_q_dq=config.block_q_dq,
        block_k_dq=config.block_k_dq,
        num_attention_heads=NUM_ATTN_HEADS,
        attention_dropout=0.0,
        head_dims=HEAD_DIM,
        partition_axis=PartitionAxis(
            batch_axis=("dp", "fsdp"),
            query_sequence_axis="sp",
            key_sequence_axis="sp",
            head_axis="tp",
            attention_dim_axis=None
        ),
        shard_attention_computation=config.shard_attention_computation,
        precision=lax.Precision("fastest"),
        force_float32_tpu=True,
        attn_mechanism="sharded_vanilla",
        dtype=jnp.float32,
        scan_ring_attention=config.scan_ring_attention,
        mesh=config.get_mesh(),
        sm_scale=1 / math.sqrt(q.shape[-1]),
        axis_name=config.attention_axis_name
    )

    with config.get_mesh():
        flash_attn_out = flash_attention(
            query_states=q,
            key_states=k,
            value_states=v,
            bias=attention_bias,
            key_value_sequence_length=CONTEXT_LENGTH,
            query_sequence_length=CONTEXT_LENGTH
        )
        normal_attn_out = normal_attention(
            query_states=q,
            key_states=k,
            value_states=v,
            bias=attention_bias,
            key_value_sequence_length=CONTEXT_LENGTH,
            query_sequence_length=CONTEXT_LENGTH
        )

    print(
        flash_attn_out.attention_outputs[0, CONTEXT_LENGTH - 5, NUM_ATTN_HEADS - 1, HEAD_DIM - 10:]
    )
    # Array([-0.05915311,  0.0078501 ,  0.03785717,  0.0134844 ,  0.08464689,
    #        0.06667967, -0.02629154, -0.0180066 , -0.02972782,  0.02833381],      dtype=float32)
    print(
        normal_attn_out.attention_outputs[0, CONTEXT_LENGTH - 5, NUM_ATTN_HEADS - 1, HEAD_DIM - 10:]
    )

    # Array([-0.0590958 ,  0.00796138,  0.03789062,  0.01350671,  0.08461153,
    #        0.06662725, -0.0262386 , -0.01806086, -0.0296791 ,  0.02824247],      dtype=float32)


if __name__ == "__main__":
    main()

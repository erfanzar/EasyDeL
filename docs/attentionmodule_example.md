# FlexibleAttentionModule: A Versatile Attention Mechanism Factory

The `FlexibleAttentionModule` class is designed to simplify the creation and execution of different attention mechanisms within
your EasyDeL models. It provides a unified interface for working with various attention types, allowing you to easily
switch between them and experiment with different configurations.

**Key Features:**

* **Mechanism Selection:** The `attn_mechanism` argument lets you choose the specific attention algorithm you want to
  use (e.g., "vanilla," "flash," "splash," "ring," "cudnn").
* **Sharding and Partitioning:** The class supports advanced JAX sharding techniques to distribute attention
  computations across multiple devices for efficient processing of large models. It handles partitioning of query, key,
  value, bias, and attention weight matrices using `PartitionSpec`.
* **Blockwise Attention:** Enables the use of blockwise attention for increased memory efficiency, especially with long
  sequences.
* **Caching Support:** Facilitates the use of attention caching to speed up inference and generation tasks.
* **Dropout and Determinism:** Allows for applying dropout to attention weights and controlling the deterministic
  behavior of the attention computation.
* **Testing Utility:**  Provides a `test_attentions` method to compare different attention mechanisms in terms of
  accuracy, gradient stability, and computation time.

**How it Works:**

1. **Initialization:**
    - During initialization, you provide the desired `attn_mechanism`, JAX `mesh` for sharding, scaling
      factor (`sm_scale`), number of attention heads, head dimensions, and other configuration parameters.
    - The class automatically sets default values for many parameters based on the chosen attention mechanism and the
      provided EasyDeL configuration (`base_module_class`).
2. **Calling the Module:**
    - When you call the `FlexibleAttentionModule` object, you pass in the query, key, and value states, along with optional
      parameters like attention masks, biases, and causal flags.
    - The module internally selects the appropriate attention function based on the specified `attn_mechanism`.
    - It performs any necessary sharding and partitioning based on the configured partition specifications.
    - The attention computation is executed, and the attention outputs (and optionally attention weights) are returned.

**Advantages:**

* **Flexibility:**  Allows you to easily switch between different attention mechanisms without major code changes.
* **Efficiency:**  Supports advanced JAX sharding for distributed computation, enabling the handling of large models.

FlexibleAttentionModule is a EasyDeL module that can perform attention operation with different strategies to help user achieve
the best possible performance and numerical stability, here are some strategies supported right now.

1. Flash Attention TPU/GPU/CPU known as "flash_attn2"
2. Ring Attention to Support higher context length such 1 Million or above known as "ring"
3. Normal Attention which use flax.linen.attention with shard map known as "vanilla"
4. Splash Attention on TPUs which is known as "splash"
5. Other Attention modules might be added you can check source code for that..

## Testing which Attention Module works best

in order to test which attention module in what axis dims works best for you you can run
```python
from easydel import FlexibleAttentionModule

print(
    FlexibleAttentionModule.test_attentions(
        axis_dims=(1, 1, 1, -1),
        sequence_length=128 * 8,
        num_attention_heads=32,
        num_key_value_heads=32,
        chunk_size=128,

    )
)
```

## Example of Using Flash Attention on TPU

```python
import jax
import flax.linen.attention as flt
from fjformer import GenerateRNG
from easydel import PartitionAxis
from easydel.modules.attention_module import FlexibleAttentionModule
from easydel.modules.easydel_modelling_utils import EDPretrainedConfig
from jax import numpy as jnp, random, lax
import math

rng_gen = GenerateRNG(seed=42)
config = EDPretrainedConfig(
    axis_dims=(1, -1, 1, 1),
    axis_names=("dp", "fsdp", "tp", "sp"),
    block_q=512,
    block_k=512
)

BATCH_SIZE = len(jax.devices())
NUM_ATTN_HEADS = 32
CONTEXT_LENGTH = 8192
HEAD_DIM = 256


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

flash_attention = FlexibleAttentionModule(

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
    attn_mechanism="flash",
    dtype=jnp.float32,
    scan_ring_attention=config.scan_ring_attention,
    mesh=config.mesh,
    sm_scale=1 / math.sqrt(q.shape[-1]),
)

normal_attention = FlexibleAttentionModule(

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
    attn_mechanism="jax_flash_attn2",
    dtype=jnp.float32,
    scan_ring_attention=config.scan_ring_attention,
    mesh=config.mesh,
    sm_scale=1 / math.sqrt(q.shape[-1]),
)

with config.mesh:
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
```
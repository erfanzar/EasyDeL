# Base Configuration System

The `EasyDeLBaseConfig` class is the foundation of EasyDeL's configuration system. Every model in EasyDeL has a configuration class that inherits from it, defining the model's architecture, sharding strategy, and runtime behavior.

## Understanding EasyDeLBaseConfig

### Basic Structure

```python
from easydel import EasyDeLBaseConfig

class MyModelConfig(EasyDeLBaseConfig):
    model_type = "my_model"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        vocab_size: int = 32000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
```

### EasyDeL-Specific Parameters

The base config accepts several EasyDeL-specific parameters through `kwargs`:

```python
config = MyModelConfig(
    # Model architecture
    hidden_size=4096,
    num_hidden_layers=32,

    # EasyDeL-specific sharding parameters (5D)
    sharding_axis_dims=(1, -1, 1, 1, 1),  # (dp, fsdp, ep, tp, sp)
    sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    sharding_dcn_axis_dims=None,          # For multi-node (DCN) setups

    # Attention configuration
    attn_dtype=jnp.bfloat16,              # Attention computation dtype
    attn_mechanism="flash",                # Attention backend
    blocksize_q=128,                       # Query block size for attention
    blocksize_k=128,                       # Key block size for attention

    # Gradient checkpointing
    gradient_checkpointing="nothing_saveable",

    # MoE-specific (for expert parallelism)
    moe_method="einsum",                  # MoE implementation method
    fsdp_is_ep_bound=False,               # Bind FSDP to expert parallelism
    sp_is_ep_bound=False,                 # Bind SP to expert parallelism

    # Hardware optimization
    hardware_abstraction=True,
)
```

## Mesh and Sharding

### Creating a Mesh

The configuration manages the JAX mesh for distributed computation:

```python
# Automatic mesh creation
mesh = config.mesh  # Creates mesh based on axis_dims and axis_names

# Manual mesh creation with specific devices
mesh = config.create_mesh(
    axis_dims=(1, 2, 1, 4, 1),  # 2-way FSDP, 4-way tensor parallel
    axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    backend="tpu",
)

# Access the JAX mesh directly
jax_mesh = config.jax_mesh()
```

### Understanding Axis Dimensions (5D Sharding)

EasyDeL uses a 5-dimensional sharding scheme:

```python
# sharding_axis_dims = (dp, fsdp, ep, tp, sp)
# dp:   Data Parallelism (batch splitting)
# fsdp: Fully Sharded Data Parallelism (parameter sharding)
# ep:   Expert Parallelism (for MoE models)
# tp:   Tensor Parallelism (layer sharding)
# sp:   Sequence Parallelism (sequence splitting)

# Example: 8 TPU v4 chips with full FSDP
config = MyModelConfig(
    sharding_axis_dims=(1, 8, 1, 1, 1),  # 8-way FSDP
)

# Example: 32 TPU v4 chips with tensor parallelism
config = MyModelConfig(
    sharding_axis_dims=(1, 4, 1, 8, 1),  # 4-way FSDP, 8-way TP
)

# Example: MoE model with expert parallelism
config = MyMoEModelConfig(
    sharding_axis_dims=(1, 4, 8, 1, 1),  # 4-way FSDP, 8-way EP
)

# Use -1 to auto-compute based on available devices
config = MyModelConfig(
    sharding_axis_dims=(1, -1, 1, 1, 1),  # FSDP uses all available devices
)

# Multi-node (DCN) setup
config = MyModelConfig(
    sharding_axis_dims=(1, 4, 1, 2, 1),      # Intra-node sharding
    sharding_dcn_axis_dims=(1, 2, 1, 1, 1),  # Inter-node (DCN) sharding
)
```

### Partition Rules

Models define how their parameters should be sharded. With 5D sharding, you can use any combination of the axes:

```python
class MyModelConfig(EasyDeLBaseConfig):
    def get_partition_rules(self, fully_sharded: bool = True):
        """Define sharding rules for model parameters."""
        return (
            # Embedding layers - shard across tp and fsdp
            ("embed_tokens/embedding", PartitionSpec("tp", "fsdp")),

            # Attention layers
            ("self_attn/q_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("self_attn/k_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("self_attn/v_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", "fsdp")),

            # MLP layers
            ("mlp/gate_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("mlp/up_proj/kernel", PartitionSpec("fsdp", "tp")),
            ("mlp/down_proj/kernel", PartitionSpec("tp", "fsdp")),

            # Default: replicate
            (".*", PartitionSpec()),
        )


class MyMoEModelConfig(EasyDeLBaseConfig):
    def get_partition_rules(self, fully_sharded: bool = True):
        """Partition rules for MoE model with expert parallelism."""
        return (
            # Regular layers same as above
            ("embed_tokens/embedding", PartitionSpec("tp", "fsdp")),
            ("self_attn/.*", PartitionSpec("fsdp", "tp")),

            # MoE expert layers - shard across expert parallel axis
            ("experts/.*/gate_proj/kernel", PartitionSpec("ep", "fsdp", "tp")),
            ("experts/.*/up_proj/kernel", PartitionSpec("ep", "fsdp", "tp")),
            ("experts/.*/down_proj/kernel", PartitionSpec("ep", "tp", "fsdp")),

            # Router stays replicated or lightly sharded
            ("router/kernel", PartitionSpec("fsdp")),

            # Default
            (".*", PartitionSpec()),
        )
```

## Attention Configuration

### Attention Mechanisms

EasyDeL supports multiple attention backends:

```python
config = MyModelConfig(
    attn_mechanism="flash",        # Flash Attention 2
    # attn_mechanism="vanilla",    # Standard attention
    # attn_mechanism="ring",       # Ring Attention (long sequences)
    # attn_mechanism="splash",     # Splash Attention (TPU)
    # attn_mechanism="cudnn",      # cuDNN attention (GPU)
    # attn_mechanism="blockwise",  # Blockwise attention
)
```

### Block Sizes

Configure attention block sizes for memory efficiency:

```python
config = MyModelConfig(
    attn_mechanism="flash",
    blocksize_q=128,      # Query block size
    blocksize_k=128,      # Key block size
    blocksize_b=1,        # Batch block size
)
```

## RoPE Configuration

### Basic RoPE Setup

```python
config = MyModelConfig(
    rope_theta=10000.0,           # RoPE base frequency
    max_position_embeddings=4096,  # Maximum sequence length
    rope_scaling=None,             # Optional scaling configuration
)
```

### RoPE Scaling for Extended Context

```python
# Linear scaling
config = MyModelConfig(
    rope_scaling={
        "type": "linear",
        "factor": 2.0,  # Extends context by 2x
    }
)

# Dynamic NTK scaling
config = MyModelConfig(
    rope_scaling={
        "type": "dynamic",
        "factor": 2.0,
    }
)

# YaRN scaling
config = MyModelConfig(
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 4096,
    }
)
```

### Accessing RoPE Values

```python
# Get precomputed frequencies
frequencies = config.get_basic_frequencies(
    head_size=128,
    rotary_dim=128,
    base=10000.0,
    max_position_embeddings=4096,
)

# Get full RoPE tensors (sin/cos)
sin, cos = config.get_basic_rope(
    dtype=jnp.float32,
    head_size=128,
    rotary_dim=128,
    max_position_embeddings=4096,
)
```

## Causal Mask Configuration

### Getting Causal Masks

```python
# Basic causal mask
causal_mask = config.get_basic_causal_mask(
    max_length=2048,
    dtype=jnp.float32,
)

# With forgetful causal masking (FCM)
fcm_mask = config.get_fcm_mask(
    key=jax.random.PRNGKey(0),
    shape=(batch_size, 1, seq_len, seq_len),
    fcm_ratio=0.1,  # 10% dropout
)
```

## Serialization

### Saving and Loading

```python
# Save to directory
config.save_pretrained("./my-model")

# Load from directory or HuggingFace Hub
config = MyModelConfig.from_pretrained("./my-model")
config = MyModelConfig.from_pretrained("organization/model-name")

# Convert to dictionary
config_dict = config.to_dict()

# Save to JSON
config.to_json_file("config.json")
```

### Handling Extra Arguments

```python
# Attach custom arguments that persist through serialization
config.attach_custom_arguments(
    my_custom_param="value",
    another_param=42,
)
```

## Best Practices

### 1. Always Specify Sharding for Large Models

```python
config = LlamaConfig.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    sharding_axis_dims=(1, -1, 1, 1, 1),  # Use FSDP (dp, fsdp, ep, tp, sp)
)

# For MoE models, consider expert parallelism
config = DeepseekV3Config.from_pretrained(
    "deepseek/DeepSeek-V3",
    sharding_axis_dims=(1, 4, 8, 1, 1),  # FSDP + Expert Parallel
)
```

### 2. Match Attention Mechanism to Hardware

```python
# TPU
config = MyModelConfig(attn_mechanism="flash")  # Pallas-based

# GPU with Triton
config = MyModelConfig(attn_mechanism="flash")  # Triton-based

# CPU or debugging
config = MyModelConfig(attn_mechanism="vanilla")
```

### 3. Use Gradient Checkpointing for Memory Efficiency

```python
config = MyModelConfig(
    gradient_checkpointing="nothing_saveable",  # Maximum memory savings
    # gradient_checkpointing="everything_saveable",  # Faster but more memory
)
```

### 4. Set Appropriate dtypes

```python
config = MyModelConfig(
    attn_dtype=jnp.float32,  # Keep attention in fp32 for stability
)
```

## Configuration Inheritance

When creating a custom model config, inherit properly:

```python
class MyCustomConfig(EasyDeLBaseConfig):
    model_type = "my_custom_model"

    def __init__(
        self,
        # Your custom parameters
        custom_param: int = 42,
        # Pass through to parent
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def get_partition_rules(self, fully_sharded: bool = True):
        # Define your sharding rules
        return (
            # Your rules here
            (".*", PartitionSpec()),
        )
```

## Next Steps

- [Base Module Guide](base_module.md) - Learn about the module system
- [Customization Guide](customization.md) - Customize configurations
- [Adding Your Own Model](adding_models.md) - Create custom model configs

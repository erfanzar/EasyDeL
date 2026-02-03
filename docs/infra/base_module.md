# Base Module System

The `EasyDeLBaseModule` class is the foundation for all neural network models in EasyDeL. Built on Flax NNX, it provides a rich set of features including automatic sharding, quantization, LoRA support, and seamless integration with the HuggingFace ecosystem.

## Understanding EasyDeLBaseModule

### Class Hierarchy

```md
flax.nnx.Module
    └── EasyDeLBaseModule
            ├── EasyGenerationMixin      (generation capabilities)
            ├── EasyBridgeMixin          (HuggingFace compatibility)
            └── OperationCacheMixin      (cache management)
```

### Basic Structure

```python
import flax.nnx as nnx
from easydel import EasyDeLBaseModule

class MyModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: MyModelConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Initialize your layers here
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )
        # ... more layers

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        **kwargs,
    ):
        # Forward pass implementation
        pass
```

## Key Properties and Methods

### Model Information

```python
model = MyModel(config, rngs=nnx.Rngs(0))

# Access configuration
model.config              # The model's config object

# Model metadata
model.model_type          # e.g., "llama", "mistral"
model.model_task          # e.g., "causal-lm", "seq2seq"

# Get the mesh for distributed computation
model.mesh                # Returns the JAX mesh
```

### Parameter Access

```python
# Get all parameters as a dictionary
params = model.params

# Get parameter shapes
shapes = model.graphtree_params_shape

# Split into graph definition and state
graphdef, state = model.split_module()

# Merge back
model = model.merge_module(graphdef, state)
```

### Graph Operations (Flax NNX)

```python
# Get graph components
graphdef = model.graphdef      # Graph definition (structure)
graphstate = model.graphstate  # Graph state (parameters + other)
graphother = model.graphother  # Non-parameter state

# Split parameters from other state
params_dict = model.split_params_dict()  # Returns dict
params_state = model.split_params()       # Returns State object

# Merge parameters back
model.merge_params_dict(params_dict)
model.merge_params(params_state)
```

## Sharding and Distribution

### Automatic Sharding

```python
# Shard model across devices
model.shard_model(
    partition_rules=None,  # Uses config.get_partition_rules() if None
    mesh=None,             # Uses config.mesh if None
)

# Gather model back to single device (for saving)
model.gather_model()
```

### Manual Sharding Control

```python
# Get partition rules
rules = model._get_partition_rules()

# Get sharding specifications
specs = model._specs_sharding(rules)

# Get named shardings
named_shardings = model._named_shardings(rules, mesh)

# Apply custom sharding
model._apply_sharding_fns(
    tree,
    partition_rules,
    mesh,
    is_gather=False,  # True to gather, False to shard
)
```

### Full Sharding Operations

```python
# Fully shard (all parameters across all devices)
model.fully_shard(mesh=None)

# Fully gather (all parameters to single device)
model.fully_gather(mesh=None)

# Get sharding functions
shard_fns = model._shard_fns(rules, mesh)
gather_fns = model._gather_fns(rules, mesh)
```

## Dtype Management

### Converting dtypes

```python
# Convert to specific dtype
model.to_dtype(jnp.bfloat16)

# Convenience methods
model.half()   # Convert to float16
model.float()  # Convert to float32

# Access current dtype
current_dtype = model.module_dtype
```

## Quantization

### Applying Quantization

```python
# Quantize the model
model.quantize(
    method="nf4",           # Quantization method
    group_size=64,          # Group size for quantization
    quantization_pattern=".*",  # Regex pattern for layers to quantize
)

# Check if model is quantized
is_quant = model.is_quantized
```

### Supported Quantization Methods

- `"nf4"` - 4-bit NormalFloat
- `"8bit"` - 8-bit quantization
- `"a8bit"` - Activation-aware 8-bit
- `"a8q"` - Activation quantization
- `"a4q"` - 4-bit activation quantization

## LoRA (Low-Rank Adaptation)

### Applying LoRA

```python
# Apply LoRA to specific layers
model.apply_lora_to_layers(
    lora_rank=8,
    lora_alpha=16,
    lora_pattern=".*q_proj.*|.*v_proj.*",  # Apply to Q and V projections
)

# Check if LoRA is enabled
has_lora = model.lora_is_enabled
```

### Managing LoRA Parameters

```python
# Split LoRA parameters from base parameters
base_params, lora_params = model.split_lora_params()

# Merge LoRA parameters back
model.merge_lora_params()

# Remove LoRA (merge weights permanently)
model.unwrap_lora_to_layers()
```

## Loss Computation

### Built-in Loss Functions

```python
# Compute loss during forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,  # Providing labels triggers loss computation
)
loss = outputs.loss

# Or compute loss separately
loss, metrics = model.compute_loss(
    labels=labels,
    logits=outputs.logits,
    config=loss_config,
)
```

### Loss Configuration

```python
from easydel.infra import LossConfig

loss_config = LossConfig(
    loss_type="cross_entropy",
    label_smoothing=0.0,
    z_loss_coefficient=0.0,
    auxiliary_loss_coefficient=0.001,  # For MoE models
)
```

## State Conversion

### Converting to EasyDeLState

```python
# Convert model to training state
state = model.to_state(
    optimizer=optax.adamw(learning_rate=1e-4),
)

# State includes:
# - Model parameters
# - Optimizer state
# - Step counter
```

### Converting to PyTorch

```python
# Convert to PyTorch format
torch_params = model.to_torch()

# This returns a dictionary compatible with
# transformers model.load_state_dict()
```

## Generation

Models inheriting generation capabilities can generate text. The `generate()` method handles cache initialization internally via `init_operations_cache`:

```python
# Simple generation - cache is handled automatically
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

# For manual cache control (advanced usage)
cache = model.init_operations_cache(
    batch_size=1,
    max_length=2048,
    starts=jnp.array([0]),  # Starting positions
)
```

## Saving and Loading

### Save Model

```python
# Save to local directory
model.save_pretrained(
    "./my-model",
    max_shard_size="5GB",
)

# Push to HuggingFace Hub
model.push_to_hub(
    "my-username/my-model",
    commit_message="Upload model",
)
```

### Load Model

```python
# Load from local or HuggingFace Hub
model = MyModel.from_pretrained(
    "organization/model-name",
    dtype=jnp.bfloat16,
    auto_shard_model=True,
)
```

## Static Arguments for JIT

When JIT compiling, some arguments should be static:

```python
# Get static argument names
static_args = model.get_static_arguments()

# Prepare inputs with static args handled correctly
prepared_inputs = model.prepare_inputs_for_call(**inputs)
```

## FLOP Calculation

```python
# Calculate FLOPs per token
flops = model.flops_per_token(
    batch_size=1,
    sequence_length=2048,
)

# Low-level FLOP calculation
flops = model._flop(config, batch_size, sequence_length)
```

## Transform Functions

For functional-style operations:

```python
# Get a pure transform function (no side effects)
transform_fn = model.pure_transform_fn()

# Use in JAX transformations
@jax.jit
def train_step(params, inputs):
    outputs = transform_fn(params, inputs)
    return outputs.loss
```

## Best Practices

### 1. Always Initialize with rngs

```python
model = MyModel(
    config=config,
    rngs=nnx.Rngs(0),  # Provide random key
)
```

### 2. Use Appropriate dtypes

```python
model = MyModel(
    config=config,
    dtype=jnp.bfloat16,      # Computation dtype
    param_dtype=jnp.float32,  # Parameter storage dtype
    rngs=nnx.Rngs(0),
)
```

### 3. Shard Before Training

```python
model = MyModel.from_pretrained("model-name")
model.shard_model()  # Distribute across devices
state = model.to_state(optimizer)
```

### 4. Gather Before Saving

```python
model.gather_model()  # Collect to single device
model.save_pretrained("./output")
```

## Next Steps

- [Customization Guide](customization.md) - Learn to customize modules
- [Adding Your Own Model](adding_models.md) - Create custom models
- [eLargeModel Guide](elarge_model.md) - High-level training API

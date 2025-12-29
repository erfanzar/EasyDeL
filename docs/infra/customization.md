# Customization Guide

EasyDeL is designed to be fully customizable. This guide covers how to override default behaviors, customize components, and extend the framework to suit your specific needs.

## Customizing Configuration

### Overriding Default Values

```python
from easydel.modules.llama import LlamaConfig

# Override defaults when creating config
config = LlamaConfig(
    hidden_size=4096,
    num_hidden_layers=32,
    # Override EasyDeL defaults (5D sharding: dp, fsdp, ep, tp, sp)
    attn_mechanism="flash",
    sharding_axis_dims=(1, -1, 1, 1, 1),
    gradient_checkpointing="nothing_saveable",
)
```

### Adding Custom Configuration Parameters

```python
class MyCustomConfig(LlamaConfig):
    model_type = "my_custom_llama"

    def __init__(
        self,
        # New custom parameters
        use_custom_attention: bool = False,
        custom_scaling_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_custom_attention = use_custom_attention
        self.custom_scaling_factor = custom_scaling_factor
```

### Custom Partition Rules

Override sharding behavior for your specific hardware. EasyDeL uses 5D sharding with axes: `(dp, fsdp, ep, tp, sp)`:

```python
class MyModelConfig(EasyDeLBaseConfig):
    def get_partition_rules(self, fully_sharded: bool = True):
        """Custom partition rules for your model."""
        if fully_sharded:
            return (
                # Shard embeddings across TP and FSDP axes
                ("embed_tokens/embedding", PartitionSpec("tp", "fsdp")),

                # Custom attention sharding
                ("self_attn/q_proj/kernel", PartitionSpec("fsdp", "tp")),
                ("self_attn/k_proj/kernel", PartitionSpec("fsdp", "tp")),
                ("self_attn/v_proj/kernel", PartitionSpec("fsdp", "tp")),
                ("self_attn/o_proj/kernel", PartitionSpec("tp", "fsdp")),

                # MLP layers
                ("mlp/gate_proj/kernel", PartitionSpec("fsdp", "tp")),
                ("mlp/up_proj/kernel", PartitionSpec("fsdp", "tp")),
                ("mlp/down_proj/kernel", PartitionSpec("tp", "fsdp")),

                # For MoE models - use expert parallel (ep) axis
                ("experts/.*/kernel", PartitionSpec("ep", "fsdp", "tp")),

                # Shard layer norms minimally
                (".*norm.*", PartitionSpec()),

                # Default: replicate everything else
                (".*", PartitionSpec()),
            )
        else:
            # Non-sharded rules (for inference)
            return ((".*", PartitionSpec()),)
```

### Custom RoPE Implementation

```python
class MyModelConfig(EasyDeLBaseConfig):
    def get_basic_frequencies(
        self,
        head_size: int,
        rotary_dim: int,
        base: float,
        max_position_embeddings: int,
    ):
        """Custom frequency computation for RoPE."""
        # Your custom implementation
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, rotary_dim, 2) / rotary_dim)
        )
        # Apply custom scaling
        inv_freq = inv_freq * self.custom_scaling_factor
        return inv_freq
```

## Customizing Modules

### Overriding Forward Pass

```python
from easydel.modules.llama import LlamaForCausalLM

class MyCustomLlama(LlamaForCausalLM):
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        **kwargs,
    ):
        # Add custom preprocessing
        input_ids = self.preprocess_inputs(input_ids)

        # Call parent implementation
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Add custom postprocessing
        outputs = self.postprocess_outputs(outputs)
        return outputs

    def preprocess_inputs(self, input_ids):
        """Custom input preprocessing."""
        return input_ids

    def postprocess_outputs(self, outputs):
        """Custom output postprocessing."""
        return outputs
```

### Custom Attention Implementation

```python
from easydel.layers import EasyAttention

class MyCustomAttention(EasyAttention):
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        attention_mask: jax.Array = None,
        **kwargs,
    ):
        # Custom attention logic
        if self.config.use_custom_attention:
            return self.custom_attention_forward(
                query, key, value, attention_mask
            )
        else:
            return super().__call__(
                query, key, value, attention_mask, **kwargs
            )

    def custom_attention_forward(self, q, k, v, mask):
        """Your custom attention implementation."""
        # Implement your custom attention here
        pass
```

### Custom Loss Function

```python
class MyModelForCausalLM(EasyDeLBaseModule):
    @property
    def loss_function(self):
        """Override the default loss function."""
        return self.custom_loss_fn

    def custom_loss_fn(
        self,
        logits: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array = None,
    ):
        """Custom loss computation."""
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Custom loss calculation
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )

        # Apply custom weighting
        if attention_mask is not None:
            mask = attention_mask[..., 1:]
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss
```

## Customizing Generation

### Custom Sampling Strategy

```python
class MyModel(LlamaForCausalLM):
    def _sample(
        self,
        input_ids,
        attention_mask,
        max_length,
        **kwargs,
    ):
        """Custom sampling implementation."""
        # Your custom sampling logic
        pass

    def _get_logits_processor(self, generation_config, **kwargs):
        """Add custom logits processors."""
        processors = super()._get_logits_processor(
            generation_config, **kwargs
        )
        # Add your custom processor
        processors.append(MyCustomLogitsProcessor())
        return processors
```

### Custom Stopping Criteria

```python
from transformers import StoppingCriteria

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores, **kwargs):
        # Check if any stop token is generated
        for stop_token in self.stop_tokens:
            if input_ids[0, -1] == stop_token:
                return True
        return False

# Use in generation
model.generate(
    input_ids,
    stopping_criteria=[MyStoppingCriteria([eos_token_id])],
)
```

## Customizing Training

### Custom Training State

```python
from easydel import EasyDeLState
import optax

class MyTrainingState(EasyDeLState):
    # Add custom fields
    custom_metrics: dict = None

    @classmethod
    def create(cls, model, optimizer, **kwargs):
        state = super().create(model=model, optimizer=optimizer)
        state = state.replace(custom_metrics={})
        return state

    def update_metrics(self, new_metrics):
        """Update custom metrics."""
        updated = {**self.custom_metrics, **new_metrics}
        return self.replace(custom_metrics=updated)
```

### Custom Optimizer Configuration

```python
def create_custom_optimizer(
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
):
    """Create a custom optimizer with schedule."""
    # Learning rate schedule
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, learning_rate, warmup_steps),
            optax.cosine_decay_schedule(learning_rate, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )

    # Optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.95,
        ),
    )

    return optimizer
```

## Customizing Sharding

### Custom Shard/Gather Functions

```python
class MyModel(EasyDeLBaseModule):
    def _shard_fns(self, rules, mesh):
        """Custom sharding functions."""
        fns = super()._shard_fns(rules, mesh)
        # Modify or add custom sharding logic
        return fns

    def _gather_fns(self, rules, mesh):
        """Custom gathering functions."""
        fns = super()._gather_fns(rules, mesh)
        # Modify or add custom gathering logic
        return fns
```

### Custom Mesh Creation

```python
class MyModelConfig(EasyDeLBaseConfig):
    def create_mesh(
        self,
        axis_dims=None,  # 5D: (dp, fsdp, ep, tp, sp)
        axis_names=None,
        backend=None,
    ):
        """Custom mesh creation logic."""
        # Custom device arrangement
        devices = jax.devices()

        # Custom mesh shape - must produce 5D mesh
        mesh_shape = self._compute_optimal_mesh_shape(devices)

        # Create mesh with 5 axes
        mesh = jax.sharding.Mesh(
            np.array(devices).reshape(mesh_shape),
            axis_names=axis_names or self.sharding_axis_names,  # 5D names
        )
        return mesh
```

## Customizing Weight Loading

### Custom Weight Transformation

```python
class MyModel(EasyDeLBaseModule):
    @classmethod
    def _from_torch_pretrained(cls, model_id, config, **kwargs):
        """Custom weight loading from PyTorch."""
        # Load weights with custom transformations
        model = super()._from_torch_pretrained(model_id, config, **kwargs)

        # Apply custom weight modifications
        model = cls._apply_custom_weight_transforms(model)
        return model

    @staticmethod
    def _apply_custom_weight_transforms(model):
        """Apply custom transformations to loaded weights."""
        # Your custom transformations
        return model
```

### Custom Parameter Renaming

```python
def custom_param_mapping(torch_key: str) -> str:
    """Map PyTorch parameter names to EasyDeL names."""
    mappings = {
        "transformer.h": "model.layers",
        "ln_1": "input_layernorm",
        "ln_2": "post_attention_layernorm",
    }
    for old, new in mappings.items():
        torch_key = torch_key.replace(old, new)
    return torch_key
```

## Customizing Quantization

### Custom Quantization Patterns

```python
# Quantize only specific layers
model.quantize(
    method="nf4",
    quantization_pattern=".*mlp.*",  # Only MLP layers
)

# Exclude certain layers
model.quantize(
    method="8bit",
    quantization_pattern="(?!.*embed)(?!.*norm).*",  # Exclude embeddings and norms
)
```

### Custom Quantization Method

```python
from easydel.layers.quantization import BaseQuantizer

class MyQuantizer(BaseQuantizer):
    def quantize(self, weight):
        """Custom quantization logic."""
        pass

    def dequantize(self, quantized_weight):
        """Custom dequantization logic."""
        pass
```

## Best Practices

### 1. Preserve Parent Functionality

```python
class MyModel(ParentModel):
    def __call__(self, **kwargs):
        # Add your logic
        result = super().__call__(**kwargs)  # Call parent
        # More logic
        return result
```

### 2. Use Configuration for Customization

```python
# Good: Configurable behavior
class MyModel(EasyDeLBaseModule):
    def __call__(self, **kwargs):
        if self.config.use_custom_feature:
            return self.custom_forward(**kwargs)
        return self.standard_forward(**kwargs)
```

### 3. Document Custom Behavior

```python
class MyCustomConfig(EasyDeLBaseConfig):
    """
    Custom configuration for MyModel.

    Args:
        custom_param: Description of what this parameter does.
            Defaults to False.
    """
    def __init__(self, custom_param: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
```

### 4. Test Custom Components

```python
def test_custom_attention():
    config = MyModelConfig(use_custom_attention=True)
    model = MyModel(config, rngs=nnx.Rngs(0))

    # Test that custom attention works
    outputs = model(input_ids=jnp.ones((1, 10), dtype=jnp.int32))
    assert outputs.logits is not None
```

## Next Steps

- [Adding Your Own Model](adding_models.md) - Create entirely new models
- [eLargeModel Guide](elarge_model.md) - High-level training API

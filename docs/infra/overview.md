# EasyDeL Infrastructure Overview

The `easydel.infra` module is the foundation upon which all EasyDeL models are built. It provides the core abstractions, utilities, and patterns that enable efficient distributed training and inference of large language models on JAX/Flax.

## Architecture Overview

```md
easydel.infra/
├── base_config.py      # Configuration system
├── base_module.py      # Base neural network module
├── base_state.py       # Training state management
├── factory.py          # Model registry system
├── loss_utils.py       # Loss computation utilities
├── modeling_outputs.py # Output dataclasses
├── utils.py            # General utilities
├── etils.py            # Enums and constants
├── errors.py           # Custom exceptions
├── mixins/             # Reusable functionality mixins
│   ├── protocol.py     # Base protocol definition
│   ├── generation.py   # Text generation capabilities
│   ├── bridge.py       # HuggingFace integration
│   └── operation_cache.py # Cache management for hybrid models
└── elarge_model/       # High-level training API
    └── elarge_model.py # Unified model builder
```

## Core Components

### 1. Configuration System (`EasyDeLBaseConfig`)

The configuration system manages all model hyperparameters and settings. Every EasyDeL model has a configuration class that inherits from `EasyDeLBaseConfig`.

```python
from easydel import EasyDeLBaseConfig

# Configurations handle:
# - Model architecture parameters (hidden_size, num_layers, etc.)
# - Sharding and partitioning settings
# - Attention mechanism selection
# - RoPE (Rotary Position Embedding) configuration
# - Mesh creation for distributed training
```

### 2. Base Module (`EasyDeLBaseModule`)

The base module class provides common functionality for all neural network models:

```python
from easydel import EasyDeLBaseModule

# Base modules provide:
# - Parameter management and sharding
# - Quantization support
# - LoRA (Low-Rank Adaptation) integration
# - Loss computation
# - State conversion (to/from PyTorch)
```

### 3. Training State (`EasyDeLState`)

The state container manages model parameters and optimizer state during training:

```python
from easydel import EasyDeLState

# State management includes:
# - Gradient application
# - Optimizer state handling
# - Checkpointing (save/load)
# - Distributed sharding
```

### 4. Model Registry

The registry system enables automatic model discovery and instantiation:

```python
from easydel.infra import register_config, register_module

# Registry allows:
# - Registering custom models
# - Auto-loading from HuggingFace
# - Task-specific model retrieval
```

## How It All Works Together

### Sharding Axes (5D)

EasyDeL uses a 5-dimensional sharding scheme:

| Axis | Name   | Purpose                             |
| ---- | ------ | ----------------------------------- |
| 0    | `dp`   | Data Parallelism (batch splitting)  |
| 1    | `fsdp` | Fully Sharded Data Parallelism      |
| 2    | `ep`   | Expert Parallelism (for MoE models) |
| 3    | `tp`   | Tensor Parallelism (layer sharding) |
| 4    | `sp`   | Sequence Parallelism                |

Default: `sharding_axis_dims=(1, -1, 1, 1, 1)` and `sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp")`

### Model Loading Flow

```md
1. User calls: AutoEasyDeLModelForCausalLM.from_pretrained("model-name")
                                    ↓
2. Registry looks up model type from config.json
                                    ↓
3. Appropriate Config class is instantiated
                                    ↓
4. Config creates mesh for distributed training
                                    ↓
5. Module class is instantiated with config
                                    ↓
6. Weights are loaded and converted from PyTorch
                                    ↓
7. Model is sharded across devices
                                    ↓
8. Ready for inference or training!
```

### Training Flow

```md
1. Create model and config
                ↓
2. Create EasyDeLState with optimizer
                ↓
3. Shard state across devices
                ↓
4. Training loop:
   a. Forward pass (model.__call__)
   b. Loss computation (compute_loss)
   c. Backward pass (JAX autodiff)
   d. Gradient application (state.apply_gradients)
                ↓
5. Save checkpoint (state.save_state)
```

### Generation Flow

```md
1. Load model with from_pretrained
                ↓
2. Call generate() method
                ↓
3. prepare_inputs_for_generation:
   - Calls init_operations_cache (handles hybrid/transformer/recurrent caches)
   - Creates MaskInfo for attention
   - Computes position_ids
                ↓
4. Generation loop (_greedy_search/_sample/_beam_search):
   a. Forward pass with cache
   b. Process logits
   c. Sample/select next token
   d. Update inputs for next iteration
   e. Check stopping criteria
                ↓
5. Return generated sequences
```

## Key Design Principles

### 1. Composability

Components are designed to be mixed and matched:

```python
# Mixins add specific functionality
class MyModel(
    EasyDeLBaseModule,
    EasyGenerationMixin,    # Adds generation capabilities
    EasyBridgeMixin,        # Adds HuggingFace compatibility
):
    pass
```

### 2. Transparency

Everything is inspectable and modifiable:

```python
# Access internals easily
model.config                    # Configuration
model.graphdef                  # Flax NNX graph definition
model.graphstate                # Current state
model._get_partition_rules()    # Sharding rules
```

### 3. Performance First

Optimizations are built-in:

- **Automatic sharding**: Models are automatically distributed across devices
- **Gradient checkpointing**: Memory-efficient training
- **Quantization**: Built-in support for NF4, 8-bit, and other formats
- **Attention backends**: Multiple optimized attention implementations

### 4. HuggingFace Compatibility

Seamless integration with the HuggingFace ecosystem:

```python
# Load any compatible HuggingFace model
import easydel as ed

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    dtype=jnp.bfloat16,
    auto_shard_model=True,
)

# Save back to HuggingFace format
model.save_pretrained("./my-model")
model.push_to_hub("my-username/my-model")
```

## Next Steps

- [Base Configuration Guide](base_config.md) - Deep dive into the configuration system
- [Base Module Guide](base_module.md) - Understanding the base module class
- [Customization Guide](customization.md) - How to customize and extend
- [Adding Your Own Model](adding_models.md) - Step-by-step guide to adding new models
- [eLargeModel Guide](elarge_model.md) - High-level training API

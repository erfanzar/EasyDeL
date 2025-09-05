# RayDistributedTrainer

## Overview

The `RayDistributedTrainer` is a specialized training orchestrator designed for distributed training with Ray. It provides a lightweight wrapper that manages model configuration, scaling, and initialization while delegating the actual training logic to EasyDeL's core trainers.

## Key Features

- **Dynamic Model Scaling**: Automatically scale model dimensions based on scaling indices for distributed training
- **Flexible Initialization**: Support multiple initialization strategies including from scratch, checkpoints, or pre-initialized models
- **Automatic Tokenizer Setup**: Handle tokenizer loading and padding configuration automatically
- **Configuration Persistence**: Save and load training configurations as JSON for reproducibility
- **Seamless Integration**: Works with any EasyDeL model and trainer implementation

## Basic Usage

### Creating a Trainer Instance

```python
from easydel.trainers.ray_scaler import RayDistributedTrainer
from easydel.infra.factory import TaskType

# Initialize trainer with model specifications
trainer = RayDistributedTrainer(
    pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
    model_task=TaskType.CAUSAL_LM,
    model_type="llama",
    offload_backend="cpu",  # or "gpu"
)

# Or initialize with a custom model class
from easydel.modules.llama import LlamaModel

trainer = RayDistributedTrainer(
    pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
    model_class=LlamaModel,  # Will infer task and type from class
)
```

### Configuration Management

The trainer supports saving and loading configurations for reproducibility:

```python
# Save configuration
trainer.save_config("trainer_config.json")

# Load from saved configuration
trainer = RayDistributedTrainer.from_config(
    "trainer_config.json",
    model_class=LlamaModel,  # Optional override
)
```

## Advanced Configuration

### Scaling Variables

The trainer distinguishes between two types of configuration variables:

1. **Scaling Variables**: Dimensions that scale with the `scaling_index` parameter
2. **Fixed Variables**: Configuration that remains constant across all scales

```python
trainer = RayDistributedTrainer(
    pretrained_model_name_or_path="model-path",
    config_scaling_variables={
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    config_variables={
        "dtype": jnp.bfloat16,
        "max_position_embeddings": 8192,
        "attn_mechanism": AttentionMechanisms.FLASH,
        "gradient_checkpointing": EasyDeLGradientCheckPointers.SELECTIVE,
    }
)

# Create scaled configuration (scaling_index=2 doubles all scaling variables)
config = trainer.create_config(scaling_index=2)
# Results in: hidden_size=512, intermediate_size=2048, etc.
```

### Model Initialization Options

The trainer supports multiple initialization strategies with clear priority:

```python
# Priority 1: Use provided state directly
trainer.train(
    scaling_index=1,
    arguments=training_args,
    dataset_train=train_dataset,
    state=existing_state,  # Highest priority
)

# Priority 2: Convert provided model to state
trainer.train(
    scaling_index=1,
    arguments=training_args,
    dataset_train=train_dataset,
    model=initialized_model,  # Converted to state
)

# Priority 3: Load from checkpoint path
trainer = RayDistributedTrainer(
    pretrained_model_name_or_path="model-path",
    bucket_path="gs://my-bucket/checkpoint",  # Cloud checkpoint
)
trainer.train(
    scaling_index=1,
    arguments=training_args,
    dataset_train=train_dataset,
)

# Priority 4: Create new model with scaling
trainer.train(
    scaling_index=2,  # Creates scaled model from scratch
    arguments=training_args,
    dataset_train=train_dataset,
)
```

## Data Processing

### Text Data Processing

```python
# Process raw text samples
processed = trainer.process_sample_data(
    sample="This is a text sample",
    max_length=512,
    padding_side="left",  # or "right"
)
```

### Chat Template Processing

```python
# Process chat messages
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]

processed = trainer.process_messages_data(
    messages=messages,
    max_length=512,
    padding_side="left",
)
```

### Dataset Column Extraction

```python
# Extract column names from dataset
columns = trainer.extract_column_names(dataset)
```

## Training with Automatic Resume

The trainer supports automatic resume from interruptions through the underlying trainer:

```python
from easydel.trainers import TrainingArguments

arguments = TrainingArguments(
    # Enable automatic resume
    resume_if_possible=True,
    save_directory="./checkpoints",

    # Other training parameters
    total_steps=10000,
    batch_size=8,
    learning_rate=1e-4,
)

trainer.train(
    scaling_index=1,
    arguments=arguments,
    dataset_train=train_dataset,
    dataset_eval=eval_dataset,
)
```

## Integration with Ray

While the `RayDistributedTrainer` itself doesn't directly interact with Ray, it's designed to work seamlessly in Ray-based distributed training workflows:

```python
import ray
from ray import train

@ray.remote
class Worker:
    def __init__(self, scaling_index):
        self.trainer = RayDistributedTrainer(
            pretrained_model_name_or_path="model-path",
            model_type="llama",
            model_task=TaskType.CAUSAL_LM,
        )
        self.scaling_index = scaling_index

    def train_model(self, dataset, args):
        return self.trainer.train(
            scaling_index=self.scaling_index,
            arguments=args,
            dataset_train=dataset,
        )

# Create workers with different scaling indices
workers = [Worker.remote(scaling_index=i) for i in [1, 2, 4, 8]]
```

## Custom Model and State Classes

You can use custom model and state classes for specialized requirements:

```python
from easydel.infra import EasyDeLBaseModule, EasyDeLState

class CustomModel(EasyDeLBaseModule):
    _model_type = "custom"
    _model_task = TaskType.CUSTOM
    # ... implementation

class CustomState(EasyDeLState):
    # ... custom state implementation

trainer = RayDistributedTrainer(
    pretrained_model_name_or_path="model-path",
    model_class=CustomModel,
    state_class=CustomState,
    trainer_module=CustomTrainer,  # Your custom trainer
)
```

## Configuration Reference

### Default Scaling Variables

```python
CONFIG_SCALING_VARIABLES = {
    "hidden_size": 256,
    "intermediate_size": 256 * 4,
    "moe_intermediate_size": 256 * 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
}
```

### Default Fixed Variables

```python
CONFIG_VARIABLES = {
    "dtype": jnp.bfloat16,
    "param_dtype": jnp.bfloat16,
    "precision": lax.Precision.DEFAULT,
    "seed": 654,
    "max_position_embeddings": 2**13,
    "gradient_checkpointing": EasyDeLGradientCheckPointers.NONE,
    "initializer_range": 0.02,
    "partition_axis": PartitionAxis(),
    "attn_mechanism": AttentionMechanisms.AUTO,
    "attn_dtype": jnp.bfloat16,
    "attn_softmax_dtype": jnp.bfloat16,
    "sharding_axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
    "sharding_axis_dims": (1, -1, 1, 1, 1),
    "sharding_dcn_axis_dims": (1, -1, 1, 1, 1),
}
```

## Best Practices

1. **Configuration Management**: Always save your configuration for reproducibility
2. **Scaling Strategy**: Start with small scaling indices for testing, then scale up
3. **Memory Management**: Use appropriate `offload_backend` based on your hardware
4. **Checkpoint Loading**: Prefer bucket paths for distributed setups
5. **Tokenizer Setup**: The trainer automatically handles padding token configuration

## API Reference

### RayDistributedTrainer Structure

Main class for distributed training orchestration.

#### `__init__(pretrained_model_name_or_path, bucket_path=None, model_task=None, model_type=None, model_class=None, state_class=None, offload_backend=None, trainer_module=None, config_scaling_variables=None, config_variables=None)`

Initialize the distributed trainer.

**Parameters:**

- `pretrained_model_name_or_path` (str): Path or identifier for the pretrained model
- `bucket_path` (str, optional): Path to load checkpoints from cloud storage
- `model_task` (TaskType, optional): Task type (inferred from model_class if not provided)
- `model_type` (str, optional): Model architecture type (inferred from model_class if not provided)
- `model_class` (type[EasyDeLBaseModule], optional): EasyDeL model class to use
- `state_class` (type[EasyDeLState], optional): State class for checkpointing
- `offload_backend` (str, optional): Backend for memory offloading ("cpu" or "gpu")
- `trainer_module` (type[BaseTrainer], optional): Trainer class to use
- `config_scaling_variables` (dict, optional): Variables to scale with scaling_index
- `config_variables` (dict, optional): Fixed configuration variables

#### `train(scaling_index, arguments, dataset_train, dataset_eval=None, data_collator=None, model=None, state=None)`

Execute distributed training with the configured model.

**Parameters:**

- `scaling_index` (int): Multiplier for model scaling
- `arguments` (TrainingArguments): Training configuration
- `dataset_train` (Dataset): Training dataset
- `dataset_eval` (Dataset, optional): Evaluation dataset
- `data_collator` (Callable, optional): Data collator for batching
- `model` (EasyDeLBaseModule, optional): Pre-initialized model
- `state` (EasyDeLState, optional): Pre-initialized state

**Returns:**

- Training results from the underlying trainer

### RayDistributedConfig

Configuration class for persisting trainer settings.

#### `_saving_preprocess()`

Convert dtypes and PartitionAxis to JSON-serializable formats before saving.

#### `_loading_postprocess()`

Convert string representations back to dtypes and PartitionAxis after loading.

## Troubleshooting

### Common Issues

1. **Model Resolution Failure**

   ```python
   # Ensure both model_type and model_task are provided
   trainer = RayDistributedTrainer(
       pretrained_model_name_or_path="model-path",
       model_type="llama",
       model_task=TaskType.CAUSAL_LM,
   )
   ```

2. **Tokenizer Padding Issues**

   ```python
   # The trainer automatically handles this, but you can override
   tokenizer = trainer.load_processor()
   tokenizer.pad_token = tokenizer.eos_token  # Manual override if needed
   ```

3. **Memory Issues with Large Models**

   ```python
   # Use lazy initialization for memory efficiency
   model = trainer.create_model(
       config=config,
       lazy=True,  # Lazy initialization
   )
   ```

## See Also

- [BaseTrainer Documentation](base_trainer.md)
- [TrainerProtocol Documentation](trainer_protocol.md)
- [Training Arguments Guide](../getting_started.md#training-arguments)
- [Model Configuration](../api_docs/apis.md#configuration)

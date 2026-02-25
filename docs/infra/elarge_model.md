# eLargeModel Guide

`eLargeModel` is EasyDeL's high-level API for building, configuring, and running large language models. It provides both a fluent Python interface and a unified YAML runner for training, evaluation, and serving.

## Overview

eLargeModel acts as a unified builder that handles:

- Model loading and configuration
- Sharding and distributed setup
- Quantization
- Dataset preparation
- Training orchestration
- Inference engine setup (eSurge)
- Evaluation with lm-evaluation-harness
- OpenAI-compatible API serving

---

## YAML Runner (`python -m easydel.scripts.elarge`)

The YAML runner provides a declarative way to configure and execute eLargeModel pipelines without writing Python code. This is ideal for reproducible experiments, cluster deployments, and configuration management.

### Basic Usage

```bash
# Run with a YAML config
python -m easydel.scripts.elarge config.yaml

# Or use --config flag
python -m easydel.scripts.elarge --config config.yaml

# Dry run (parse and print config without executing)
python -m easydel.scripts.elarge config.yaml --dry-run
```

### YAML Structure

A YAML configuration file has two main parts:

1. **Configuration** - Model, sharding, data, and trainer settings
2. **Actions** - Sequential operations to execute

```yaml
# Configuration section (can use "config:", "elm:", "elarge_model:", or top-level keys)
model:
  name_or_path: "meta-llama/Llama-2-7b-hf"
  task: causal-language-model

loader:
  dtype: bf16
  param_dtype: bf16

sharding:
  axis_dims: [1, 1, 1, -1, 1]  # (dp, fsdp, ep, tp, sp)

# Actions to execute (required)
actions:
  - validate
  - train
```

### Available Actions

| Action                   | Description                               | Parameters                                                       |
| ------------------------ | ----------------------------------------- | ---------------------------------------------------------------- |
| `validate`               | Validate configuration before execution   | None                                                             |
| `train`                  | Run training with configured trainer      | None                                                             |
| `eval`                   | Run evaluation with lm-evaluation-harness | `tasks`, `engine`, `num_fewshot`, `output_path`, `print_results` |
| `serve` / `server`       | Start OpenAI-compatible API server        | `host`, `port`, `log_level`, etc.                                |
| `print` / `show`         | Print eLargeModel summary                 | None                                                             |
| `dump_config` / `config` | Print normalized configuration            | None                                                             |
| `to_json` / `save_json`  | Save config to JSON file                  | `path` or string path                                            |
| `to_yaml` / `save_yaml`  | Save config to YAML file                  | `path` or string path                                            |

---

## Complete YAML Reference

### Model Configuration (`model`)

```yaml
model:
  name_or_path: "meta-llama/Llama-2-7b-hf"  # Required: HuggingFace ID or local path
  tokenizer: "custom/tokenizer"              # Optional: custom tokenizer path
  task: causal-language-model                # Optional: task type override
  extra_kwargs:                              # Optional: additional loading args
    trust_remote_code: true
```

**Task types:** `causal-language-model`, `vision-language-model`, `image-text-to-text`, `sequence-to-sequence`, `speech-sequence-to-sequence`, `image-classification`, `auto-bind`

### Teacher/Reference Models

For distillation and preference optimization (DPO, ORPO):

```yaml
teacher_model:
  name_or_path: "meta-llama/Llama-2-70b-hf"

reference_model:
  name_or_path: "meta-llama/Llama-2-7b-hf"
```

### Loader Configuration (`loader`)

```yaml
loader:
  dtype: bf16           # Computation dtype: fp32, fp16, bf16
  param_dtype: bf16     # Parameter storage dtype
  precision: high       # JAX precision: default, high, highest
  verbose: false        # Enable verbose loading
  from_torch: true      # Convert from PyTorch checkpoint
  trust_remote_code: true
```

### Sharding Configuration (`sharding`)

EasyDeL uses 5D sharding: `(dp, fsdp, ep, tp, sp)`

```yaml
sharding:
  axis_dims: [1, -1, 1, 1, 1]        # Sharding dimensions (-1 = auto)
  axis_names: [dp, fsdp, ep, tp, sp]
  auto_shard_model: true
  use_ring_of_experts: false         # Ring topology for MoE experts
  fsdp_is_ep_bound: false            # Fold FSDP into expert axis
  sp_is_ep_bound: false              # Fold SP into expert axis
```

**Common configurations:**

- `[1, -1, 1, 1, 1]` - Full FSDP sharding
- `[1, 4, 1, 2, 1]` - 4-way FSDP + 2-way tensor parallel
- `[2, -1, 1, 1, 1]` - 2-way data parallel + FSDP
- `[1, 4, 8, 1, 1]` - 4-way FSDP + 8-way expert parallel (MoE)

### Quantization Configuration (`quantization`)

```yaml
quantization:
  platform: triton  # or jax, pallas
  apply_quantization: true
  kv_cache:
    quantization_method: 8bit
    group_size: 128
  model:
    quantization_method: nf4
    group_size: 64
```

**Methods:** `nf4`, `8bit`, `a8bit`, `a8q`, `a4q`

### Base Configuration (`base_config`)

Override model configuration values and operation configs:

```yaml
base_config:
  values:
    attn_dtype: float32
    use_scan_mlp: true
  operation_configs:
    flash_attn2:
      blocksize_q: 128
      blocksize_k: 128
```

### eSurge Inference Configuration (`esurge`)

```yaml
esurge:
  max_model_len: 8192
  max_num_seqs: 256
  hbm_utilization: 0.85
  page_size: 128
  use_aot_forward: true
  enable_prefix_caching: true
  compile_runner: true
  overlap_execution: false
  silent_mode: false

  # Context handling
  auto_truncate_prompt: true
  auto_cap_new_tokens: true
  strict_context: false
  truncate_mode: left  # left, right, middle

  # External workers
  tokenizer_endpoint: null
  detokenizer_endpoint: null
```

### Dataset Mixture Configuration (`mixture`)

```yaml
mixture:
  informs:
    # JSON files
    - type: json
      data_files: "train/*.json"
      content_field: text
      split: train

    # Parquet files
    - type: parquet
      data_files: ["data/part1.parquet", "data/part2.parquet"]
      content_field: content
      additional_fields: [metadata, source]

    # HuggingFace dataset
    - type: databricks/dolly-15k
      dataset_split_name: train
      content_field: text
      num_rows: 10000  # Optional: limit rows
      format_fields:
        question: prompt
        answer: response

    # Visual dataset
    - type: visual
      data_files: "images/*.json"
      image_field: image_path
      text_field: caption

  # Batching
  batch_size: 32
  shuffle_buffer_size: 10000
  seed: 42

  # Token packing (for efficient training)
  pack_tokens: true
  pack_seq_length: 2048
  pack_eos_token_id: 2
  pack_shuffle: true
  pack_shuffle_buffer_factor: 10

  # On-the-fly tokenization and packing
  pack_on_the_fly: true

  # Block mixture (deterministic mixing)
  block_mixture: true
  mixture_block_size: 1000
  stop_strategy: first_exhausted  # or longest
  mixture_weights:
    train_data: 0.8
    valid_data: 0.2

  # Prefetch and caching
  prefetch_workers: 4
  prefetch_buffer_size: 100
  cache_remote_files: true

  # Tokenization settings
  tokenization:
    max_length: 2048
    truncation: true
    padding: max_length

  # Save processed dataset
  save:
    path: ./processed_dataset
    format: parquet
```

### Trainer Configuration (`trainer`)

```yaml
trainer:
  trainer_type: sft  # sft, dpo, orpo, grpo, reward, distillation

  # Core training params
  learning_rate: 2.0e-5
  num_train_epochs: 3
  total_batch_size: 32
  gradient_accumulation_steps: 4

  # Optimization
  warmup_steps: 100
  weight_decay: 0.01
  clip_grad: 1.0
  optimizer: adamw  # adamw, lion, adafactor
  scheduler: cosine  # cosine, linear, constant

  # Logging and saving
  log_steps: 10
  evaluation_steps: 500
  save_steps: 500
  save_directory: ./output
  save_total_limit: 3

  # DPO-specific
  beta: 0.1
  label_smoothing: 0.0

  # GRPO-specific
  group_size: 4
  kl_coef: 0.1
```

### Evaluation Configuration (`eval`)

```yaml
eval:
  max_new_tokens: 2048
  temperature: 0.0
  top_p: 0.95
  batch_size: 32
  use_tqdm: true
  limit: null  # Limit examples per task
  cache_requests: true
  apply_chat_template: true
  system_instruction: "You are a helpful assistant."
```

---

## Example YAML Configurations

### SFT Training

```yaml
model:
  name_or_path: meta-llama/Llama-2-7b-hf

loader:
  dtype: bf16
  param_dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: json
      data_files: "train/*.json"
      content_field: text
  batch_size: 32
  pack_tokens: true
  pack_seq_length: 2048

trainer:
  trainer_type: sft
  learning_rate: 2.0e-5
  num_train_epochs: 3
  total_batch_size: 32
  gradient_accumulation_steps: 4
  warmup_steps: 100
  save_directory: ./llama-sft-output
  save_steps: 500
  log_steps: 10

actions:
  - validate
  - train
```

### DPO Training

```yaml
model:
  name_or_path: ./my-sft-model

reference_model:
  name_or_path: ./my-sft-model

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: json
      data_files: "preference_data.json"
      content_field: prompt
      additional_fields: [chosen, rejected]
  batch_size: 16

trainer:
  trainer_type: dpo
  learning_rate: 5.0e-7
  num_train_epochs: 1
  total_batch_size: 16
  beta: 0.1
  save_directory: ./llama-dpo-output

actions:
  - validate
  - train
```

### Evaluation Pipeline

```yaml
model:
  name_or_path: meta-llama/Llama-2-7b-hf

loader:
  dtype: bf16

sharding:
  axis_dims: [1, 1, 1, -1, 1]

esurge:
  max_model_len: 4096
  max_num_seqs: 64
  hbm_utilization: 0.9
  enable_prefix_caching: true

eval:
  max_new_tokens: 512
  temperature: 0.0
  batch_size: 32

actions:
  - validate
  - eval:
      tasks: [hellaswag, winogrande, arc_easy, arc_challenge]
      engine: esurge
      num_fewshot: 0
      output_path: ./eval_results.json
      print_results: true
```

### API Server

```yaml
model:
  name_or_path: meta-llama/Llama-2-7b-chat-hf

loader:
  dtype: bf16

sharding:
  axis_dims: [1, 1, 1, -1, 1]

esurge:
  max_model_len: 4096
  max_num_seqs: 256
  hbm_utilization: 0.85
  enable_prefix_caching: true

actions:
  - validate
  - serve:
      host: 0.0.0.0
      port: 8000
      log_level: info
      enable_function_calling: true
      tool_parser_name: hermes
      oai_like_processor: true
      require_api_key: false
```

### Quantized Training (QLoRA-style)

```yaml
model:
  name_or_path: meta-llama/Llama-2-7b-hf

loader:
  dtype: bf16
  param_dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

quantization:
  model:
    quantization_method: nf4
    group_size: 64

mixture:
  informs:
    - type: databricks/dolly-15k
      content_field: text
  batch_size: 64
  pack_tokens: true

trainer:
  trainer_type: sft
  learning_rate: 2.0e-4  # Higher LR for quantized
  num_train_epochs: 1
  total_batch_size: 64
  save_directory: ./llama-qlora-output

actions:
  - validate
  - train
```

### Knowledge Distillation

```yaml
model:
  name_or_path: meta-llama/Llama-2-7b-hf

teacher_model:
  name_or_path: meta-llama/Llama-2-70b-hf

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: json
      data_files: "train.json"
      content_field: text
  batch_size: 16

trainer:
  trainer_type: distillation
  learning_rate: 1.0e-5
  num_train_epochs: 3
  save_directory: ./distilled-model

actions:
  - validate
  - train
```

---

## Python API

### Basic Usage elarge

```python
from easydel.infra import eLargeModel

# Create and configure a training pipeline
elm = (
    eLargeModel
    .from_pretrained("meta-llama/Llama-2-7b-hf")
    .set_dtype(dtype="bf16", param_dtype="bf16")
    .set_sharding(axis_dims=(1, -1, 1, 1, 1))
    .add_dataset(
        dataset_name="your-dataset",
        split="train",
        max_length=2048,
    )
    .set_trainer(
        trainer_type="sft",
        learning_rate=2e-5,
        num_train_epochs=3,
        total_batch_size=32,
    )
)

# Start training
trainer_output = elm.train()
```

### Creating an eLargeModel

#### From Pretrained

```python
# From HuggingFace Hub
elm = eLargeModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# From local path
elm = eLargeModel.from_pretrained("./my-local-model")

# With specific task type
elm = eLargeModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    task="causal-lm",
)
```

#### From YAML/JSON Configuration

```python
# Load from YAML
elm = eLargeModel.from_yaml("training_config.yaml")

# Load from JSON
elm = eLargeModel.from_json("training_config.json")

# Save configuration
elm.to_yaml("config.yaml")
elm.to_json("config.json")
```

#### From Dictionary

```python
config = {
    "model": {"name_or_path": "meta-llama/Llama-2-7b-hf"},
    "loader": {"dtype": "bf16"},
    "sharding": {"axis_dims": [1, -1, 1, 1, 1]},
}
elm = eLargeModel(config)
```

### Configuration Methods

All setter methods return `self` for method chaining:

```python
elm = (
    eLargeModel.from_pretrained("model-name")
    .set_dtype(dtype="bf16", param_dtype="bf16")
    .set_sharding(axis_dims=(1, -1, 1, 1, 1))
    .set_quantization(method="nf4", group_size=64)
    .set_operation_configs(attn_mechanism="flash")
    .set_esurge(max_model_len=4096, enable_prefix_caching=True)
    .set_mixture(batch_size=32, pack_tokens=True)
    .add_dataset(dataset_name="dataset", split="train")
    .set_eval(max_new_tokens=512, temperature=0.0)
    .set_trainer(trainer_type="sft", learning_rate=2e-5)
)
```

### Building Components

```python
# Build model (loads weights, applies sharding)
model = elm.build_model()

# Build tokenizer
tokenizer = elm.build_tokenizer()

# Build eSurge inference engine
esurge = elm.build_esurge()

# Build dataset
dataset = elm.build_dataset()

# Build trainer
trainer = elm.build_trainer()

# Build training arguments
args = elm.build_training_arguments()

# For distillation/DPO
teacher = elm.build_teacher_model()
reference = elm.build_reference_model()
```

### Training and Evaluation

```python
# Training
output = elm.train()
print(f"Training completed. Final loss: {output.metrics['train_loss']}")

# Evaluation with lm-evaluation-harness
results = elm.eval(
    tasks=["hellaswag", "mmlu", "gsm8k"],
    engine="esurge",
    num_fewshot=5,
    output_path="eval_results.json",
)

for task, metrics in results["results"].items():
    print(f"{task}: {metrics.get('acc', 'N/A')}")
```

### Validation

```python
# Validate configuration before running
elm.validate()

# Clear cached models/datasets
elm.clear_cache()
```

---

## Trainer Types

| Type           | Description                        | Key Parameters                      |
| -------------- | ---------------------------------- | ----------------------------------- |
| `sft`          | Supervised Fine-Tuning             | `learning_rate`, `num_train_epochs` |
| `dpo`          | Direct Preference Optimization     | `beta`, `label_smoothing`           |
| `orpo`         | Odds Ratio Preference Optimization | `beta`, `lambda_orpo`               |
| `grpo`         | Group Relative Policy Optimization | `group_size`, `kl_coef`             |
| `reward`       | Reward Model Training              | -                                   |
| `distillation` | Knowledge Distillation             | `alpha`, `temperature`              |

---

## Best Practices

1. **Always validate before execution:**

   ```yaml
   actions:
     - validate  # First action
     - train
   ```

2. **Use dry-run for debugging:**

   ```bash
   python -m easydel.scripts.elarge config.yaml --dry-run
   ```

3. **Save configurations for reproducibility:**

   ```yaml
   actions:
     - to_yaml: ./saved_config.yaml
     - train
   ```

4. **Start with smaller test runs:**

   ```yaml
   trainer:
     num_train_epochs: 1
     max_training_steps: 100  # Quick test
   ```

5. **Use appropriate sharding for your hardware:**

   ```yaml
   # Single GPU/TPU
   sharding:
     axis_dims: [1, 1, 1, 1, 1]

   # Multi-GPU/TPU with FSDP
   sharding:
     axis_dims: [1, -1, 1, 1, 1]

   # Large MoE models
   sharding:
     axis_dims: [1, 4, 8, 1, 1]
   ```

6. **Enable token packing for efficiency:**

   ```yaml
   mixture:
     pack_tokens: true
     pack_seq_length: 2048
   ```

---

## Next Steps

- [Overview](overview.md) - Understand the infrastructure
- [Adding Your Own Model](adding_models.md) - Add custom models
- [Customization Guide](customization.md) - Customize eLargeModel behavior

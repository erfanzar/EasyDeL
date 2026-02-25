
<!-- markdownlint-disable MD033 MD045 MD041 -->
<div align="center">
 <div style="margin-bottom: 50px;">
  <a href="">
  <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/easydel-logo-with-text.png" height="80">
  </a>
 </div>
 <div>
 <a href="https://discord.gg/FCAMNqnGtt"> <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/discord-button.png" height="48"></a>&nbsp;&nbsp;
 <a href="https://easydel.readthedocs.io/en/latest/"><img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/documentation-button.png" height="48"></a>&nbsp;&nbsp;
 <a href="https://easydel.readthedocs.io/en/latest/install.html"><img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/quick-start-button.png" height="48"></a>
 </div>
</div>

--

# Getting Started with EasyDeL on TPU Research Cloud (TRC)

Welcome to the TPU Research Cloud (TRC) platform! This guide will help you set up EasyDeL on Google Cloud TPUs for high-performance model training and fine-tuning. TRC provides free access to state-of-the-art TPU accelerators, enabling efficient training of large language models with EasyDeL's JAX-based framework.

## Why EasyDeL on TRC?

EasyDeL is designed for maximum performance and flexibility on TPU hardware:

- **High Performance**: Optimized JAX implementation for multi-host TPU training
- **Efficient Memory Usage**: Advanced sharding strategies and mixed precision support
- **Production Ready**: Streamlined workflows from research to deployment
- **Fully Customizable**: Build your own training pipelines or use ready-made scripts

## Initial Setup

### 1. Run the TPU setup script

Use the official setup script to install and configure everything needed on TRC (eopod, virtualenvs, EasyDeL TPU deps, Ray) and run a quick health check.

```shell
bash <(curl -sL https://raw.githubusercontent.com/erfanzar/EasyDeL/refs/heads/main/scripts/tpu_setup.sh)
```

If `eopod` is not available right away after the script finishes, restart your shell (or `source ~/.bashrc` / `source ~/.zshrc`) to pick up the alias it adds.

### 2. Set Up Authentication

Connect to your experiment tracking and model hosting accounts:

```shell
# Login to Hugging Face Hub
eopod run "python -c 'from huggingface_hub import login; login(token=\"YOUR_HF_TOKEN\")'"

# Login to Weights & Biases
eopod run python -m wandb login YOUR_WANDB_TOKEN
```

## eLargeModel & YAML Runner

EasyDeL provides `eLargeModel`, a unified high-level API for training, evaluation, and serving. The YAML runner (`python -m easydel.scripts.elarge`) lets you define complete pipelines declaratively.

### Basic Usage

```shell
# Run a YAML config
eopod run python -m easydel.scripts.elarge config.yaml

# Dry run (parse and validate without executing)
eopod run python -m easydel.scripts.elarge config.yaml --dry-run

# Get help
eopod run python -m easydel.scripts.elarge --help
```

### YAML Structure

Every config file has two parts:

1. **Configuration** - Model, sharding, data, trainer settings
2. **Actions** - Operations to execute sequentially

```yaml
# Model and training configuration
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]  # (dp, fsdp, ep, tp, sp)

# Actions to execute
actions:
  - validate
  - train
```

### Available Actions

| Action                | Description                             |
| --------------------- | --------------------------------------- |
| `validate`            | Validate configuration before execution |
| `train`               | Run training with configured trainer    |
| `eval`                | Evaluate with lm-evaluation-harness     |
| `serve`               | Start OpenAI-compatible API server      |
| `to_yaml` / `to_json` | Save normalized config                  |

## Training Examples

### SFT (Supervised Fine-Tuning)

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16
  param_dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: databricks/dolly-15k
      content_field: text
      split: train
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
  save_directory: ./sft-output
  save_steps: 500
  log_steps: 10

actions:
  - validate
  - train
```

### DPO (Direct Preference Optimization)

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

reference_model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: trl-lib/ultrafeedback_binarized
      split: "train[:90%]"
  batch_size: 16

trainer:
  trainer_type: dpo
  beta: 0.1
  learning_rate: 5.0e-7
  num_train_epochs: 1
  total_batch_size: 16
  save_directory: ./dpo-output

actions:
  - validate
  - train
```

### ORPO (Odds Ratio Preference Optimization)

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: argilla/distilabel-capybara-dpo-7k-binarized
      split: train
  batch_size: 16

trainer:
  trainer_type: orpo
  beta: 0.1
  learning_rate: 8.0e-6
  num_train_epochs: 1
  total_batch_size: 16
  save_directory: ./orpo-output

actions:
  - validate
  - train
```

### Reward Model Training

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: Anthropic/hh-rlhf
      split: train
  batch_size: 16

trainer:
  trainer_type: reward
  learning_rate: 1.0e-5
  num_train_epochs: 1
  save_directory: ./reward-output

actions:
  - validate
  - train
```

### Knowledge Distillation

```yaml
model:
  name_or_path: meta-llama/Llama-3.2-1B

teacher_model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]

mixture:
  informs:
    - type: json
      data_files: "train/*.json"
      content_field: text
  batch_size: 32

trainer:
  trainer_type: distillation
  learning_rate: 1.0e-5
  num_train_epochs: 3
  save_directory: ./distilled-output

actions:
  - validate
  - train
```

## Evaluation

Run benchmarks using lm-evaluation-harness:

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

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
      tasks: [hellaswag, winogrande, arc_easy, arc_challenge, mmlu]
      engine: esurge
      num_fewshot: 5
      output_path: ./eval_results.json
      print_results: true
```

## Serving (OpenAI-Compatible API)

Deploy a model as an API server:

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct

loader:
  dtype: bf16

sharding:
  axis_dims: [1, 1, 1, -1, 1]

esurge:
  max_model_len: 8192
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
```

Then query it with any OpenAI-compatible client:

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration Reference

### Model (`model`)

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct  # Required
  tokenizer: custom/tokenizer                      # Optional
  task: causal-language-model                      # Optional
```

### Loader (`loader`)

```yaml
loader:
  dtype: bf16           # bf16, fp16, fp32
  param_dtype: bf16     # Parameter storage dtype
  precision: high       # JAX precision level
  trust_remote_code: true
```

### Sharding (`sharding`)

EasyDeL uses 5D sharding: `(dp, fsdp, ep, tp, sp)`

```yaml
sharding:
  axis_dims: [1, -1, 1, 1, 1]  # Full FSDP
  auto_shard_model: true
```

Common configurations:

- `[1, -1, 1, 1, 1]` - Full FSDP (memory efficient)
- `[1, 4, 1, 2, 1]` - FSDP + Tensor Parallel
- `[1, 4, 8, 1, 1]` - FSDP + Expert Parallel (MoE)

### Quantization (`quantization`)

```yaml
quantization:
  model:
    quantization_method: nf4  # nf4, 8bit
    block_size: 64
  kv_cache:
    quantization_method: 8bit
    block_size: 128
```

### eSurge Inference (`esurge`)

```yaml
esurge:
  max_model_len: 8192
  max_num_seqs: 256
  hbm_utilization: 0.85
  page_size: 128
  enable_prefix_caching: true
  use_aot_forward: true
```

### Dataset Mixture (`mixture`)

```yaml
mixture:
  informs:
    - type: databricks/dolly-15k  # HuggingFace dataset
      content_field: text
      split: train
    - type: json                   # Local JSON files
      data_files: "data/*.json"
      content_field: text
  batch_size: 32
  pack_tokens: true
  pack_seq_length: 2048
```

### Trainer (`trainer`)

```yaml
trainer:
  trainer_type: sft  # sft, dpo, orpo, grpo, reward, distillation
  learning_rate: 2.0e-5
  num_train_epochs: 3
  total_batch_size: 32
  gradient_accumulation_steps: 4
  warmup_steps: 100
  weight_decay: 0.01
  clip_grad: 1.0
  save_directory: ./output
  save_steps: 500
  log_steps: 10
```

## GRPO (Group Relative Policy Optimization)

GRPO requires Python reward functions and is best used with the programmatic API:

```python
from easydel.infra import eLargeModel

def reward_func(completions, prompts, **kwargs):
    # Your reward logic
    return [compute_reward(c) for c in completions]

elm = (
    eLargeModel.from_yaml("grpo_config.yaml")
    .set_trainer(trainer_type="grpo", group_size=4, kl_coef=0.1)
)

elm.train(reward_funcs=[reward_func])
```

## Best Practices for TRC

1. **Always validate first**: Add `validate` as your first action to catch config errors early.

2. **Use dry-run for debugging**:

   ```shell
   eopod run python -m easydel.scripts.elarge config.yaml --dry-run
   ```

3. **Enable token packing** for efficient training:

   ```yaml
   mixture:
     pack_tokens: true
     pack_seq_length: 2048
   ```

4. **Use appropriate sharding** for your TPU pod:

   ```yaml
   # v4-8 (single host)
   sharding:
     axis_dims: [1, 1, 1, -1, 1]

   # v4-32+ (multi-host)
   sharding:
     axis_dims: [1, -1, 1, 1, 1]
   ```

5. **Save configs for reproducibility**:

   ```yaml
   actions:
     - to_yaml: ./saved_config.yaml
     - train
   ```

## Getting Help

If you encounter any issues or have questions:

- Join our [Discord community](https://discord.gg/FCAMNqnGtt) for direct support
- Check the [documentation](https://easydel.readthedocs.io/) for detailed guides
- See the [eLargeModel Guide](https://easydel.readthedocs.io/en/latest/infra/elarge_model.html) for complete YAML reference
- Explore the [GitHub repository](https://github.com/erfanzar/easydel) for examples and source code

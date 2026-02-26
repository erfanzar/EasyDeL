# eLarge CLI

YAML-driven runner for `eLargeModel` — train, evaluate, and serve models from a single config file.

## Script Location

```bash
# Run from easydel/scripts
python easydel/scripts/elarge.py --config config.yaml

# Or with positional argument
python easydel/scripts/elarge.py config.yaml
```

## Basic Usage

```bash
# Run a training + eval + serve pipeline
python easydel/scripts/elarge.py --config train.yaml

# Dry run (parse config, print actions, don't execute)
python easydel/scripts/elarge.py --config train.yaml --dry-run
```

## YAML Structure

A config file has two parts:

1. **Configuration** — model, sharding, data, trainer settings
2. **Actions** — sequential operations to execute

```yaml
# Configuration (under "config:", "elm:", or top-level)
model:
  name_or_path: Qwen/Qwen3-8B
  task: causal-language-model

loader:
  dtype: bf16
  param_dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]
  axis_names: [dp, fsdp, ep, tp, sp]
  auto_shard_model: true

trainer:
  trainer_type: sft

mixture:
  informs:
    - type: json
      data_files: train.jsonl
      content_field: text

# Actions to execute
actions:
  - validate
  - print
  - train
```

## Available Actions

| Action                | Description                        |
| --------------------- | ---------------------------------- |
| `validate`            | Validate configuration             |
| `print` / `show`      | Print model info                   |
| `dump_config`         | Print full config dict             |
| `train`               | Run training                       |
| `eval`                | Run evaluation (lm-eval)           |
| `serve`               | Start OpenAI-compatible API server |
| `to_json` / `to_yaml` | Save config to file                |

### Eval Action

```yaml
actions:
  - eval:
      tasks: ["gsm8k", "mmlu"]
      num_fewshot: 5
      output_path: results.json
      print_results: true
```

### Serve Action

```yaml
actions:
  - serve:
      host: "0.0.0.0"
      port: 11556
      workers: 1
      log_level: info
      enable_function_calling: true
      tool_parser_name: hermes
```

## Example Configs

### Minimal Training Config

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]
  axis_names: [dp, fsdp, ep, tp, sp]
  auto_shard_model: true

trainer:
  trainer_type: sft
  num_train_epochs: 1
  learning_rate: 2e-5

mixture:
  informs:
    - type: json
      data_files: data/train.jsonl
      content_field: text

actions:
  - validate
  - train
```

### Eval-Only Config

```yaml
model:
  name_or_path: EasyDeL/Qwen3-8B

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]
  axis_names: [dp, fsdp, ep, tp, sp]
  auto_shard_model: true

actions:
  - validate
  - eval:
      tasks: ["hellaswag", "arc_easy"]
      num_fewshot: 0
      output_path: eval_results.json
```

### Serve-Only Config

```yaml
model:
  name_or_path: EasyDeL/Llama-3.1-8B

loader:
  dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]
  axis_names: [dp, fsdp, ep, tp, sp]
  auto_shard_model: true

actions:
  - validate
  - serve:
      host: "0.0.0.0"
      port: 8000
```

## Key Flags

| Flag              | Description                              |
| ----------------- | ---------------------------------------- |
| `--config` / `-c` | Path to YAML config file                 |
| `--dry-run`       | Parse and print config without executing |

## Related Documentation

- [eLargeModel Guide](../infra/elarge_model.md) — full eLargeModel API and YAML reference
- [Base Configuration](../infra/base_config.md) — sharding and model config options
- [SFT Trainer](../trainers/sft.md) — supervised fine-tuning details

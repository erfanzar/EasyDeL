# Model Conversion Scripts

Convert Hugging Face PyTorch checkpoints to EasyDeL's JAX-native format.

## Scripts

| Script                           | Purpose                        |
| -------------------------------- | ------------------------------ |
| `convert_hf_to_easydel.py`       | Convert a single model         |
| `convert_hf_to_easydel_batch.py` | Batch convert from a list file |

---

## convert_hf_to_easydel.py

Converts a HuggingFace PyTorch checkpoint to EasyDeL format and optionally pushes to HF Hub.

### Basic Usage

```bash
# Sequential mode (recommended for large models)
python scripts/convert_hf_to_easydel.py \
  --source meta-llama/Llama-3.1-8B \
  --out /mnt/gcs/easydel/Llama-3.1-8B \
  --convert-mode sequential \
  --torch-streaming-cache temp \
  --token $HF_TOKEN

# Push to HF Hub
python scripts/convert_hf_to_easydel.py \
  --source meta-llama/Llama-3.1-8B \
  --out /mnt/gcs/easydel/Llama-3.1-8B \
  --repo-id EasyDeL/Llama-3.1-8B \
  --convert-mode sequential \
  --token $HF_TOKEN
```

### Key Flags

| Flag                        | Default      | Description                                                               |
| --------------------------- | ------------ | ------------------------------------------------------------------------- |
| `--source`                  | *required*   | HF repo id or local path                                                  |
| `--out`                     | *required*   | Output directory (can be gcsfuse mount)                                   |
| `--convert-mode`            | `sequential` | `sequential` streams shards (low RAM); `from_pretrained` loads full model |
| `--no-push-to-hub`          | push enabled | Skip pushing to HF Hub even if `--repo-id` is set                         |
| `--torch-streaming-cache`   | `hf_cache`   | `temp` downloads one shard at a time; `hf_cache` uses HF cache            |
| `--torch-streaming-tmp-dir` | system temp  | Directory for temp shard downloads (with `--torch-streaming-cache temp`)  |
| `--cache-dir`               | HF default   | Redirect HF cache to avoid filling root disk                              |
| `--enable-hf-transfer`      | off          | Use fast `hf_transfer` downloads (requires `pip install hf_transfer`)     |

### Sharding Flags

| Flag                    | Default            | Description                            |
| ----------------------- | ------------------ | -------------------------------------- |
| `--sharding-axis-dims`  | `1,-1,1,1,1`       | 5D mesh dimensions: `dp,fsdp,ep,tp,sp` |
| `--sharding-axis-names` | `dp,fsdp,ep,tp,sp` | Axis names for the mesh                |
| `--auto-shard-model`    | `True`             | Enable automatic sharding              |

---

## Sharding Quick Guide

EasyDeL uses a 5D mesh for distributed training and inference:

| Axis   | Name                        | Purpose                        |
| ------ | --------------------------- | ------------------------------ |
| `dp`   | Data Parallel               | Replicate model, shard data    |
| `fsdp` | Fully Sharded Data Parallel | Shard model parameters         |
| `ep`   | Expert Parallel             | For MoE models                 |
| `tp`   | Tensor Parallel             | Shard individual tensors       |
| `sp`   | Sequence Parallel           | Shard along sequence dimension |

### Choosing Dimensions

Use `-1` to auto-infer a dimension from available devices:

```bash
# Single device: no sharding
--sharding-axis-dims 1,1,1,1,1

# 8 devices: FSDP across all
--sharding-axis-dims 1,-1,1,1,1   # -1 becomes 8

# 8 devices: DP=2, FSDP=4
--sharding-axis-dims 2,4,1,1,1

# 16 devices: DP=2, FSDP=8
--sharding-axis-dims 2,8,1,1,1

# 16 devices: FSDP=4, TP=4
--sharding-axis-dims 1,4,1,4,1
```

### Common Patterns

| Devices | Pattern      | Use Case             |
| ------- | ------------ | -------------------- |
| 1       | `1,1,1,1,1`  | Development/testing  |
| 8       | `1,-1,1,1,1` | Single-host TPU v4-8 |
| 8       | `1,4,1,2,1`  | Single-host with TP  |
| 16      | `1,-1,1,1,1` | 2-host TPU pod slice |
| 64+     | `1,8,1,8,1`  | Large multi-host     |

For detailed sharding configuration, see:

- [Base Configuration](../infra/base_config.md) — sharding parameters
- [eLargeModel Guide](../infra/elarge_model.md) — sharding in eLarge workflows

---

## convert_hf_to_easydel_batch.py

Batch wrapper that reads a list of models and converts each sequentially.

### Models File Format

Create a text file with one model per line:

```text
# models.txt
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.1-8B-Instruct -> EasyDeL/Llama-3.1-8B-Instruct
mistralai/Mistral-7B-v0.3, EasyDeL/Mistral-7B-v0.3
```

Supported formats:

- `source` — auto-generates repo-id as `<repo-owner>/name`
- `source -> owner/name` — explicit target repo
- `source, owner/name` — CSV-style

### Basic Usage Script

```bash
python scripts/convert_hf_to_easydel_batch.py \
  --models-file models.txt \
  --out-root /mnt/gcs/easydel \
  --convert-mode sequential \
  --torch-streaming-cache temp \
  --no-push-to-hub \
  --token $HF_TOKEN
```

### Key Flags Script

| Flag                  | Default    | Description                                           |
| --------------------- | ---------- | ----------------------------------------------------- |
| `--models-file`       | —          | Path to models list file                              |
| `--out-root`          | *required* | Output root; each model writes to `<out-root>/<name>` |
| `--repo-owner`        | `EasyDeL`  | Default HF org when repo-id not specified             |
| `--dry-run`           | off        | Print commands without executing                      |
| `--skip-existing`     | off        | Skip if output directory exists and is non-empty      |
| `--continue-on-error` | off        | Continue with remaining models if one fails           |

All other flags are forwarded to `convert_hf_to_easydel.py`.

### Example Workflow

```bash
# 1. Preview what will run
python scripts/convert_hf_to_easydel_batch.py \
  --models-file models.txt \
  --out-root /mnt/gcs/easydel \
  --dry-run

# 2. Run conversion (skip already converted)
python scripts/convert_hf_to_easydel_batch.py \
  --models-file models.txt \
  --out-root /mnt/gcs/easydel \
  --skip-existing \
  --continue-on-error \
  --convert-mode sequential \
  --torch-streaming-cache temp
```

---

## Disk Usage Notes

See [Disk Usage & Cleanup](hf_download_to_gcs.md#disk-usage--cleanup) for managing caches.

**TL;DR for conversions:**

```bash
# Avoid filling disk with downloaded shards
--torch-streaming-cache temp
--torch-streaming-tmp-dir /tmp/hf-shards

# Redirect HF cache to mounted storage
--cache-dir /mnt/gcs/hf-cache
```

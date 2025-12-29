# CLI & Scripts

EasyDeL includes command-line scripts for model conversion, data downloading, model card management, and development workflows. All scripts are designed to be run via Python directly.

## Running Scripts

Most scripts have two locations:

- **`scripts/`** (recommended) — wrapper scripts at the repo root
- **`easydel/scripts/`** — package-internal implementations

Notable exceptions:

- `easydel/scripts/elarge.py` is only available under `easydel/scripts/`
- `scripts/update_hf_model_readmes.py` is a standalone script (no `easydel/scripts/` counterpart)

```bash
# Preferred (uses repo-local wrappers)
python scripts/convert_hf_to_easydel.py --help

# Alternative (from package)
python easydel/scripts/convert_hf_to_easydel.py --help
```

## Script Categories

| Category                                    | Scripts                                                                     | Purpose                                     |
| ------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------- |
| [Model Conversion](model_conversion.md)     | `convert_hf_to_easydel.py`, `convert_hf_to_easydel_batch.py`                | Convert HF PyTorch models to EasyDeL format |
| [HF Download to GCS](hf_download_to_gcs.md) | `download_hf_repo_chunked_to_gcs.py`, `download_hf_large_weights_to_gcs.py` | Download large weights to GCS               |
| [eLarge CLI](elarge_cli.md)                 | `elarge.py`                                                                 | YAML-driven train/eval/serve workflow       |
| [Model Cards](model_cards.md)               | `update_hf_model_readmes.py`                                                | Auto-generate HF model card READMEs         |
| [Dev Tools](dev_tools.md)                   | `format_and_generate_docs.py`                                               | Format code and generate API docs           |

## Quick Reference

```bash
# Convert a single model
python scripts/convert_hf_to_easydel.py \
  --source meta-llama/Llama-3.1-8B \
  --out /mnt/gcs/easydel/Llama-3.1-8B \
  --convert-mode sequential

# Batch convert from a list
python scripts/convert_hf_to_easydel_batch.py \
  --models-file models.txt \
  --out-root /mnt/gcs/easydel

# Download Zarr weights to GCS
python scripts/download_hf_repo_chunked_to_gcs.py \
  --repo-id owner/repo \
  --out-root gs://bucket/weights \
  --only-zarr

# Run eLarge from YAML
python easydel/scripts/elarge.py --config train.yaml

# Update HF model cards
python scripts/update_hf_model_readmes.py --author EasyDeL

# Format code + generate docs
python scripts/format_and_generate_docs.py --all
```

## Environment Variables

| Variable                    | Purpose                          | Example                                |
| --------------------------- | -------------------------------- | -------------------------------------- |
| `HF_TOKEN`                  | Hugging Face auth token          | `export HF_TOKEN=hf_...`               |
| `HF_HOME`                   | HF cache root directory          | `export HF_HOME=/mnt/data/hf`          |
| `HF_HUB_CACHE`              | HF Hub cache (subset of HF_HOME) | `export HF_HUB_CACHE=/mnt/data/hf/hub` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Enable fast downloads            | Set via `--enable-hf-transfer` flag    |

## Disk Usage & Cleanup

See [Disk Usage & Cleanup](hf_download_to_gcs.md#disk-usage--cleanup) for detailed guidance on managing cache directories and avoiding disk space issues (especially important when using gcsfuse).

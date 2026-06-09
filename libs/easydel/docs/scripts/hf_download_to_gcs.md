# HF Download to GCS Scripts

Download Hugging Face model weights to GCS (directly or via gcsfuse mount).

## Scripts

| Script                                | Purpose                                             |
| ------------------------------------- | --------------------------------------------------- |
| `download_hf_repo_chunked_to_gcs.py`  | Chunked download for large repos (Zarr, many files) |
| `download_hf_large_weights_to_gcs.py` | Download large individual weight files              |

---

## download_hf_repo_chunked_to_gcs.py

Downloads Hugging Face repos in chunks, syncing to GCS after each batch. Ideal for directory-style weights like Zarr with thousands of small files.

### How It Works

1. Lists files in HF repo
2. Downloads ~N GiB to local staging directory
3. Syncs staging to destination (via `gsutil rsync` or local `rsync`)
4. Deletes staging
5. Repeats until done

### Basic Usage

```bash
# Direct to GCS (no gcsfuse)
python scripts/download_hf_repo_chunked_to_gcs.py \
  --repo-id EasyDeL/Llama-3.1-8B \
  --out-root gs://my-bucket/weights \
  --only-zarr \
  --chunk-gb 10 \
  --token $HF_TOKEN

# To gcsfuse mount
python scripts/download_hf_repo_chunked_to_gcs.py \
  --repo-id EasyDeL/Llama-3.1-8B \
  --out-root /mnt/gcs/weights \
  --only-zarr
```

### Key Flags

| Flag                   | Default                 | Description                                     |
| ---------------------- | ----------------------- | ----------------------------------------------- |
| `--repo-id`            | —                       | HF repo id (repeatable)                         |
| `--repos-file`         | —                       | File with one repo id per line                  |
| `--out-root`           | *required*              | Destination: local path or `gs://bucket/prefix` |
| `--staging-dir`        | `/tmp/easydel-hf-stage` | Local staging directory for batches             |
| `--chunk-gb`           | `10`                    | Target batch size in GiB                        |
| `--download-workers`   | `8`                     | Parallel download threads per batch             |
| `--only-zarr`          | off                     | Only download files under `*.zarr/` paths       |
| `--include`            | —                       | Glob pattern to include (repeatable)            |
| `--exclude`            | —                       | Glob pattern to exclude (repeatable)            |
| `--dry-run`            | off                     | Print actions without executing                 |
| `--skip-existing`      | off                     | Skip files that already exist at destination    |
| `--keep-staging`       | off                     | Don't delete staging after sync (debugging)     |
| `--enable-hf-transfer` | off                     | Use fast `hf_transfer` downloads                |

### Example: Download Only Zarr Weights

```bash
python scripts/download_hf_repo_chunked_to_gcs.py \
  --repo-id EasyDeL/Qwen-3-8B \
  --out-root gs://my-bucket/easydel-weights \
  --only-zarr \
  --chunk-gb 5 \
  --download-workers 16 \
  --staging-dir /tmp/zarr-stage
```

### Safe Cleanup After Run

The script cleans up staging automatically. If interrupted:

```bash
# Manual cleanup
rm -rf /tmp/easydel-hf-stage/*
```

---

## download_hf_large_weights_to_gcs.py

Downloads large (non-PyTorch) weight files based on size threshold. Good for GGUF, GGML, or other single-file weights.

### What "Large Weights" Means

By default:

- Files **>= 500 MiB** are selected
- PyTorch files (`.bin`, `.safetensors`, `.pt`, `.pth`, `.ckpt`) are **excluded**
- Metadata files (`.json`, `.yaml`, `.md`, etc.) are excluded

Use `--include-pytorch` to also download PyTorch weights.

### Basic Usage Script

```bash
# Download GGUF files from a repo
python scripts/download_hf_large_weights_to_gcs.py \
  --repo-id TheBloke/Llama-2-7B-GGUF \
  --out-root /mnt/gcs/gguf-weights \
  --include "*.gguf" \
  --token $HF_TOKEN

# Download large files from a collection
python scripts/download_hf_large_weights_to_gcs.py \
  --collection https://huggingface.co/collections/Qwen/qwen3 \
  --out-root /mnt/gcs/qwen-weights \
  --min-size-mb 1000
```

### Key Flags Script

| Flag                   | Default    | Description                                    |
| ---------------------- | ---------- | ---------------------------------------------- |
| `--repo-id`            | —          | Model repo id (repeatable)                     |
| `--repos-file`         | —          | File with one repo id per line                 |
| `--collection`         | —          | HF collection URL or `owner/slug` (repeatable) |
| `--out-root`           | *required* | Output directory (local or gcsfuse mount)      |
| `--min-size-mb`        | `500`      | Minimum file size in MiB                       |
| `--include`            | —          | Glob to include (e.g., `*.gguf`)               |
| `--exclude`            | —          | Glob to exclude                                |
| `--include-pytorch`    | off        | Also download `.bin`/`.safetensors`/`.pt`      |
| `--cache-dir`          | HF default | Redirect HF cache                              |
| `--dry-run`            | off        | Print what would be downloaded                 |
| `--enable-hf-transfer` | off        | Use fast `hf_transfer` downloads               |

### Default Exclusions

PyTorch weights excluded by default:

- `*.bin`, `*.bin.*`
- `*.pt`, `*.pth`, `*.ckpt`
- `*.safetensors`, `*.safetensors.*`

Metadata excluded:

- `*.md`, `*.txt`, `*.rst`
- `*.json`, `*.yaml`, `*.yml`, `*.toml`
- Image files (`*.png`, `*.jpg`, etc.)

### Caching Behavior

Downloads use HF Hub's cache. To avoid filling your root disk:

```bash
# Redirect cache to mounted storage
--cache-dir /mnt/gcs/hf-cache

# Or set environment variable
export HF_HUB_CACHE=/mnt/gcs/hf-cache
```

---

## Disk Usage & Cleanup

### Common Cache Locations

| Location            | Purpose                  | Default Path                                 |
| ------------------- | ------------------------ | -------------------------------------------- |
| HF Hub cache        | Downloaded HF files      | `~/.cache/huggingface/hub`                   |
| Transformers cache  | Model configs/tokenizers | `~/.cache/huggingface/transformers`          |
| Staging directory   | Temp batch downloads     | `/tmp/easydel-hf-stage`                      |
| Torch streaming tmp | Temp shard downloads     | System temp (or `--torch-streaming-tmp-dir`) |

### Redirecting Caches

**Via flags:**

```bash
--cache-dir /mnt/gcs/hf-cache
--staging-dir /mnt/gcs/staging
--torch-streaming-tmp-dir /mnt/gcs/torch-tmp
```

**Via environment:**

```bash
export HF_HOME=/mnt/gcs/hf
export HF_HUB_CACHE=/mnt/gcs/hf/hub
```

### What's Safe to Delete

| Directory                   | Safe to Delete | Notes                                                   |
| --------------------------- | -------------- | ------------------------------------------------------- |
| Staging dirs                | Yes            | Auto-cleaned after each run                             |
| `--torch-streaming-tmp-dir` | Yes            | One-time temp files                                     |
| HF Hub cache                | Partially      | Re-downloads if needed; check for in-progress downloads |
| Model outputs               | No             | Your converted checkpoints                              |

### Avoiding Disk Issues with gcsfuse

When writing to a gcsfuse mount, local disk is NOT used for the final output. However:

1. **Staging is local** — ensure staging dir has space for `--chunk-gb`
2. **HF cache is local by default** — redirect with `--cache-dir` or `HF_HUB_CACHE`
3. **Check mount status:**

   ```bash
   # The script warns if /mnt/gcs isn't mounted
   mount | grep gcsfuse
   ```

### Recommended Setup for Large Downloads

```bash
# Mount GCS bucket
gcsfuse --implicit-dirs my-bucket /mnt/gcs

# Set up caches on mounted storage
export HF_HUB_CACHE=/mnt/gcs/cache/hf

# Run with local staging on SSD
python scripts/download_hf_repo_chunked_to_gcs.py \
  --repo-id owner/repo \
  --out-root /mnt/gcs/weights \
  --staging-dir /tmp/staging \
  --chunk-gb 10
```

### Cleanup Commands

```bash
# Clear HF cache (will re-download as needed)
rm -rf ~/.cache/huggingface/hub/*

# Clear staging (safe anytime)
rm -rf /tmp/easydel-hf-stage

# Clear torch streaming temp
rm -rf /tmp/hf-shards
```

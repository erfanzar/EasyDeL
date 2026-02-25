# Ray 2.42 → 2.53 Upgrade Notes (EasyDeL)

This repo pins Ray and Docker build versions to keep TPU/GCP workflows stable. If your setup worked on Ray 2.42 but started failing on 2.51+, the most common cause is **version drift** between:

- the Ray CLI you run locally (e.g. `ray up`), and
- the Ray version inside the cluster image / virtualenv.

## What EasyDeL Pins

- **Ray**: `ray[default]==2.53.0` in `pyproject.toml`.
- **Docker images**: install `ray[default,gcp]==2.53.0` in `Dockerfile` (adds GCP autoscaler extras).
- **Docker Python**: defaults to `PYTHON_VERSION=3.13` (override via build arg).

## Recommended Workflow (Avoid CLI/Cluster Mismatches)

Use `uv` to run the Ray CLI with the repo-pinned environment:

```bash
uv run --python 3.13 python autoscale/generate-cluster-configs.py --project-id YOUR_PROJECT_ID --output-dir autoscale
CLUSTER_YAML=autoscale/easydel-<YOUR_ZONE>.yaml
uv run --python 3.13 ray up "$CLUSTER_YAML" --no-config-cache
uv run --python 3.13 ray attach "$CLUSTER_YAML"
uv run --python 3.13 ray down "$CLUSTER_YAML"
```

## Ray Changes That Commonly Affect TPU/GCP Users

### 1) `uv run` runtime environment behavior (Ray 2.47+)

Ray 2.47 enables `uv run` runtime-env integration by default. If you relied on the older behavior (driver runs in `uv`, workers do not), set:

```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
```

### 2) GCP autoscaler SSH key naming/paths (Ray 2.51+)

Ray 2.51 updates how the GCP autoscaler generates SSH key names and paths.

If you hard-coded `auth.ssh_private_key` in your cluster YAML, update it to match the new key path (private key no longer uses the `.pem` suffix).

If you did not hard-code keys, Ray will generate new ones under `~/.ssh/` as needed.

### 3) Controlling uploads: `.rayignore` (Ray 2.53)

Ray 2.53 supports `.rayignore` to control what gets uploaded during cluster operations. This repo includes a `.rayignore` at the project root.

For **runtime environment** uploads (e.g. `ray.init(runtime_env={"working_dir": "."})`), still prefer explicit excludes:

```python
runtime_env = {
    "working_dir": ".",
    "excludes": [".git/**", ".venv/**", "**/__pycache__", "*.pt", "*.safetensors", "data/", "checkpoints/", "wandb/"],
}
```

This prevents large `.git` pack files from being uploaded.

## Troubleshooting

### “Missing `cluster_synced_files` field … version running in the cluster …”

This usually indicates one of:

- You validated a raw cluster YAML without Ray filling defaults (Ray adds `cluster_synced_files` during config preparation), or
- Your local Ray CLI and the cluster Ray version are out of sync.

Fix: run Ray commands via `uv run --python 3.13 ray ...` so versions match.

### “File … pack-*.pack is very large … consider adding to excludes”

Add `.git/**` (and other large paths) to `runtime_env["excludes"]`, or avoid using `working_dir` uploads when you don’t need them.

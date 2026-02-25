# EasyDeL TPU Autoscale (Ray on Google Cloud)

This folder contains a small toolchain to discover TPU availability by zone and generate Ray cluster configs for EasyDeL workloads.

## What is in this directory

- `generate-cluster-configs.py`: Queries Google Cloud TPU APIs and generates one Ray cluster YAML per zone.
- `easydel-cluster-template.yaml`: Base template used by the generator.
- `easydel-*.yaml`: Generated zone-specific cluster configs (local-only, gitignored).

## Prerequisites

- Python environment with project dependencies installed.
- `gcloud` CLI installed and authenticated.
- Google Cloud APIs enabled in your project:
  - `compute.googleapis.com`
  - `tpu.googleapis.com`
  - `serviceusage.googleapis.com`
  - `cloudresourcemanager.googleapis.com`
- IAM permissions to read project metadata, TPU locations/types, and Service Usage quota metrics.

## Google Cloud authentication (ADC)

`generate-cluster-configs.py` uses `google.auth.default()`, so Application Default Credentials (ADC) must be configured.

### Local development (interactive)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

### Automation / CI (service account)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
export GCP_PROJECT_ID=YOUR_PROJECT_ID
```

Quick check:

```bash
python -c "import google.auth; print(google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform']))"
```

## Generate cluster YAMLs for your own project

From repo root:

```bash
uv run --python 3.13 python autoscale/generate-cluster-configs.py \
  --project-id YOUR_PROJECT_ID \
  --output-dir autoscale \
  --families v4 v5e v5p v6e \
  --print-summary
```

Notes:

- `--project-id` overrides project auto-detection.
- If omitted, the script uses `GCP_PROJECT_ID`, then the ADC default project.
- Use `--verbose` for detailed API diagnostics.
- Use `--try-all-services` if quota metrics are not being classified correctly.

## Launch, inspect, and tear down a cluster

Pick one generated config from your own project output:

```bash
CLUSTER_YAML=autoscale/easydel-<YOUR_ZONE>.yaml
```

```bash
uv run --python 3.13 ray up "$CLUSTER_YAML" --no-config-cache
uv run --python 3.13 ray attach "$CLUSTER_YAML"
uv run --python 3.13 ray dashboard "$CLUSTER_YAML" --port 8265
uv run --python 3.13 ray exec "$CLUSTER_YAML" "ray status"
uv run --python 3.13 ray down "$CLUSTER_YAML"
```

## Generated config behavior

- TPU families currently generated: `v4`, `v5e`, `v5p`, `v6e`.
- TPU node types are generated under `available_node_types` as:
  - `tpu_base_<family>_<size>`
  - `tpu_slice_<family>_<size>`
- All TPU worker node configs are preemptible (`schedulingConfig.preemptible: true`).
- Runtime versions are selected per family in `generate-cluster-configs.py` (`GENERATION_CONFIGS`).
- Head and worker setup attempt to read a Secret Manager secret named `HF_TOKEN`; failures are ignored (`|| true`).

## Common custom edits

Edit generated YAML when needed:

- Increase `min_workers` on a TPU node type to keep capacity warm.
- Cap `max_workers` to control spend.
- Disable preemptible in `node_config.schedulingConfig.preemptible` for non-spot capacity.
- Change Docker image under `docker.image` if you need a custom runtime.

## Troubleshooting

### `google.auth.exceptions.DefaultCredentialsError`

ADC is missing. Run:

```bash
gcloud auth application-default login
```

Then rerun with explicit project:

```bash
uv run --python 3.13 python autoscale/generate-cluster-configs.py --project-id YOUR_PROJECT_ID --print-summary
```

### Wrong project used by Ray config

Check `provider.project_id` in the generated YAML. If it is not your project, regenerate with `--project-id YOUR_PROJECT_ID`.

### No zones or no TPU families found

- Verify `tpu.googleapis.com` is enabled.
- Verify your account/service account can list TPU locations and accelerator types.
- Some zones may legitimately return no available TPU types for your project/quota.

### Permission errors on quotas or services

The script reads from Service Usage and Cloud Resource Manager. Missing permissions can reduce summary quality or fail generation. Re-run with `--verbose` to see the failing API.

### `ray up` fails after config generation

- Confirm quota and capacity in that exact zone.
- Try another generated zone file.
- If preemptible capacity is unstable, set `preemptible: false` for critical workloads.

## References

- [Ray Cluster Launcher Docs](https://docs.ray.io/en/latest/cluster/vms/index.html)
- [Cloud TPU Docs](https://cloud.google.com/tpu/docs)
- [Google Cloud ADC Setup](https://cloud.google.com/docs/authentication/provide-credentials-adc)

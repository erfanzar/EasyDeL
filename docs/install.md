# EasyDeL Installation and Usage Guide

## Installation

### Kaggle or Colab Installation

To install EasyDeL in a Kaggle or Colab environment, follow these steps:

```shell
pip uninstall torch-xla -y -q  # Remove pre-installed torch-xla (for TPUs)
pip install -U "easydel[tpu,torch,podutils]"  # Install EasyDeL with TPU + PyTorch + podutils extras
```

### Configuring TPU Hosts for Multi-Host or Multi-Slice Usage

To set up TPU hosts for multi-host or multi-slice environments, install `eopod`:

```shell
pip install eopod
```

If you encounter an error where `eopod` is not found, add the local bin path to your shell profile:

```shell
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # Apply changes immediately
```

#### Configuring TPU with EOpod

Next, configure `eopod` with your Google Cloud project details:

```shell
eopod configure --project-id YOUR_PROJECT_ID --zone YOUR_ZONE --tpu-name YOUR_TPU_NAME
```

#### Installing Required Packages on TPU Hosts

Use `eopod` to install the necessary packages on all TPU slices:

```shell
eopod run pip uninstall torch-xla -y -q
eopod run pip install -U "easydel[tpu,torch,podutils]"
```

## Using EasyDeL with Ray

EasyDeL supports distributed execution with Ray, particularly for multi-host and multi-slice TPU environments. For GPUs, manual configuration is required, but TPUs can leverage `eformer`, an EasyDeL utility for cluster management.

### Ray Version Compatibility (2.42 → 2.53)

EasyDeL pins Ray to a known-good version to avoid cluster/CLI mismatches:

- Project dependency: `ray[default]==2.53.0` (see `pyproject.toml`).
- Docker images also install `ray[default,gcp]==2.53.0` (see `Dockerfile`).

If you use Ray autoscaler (`ray up/attach/down`), run the CLI with the same pinned version:

```shell
uv run --python 3.13 python autoscale/generate-cluster-configs.py --project-id YOUR_PROJECT_ID --output-dir autoscale
CLUSTER_YAML=autoscale/easydel-<YOUR_ZONE>.yaml
uv run --python 3.13 ray up "$CLUSTER_YAML" --no-config-cache
uv run --python 3.13 ray attach "$CLUSTER_YAML"
```

Notes:

- Ray 2.47+ enables `uv run` runtime environment support by default; set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` to restore the older behavior.
- Ray 2.53 adds `.rayignore` support for controlling cluster uploads (this repo includes `.rayignore`). For `runtime_env` uploads, prefer `runtime_env={"excludes":[...]}`
  to avoid sending `.git/` and other large folders.

### Setting Up Ray with TPU Clusters

For a 2x v4-64 TPU setup, run:

```shell
eopod run "python -m eformer.executor.tpu_patch_ray --tpu-version TPU-VERSION --tpu-slice TPU-SLICES --num-slices NUM_SLICES --internal-ips INTERNAL_IP1-SLICE1,INTERNAL_IP2-SLICE1,INTERNAL_IP3-SLICE1,INTERNAL_IP4-SLICE1,INTERNAL_IP1-SLICE2,INTERNAL_IP2-SLICE2,INTERNAL_IP3-SLICE2,INTERNAL_IP4-SLICE2 --self-job"
```

For a v4-256 TPU:

```shell
eopod run "python -m eformer.executor.tpu_patch_ray --tpu-version v4 --tpu-slice 256 --num-slices 1 --internal-ips <comma-separated-TPU-IPs> --self-job"
```

### Automated TPU VM Setup (Optional)

This repo also ships a helper script to install EasyDeL + Ray on TPU VMs and run `eopod auto-config-ray`:

```shell
./scripts/tpu_setup.sh
```

## Usage Example

Once Ray is configured, you can use `eformer.escale.tpexec` instead of `eopod` for executing distributed code and benefiting from Ray's capabilities.

### Authenticating with Hugging Face and Weights & Biases

Before training, log in to Hugging Face and Weights & Biases:

```shell
eopod run "python -c 'from huggingface_hub import login; login(token=\"<API-TOKEN-HERE>\")'"
eopod run python -m wandb login <API-TOKEN-HERE>
```

### Training a Model with EasyDeL-DPO

Run the following command to fine-tune a model using EasyDeL's DPO framework:

```shell
eopod run python -m easydel.scripts.elarge --config dpo.yaml
```

This setup ensures proper installation and configuration for training large models using EasyDeL with TPUs and distributed environments.

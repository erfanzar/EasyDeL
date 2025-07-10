# EasyDeL Installation and Usage Guide

## Installation

### Kaggle or Colab Installation

To install EasyDeL in a Kaggle or Colab environment, follow these steps:

```shell
pip uninstall torch-xla -y -q  # Remove pre-installed torch-xla (for TPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -qU  # Install PyTorch for model conversion
pip install git+https://github.com/erfanzar/easydel -qU  # Install EasyDeL from the latest source
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -qU  # Install JAX for TPUs
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
eopod run pip install torch --index-url https://download.pytorch.org/whl/cpu -qU  # Required for model conversion
eopod run pip install git+https://github.com/erfanzar/easydel -qU  # Install EasyDeL from the latest source
eopod run pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -qU  # Install JAX for TPUs
```

## Using EasyDeL with Ray

EasyDeL supports distributed execution with Ray, particularly for multi-host and multi-slice TPU environments. For GPUs, manual configuration is required, but TPUs can leverage `eformer`, an EasyDeL utility for cluster management.

### Setting Up Ray with TPU Clusters

For a 2x v4-64 TPU setup, run:

```shell
eopod run "python -m eformer.executor.tpu_patch_ray --tpu-version TPU-VERSION --tpu-slice TPU-SLICES --num-slices NUM_SLICES --internal-ips INTERNAL_IP1-SLICE1,INTERNAL_IP2-SLICE1,INTERNAL_IP3-SLICE1,INTERNAL_IP4-SLICE1,INTERNAL_IP1-SLICE2,INTERNAL_IP2-SLICE2,INTERNAL_IP3-SLICE2,INTERNAL_IP4-SLICE2 --self-job"
```

For a v4-256 TPU:

```shell
eopod run "python -m eformer.executor.tpu_patch_ray --tpu-version v4 --tpu-slice 256 --num-slices 1 --internal-ips <comma-separated-TPU-IPs> --self-job"
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
eopod run python -m easydel.scripts.finetune.dpo \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --dataset_split "train[:90%]" \
  --refrence_model_repo_id meta-llama/Llama-3.3-70B-Instruct \
  --attn_mechanism vanilla \
  --beta 0.08 \
  --loss_type sigmoid \
  --max_length 2048 \
  --max_prompt_length 1024 \
  --ref_model_sync_steps 128 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --learning_rate_end 6e-7 \
  --log_steps 50 \
  --shuffle_train_dataset \
  --report_steps 1 \
  --progress_bar_type tqdm \
  --num_train_epochs 3 \
  --auto_shard_states \
  --optimizer adamw \
  --scheduler linear \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

This setup ensures proper installation and configuration for training large models using EasyDeL with TPUs and distributed environments.

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

-----

# Getting Started with EasyDeL on TPU Research Cloud (TRC)

Welcome to the TPU Research Cloud (TRC) platform! This guide will help you set up EasyDeL on Google Cloud TPUs for high-performance model training and fine-tuning. TRC provides free access to state-of-the-art TPU accelerators, enabling efficient training of large language models with EasyDeL's JAX-based framework.

## Why EasyDeL on TRC?

EasyDeL is designed for maximum performance and flexibility on TPU hardware:

- **High Performance**: Optimized JAX implementation for multi-host TPU training
- **Efficient Memory Usage**: Advanced sharding strategies and mixed precision support
- **Production Ready**: Streamlined workflows from research to deployment
- **Fully Customizable**: Build your own training pipelines or use ready-made scripts

## Initial Setup

### 1. Install and Configure `eopod`

First, install `eopod`, the command-line tool for managing TPU pods:

```shell
pip install eopod
```

> **Troubleshooting**: If you encounter a "command not found" error, add your local bin to PATH:
> ```shell
> echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```

Configure `eopod` with your TPU project details:

```shell
eopod configure --project-id YOUR_PROJECT_ID --zone YOUR_ZONE --tpu-name YOUR_TPU_NAME
```

### 2. Install Required Dependencies

Install the necessary packages for model training and conversion:

```shell
# Install required dependencies
eopod run pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install EasyDeL from the latest source
eopod run pip install git+https://github.com/erfanzar/easydel
```

### 3. Set Up Authentication

Connect to your experiment tracking and model hosting accounts:

```shell
# Login to Hugging Face Hub
eopod run "python -c 'from huggingface_hub import login; login(token=\"YOUR_HF_TOKEN\")'"

# Login to Weights & Biases
eopod run python -m wandb login YOUR_WANDB_TOKEN
```

## Fine-Tuning Methods

EasyDeL uses a single YAML-driven entrypoint for training and evaluation: `easydel.scripts.elarge`.

> **Tip**: The unified runner has a small CLI and reads the full configuration from YAML:
> ```shell
> eopod run python -m easydel.scripts.elarge --help
> ```

### Configure via YAML

Create a YAML file (e.g. `run.yaml`) with an `eLargeModel` config plus `actions`:

```yaml
config:
  model:
    name_or_path: meta-llama/Llama-3.1-8B-Instruct
  mixture:
    informs:
      - type: hf
        data_files: trl-lib/ultrafeedback_binarized
        split: "train[:90%]"
  trainer:
    trainer_type: dpo
    beta: 0.08
actions:
  - train
```

Run it:

```shell
eopod run python -m easydel.scripts.elarge --config run.yaml
```

### Method Notes

- **DPO**: set `trainer.trainer_type: dpo` and `reference_model.name_or_path`.
- **ORPO**: set `trainer.trainer_type: orpo`.
- **SFT**: set `trainer.trainer_type: sft` and configure `mixture.informs` to point at your text dataset.
- **GRPO**: requires Python reward functions; use the programmatic API (GRPOTrainer) or a custom wrapper that calls `eLargeModel.train(reward_funcs=...)`.
- **Reward model**: set `trainer.trainer_type: reward`.

## Common Parameters Explained

Most knobs map 1:1 to `eLargeModel` configuration keys:

- `config.model.name_or_path`: Hugging Face model repo (or local path)
- `config.mixture.informs`: Dataset sources (HF datasets via `type: hf` + `data_files: <dataset_name>`)
- `config.base_config.values.attn_mechanism`: Attention mechanism selection
- `config.sharding.axis_dims`: Device mesh sharding dims
- `config.trainer.total_batch_size`: Global batch size
- `config.trainer.learning_rate`: Optimizer LR
- `config.quantization.kv_cache`: KV cache quantization config (when supported)

## Advanced Usage

For advanced scenarios, the EasyDeL library offers full programmatic access to customize training loops, architectures, and optimization strategies. Check out the [documentation](https://easydel.readthedocs.io/) for more examples and API details.

## Getting Help

If you encounter any issues or have questions:

- Join our [Discord community](https://discord.gg/FCAMNqnGtt) for direct support
- Check the [documentation](https://easydel.readthedocs.io/) for detailed guides
- Explore the [GitHub repository](https://github.com/erfanzar/easydel) for examples and source code

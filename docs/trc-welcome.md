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

EasyDeL provides several state-of-the-art fine-tuning techniques. Below you'll find examples for each method with explanations of their use cases.

> **Tip**: To explore all available parameters for any script, use the `--help` flag:
> ```shell
> eopod run python -m easydel.scripts.finetune.dpo --help
> ```

### Direct Preference Optimization (DPO)

**Use Case**: Fine-tune a model from human preference data to align with human preferences without needing a separate reward model.

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

### Odds Ratio Policy Optimization (ORPO)

**Use Case**: An enhanced version of DPO that uses odds ratios to stabilize training and often results in better preference alignment.

```shell
eopod run python -m easydel.scripts.finetune.orpo \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --dataset_split "train" \
  --attn_mechanism vanilla \
  --beta 0.12 \
  --max_length 2048 \
  --max_prompt_length 1024 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --learning_rate_end 6e-7 \
  --log_steps 50 \
  --shuffle_train_dataset \
  --report_steps 1 \
  --progress_bar_type json \
  --num_train_epochs 3 \
  --auto_shard_states \
  --optimizer adamw \
  --scheduler linear \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

### Supervised Fine-Tuning (SFT)

**Use Case**: The most basic fine-tuning approach for adapting a model to a specific task or dataset, including multimodal models.

```shell
eopod run python -m easydel.scripts.finetune.sft \
  --repo_id Qwen/Qwen2.5-VL-72B-Instruct \
  --dataset_name trl-lib/Capybara \
  --dataset_split "train" \
  --dataset_text_field messages \
  --sharding_axis 1,-1,1,1 \
  --attn_mechanism vanilla \
  --max_sequence_length 2048 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --learning_rate_end 6e-7 \
  --log_steps 50 \
  --shuffle_train_dataset \
  --report_steps 1 \
  --progress_bar_type json \
  --num_train_epochs 3 \
  --auto_shard_states \
  --optimizer adamw \
  --scheduler linear \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

### Group Relative Policy Optimization for GSM8K (GRPO)

**Use Case**: Specifically designed for enhancing mathematical reasoning capabilities by using group-based preference comparisons on the GSM8K dataset.

```shell
eopod run python -m easydel.scripts.finetune.gsm8k_grpo \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --attn_mechanism vanilla \
  --sharding_axis 1,1,1,-1 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --beta 0.04 \
  --top_p 0.95 \
  --top_k 50 \
  --num_return_sequences 4 \
  --xml_reward 0.125 \
  --xml_full_match_reward 0.5 \
  --xml_full_match_reject 0.0 \
  --correctness_reward 2.0 \
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
  --use_wandb \
  --kv-cache-quantization 8bit
```

### Reward Model Training

**Use Case**: Train a reward model that can later be used for RLHF (Reinforcement Learning from Human Feedback) pipelines.

```shell
eopod run python -m easydel.scripts.finetune.reward \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --dataset_split "train" \
  --attn_mechanism vanilla \
  --max_sequence_length 2048 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --learning_rate_end 6e-7 \
  --log_steps 50 \
  --shuffle_train_dataset \
  --report_steps 1 \
  --progress_bar_type json \
  --num_train_epochs 3 \
  --auto_shard_states \
  --optimizer adamw \
  --scheduler linear \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

## Common Parameters Explained

The fine-tuning scripts share many parameters:

- `--repo_id`: The Hugging Face model repository to fine-tune
- `--dataset_name`: Dataset from Hugging Face Hub for training
- `--attn_mechanism`: Attention computation method (vanilla, flash, etc.)
- `--total_batch_size`: Total batch size across all TPU devices
- `--learning_rate` & `--learning_rate_end`: Initial and final learning rates for scheduling
- `--auto_shard_states`: Automatically shard model parameters across TPU devices
- `--save_steps`: Number of steps between model checkpoints
- `--kv-cache-quantization`: Enable KV cache quantization (e.g., 8bit) to reduce memory usage

## Advanced Usage

For advanced scenarios, the EasyDeL library offers full programmatic access to customize training loops, architectures, and optimization strategies. Check out the [documentation](https://easydel.readthedocs.io/) for more examples and API details.

## Getting Help

If you encounter any issues or have questions:

- Join our [Discord community](https://discord.gg/FCAMNqnGtt) for direct support
- Check the [documentation](https://easydel.readthedocs.io/) for detailed guides
- Explore the [GitHub repository](https://github.com/erfanzar/easydel) for examples and source code

# TRC Welcome Page

Welcome to the TRC platform! Let's get started by setting up `eopod` and running some fine-tuning examples.

**Note** :

`EasyDeL` is more than just a collection of scripts. It was designed to be both hackable and performant. To unlock the full power of EasyDeL—its performance, speed, and flexibility—we encourage you to write your own code, scripts, or even create customized models and runtime environments. This approach allows you to tailor EasyDeL to your specific needs and fully leverage its capabilities.

## Installation and Configuration

First, install `eopod` using pip:

```shell
pip install eopod
```

Next, configure `eopod` with your project details:

```shell
eopod configure --project-id YOUR_PROJECT_ID --zone YOUR_ZONE --tpu-name YOUR_TPU_NAME
```

Install the necessary dependencies and `easydel` from the latest source:

```shell
eopod run pip install tensorflow tensorflow-datasets  # Required for training
eopod run pip install torch --index-url https://download.pytorch.org/whl/cpu  # Required for model conversion
eopod run pip install git+https://github.com/erfanzar/easydel  # Install easydel from the latest source
```

Login to your wandb and huggingface account:

```shell
eopod run "python -c 'from huggingface_hub import login; login(token=\"<API-TOKEN-HERE>\")'"
eopod run wandb login <API-TOKEN-HERE>
```

## Notes

Each of the following examples includes customizable parameters. To explore available options, use the `--help` flag. For example:

```shell
python -m easydel.scripts.finetune.dpo --help
```

## Fine-Tuning Examples

### 1. DPO Fine-Tuning

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

### 2. ORPO Fine-Tuning

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

### 3. Supervised Fine-Tuning (SFT)

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

### 4. GRPO GSM8K-OAI Fine-Tuning

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

### 5. Reward Model Training

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

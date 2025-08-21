
# Tutorial: Fine-Tuning with DPO on TPUs using EasyDeL & Ray

This tutorial will guide you through fine-tuning a large language model, specifically **Qwen3-14B**, on a preference dataset, `mlabonne/orpo-dpo-mix-40k`, using the **Direct Preference Optimization (DPO)** algorithm. We'll leverage the EasyDeL library for efficient JAX-based training and Ray for distributed execution on Google TPUs.

**What is DPO?**
Direct Preference Optimization (DPO) is an elegant and effective algorithm for aligning large language models with human preferences. Unlike traditional Reinforcement Learning from Human Feedback (RLHF) methods like PPO or GRPO, DPO does not require training a separate reward model or extensive sampling. Instead, DPO directly optimizes the language model's policy to increase the likelihood of "chosen" (preferred) responses while decreasing the likelihood of "rejected" (dispreferred) responses, based on a dataset of human preferences. This makes DPO simpler to implement, more stable, and often more computationally efficient.

**Key Technologies Used:**

* **EasyDeL:** A JAX-based library designed for easy and efficient training/fine-tuning of large language models, especially on TPUs. It provides optimized trainers and model sharding.
* **Ray:** An open-source framework for building distributed applications, perfect for managing and coordinating compute resources like TPUs.
* **JAX:** A Python library for high-performance numerical computing, providing automatic differentiation and just-in-time compilation, making it ideal for accelerators like TPUs.
* **Hugging Face Transformers & Datasets:** For seamless model (tokenizer/processor) loading and efficient dataset access from the Hugging Face Hub.

---

## Prerequisites

1. **Google Cloud TPU:** You need access to a Google Cloud TPU environment (e.g., a TPU VM). This script is specifically configured for a `v4-64` TPU pod slice. Adjust the configuration if you have a different TPU type or size.
2. **Google Cloud Account & Project:** Properly configured for TPU usage, including necessary IAM permissions.
3. **Basic Python & ML Knowledge:** Familiarity with Python, virtual environments, and fundamental machine learning concepts.

---

## Step 1: Setting up your TPU Environment

The provided script relies on a setup script from EasyDeL to prepare the TPU environment with all necessary dependencies.

1. **SSH into your TPU VM.**
2. **Run the EasyDeL setup script:**

    ```bash
    bash <(curl -sL https://raw.githubusercontent.com/erfanzar/EasyDeL/refs/heads/main/tpu_setup.sh)
    ```

    This command will download and execute a script that installs required packages such as JAX, EasyDeL, Ray, Hugging Face libraries, and other Python dependencies optimized for TPU operation. This process might take several minutes.

3. **Set Environment Variables:**
    The Python script expects certain sensitive tokens (like Hugging Face and Weights & Biases API keys) to be available as environment variables. You should set these in your TPU VM's environment.

    * `HF_TOKEN_FOR_EASYDEL`: Your Hugging Face token. This is required to download models and datasets, especially if they are private or if you want to avoid rate limits. Generate one from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    * `WANDB_API_KEY_FOR_EASYDEL`: Your Weights & Biases API key. This is needed if you plan to use WandB for experiment tracking and logging. Get one from [wandb.ai/authorize](https://wandb.ai/authorize).

    You can set them temporarily for your current shell session:

    ```bash
    export HF_TOKEN_FOR_EASYDEL="hf_YOUR_HUGGINGFACE_TOKEN_HERE"
    export WANDB_API_KEY_FOR_EASYDEL="YOUR_WANDB_API_KEY_HERE"
    ```

    To make these environment variables persistent across reboots or new SSH sessions, add them to your shell's profile file (e.g., `~/.bashrc` or `~/.zshrc`):

    ```bash
    echo 'export HF_TOKEN_FOR_EASYDEL="hf_YOUR_HUGGINGFACE_TOKEN_HERE"' >> ~/.bashrc
    echo 'export WANDB_API_KEY_FOR_EASYDEL="YOUR_WANDB_API_KEY_HERE"' >> ~/.bashrc
    source ~/.bashrc # Apply changes
    ```

    The Python script will then retrieve these values using `os.environ.get()`.

---

## Step 2: Understanding the Python Script

Save the updated Python script (provided above) as `dpo_finetune.py` (or any other `.py` name) on your TPU VM. Let's walk through its key components.

```python
# ... (license header) ...

import os
import pprint
import ray
from typing import Any

from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray: This is the entry point for Ray.
# On a single TPU VM, it typically starts a local Ray instance.
# In a Ray cluster, it connects to the head node.
ray.init()

# --- Configuration Constants ---
# MODEL_ID: The pre-trained model to fine-tune. Qwen3-14B is a large, capable model.
MODEL_ID = "Qwen/Qwen3-14B"
# DATASET_ID: The DPO-formatted dataset. This dataset contains 'chosen' and 'rejected'
# conversation pairs, which DPO uses directly for training.
DATASET_ID = "mlabonne/orpo-dpo-mix-40k"
WANDB_ENTITY = None

# TPU_EXECUTION_ENV_VARS: A dictionary of environment variables to be set for each Ray worker
# running on the TPUs. These ensure components like Hugging Face libraries find necessary tokens
# and use shared memory for caches, speeding up I/O.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1", # Enables EasyDeL's automatic configuration for sharding.
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset", # Using /dev/shm for faster cache.
    "HF_HOME": "/dev/shm/huggingface", # Using /dev/shm for Hugging Face home.
    "HF_DATASETS_OFFLINE": "0", # Allow online access to datasets.
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}

# TPU_PIP_PACKAGES: A list of additional Python packages to be installed in the Ray worker
# environments. For DPO with this dataset, no extra packages are typically needed beyond
# what the EasyDeL setup script installs.
TPU_PIP_PACKAGES = []

# Print the environment variables for verification.
pprint.pprint(TPU_EXECUTION_ENV_VARS)

# --- TPU Accelerator Configuration ---
# tpu_config: Defines the TPU environment for Ray.
tpu_config = TpuAcceleratorConfig(
    "v4-64", # Specifies a TPU v4 pod slice with 64 chips. Change this to match your TPU.
    execution_env={ # Details for the execution environment on each Ray worker.
        "env_vars": TPU_EXECUTION_ENV_VARS,
        "pip": TPU_PIP_PACKAGES
    },
)

# --- Main Training Function (Decorated for Distributed Execution) ---
# @execute(tpu_config): This EasyDeL decorator handles the provisioning and setup
# of TPU resources via Ray, ensuring the decorated function runs on them.
# @ray.remote: This standard Ray decorator makes `main` a remote function,
# allowing Ray to manage its execution across the cluster.
@execute(tpu_config)
@ray.remote
def main():
    # Imports inside main(): These libraries are imported within the Ray worker processes
    # to ensure they are available in that distributed context.
    import jax
    from datasets import load_dataset
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    import easydel as ed

    # EasyDeL logger: Provides informative messages during the training process.
    logger = ed.utils.get_logger("DPO-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    # --- DPO-Specific Length Configurations ---
    # sequence_length: Max length for the prompt part of the input.
    sequence_length = 2048
    # max_length: Total maximum length for the combined prompt and completion.
    # Often twice the prompt length to accommodate full conversational turns.
    max_length = sequence_length * 2
    # total_batch_size: The effective batch size across all TPU devices.
    total_batch_size = 32

    # --- Tokenizer Setup ---
    # Load the tokenizer (referred to as 'processor' for Qwen models).
    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    # Crucially, ensure `pad_token_id` is set, typically to `eos_token_id`, for consistent padding.
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {processor.pad_token_id}")

    # --- Model Loading ---
    logger.info(f"Loading model: {MODEL_ID}")
    # AutoEasyDeLModelForCausalLM: EasyDeL's wrapper for loading and sharding
    # Hugging Face causal language models for JAX/TPU.
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16, # Use bfloat16 for computations (good for TPUs).
        param_dtype=jnp.bfloat16, # Store model parameters in bfloat16 for memory efficiency.
        precision=jax.lax.Precision.DEFAULT, # Default JAX precision for matrix multiplications.
        auto_shard_model=True, # EasyDeL automatically shards the model across TPU devices.
        # sharding_axis_dims: Defines how the model is sharded.
        # (1, -1, 1, 1, 1) typically corresponds to Fully Sharded Data Parallel (FSDP)-like sharding.
        # but let try sequence_parallel
        sharding_axis_dims=(1, 1, 1, 1, -1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            # Override specific model configuration parameters.
            freq_max_position_embeddings=max_length, # For RoPE-based models, sets max position for frequency encoding.
            mask_max_position_embeddings=max_length, # Max length for attention mask.
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE, # Disables quantization for KV cache.
            attn_mechanism=ed.AttentionMechanisms.AUTO, # EasyDeL selects the best attention mechanism (e.g., FlashAttention).
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE, # change this if u go OOM # Memory-saving technique for gradients.
        ),
        # partition_axis: Provides finer control over sharding, e.g., sharding KV heads via tensor parallelism.
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE, # Disables quantization for model weights.
    )
    logger.info("Model loaded successfully.")

    # --- Dataset Preparation ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    # load_dataset: Loads the 'mlabonne/orpo-dpo-mix-40k' dataset.
    # This dataset is already in a DPO-compatible format, with 'chosen' and 'rejected'
    # columns, which are conversation lists.
    # We load both 'train' and 'test' splits for training and evaluation.
    train_dataset, test_dataset = load_dataset(DATASET_ID, split=["train", "test"])
    logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # --- DPO Configuration (Hyperparameters for DPO Training) ---
    # DPOConfig: Defines all the hyperparameters for the DPO training process.
    arguments = ed.DPOConfig(
        num_train_epochs=1, # Number of training epochs.
        total_batch_size=total_batch_size, # Total batch size used across all devices.
        gradient_accumulation_steps=1, # Number of updates steps to accumulate gradients for.
        do_eval=True, # Enable evaluation on the test set during training.
        use_wandb=True,
        wandb_entity=WANDB_ENTITY,
        do_last_save=True, # Save the final model checkpoint after training.
        max_prompt_length=sequence_length, # Max length for the prompt.
        max_length=max_length, # Max total sequence length.
        max_completion_length=max_length - sequence_length, # Max length for the completion.
        max_training_steps=None, # Max number of training steps (None for full dataset).
        max_evaluation_steps=None, # Max number of evaluation steps (None for full test set).
        max_sequence_length=max_length, # Redundant but for clarity.
        loss_config=ed.LossConfig(z_loss=0.0), # DPO loss configuration, z_loss regularizes logits.
        track_memory=False, # Set to True to track memory usage (can add minor overhead).
        save_steps=1_000, # Save checkpoint every 1000 steps.
        save_total_limit=0, # 0 means save all checkpoints, otherwise limit number of checkpoints.
        save_optimizer_state=False, # Whether to save optimizer state with checkpoints (turn on for resuming training).
        per_epoch_training_steps=None, # Steps per epoch (None for automatic).
        per_epoch_evaluation_steps=None, # Evaluation steps per epoch (None for automatic).
        learning_rate=1e-5, # Initial learning rate.
        learning_rate_end=5e-7, # End learning rate for linear scheduler (not used with cosine).
        beta=0.1, # The DPO beta parameter. Controls how much the policy can deviate from the reference model.
                  # Higher beta means stronger regularization towards the reference.
        optimizer=ed.EasyDeLOptimizers.ADAMW, # Optimizer to use (AdamW).
        scheduler=ed.EasyDeLSchedulers.COSINE, # Learning rate scheduler (Cosine annealing).
        clip_grad=1.0, # Gradient clipping value.
        weight_distribution_log_steps=100, # Log weight distributions every 100 steps.
        warmup_steps=0, # Number of warmup steps for the scheduler.
        report_steps=10, # Report to WandB (if enabled) every 10 steps.
        log_steps=5, # Log metrics to console/logger every 5 steps.
        progress_bar_type="json", # Type of progress bar display.
        # save_directory="gs://your-bucket/dpo-qwen-mix" # Optional: Specify a GCS bucket for saving.
    )

    # --- Trainer Setup and Execution ---
    logger.info("Initializing DPOTrainer.")
    # DPOTrainer: EasyDeL's trainer class for DPO.
    trainer = ed.DPOTrainer(
        arguments=arguments, # Pass the DPO configuration.
        model=model, # The EasyDeL model instance.
        processing_class=processor, # The tokenizer/processor (DPOTrainer handles internal tokenization).
        train_dataset=train_dataset, # The dataset for training.
        eval_dataset=test_dataset, # The dataset for evaluation.
    )

    logger.info("Starting training...")
    trainer.train() # Start the DPO training loop.
    logger.info("Training finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # When the script is executed, `main()` is called. The decorators ensure
    # it runs correctly in the distributed Ray/TPU environment.
    main()
```

---

## Step 3: Running the Script

1. **Save the Code:** Ensure the Python script (`dpo_finetune.py`) is saved on your TPU VM.
2. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python dpo_finetune.py
    ```

    * Ray will initialize, and the `@execute(tpu_config)` decorator will provision the Ray actors on the TPUs according to your `tpu_config`.
    * The `main()` function will then run on these Ray actors.
    * You'll see logs related to dataset loading, model sharding, and then the DPO training progress (loss, rewards, etc.).
    * If Weights & Biases (WandB) is enabled (`WANDB_ENTITY` is set), metrics will be logged there.
    * Checkpoints will be saved according to `save_steps` and `save_directory` (if specified, defaults to `./EasyDeL-Checkpoints/`).

---

## Key Points and Customization

* **TPU Configuration (`tpu_config`):**
  * `accelerator`: Change `"v4-64"` to match your specific TPU setup (e.g., `"v3-8"`, `"v4-128"`, `"v5e-128"`). Ensure your Cloud TPU VM type matches this configuration.
  * `execution_env`: If you have other environment variables or pip packages specific to your Ray workers, add them here.
* **Model and Dataset (`MODEL_ID`, `DATASET_ID`):**
  * You can easily swap `Qwen/Qwen3-14B` for other Hugging Face causal language models (e.g., Llama, Mistral, Gemma).
  * Similarly, replace `mlabonne/orpo-dpo-mix-40k` with any other DPO-formatted dataset from the Hugging Face Hub. DPO datasets are expected to have `chosen` and `rejected` columns, where each is a list of dictionaries representing a conversation.
* **DPO Configuration (`ed.DPOConfig`):**
  * `total_batch_size`: Adjust based on your TPU memory and number of devices. A higher batch size can be more efficient but requires more memory.
  * `num_train_epochs`, `learning_rate`: Standard hyperparameters to tune for training duration and convergence.
  * `beta`: This is the **key DPO parameter**. It controls the "strength" of the preference learning. A higher beta (e.g., 0.5) enforces a stronger penalty for violating preferences, while a lower beta (e.g., 0.01) allows more deviation. Common values are between 0.01 and 0.5.
  * `max_prompt_length`, `max_length`, `max_completion_length`: These parameters are critical for tokenization and ensuring inputs fit your model's context window. Adjust them based on your dataset's typical conversation lengths. `max_length` should be `max_prompt_length + max_completion_length`.
  * `save_optimizer_state`: Set this to `True` if you intend to resume training from a checkpoint.
* **Sharding (`sharding_axis_dims`, `partition_axis`):**
  * The `(1, -1, 1, 1, 1)` in `sharding_axis_dims` is a common setting for Fully Sharded Data Parallel (FSDP)-like behavior in EasyDeL. For very large models or different TPU topologies, you might need to adjust this.
  * `partition_axis=ed.PartitionAxis(kv_head_axis="tp")` indicates tensor parallelism (TP) specifically for Key/Value heads. This is an advanced sharding strategy for better memory utilization.
* **Memory (`/dev/shm`):**
  * Using `/dev/shm` (shared memory) for Hugging Face caches (`HF_DATASETS_CACHE`, `HF_HOME`) can significantly speed up dataset loading and model initialization if these files are accessed repeatedly, as `/dev/shm` is a RAM disk.
* **Saving (`save_directory`):**
  * By default, models save to a local directory. For larger models or more permanent storage, consider setting `save_directory` in `DPOConfig` to a Google Cloud Storage (GCS) bucket path (e.g., `"gs://your-bucket-name/dpo-checkpoints"`). Ensure your TPU VM has write access to the bucket.

---

This tutorial provides a comprehensive overview of the DPO script and how to run it on TPUs. DPO is a powerful and efficient technique for model alignment, and EasyDeL + Ray makes it accessible for large-scale training. Happy fine-tuning!

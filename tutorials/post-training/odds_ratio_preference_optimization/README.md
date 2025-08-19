
# Tutorial: Fine-Tuning with ORPO on TPUs using EasyDeL & Ray

This tutorial will guide you through fine-tuning a large language model, **Qwen3-14B**, on a preference dataset using the **Odds Ratio Preference Optimization (ORPO)** algorithm. We'll leverage the EasyDeL library for efficient JAX-based training and Ray for distributed execution on Google TPUs.

**What is ORPO?**
ORPO is a novel fine-tuning technique that combines standard instruction tuning (Supervised Fine-Tuning, SFT) and preference alignment into a single, elegant training stage. Unlike DPO, which requires a model to first be instruction-tuned, ORPO uses a unique loss function that simultaneously maximizes the likelihood of the "chosen" (preferred) response while penalizing the "rejected" (dispreferred) response.

**Key Advantages of ORPO:**

* **Simplicity & Efficiency:** It merges SFT and preference tuning into one step, simplifying the fine-tuning pipeline and reducing computational requirements.
* **Improved Performance:** By learning from both the "chosen" response and the preference pair, ORPO can achieve strong results without the need for complex, multi-stage training.
* **Stability:** It avoids some potential pitfalls of traditional RLHF methods by using a direct loss function.

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

    This command downloads and executes a script that installs required packages such as JAX, EasyDeL, Ray, Hugging Face libraries, and other Python dependencies optimized for TPU operation. This process might take several minutes.

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

Save the provided Python script as `orpo_finetune.py` (or any other `.py` name) on your TPU VM. Let's walk through its key components.

```python
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# ... (license header) ...

import os
import pprint

import ray  # For distributed computing
from eformer.executor.ray import TpuAcceleratorConfig, execute  # EasyDeL's Ray utilities

# Initialize Ray: This is the entry point for Ray. On a single TPU VM,
# it starts a local Ray instance.
ray.init()

# --- Configuration Constants ---
# MODEL_ID: The pre-trained model to fine-tune.
MODEL_ID = "Qwen/Qwen3-14B"
# DATASET_ID: The dataset for ORPO training. This dataset contains 'chosen'
# and 'rejected' response pairs, which ORPO uses for its combined loss.
DATASET_ID = "mlabonne/orpo-dpo-mix-40k"
# WANDB_ENTITY: Your Weights & Biases entity (username/org). Set to None if not using.
WANDB_ENTITY = None

# TPU_EXECUTION_ENV_VARS: A dictionary of environment variables to be set for each Ray worker
# running on the TPUs. This ensures components like Hugging Face libraries find necessary tokens
# and use shared memory (/dev/shm) for caches, which speeds up I/O.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding configuration.
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),  # Your HuggingFace token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",
    "HF_HOME": "/dev/shm/huggingface",
    "HF_DATASETS_OFFLINE": "0",  # Allow online access to datasets.
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}

# TPU_PIP_PACKAGES: A list of additional Python packages to install in the Ray worker environments.
# For ORPO with this dataset, no extra packages are typically needed.
TPU_PIP_PACKAGES = []

# Pretty print the environment variables for verification.
pprint.pprint(TPU_EXECUTION_ENV_VARS)

# --- TPU Accelerator Configuration ---
# TpuAcceleratorConfig defines the TPU environment for Ray.
tpu_config = TpuAcceleratorConfig(
    "v4-64",  # Specifies a TPU v4 pod slice with 64 chips. Change this to match your TPU.
    execution_env={
        "env_vars": TPU_EXECUTION_ENV_VARS,
        "pip": TPU_PIP_PACKAGES,
    },
)


# --- Main Training Function (Decorated for Distributed Execution) ---
# @execute(tpu_config): This EasyDeL decorator handles provisioning and setup
# of TPU resources via Ray, ensuring the decorated function runs on them.
# @ray.remote: This standard Ray decorator makes `main` a remote function,
# allowing Ray to manage its execution across the cluster.
@execute(tpu_config)
@ray.remote
def main():
    # Imports inside main(): These libraries are imported within the Ray worker processes
    # to ensure they are available in that distributed context.
    import easydel as ed
    import jax
    from datasets import load_dataset
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    # EasyDeL logger provides informative messages during the training process.
    logger = ed.utils.get_logger("ORPO-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    # --- ORPO Specific Length Configurations ---
    sequence_length = 2048
    max_completion_length = 2048
    max_length = sequence_length + max_completion_length
    total_batch_size = 32

    # --- Tokenizer Setup ---
    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {processor.pad_token_id}")

    # --- Model Loading ---
    logger.info(f"Loading model: {MODEL_ID}")
    # AutoEasyDeLModelForCausalLM: EasyDeL's wrapper for loading and sharding
    # Hugging Face causal language models for JAX/TPU.
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,  # Use bfloat16 for computations (efficient on TPUs).
        param_dtype=jnp.bfloat16,  # Store model parameters in bfloat16.
        precision=jax.lax.Precision.DEFAULT,  # Default JAX precision for matmuls.
        auto_shard_model=True,  # EasyDeL automatically shards the model.
        # sharding_axis_dims: Defines how the model is sharded. `(1, -1, 1, 1, 1)` is a common
        # setting for Fully Sharded Data Parallel (FSDP)-like behavior.
        sharding_axis_dims=(1, -1, 1, 1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            # Override specific model configuration parameters.
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO, # EasyDeL picks the best attention mechanism.
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE, # change this if u go OOM
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"), # Advanced sharding for tensor parallelism.
        quantization_method=ed.EasyDeLQuantizationMethods.NONE, # No weight quantization.
    )
    logger.info("Model loaded successfully.")

    # --- Dataset Preparation ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    # We only load the 'train' split as per the script.
    # The dataset must contain 'chosen' and 'rejected' fields for ORPO.
    train_dataset = load_dataset(DATASET_ID, split="train")
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # --- ORPO Configuration (Hyperparameters for ORPO Training) ---
    # ORPOConfig: Defines all hyperparameters for the ORPO training process.
    arguments = ed.ORPOConfig(
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        gradient_accumulation_steps=1,
        do_eval=True,  # Note: Evaluation is enabled, but no eval_dataset is passed to the trainer.
                         # This will result in evaluation metrics on the training set.
        use_wandb=True,
        wandb_entity=WANDB_ENTITY,
        do_last_save=True,
        max_prompt_length=sequence_length,
        max_length=max_length,
        max_completion_length=max_completion_length,
        learning_rate=8e-6,  # A slightly higher learning rate is often used for ORPO.
        learning_rate_end=1e-6,
        beta=0.1,  # The ORPO beta parameter, balancing the SFT and preference loss components.
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        # ... other standard training arguments ...
        loss_config=ed.LossConfig(z_loss=0.0),
        track_memory=False,
        save_steps=1_000,
        save_total_limit=0,
        save_optimizer_state=False,
        clip_grad=1.0,
        report_steps=10,
        log_steps=5,
        progress_bar_type="json",
    )

    # --- Trainer Setup and Execution ---
    logger.info("Initializing ORPOTrainer.")
    # ORPOTrainer: EasyDeL's dedicated trainer class for the ORPO algorithm.
    trainer = ed.ORPOTrainer(
        arguments=arguments,  # Pass the ORPO configuration.
        model=model,  # The EasyDeL model instance.
        processing_class=processor,  # The tokenizer (ORPOTrainer handles tokenization internally).
        train_dataset=train_dataset,  # The dataset for training.
        eval_dataset=None,  # No separate evaluation dataset is provided in this script.
    )

    logger.info("Starting training...")
    trainer.train()  # Initiate the ORPO training process.
    logger.info("Training finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # When the script is executed, `main()` is called. The decorators ensure
    # it runs correctly in the distributed Ray/TPU environment.
    main()
```

---

## Step 3: Running the Script

1. **Save the Code:** Ensure the Python script (`orpo_finetune.py`) is saved on your TPU VM.
2. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python orpo_finetune.py
    ```

    * Ray will initialize, and the `@execute(tpu_config)` decorator will provision the Ray actors on the TPUs.
    * The `main()` function will then run on these actors.
    * You'll see logs related to dataset loading, model sharding, and then the ORPO training progress (loss, log-probabilities, etc.).
    * If Weights & Biases (WandB) is enabled, metrics will be logged to your WandB dashboard.
    * Checkpoints will be saved according to `save_steps` and `save_directory`.

---

## Key Points and Customization

* **TPU Configuration (`tpu_config`):**
  * `accelerator`: Change `"v4-64"` to match your specific TPU setup (e.g., `"v3-8"`, `"v4-128"`, `"v5e-128"`). Ensure your Cloud TPU VM type matches this configuration.
* **Model and Dataset (`MODEL_ID`, `DATASET_ID`):**
  * You can easily swap `Qwen/Qwen3-14B` for other Hugging Face causal language models.
  * Similarly, replace `mlabonne/orpo-dpo-mix-40k` with any other dataset formatted with `chosen` and `rejected` columns.
* **ORPO Configuration (`ed.ORPOConfig`):**
  * `learning_rate`: ORPO can often tolerate slightly higher learning rates than DPO because it includes an SFT-like objective. `8e-6` is a good starting point.
  * `beta`: This is the **key ORPO hyperparameter**. It controls the trade-off between the SFT loss (fitting the "chosen" response) and the odds ratio preference loss. A higher beta gives more weight to the preference component. The default of `0.1` is a common starting point.
  * `do_eval` and `eval_dataset`: In the provided script, `do_eval` is `True` but `eval_dataset` is `None`. This means EasyDeL will perform evaluation, but it will do so on a subset of the *training dataset*. To get a more reliable measure of generalization, you should load a separate test/validation split and pass it to `eval_dataset`. For example:

    ```python
    # In main():
    dataset = load_dataset(DATASET_ID, split="train")
    eval_dataset = load_dataset(DATASET_ID, split="test")

    # In ORPOTrainer():
    trainer = ed.ORPOTrainer(
        #...
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )
    ```

* **Sharding (`sharding_axis_dims`):**
  * The `(1, -1, 1, 1, 1)` setting is a robust default for FSDP-like sharding. For very large models or different TPU topologies, you might need to adjust this. Consult EasyDeL documentation for advanced sharding strategies.
* **Saving (`save_directory`):**
  * For permanent storage, especially with large models, set `save_directory` in `ORPOConfig` to a Google Cloud Storage (GCS) bucket path (e.g., `"gs://your-bucket-name/orpo-checkpoints"`). Ensure your TPU VM has write access to the bucket.

---

This tutorial provides a comprehensive overview of the ORPO script and how to run it on TPUs. ORPO is a very promising and efficient method for model alignment, and the combination of EasyDeL and Ray makes it accessible for large-scale training. Happy fine-tuning

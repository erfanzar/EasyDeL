# Tutorial: Supervised Fine-Tuning (SFT) on TPUs with EasyDeL & Ray

This tutorial will guide you through the process of **Supervised Fine-Tuning (SFT)** a large language model, **Qwen3-14B**, on a conversational dataset. We will use the EasyDeL library for efficient JAX-based training and Ray for distributed execution on Google TPUs.

**What is Supervised Fine-Tuning (SFT)?**
SFT is the most common and fundamental method for adapting a pre-trained language model to specific tasks or styles. Unlike preference-tuning methods (like DPO or ORPO) that learn from "chosen" vs. "rejected" pairs, SFT simply teaches the model to imitate high-quality examples. You provide the model with prompts and their desired responses, and the model learns to generate those responses by training on a standard "next-token prediction" loss.

SFT is essential for:

* Teaching a model to follow instructions.
* Adapting a model to a specific chat format (e.g., User/Assistant roles).
* Infusing a model with domain-specific knowledge in a particular style.

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
    * `WANDB_API_KEY_FOR_EASYDEL`: Your Weights & Biases API key. This is needed if you plan to use WandB for experiment tracking and logging.
    * `WANDB_ENTITY`: Your Weights & Biases username or organization. This can also be set here.

    You can set them temporarily for your current shell session:

    ```bash
    export HF_TOKEN_FOR_EASYDEL="hf_YOUR_HUGGINGFACE_TOKEN_HERE"
    export WANDB_API_KEY_FOR_EASYDEL="YOUR_WANDB_API_KEY_HERE"
    export WANDB_ENTITY="your_wandb_username"
    ```

    To make these environment variables persistent across reboots or new SSH sessions, add them to your shell's profile file (e.g., `~/.bashrc` or `~/.zshrc`):

    ```bash
    echo 'export HF_TOKEN_FOR_EASYDEL="hf_YOUR_HUGGINGFACE_TOKEN_HERE"' >> ~/.bashrc
    echo 'export WANDB_API_KEY_FOR_EASYDEL="YOUR_WANDB_API_KEY_HERE"' >> ~/.bashrc
    echo 'export WANDB_ENTITY="your_wandb_username"' >> ~/.bashrc
    source ~/.bashrc # Apply changes
    ```

    The Python script will then retrieve these values using `os.environ.get()`.

---

## Step 2: Understanding the Python Script

Save the provided Python script as `sft_finetune.py` (or any other `.py` name) on your TPU VM. Let's walk through its key components, focusing on what makes SFT unique.

```python
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# ... (license header) ...

import os
import pprint

import ray  # For distributed computing
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray. This ensures Ray is ready to manage distributed tasks.
ray.init()

# --- Configuration Constants ---
# MODEL_ID: Specifies the base model to be fine-tuned.
MODEL_ID = "Qwen/Qwen3-14B"
# DATASET_ID: The dataset for SFT. 'allenai/WildChat' is a dataset with
# conversational data suitable for instruction/chat tuning.
DATASET_ID = "allenai/WildChat"
# WANDB_ENTITY: Your Weights & Biases entity (username or organization).
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# Environment variables that will be passed to each Ray worker process on the TPUs.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",
    "HF_HOME": "/dev/shm/huggingface",
    "HF_DATASETS_OFFLINE": "0",
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}

# Additional pip packages to be installed on each Ray worker.
TPU_PIP_PACKAGES = []

pprint.pprint(TPU_EXECUTION_ENV_VARS)

# --- TPU Accelerator Configuration ---
tpu_config = TpuAcceleratorConfig(
    "v4-64",  # Specifies a TPU v4 pod slice with 64 chips.
    execution_env={
        "env_vars": TPU_EXECUTION_ENV_VARS,
        "pip": TPU_PIP_PACKAGES,
    },
)

# --- Main Training Function (Decorated for Ray and TPU execution) ---
@execute(tpu_config)
@ray.remote
def main():
    import easydel as ed
    import jax
    from datasets import load_dataset
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    logger = ed.utils.get_logger("SFT-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    max_length = 4096
    total_batch_size = 32

    # --- Tokenizer Setup ---
    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    # Left padding is important for causal language models during training.
    processor.padding_side = "left"
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {processor.pad_token_id}")

    # --- Model Loading ---
    logger.info(f"Loading model: {MODEL_ID}")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1), # FSDP-like sharding
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE, # change this if u go OOM
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
    )
    logger.info("Model loaded successfully.")

    # --- Dataset Preparation ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    train_dataset = load_dataset(DATASET_ID, split="train")
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # --- SFT Configuration (Hyperparameters for SFT training) ---
    arguments = ed.SFTConfig(
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        gradient_accumulation_steps=1,
        do_eval=True, # Will evaluate on the training set as no eval_dataset is passed.
        use_wandb=WANDB_ENTITY is not None,
        wandb_entity=WANDB_ENTITY,
        do_last_save=True,
        max_sequence_length=max_length,
        learning_rate=1e-5,
        learning_rate_end=7e-6,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        # packing: Set to True to combine multiple short examples into one sequence, improving
        # training efficiency for datasets with many short conversations.
        packing=False,
        # ... other standard training arguments ...
        save_steps=1_000,
        save_total_limit=0,
        save_optimizer_state=False,
        clip_grad=1.0,
        report_steps=10,
        log_steps=5,
    )

    # --- Trainer Setup and Execution ---
    logger.info("Initializing SFTTrainer.")
    # SFTTrainer: EasyDeL's dedicated trainer class for Supervised Fine-Tuning.
    trainer = ed.SFTTrainer(
        arguments=arguments,  # Pass the SFT configuration.
        model=model,  # The EasyDeL model instance.
        processing_class=processor,  # The tokenizer/processor.
        train_dataset=train_dataset,  # The dataset for training.
        eval_dataset=None,  # No separate evaluation dataset is provided.
        # formatting_func: This is a KEY component for SFT.
        # It's a function that takes a batch of examples and formats them into
        # a single string that the model will be trained on.
        # Here, we use the tokenizer's `apply_chat_template` method, which correctly
        # formats the conversational data from the 'conversation' column of the dataset
        # into the specific chat format expected by the Qwen3 model (e.g., adding
        # special tokens for user/assistant roles).
        formatting_func=lambda batch: processor.apply_chat_template(
            batch["conversation"], tokenize=False
        ),
    )

    logger.info("Starting training...")
    trainer.train()  # Initiate the SFT training process.
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
```

---

## Step 3: Running the Script

1. **Save the Code:** Ensure the Python script (`sft_finetune.py`) is saved on your TPU VM.
2. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python sft_finetune.py
    ```

    * Ray will initialize, and the `@execute(tpu_config)` decorator will provision the Ray actors on the TPUs.
    * The `main()` function will then run on these actors.
    * You'll see logs related to dataset loading, model sharding, and then the SFT training progress (loss, perplexity, etc.).
    * If Weights & Biases (WandB) is enabled, metrics will be logged to your WandB dashboard.
    * Checkpoints will be saved according to `save_steps`.

---

## Key Points and Customization

* **`formatting_func` is Crucial:** This is the most important part to understand for SFT. The `SFTTrainer` uses this function to prepare your data.
  * The provided `lambda batch: processor.apply_chat_template(batch["conversation"], tokenize=False)` is perfect for datasets that have a `conversation` column with a list of chat turns (e.g., `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`).
  * If your dataset has different column names, like `prompt` and `response`, you would need to change this function. For example:

    ```python
    def format_prompt_response(batch):
        # Create a list of formatted strings for the batch
        formatted_texts = []
        for prompt, response in zip(batch['prompt'], batch['response']):
            # Manually create the chat format if needed
            text = f"User: {prompt}\nAssistant: {response}"
            formatted_texts.append(text)
        return formatted_texts

    # Then in the SFTTrainer:
    trainer = ed.SFTTrainer(
        # ...
        formatting_func=format_prompt_response,
    )
    ```

* **`packing=True` for Efficiency:** If your dataset contains many short examples (e.g., single-turn Q&A), setting `packing=True` in `SFTConfig` can significantly speed up training. It works by concatenating multiple short examples into a single sequence of `max_sequence_length`, separated by an EOS token. This ensures the model is always processing full-length sequences, maximizing TPU utilization.
* **Evaluation Dataset:** For a more meaningful evaluation of how your model is learning to generalize, you should always use a separate validation or test split.

    ```python
    # In main():
    train_dataset = load_dataset(DATASET_ID, split="train")
    eval_dataset = load_dataset(DATASET_ID, split="test") # Or another eval split

    # In SFTTrainer():
    trainer = ed.SFTTrainer(
        #...
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #...
    )
    ```

* **Hyperparameters:**
  * `learning_rate`: `1e-5` is a common and safe starting point for SFT. You can experiment with slightly higher or lower values.
  * `total_batch_size`: Adjust based on your TPU memory and the number of devices.
* **Model and Dataset:** You can easily swap `MODEL_ID` and `DATASET_ID` with any Hugging Face model or dataset. Just make sure your `formatting_func` is compatible with the new dataset's structure.
* **Saving (`save_directory`):** For permanent storage, especially with large models, set `save_directory` in `SFTConfig` to a Google Cloud Storage (GCS) bucket path (e.g., `"gs://your-bucket-name/sft-checkpoints"`). Ensure your TPU VM has write access to the bucket.

---

This tutorial provides a comprehensive guide to performing SFT on TPUs. It is the foundational step for almost all model customization and a critical skill for any LLM practitioner. The combination of EasyDeL's `SFTTrainer` and Ray's distributed execution makes this process efficient and scalable. Happy fine-tuning

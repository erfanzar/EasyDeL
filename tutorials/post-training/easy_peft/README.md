
# The Universal PEFT Recipe for EasyDeL: Adding LoRA to Any Trainer (SFT, DPO, ORPO, GRPO)

This tutorial showcases one of the most powerful and efficient features of the EasyDeL library: the ability to apply **Parameter-Efficient Fine-Tuning (PEFT)**, specifically **LoRA**, to *any* of its training methods with a **single line of code**.

Whether you are doing Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), Odds Ratio Preference Optimization (ORPO), or Group Relative Policy Optimization (GRPO), the process for enabling memory-efficient LoRA training is exactly the same.

**What is PEFT and LoRA?**
**PEFT (Parameter-Efficient Fine-Tuning)** is a family of techniques designed to fine-tune large models without updating all of their billions of parameters. **LoRA (Low-Rank Adaptation)** is a star player in this family. It freezes the original model and injects small, trainable "adapter" layers. We only train these tiny adapters, which offers huge benefits:

* **Massive Memory Savings:** Drastically reduces the memory needed for gradients and optimizers.
* **Faster Training:** Fewer parameters to update can speed up training iterations.
* **Tiny Checkpoints:** Instead of saving a 28GB model, you save a few megabytes of adapter weights.
* **Modularity:** Use one base model with many different LoRA adapters for various tasks.

## The "One-Line" Magic of PEFT in EasyDeL

The core philosophy in EasyDeL is simplicity and power. To convert *any* full fine-tuning script into a PEFT script, you only need to do one thing: apply the LoRA layers to the model after it's loaded.

**Before (Full Fine-Tuning):**

```python
# 1. Load the full model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(MODEL_ID, ...)

# 2. Create the trainer with the full model
trainer = ed.SFTTrainer(
    model=model,
    ...
)
```

**After (PEFT/LoRA Fine-Tuning):**

```python
# 1. Load the full model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(MODEL_ID, ...)

# 2. THE MAGIC LINE: Apply LoRA adapters
model = model.apply_lora_to_layers(LORA_RANK, LORA_PATTERN)

# 3. Create the trainer with the PEFT-modified model
# The trainer automatically knows to only train the LoRA parameters!
trainer = ed.SFTTrainer(
    model=model,
    ...
)
```

That's it. The trainer (whether it's `SFTTrainer`, `DPOTrainer`, `ORPOTrainer`, or `GRPOTrainer`) intelligently detects the trainable adapter parameters and handles the rest.

---

## Example in Action: PEFT with an SFT Trainer

Let's see this with the complete, runnable script you provided. This script performs SFT, but with LoRA enabled.

### The PEFT/LoRA Script for SFT

```python
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# ... (license header) ...

import os
import pprint

import ray  # For distributed computing
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray.
ray.init()

# --- NEW: LoRA Configuration Constants ---
# LORA_RANK: The dimension of the LoRA adapter matrices. A higher rank means more
# trainable parameters and more expressive power, at the cost of memory.
LORA_RANK = 64
# LORA_PATTERN: A regular expression to select which layers to apply LoRA to.
# This pattern targets the query, key, value, and output projection layers in attention blocks.
LORA_PATTERN = ".*(q_proj|k_proj|v_proj|o_proj).*"

# --- Configuration Constants ---
MODEL_ID = "Qwen/Qwen3-14B"
DATASET_ID = "LDJnr/Pure-Dove"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Environment and TPU Configuration (Same as before) ---
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1", "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset", "HF_HOME": "/dev/shm/huggingface",
    "HF_DATASETS_OFFLINE": "0", "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}
TPU_PIP_PACKAGES = []
tpu_config = TpuAcceleratorConfig("v4-64", execution_env={"env_vars": TPU_EXECUTION_ENV_VARS, "pip": TPU_PIP_PACKAGES})


@execute(tpu_config)
@ray.remote
def main():
    import easydel as ed
    import jax
    from datasets import load_dataset
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    logger = ed.utils.get_logger("PEFT-SFT-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    max_length = 4096
    total_batch_size = 32

    # --- Tokenizer and Model Loading (Same as before) ---
    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    processor.padding_side = "left"
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    logger.info(f"Loading base model: {MODEL_ID}")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
        # ... other config options ...
    )
    logger.info("Base model loaded successfully.")

    # --- Dataset Preparation (Same as before) ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    train_dataset = load_dataset(DATASET_ID, split="train")

    # ==============================================================================
    # --- THE UNIVERSAL PEFT STEP ---
    # This single line converts our full model into a memory-efficient PEFT model.
    logger.info(f"Applying LoRA with rank={LORA_RANK} to pattern='{LORA_PATTERN}'")
    model = model.apply_lora_to_layers(LORA_RANK, LORA_PATTERN)
    # ==============================================================================

    # --- SFT Configuration (No changes needed for LoRA) ---
    arguments = ed.SFTConfig(
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        max_sequence_length=max_length,
        learning_rate=1e-5, # LoRA can sometimes use a higher LR, e.g., 1e-4
        use_wandb=WANDB_ENTITY is not None,
        wandb_entity=WANDB_ENTITY,
        do_last_save=True,
        save_steps=1_000,
        packing=False,
        # ... other arguments ...
    )

    # --- Trainer Setup and Execution (Works seamlessly) ---
    logger.info("Initializing SFTTrainer for PEFT.")
    trainer = ed.SFTTrainer(
        arguments=arguments,
        model=model, # Pass the LoRA-modified model
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=None,
        formatting_func=lambda batch: processor.apply_chat_template(
            batch["conversation"], tokenize=False
        ),
    )

    logger.info("Starting PEFT training...")
    trainer.train()
    logger.info("PEFT training finished.")


if __name__ == "__main__":
    main()
```

---

## How to Apply This to DPO, ORPO, or GRPO

This is where the true power lies. To convert your DPO, ORPO, or GRPO scripts to use PEFT, you just add the **exact same line**.

### Example: Modifying a DPO Script for PEFT

Imagine you have a DPO training script. You would find the model loading section and insert the LoRA line right after it.

```python
# Inside your DPO script's main() function...

# 1. Load the model as usual
logger.info(f"Loading base model: {MODEL_ID}")
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    # ... your model config ...
)
logger.info("Base model loaded successfully.")

# 2. Add the magic line
logger.info(f"Applying LoRA with rank={LORA_RANK} to pattern='{LORA_PATTERN}'")
model = model.apply_lora_to_layers(LORA_RANK, LORA_PATTERN)

# 3. Load your dataset
train_dataset, test_dataset = load_dataset(DATASET_ID, split=["train", "test"])

# 4. Create your DPOTrainer as usual. It will automatically handle PEFT.
trainer = ed.DPOTrainer(
    arguments=dpo_arguments,
    model=model, # Pass the LoRA-modified model
    processing_class=processor,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
```

### Example: Modifying an ORPO or GRPO Script for PEFT

The process is identical for ORPO and GRPO.

```python
# Inside your ORPO/GRPO script's main() function...

# 1. Load the model as usual
logger.info(f"Loading base model: {MODEL_ID}")
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    MODEL_ID,
    # ... your model config ...
)

# 2. Add the magic line
logger.info(f"Applying LoRA with rank={LORA_RANK} to pattern='{LORA_PATTERN}'")
model = model.apply_lora_to_layers(LORA_RANK, LORA_PATTERN)

# 3. Create your ORPOTrainer or GRPOTrainer as usual.
trainer = ed.ORPOTrainer( # Or ed.GRPOTrainer(...)
    arguments=orpo_arguments,
    model=model, # Pass the LoRA-modified model
    # ... other trainer arguments ...
)

trainer.train()
```

---

## Running the Script and Key Takeaways

* **Execution:** Run the script from your TPU VM terminal: `python peft_sft_finetune.py`.
* **Observe the Difference:** You will notice significantly lower memory usage and your saved checkpoints will be tiny (megabytes instead of gigabytes), containing only the trained adapter weights.
* **Universality is Key:** This "one-line" approach works for **all EasyDeL trainers**. This makes experimentation incredibly fast and efficient. You can try full fine-tuning, then switch to LoRA by adding one line, without changing any other logic in your trainer setup.
* **Tune `LORA_RANK` and `LORA_PATTERN`:** These are your main PEFT hyperparameters.
  * `LORA_RANK`: Controls the capacity of the adapter. Start with `16` or `32` and increase if the model underfits.
  * `LORA_PATTERN`: Controls which layers get adapters. Targeting attention projections (`q_proj`, `k_proj`, etc.) is the most common and effective strategy.
* **Learning Rate:** You can often use a slightly higher learning rate for LoRA (e.g., `1e-4`) than for full fine-tuning, as you are training far fewer parameters.

By mastering this simple, universal recipe, you unlock the ability to efficiently fine-tune massive language models on a wide variety of tasks and training paradigms, all within the powerful and streamlined EasyDeL ecosystem.

# Tutorial: Fine-Tuning with GRPO on TPUs using EasyDeL & Ray

This tutorial will guide you through fine-tuning a language model (Cohere Aya-8B) on a math problem-solving dataset (NuminaMath) using the Group Relative Policy Optimization (GRPO) algorithm. We'll leverage the EasyDeL library for efficient JAX-based training and Ray for distributed execution on TPUs.

**What is GRPO?**
GRPO is an algorithm similar to Reinforcement Learning from Human Feedback (RLHF) methods like PPO, but often simpler to implement and tune for generative tasks. It aims to steer the model towards generating responses that maximize a given reward signal, often by comparing multiple generated responses.

**Key Technologies Used:**

* **EasyDeL:** A JAX-based library designed for easy and efficient training/fine-tuning of large language models, especially on TPUs.
* **Ray:** An open-source framework for building distributed applications, perfect for managing TPU resources.
* **JAX:** A Python library for high-performance numerical computing, especially well-suited for accelerators like TPUs.
* **Hugging Face Transformers & Datasets:** For model loading (tokenizer) and dataset access.
* **`math_verify`:** A utility to check the correctness of mathematical solutions.

---

## Prerequisites

1. **Google loud TPU:** You need access to a Google Cloud TPU environment (e.g., a TPU VM). This script is configured for a `v4-64` TPU pod slice.
2. **Google Cloud Account & Project:** Properly configured for TPU usage.
3. **Basic Python & ML Knowledge:** Familiarity with Python, virtual environments, and basic machine learning concepts.

---

## Step 1: Setting up your TPU Environment

The provided script uses a setup script from EasyDeL to prepare the TPU environment.

1. **SSH into your TPU VM.**
2. **Run the setup script:**

    ```bash
    bash <(curl -sL https://raw.githubusercontent.com/erfanzar/EasyDeL/refs/heads/main/tpu_setup.sh)
    ```

    This script will install necessary dependencies like JAX, EasyDeL, Ray, and other Python packages required for TPU operation. It might take a few minutes.

3. **Set Environment Variables:**
    The script requires certain environment variables, especially API keys. You should set these in your TPU VM's environment.
    * `HF_TOKEN_FOR_EASYDEL`: Your Hugging Face token (if accessing private models/datasets, or to avoid rate limits). Get one from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    * `WANDB_API_KEY_FOR_EASYDEL`: Your Weights & Biases API key if you want to use WandB for logging. Get one from [wandb.ai/authorize](https://wandb.ai/authorize).

    You can set them in your shell session:

    ```bash
    export HF_TOKEN_FOR_EASYDEL="your_hf_token_here"
    export WANDB_API_KEY_FOR_EASYDEL="your_wandb_api_key_here"
    # To make them persistent, add them to your ~/.bashrc or ~/.zshrc
    # echo 'export HF_TOKEN_FOR_EASYDEL="your_hf_token_here"' >> ~/.bashrc
    # echo 'export WANDB_API_KEY_FOR_EASYDEL="your_wandb_api_key_here"' >> ~/.bashrc
    # source ~/.bashrc
    ```

    The script will pick these up using `os.environ.get()`.

---

## Step 2: Understanding the Python Script

Let's break down the provided Python script. Save it as `launch.py` (or any name you prefer) on your TPU VM.

```python
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# ... (license header) ...

import os
import re
from typing import Any

import ray # For distributed computing
from eformer.executor.ray import TpuAcceleratorConfig, execute # EasyDeL's Ray utilities

# Initialize Ray. This should be done once per application.
# If running in a Ray cluster, this might connect to an existing cluster.
# On a single TPU VM, it starts a local Ray instance.
ray.init()

# --- Configuration Constants ---
# These define the model, dataset, and WandB entity (if used)
MODEL_ID = "cohereLabs/aya-expanse-8b"
DATASET_ID = "AI-MO/NuminaMath-TIR"
WANDB_ENTITY = None # Set this to your WandB username or org if you use WandB

# Environment variables for Ray workers on TPUs.
# These ensure that the TPU environment has access to necessary tokens and paths.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1", # Enables EasyDeL specific auto-configurations
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""), # HuggingFace Token
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset", # Use shared memory for faster dataset caching
    "HF_HOME": "/dev/shm/huggingface", # Use shared memory for HuggingFace home
    "HF_DATASETS_OFFLINE": "0", # Allow online dataset fetching
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""), # WandB API Key
}

# Additional pip packages to be installed in the Ray worker environments.
TPU_PIP_PACKAGES = ["math_verify"] # For the accuracy reward function

# System prompt to guide the model's generation style.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the think process in the mind and then provides the user with the answer. "
    "The think process and answer are enclosed within <think> </think> and answer needs no tag tags, respectively, i.e., "
    "<think> think process here </think> answer here"
)


# --- TPU Accelerator Configuration ---
# This configures the Ray execution environment for TPUs.
tpu_config = TpuAcceleratorConfig(
    "v4-64", # Specifies the TPU type and size (e.g., v4 pod with 64 chips)
    execution_env={ # Defines the environment for each Ray worker
        "env_vars": TPU_EXECUTION_ENV_VARS, # Passes our defined environment variables
        "pip": TPU_PIP_PACKAGES # Installs specified pip packages
    },
)

# --- Main Training Function (Decorated for Ray) ---
# @execute(tpu_config): EasyDeL decorator to run this function on TPUs managed by Ray.
# @ray.remote: Standard Ray decorator to make this function a remote task.
@execute(tpu_config)
@ray.remote
def main():
    # Imports are inside main() because this function will run in a separate Ray worker process.
    # This ensures these libraries are imported in that specific environment.
    import jax
    from datasets import load_dataset
    from jax import numpy as jnp
    from math_verify import LatexExtractionConfig, parse, verify  # type: ignore
    from transformers import AutoTokenizer

    import easydel as ed # EasyDeL library

    # Get a logger instance from EasyDeL utilities
    logger = ed.utils.get_logger("GRPO-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    # --- Tokenizer Setup ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left" # Crucial for causal LMs; new tokens are added to the right.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id # Set pad token if not defined
        logger.info(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")

    # --- GRPO Configuration ---
    # These are hyperparameters for the GRPO algorithm and training loop.
    total_batch_size = 8
    num_return_sequences = 4 # How many responses to generate per prompt for GRPO
    top_k = 50
    top_p = 0.95
    temperature = 0.7
    max_prompt_length = 2048 # Max length of the input prompt tokens
    max_completion_length = 2048 # Max length of the generated response tokens

    grpo_config = ed.GRPOConfig(
        total_batch_size=total_batch_size,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        learning_rate=1e-6,
        learning_rate_end=6e-7, # For linear scheduler
        log_steps=5, # Log metrics every 5 steps
        report_steps=10, # Report to WandB (if enabled) every 10 steps
        progress_bar_type="json", # Type of progress bar
        num_train_epochs=3,
        optimizer=ed.EasyDeLOptimizers.ADAMW, # Optimizer type
        scheduler=ed.EasyDeLSchedulers.LINEAR, # Learning rate scheduler
        do_last_save=True, # Save model at the end of training
        track_memory=False, # Set to True to track memory usage (can add overhead)
        save_steps=1000, # Save checkpoint every 1000 steps
        save_total_limit=0, # 0 means save all checkpoints, otherwise limit number of checkpoints
        save_optimizer_state=False, # Whether to save optimizer state with checkpoints
        use_wandb=(WANDB_ENTITY is not None), # Enable WandB if WANDB_ENTITY is set
        wandb_entity=WANDB_ENTITY,
        clip_grad=1.0, # Gradient clipping value
        weight_distribution_log_steps=100, # Log weight distributions (for debugging)
        warmup_steps=0, # Number of warmup steps for the scheduler
        beta=0.04, # GRPO beta parameter (controls KL divergence from reference model, implicitly self)
        num_return_sequences=num_return_sequences, # Must match generation param
        top_p=top_p, # Sampling param for generation
        top_k=top_k, # Sampling param for generation
        temperature=temperature, # Sampling param for generation
        # save_directory="gs://your-bucket/grpo-aya-math" # Optional: specify a GCS bucket for saving
    )

    # Calculate total sequence length based on prompt and completion lengths
    max_sequence_length = grpo_config.max_completion_length + grpo_config.max_prompt_length

    # --- Model Loading ---
    logger.info(f"Loading model: {MODEL_ID}")
    # AutoEasyDeLModelForCausalLM handles sharding and precision automatically for TPUs.
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        auto_shard_model=True, # Automatically shard the model across TPU devices
        # Sharding axis dimensions: (data_parallel, fully_sharded_data_parallel, tensor_parallel_mlp, tensor_parallel_attention_heads, sequence_parallel)
        # These depend on the model architecture and TPU topology. (1, -1, 1, 1, 1) means FSDP-like sharding.
        sharding_axis_dims=(1, -1, 1, 1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            # Override model config parameters for this specific training run
            freq_max_position_embeddings=max_sequence_length, # For RoPE scaling if model uses it
            mask_max_position_embeddings=max_sequence_length, # Max sequence length for attention mask
            attn_dtype=jnp.bfloat16, # Use bfloat16 for attention computations (good for TPUs)
            attn_softmax_dtype=jnp.bfloat16, # Softmax dtype in attention
            kvdtype=jnp.bfloat16, # Key/Value cache dtype
            attn_mechanism=ed.AttentionMechanisms.AUTO, # Let EasyDeL choose best attention (e.g., Flash Attention)
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE, # change this if u go OOM # Checkpointing strategy for memory saving
        ),
        param_dtype=jnp.bfloat16, # Model parameters in bfloat16
        dtype=jnp.bfloat16, # Default dtype for computations
        precision=jax.lax.Precision.DEFAULT, # JAX precision for matmuls
    )
    logger.info("Model loaded successfully.")

    # --- Reward Functions ---
    # These functions calculate rewards for the generated completions.
    # GRPO will try to maximize these rewards.

    def format_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
        """
        Reward function that checks if the completion starts with a <think>...</think> block.
        This encourages the model to follow the specified output format.
        'completions' is a list of lists, outer list for batch, inner for num_return_sequences.
        Each item in inner list is like [{"role": "assistant", "content": "..."}]
        """
        pattern = r"^<think>.*?</think>" # Regex to find <think>...</think> at the start
        rewards = []
        # completions will be of shape (batch_size * num_return_sequences, 1, dictionary)
        # The trainer reshapes it before passing to reward function, so it becomes
        # (batch_size, num_return_sequences, dictionary)
        for completion_group in completions: # Iterates through each prompt's set of generations
            # We only care about the first (and only) message in the assistant's turn here.
            # And for this specific reward, we only check the first generated sequence if there are multiple.
            # GRPO often uses the *first* of the `num_return_sequences` as the primary completion to reward.
            # However, the script as written (and common GRPO practice) might compute rewards for *all*
            # generated sequences and then GRPO internally decides how to use these (e.g. best of N).
            # Let's assume the trainer expects a reward for *each* of the `num_return_sequences`.
            # If `completions` is already (batch_size * num_return_sequences, 1, dict), then:
            # rewards.append(1.0 if re.match(pattern, completion_group[0]["content"], re.DOTALL) else 0.0)

            # Looking at the GRPOTrainer, it seems it expects a flat list of rewards,
            # one for each generated sequence across the batch.
            # So if completions is (batch_size, num_return_sequences, dict_list)
            # We need to flatten the rewards.
            # Let's re-evaluate based on GRPOTrainer logic. It seems it passes `completions`
            # as a list where each element corresponds to one generated sequence.
            # The structure is `list[list[dict[str, str]]]`. The outer list is for each sample in the effective batch
            # (original_batch_size * num_return_sequences). The inner list is the conversation history for that sample
            # (usually just one turn from the assistant).
            if completion_group and "content" in completion_group[0]: # completion_group is one generated sequence
                content = completion_group[0]["content"]
                rewards.append(1.0 if re.match(pattern, content, re.DOTALL) else 0.0)
            else:
                rewards.append(0.0) # Should not happen if generation is successful
        return rewards

    def accuracy_reward(prompts, completions, batch, **kwargs):
        """
        Reward function that checks mathematical accuracy using the 'math_verify' library.
        'batch' contains the original tokenized input data, including 'solution_ids'.
        """
        # Decode the ground truth solutions from the batch
        # Each prompt in the original batch has `num_return_sequences` completions generated for it.
        # So we need to duplicate the solutions accordingly.
        solutions = tokenizer.batch_decode(batch["solution_ids"]) * num_return_sequences

        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions, strict=False):
            # Parse the ground truth solution
            gold_parsed = parse(
                solution,
                extraction_mode="first_match", # Extract first LaTeX block
                extraction_config=[LatexExtractionConfig()],
            )
            # Parse the model's generated answer
            answer_parsed = parse(
                content,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )

            if len(gold_parsed) != 0: # If a LaTeX answer exists in the gold solution
                try:
                    # Verify if the parsed answer matches the parsed gold solution
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0) # Penalty if verification fails
            else:
                # If no LaTeX in gold, what's the desired behavior?
                # Here, it gives a reward of 1.0. This might need adjustment based on dataset.
                # It could mean the answer is non-mathematical or simple.
                rewards.append(1.0)
        return rewards

    # --- Dataset Preparation ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    # Load train and test splits. Using [:100%] for full dataset.
    # For quick testing, you can use smaller slices like "train[:1%]"
    train_dataset, test_dataset = load_dataset(
        DATASET_ID,
        split=["train[:100%]", "test[:100%]"],
    )
    logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    def map_conversation_format(example: dict[str, Any]) -> dict[str, Any]:
        """Converts dataset examples to the conversational format EasyDeL expects."""
        # The format is a list of dictionaries, each with "role" and "content".
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT}, # Add the system prompt
                {"role": "user", "content": example["problem"]}, # User's problem
            ],
            "solution": example["solution"], # Ground truth solution (used for accuracy reward and tokenization)
        }

    # Apply the mapping function to format the datasets.
    # `remove_columns` removes original columns that are no longer needed.
    train_dataset = train_dataset.map(map_conversation_format, remove_columns=["messages"])
    test_dataset = test_dataset.map(map_conversation_format, remove_columns=["messages"])

    def data_tokenize_fn(
        batch: dict[str, Any], # A batch of examples from the mapped dataset
        tokenizer: AutoTokenizer, # The tokenizer
        tools,
    ) -> dict[str, Any]:
        """Tokenizes prompts and solutions for GRPO training."""
        # Tokenize prompts (system + user message)
        # `tokenizer(batch["prompt"])` works because HuggingFace tokenizers can
        # automatically format conversational inputs if `add_special_tokens` is appropriate
        # or if the model's chat template is used. Here, `add_special_tokens=False` suggests
        # manual control or that the model doesn't need special chat tokens prepended/appended
        # beyond what's in the roles.
        prompt_ids = tokenizer(
            batch["prompt"], # List of conversation turns
            return_tensors="np", # Return NumPy arrays
            padding="max_length", # Pad to max_prompt_length
            padding_side="left", # Pad on the left for decoder-only models
            max_length=grpo_config.max_prompt_length,
            truncation=True,
            add_special_tokens=False, # Usually False for pre-formatted conversational data,
                                      # assuming EOS/BOS handled by model/training if needed elsewhere.
                                      # For GRPO, often only the prompt text itself is tokenized.
        )

        # Tokenize solutions (ground truth answers)
        solution_ids_tokenized = tokenizer(
            batch["solution"], # Text of the solutions
            return_tensors="np",
            padding="max_length",
            padding_side="left", # Important: For GRPO, reference answer tokens are also often left-padded.
                                 # This might be specific to EasyDeL's GRPO implementation.
                                 # Typically, labels are right-padded. Double-check EasyDeL docs if issues arise.
            max_length=grpo_config.max_completion_length, # Use completion length for solutions
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False, # Attention mask not needed for solution_ids here
        )
        # Store tokenized solutions. This key 'solution_ids' is used by `accuracy_reward`.
        prompt_ids["solution_ids"] = solution_ids_tokenized["input_ids"]
        return prompt_ids

    # --- Trainer Setup and Execution ---
    logger.info("Initializing GRPOTrainer.")
    trainer = ed.GRPOTrainer(
        model=model, # The EasyDeL model instance
        # List of reward functions. Their outputs will be combined (usually summed or weighted).
        reward_funcs=[format_reward, accuracy_reward],
        processing_class=tokenizer, # The tokenizer instance
        eval_dataset=test_dataset, # Evaluation dataset
        train_dataset=train_dataset, # Training dataset
        arguments=grpo_config, # The GRPOConfig object
        # `data_collator_or_fn` is used to process batches from the dataset.
        # Here, it's a lambda calling our tokenization function.
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        arguments=grpo_config,
        data_tokenize_fn=data_tokenize_fn,  # Pass tokenizer and config
    )

    logger.info("Starting training...")
    trainer.train() # This starts the GRPO training loop
    logger.info("Training finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # This calls the main function, which is decorated to run on Ray TPUs.
    # Ray will handle distributing this call to the configured TPU resources.
    # Correction: The `eformer.executor.ray.execute` decorator handles launching.
    # So, a direct call to `main()` is what's intended IF the script itself
    # is what `ray job submit` or similar would target.

    # Let's stick to the original `main()` as it seems to be the pattern for `eformer.executor`.
    main()
```

---

## Step 3: Running the Script

1. **Save the Code:** Ensure the Python script (`grpo_math_finetune.py`) is on your TPU VM.
2. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python grpo_math_finetune.py
    ```

    * Ray will initialize.
    * The `@execute(tpu_config)` decorator will provision the Ray actors on the TPUs according to `tpu_config`.
    * The `main()` function will then run on these Ray actors.
    * You'll see logs related to dataset loading, model sharding, and then the training progress (loss, rewards, etc.).
    * If WandB is enabled, metrics will be logged there.
    * Checkpoints will be saved according to `save_steps` and `save_directory` (if specified, defaults to `./EasyDeL-Checkpoints/`).

---

## Key Points and Customization

* **TPU Configuration (`tpu_config`):**
  * `accelerator`: Change `"v4-64"` to match your TPU setup (e.g., `"v3-8"`, `"v4-128"`, `"v6-128"`).
  * `execution_env`: If you have other environment variables or pip packages specific to your Ray workers, add them here.
* **Model and Dataset (`MODEL_ID`, `DATASET_ID`):**
  * You can easily swap these out for other Hugging Face models or datasets. Ensure your dataset mapping and tokenization functions are adjusted accordingly.
* **GRPO Configuration (`grpo_config`):**
  * `total_batch_size`: Adjust based on your TPU memory.
  * `learning_rate`, `num_train_epochs`: Standard hyperparameters to tune.
  * `beta`: This is a key GRPO parameter. It controls how much the policy can deviate from a reference (in this case, implicitly, its own previous state or a frozen copy). Higher beta means stronger regularization towards the reference.
  * `num_return_sequences`: GRPO generates multiple candidate responses and uses rewards to select/rank them. This is crucial for its operation.
* **Reward Functions:**
  * This is where GRPO's power lies. You can define custom Python functions to score the generated text based on any criteria: fluency, relevance, safety, specific formatting, factual accuracy, etc.
  * The script uses two: one for format (`<think>...</think>`) and one for math accuracy. The trainer will likely sum these rewards (or you can implement weighted summing).
* **Sharding (`sharding_axis_dims`):**
  * The `(1, -1, 1, 1, 1)` is a common setting for Fully Sharded Data Parallel (FSDP)-like behavior. For very large models or different TPU topologies, you might need to adjust this. Consult EasyDeL documentation for advanced sharding.
* **Memory (`/dev/shm`):**
  * Using `/dev/shm` for Hugging Face cache can significantly speed up dataset loading if datasets are repeatedly accessed, as it's a RAM disk.
* **Saving (`save_directory`):**
  * By default, models save to a local directory. For larger models or more permanent storage, consider setting `save_directory` in `GRPOConfig` to a Google Cloud Storage (GCS) bucket path (e.g., `"gs://your-bucket-name/grpo-checkpoints"`). Ensure your TPU VM has write access to the bucket.

---

This tutorial provides a comprehensive overview of the script and how to run it. GRPO is a powerful technique, and EasyDeL + Ray makes it accessible for large-scale training on TPUs. Happy fine-tuning!

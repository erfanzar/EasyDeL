# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pprint

import ray  # For distributed computing
from eformer.executor.ray import TpuAcceleratorConfig, execute  # EasyDeL's Ray utilities

# Initialize Ray. This ensures Ray is ready to manage distributed tasks.
ray.init()

# --- Configuration Constants ---
# MODEL_ID: Specifies the base model to be fine-tuned.
# We are using Qwen/Qwen3-14B, a powerful large language model.
MODEL_ID = "Qwen/Qwen3-14B"
# DATASET_ID: The dataset for DPO training.
# 'mlabonne/orpo-dpo-mix-40k' is a common dataset formatted for DPO,
# containing 'chosen' and 'rejected' response pairs.
DATASET_ID = "mlabonne/orpo-dpo-mix-40k"
# WANDB_ENTITY: Your Weights & Biases entity (username or organization).
# Set to None if you don't want to use WandB for logging.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# Environment variables that will be passed to each Ray worker process on the TPUs.
# These are crucial for HuggingFace and EasyDeL to function correctly in a distributed setup.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding and configuration.
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),  # Your HuggingFace token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # Uses shared memory for dataset caching, speeding up access.
    "HF_HOME": "/dev/shm/huggingface",  # Uses shared memory for HuggingFace model/token caches.
    "HF_DATASETS_OFFLINE": "0",  # Ensures datasets can be downloaded from HuggingFace Hub.
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),  # Your Weights & Biases API key.
}

# Additional pip packages to be installed on each Ray worker.
# For DPO, we don't require external reward verification libraries like 'math_verify'.
TPU_PIP_PACKAGES = []

# Pretty print the environment variables to confirm they are loaded.
pprint.pprint(TPU_EXECUTION_ENV_VARS)

# --- TPU Accelerator Configuration ---
# TpuAcceleratorConfig defines the type of TPU and the environment settings for Ray workers.
tpu_config = TpuAcceleratorConfig(
    "v4-64",  # Specifies a TPU v4 pod slice with 64 chips. Adjust to your available TPU size.
    execution_env={  # Configuration for the Ray worker environment.
        "env_vars": TPU_EXECUTION_ENV_VARS,  # Passes the environment variables defined above.
        "pip": TPU_PIP_PACKAGES,  # Installs any additional pip packages.
    },
)


# --- Main Training Function (Decorated for Ray and TPU execution) ---
# @execute(tpu_config): EasyDeL decorator that orchestrates running this function
# on TPUs managed by Ray, using the specified tpu_config.
# @ray.remote: Standard Ray decorator that turns this function into a remote task
# that can be executed on Ray workers.
@execute(tpu_config)
@ray.remote
def main():
    # Imports are placed inside the `main` function. This ensures that these libraries
    # are imported within the Ray remote worker's process, which is important for
    # dependency management in distributed contexts.
    import jax  # JAX: numerical computation library, backbone of EasyDeL.
    from datasets import load_dataset  # Hugging Face Datasets library.
    from jax import numpy as jnp  # JAX's NumPy-like API.
    from transformers import AutoTokenizer  # Hugging Face Transformers for tokenizer.

    import easydel as ed  # The EasyDeL library.

    # Initialize EasyDeL's logger for informative output during training.
    logger = ed.utils.get_logger("DPO-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    # --- DPO Specific Configuration Variables ---
    # `sequence_length` is the maximum length for the prompt part of the input.
    sequence_length = 2048
    # `max_length` is the maximum total sequence length for both prompt and completion.
    # For DPO, it's often set to twice the prompt length to allow for full conversations.
    max_completion_length = 2048
    max_length = sequence_length + max_completion_length
    # `total_batch_size` is the effective batch size across all TPU devices.
    total_batch_size = 16

    # --- Tokenizer Setup ---
    # Load the tokenizer (referred to as 'processor' for Qwen models).
    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    # Ensure a `pad_token_id` is set, typically to the `eos_token_id`,
    # which is crucial for consistent padding during training.
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {processor.pad_token_id}")

    # --- Model Loading ---
    logger.info(f"Loading model: {MODEL_ID}")
    # AutoEasyDeLModelForCausalLM intelligently loads and shards the model for JAX/TPU.
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,  # Use bfloat16 for computations for efficiency on TPUs.
        param_dtype=jnp.bfloat16,  # Store model parameters in bfloat16.
        precision=jax.lax.Precision.DEFAULT,  # Default precision for JAX operations.
        auto_shard_model=True,  # EasyDeL will automatically shard the model parameters across devices.
        # Sharding axis dimensions:
        #  (data_parallel, fully_sharded_data_parallel, expert_parallel, tensor_parallel, sequence_parallel)
        # `(1, -1, 1, 1, 1)` is a common setting for FSDP-like sharding across batch and model.
        sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),  # combine FSDP and TP
        config_kwargs=ed.EasyDeLBaseConfigDict(
            # Override specific model configuration parameters.
            freq_max_position_embeddings=max_length,  # For RoPE-based models, sets max position for frequency encoding.
            mask_max_position_embeddings=max_length,  # Max length for attention mask.
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,  # Disables KV cache quantization.
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            # EasyDeL selects the best attention mechanism (e.g., FlashAttention).
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,  # change this if u go OOM
            # Memory-saving technique for gradients.
        ),
        # `partition_axis` provides finer control over sharding, e.g., sharding KV heads via tensor parallelism.
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,  # Disables quantization for model weights.
    )
    logger.info("Model loaded successfully.")

    # --- Dataset Preparation ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    # The 'mlabonne/orpo-dpo-mix-40k' dataset already provides 'chosen' and 'rejected' conversation
    # lists, which is the required format for DPOTrainer. We load both train and test splits.
    train_dataset = load_dataset(DATASET_ID, split="train")
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # --- DPO Configuration (Hyperparameters for DPO training) ---
    arguments = ed.DPOConfig(
        num_train_epochs=1,  # Number of full passes over the training dataset.
        total_batch_size=total_batch_size,  # Total batch size used across all TPU devices.
        gradient_accumulation_steps=1,  # Number of gradient accumulation steps .
        do_eval=True,  # Enable evaluation on the test set during training.
        use_wandb=True,  # Automatically enable WandB logging if entity is provided.
        wandb_entity=WANDB_ENTITY,
        do_last_save=True,  # Save the final model checkpoint after training.
        max_prompt_length=sequence_length,  # Max length for the prompt part of the input.
        max_length=max_length,  # Max total sequence length (prompt + completion) for tokenization.
        max_completion_length=max_completion_length,  # Max length for the completion.
        max_training_steps=None,  # Maximum number of training steps (None means train until epochs are done).
        max_evaluation_steps=None,  # Maximum number of evaluation steps (None means evaluate full test set).
        max_sequence_length=max_length,  # Redundant with max_length, but good for clarity.
        loss_config=ed.LossConfig(z_loss=0.0),  # Z-loss regularization term in DPO loss (0.0 means off).
        track_memory=False,  # Set to True to enable memory tracking (can add minor overhead).
        save_steps=1_000,  # Save a model checkpoint every 1000 training steps.
        save_total_limit=0,  # Maximum number of checkpoints to keep (0 means keep all).
        save_optimizer_state=False,  # Whether to save optimizer state with checkpoints (turn True to resume training).
        per_epoch_training_steps=None,  # Number of training steps per epoch (None for automatic calculation).
        per_epoch_evaluation_steps=None,  # Number of evaluation steps per epoch (None for automatic calculation).
        learning_rate=1e-5,  # Initial learning rate for the optimizer.
        learning_rate_end=5e-7,  # End learning rate for linear scheduler (not active with COSINE scheduler).
        beta=0.1,  # The DPO beta parameter. Controls how much the policy
        # can deviate from the reference model's behavior.
        # Higher beta means stronger regularization (less divergence).
        optimizer=ed.EasyDeLOptimizers.ADAMW,  # Optimizer to use (AdamW is common).
        scheduler=ed.EasyDeLSchedulers.COSINE,  # Learning rate scheduler (Cosine annealing).
        clip_grad=1.0,  # Gradients will be clipped to this maximum L2 norm.
        weight_distribution_log_steps=100,  # Log weight distribution histograms every 100 steps (for debugging).
        warmup_steps=0,  # Number of warmup steps for the learning rate scheduler.
        report_steps=10,  # Log metrics to WandB (if enabled) every 10 steps.
        log_steps=5,  # Log metrics to console/logger every 5 steps.
        progress_bar_type="json",  # Type of progress bar display.
        # save_directory="gs://your-bucket/dpo-qwen-mix"
        # # Optional: specify a Google Cloud Storage bucket path for saving.
    )

    # --- Trainer Setup and Execution ---
    logger.info("Initializing DPOTrainer.")
    trainer = ed.DPOTrainer(
        arguments=arguments,  # Pass the configured DPO hyperparameters.
        model=model,  # The EasyDeL model instance to be trained.
        processing_class=processor,  # The tokenizer/processor instance for internal data handling.
        train_dataset=train_dataset,  # The dataset used for training.
        eval_dataset=None,  # The dataset used for evaluation.
    )

    logger.info("Starting training...")
    trainer.train()  # Initiate the DPO training process.
    logger.info("Training finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # When the script is executed, `main()` is called. The `@execute` and `@ray.remote`
    # decorators ensure that this function is launched as a distributed task on the TPUs
    # configured via Ray.
    main()

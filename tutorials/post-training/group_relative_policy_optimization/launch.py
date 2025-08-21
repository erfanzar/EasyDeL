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
import re
from typing import Any

import ray  # type: ignore
from eformer.executor.ray import TpuAcceleratorConfig, execute  # type: ignore

ray.init()
# --- Configuration Constants ---
MODEL_ID = "cohereLabs/aya-expanse-8b"
DATASET_ID = "AI-MO/NuminaMath-TIR"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# For TPU execution environment - consider fetching sensitive tokens from environment variables
# For example: HF_TOKEN = os.environ.get("HF_TOKEN")
# Make sure these are set in the environment where the script/Ray workers run.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",
    "HF_HOME": "/dev/shm/huggingface",
    "HF_DATASETS_OFFLINE": "0",
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}

TPU_PIP_PACKAGES = ["math_verify"]

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the think process in the mind and then provides the user with the answer. "
    "The think process and answer are enclosed within <think> </think> and answer needs no tag tags, respectively, i.e.,"
    " <think> think process here </think> answer here"
)


# --- TPU Accelerator Configuration ---
tpu_config = TpuAcceleratorConfig(
    "v4-64",
    execution_env={"env_vars": TPU_EXECUTION_ENV_VARS, "pip": TPU_PIP_PACKAGES},
)

pprint.pprint(TPU_EXECUTION_ENV_VARS)


@execute(tpu_config)
@ray.remote
def main():
    # Imports inside main to ensure they are available in the Ray remote environment
    import jax  # type: ignore
    from datasets import load_dataset  # type: ignore
    from jax import numpy as jnp  # type: ignore
    from math_verify import LatexExtractionConfig, parse, verify  # type: ignore
    from transformers import AutoTokenizer

    import easydel as ed

    logger = ed.utils.get_logger("GRPO-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    # --- Tokenizer Setup ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")

    # --- GRPO Configuration ---
    # Hyperparameters (consider making these more configurable if varied often)
    total_batch_size = 8
    num_return_sequences = 4
    top_k = 50
    top_p = 0.95
    temperature = 0.7
    max_prompt_length = 2048
    max_completion_length = 2048

    grpo_config = ed.GRPOConfig(
        total_batch_size=total_batch_size,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        learning_rate=1e-6,
        learning_rate_end=6e-7,
        log_steps=5,
        report_steps=10,
        progress_bar_type="json",
        num_train_epochs=3,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        do_last_save=True,
        track_memory=False,  # Set to True if memory tracking is needed
        save_steps=1000,
        save_total_limit=0,
        save_optimizer_state=False,
        use_wandb=True,
        wandb_entity=WANDB_ENTITY,
        clip_grad=1.0,
        weight_distribution_log_steps=100,
        warmup_steps=0,
        beta=0.04,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        # save_directory="gs://somewhere" maybe
    )

    max_sequence_length = grpo_config.max_completion_length + grpo_config.max_prompt_length

    # --- Model Loading ---
    logger.info(f"Loading model: {MODEL_ID}")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),  # Specific to model architecture and TPU topology
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=jnp.bfloat16,
            attn_softmax_dtype=jnp.bfloat16,
            kvdtype=jnp.bfloat16,
            # kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE, # Default
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,  # change this if u go OOM
        ),
        # quantization_method=ed.EasyDeLQuantizationMethods.NONE, # Default
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        # partition_axis=ed.PartitionAxis(), # Default
    )
    logger.info("Model loaded successfully.")

    # --- Reward Functions ---
    def format_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
        """
        Reward function that checks if the completion starts with a <think>...</think> block.
        """
        pattern = r"^<think>.*?</think>"  # Simplified: only checks start, not full structure
        rewards = []
        for completion_group in completions:
            if completion_group and "content" in completion_group[0]:
                content = completion_group[0]["content"]
                rewards.append(1.0 if re.match(pattern, content, re.DOTALL) else 0.0)
            else:
                rewards.append(0.0)
        return rewards

    def accuracy_reward(prompts, completions, batch, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        # solutions = kwargs["solution"]
        solutions = tokenizer.batch_decode(batch["solution_ids"]) * num_return_sequences
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions, strict=False):
            gold_parsed = parse(
                solution,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            answer_parsed = parse(
                content,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        return rewards

    # --- Dataset Preparation ---
    logger.info(f"Loading dataset: {DATASET_ID}")
    # Consider using streaming for very large datasets if memory is an issue
    train_dataset, test_dataset = load_dataset(DATASET_ID, split=["train[:100%]", "test[:100%]"])
    logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    def map_conversation_format(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        """Converts dataset examples to the required conversational format."""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example["solution"],
        }

    train_dataset = train_dataset.map(map_conversation_format, remove_columns=["messages"])
    test_dataset = test_dataset.map(map_conversation_format, remove_columns=["messages"])

    def data_tokenize_fn(
        batch: dict[str, Any],
        tokenizer: AutoTokenizer,
        tools,
    ) -> dict[str, Any]:
        """Tokenizes prompts and solutions for GRPO training."""
        # Tokenize prompts
        prompt_ids = tokenizer(
            batch["prompt"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",  # Important for causal LM
            max_length=grpo_config.max_prompt_length,
            truncation=True,
            add_special_tokens=False,  # Usually False for pre-formatted conversations
        )

        # Tokenize solutions (targets)
        solution_ids_tokenized = tokenizer(
            batch["solution"],
            return_tensors="np",
            padding="max_length",  # Or "longest" if dynamic padding is handled later
            padding_side="left",  # Typically 'right' for labels, but check GRPO requirements
            max_length=grpo_config.max_completion_length,  # Use completion length for solutions
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,  # Usually not needed for labels
        )
        # The key for solution IDs should match what `accuracy_reward` expects
        prompt_ids["solution_ids"] = solution_ids_tokenized["input_ids"]
        return prompt_ids

    # --- Trainer Setup and Execution ---
    logger.info("Initializing GRPOTrainer.")
    trainer = ed.GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        processing_class=tokenizer,  # Pass the instance
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        arguments=grpo_config,
        data_tokenize_fn=data_tokenize_fn,  # Pass tokenizer and config
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()

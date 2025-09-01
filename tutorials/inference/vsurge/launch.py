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


import json
import os
import pprint

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize the Ray cluster. This should be run on the head node.
ray.init()

# --- Evaluation Configuration ---

# The Hugging Face model ID to be evaluated.
MODEL_ID = "Qwen/Qwen3-14B"
# A list of benchmark tasks to run, compatible with the `lm-eval` library.
BENCHMARKS = ["gsm8k"]
# Optional: Your Weights & Biases entity for logging.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Model and Inference Parameters ---

# Number of few-shot examples to provide in the prompt for evaluation tasks.
num_fewshot = 3
# Maximum number of tokens to generate during decoding.
max_decodes_length = 4096
# Maximum number of tokens in the prefill (prompt) part of the input.
max_prefill_length = 4096
# Total maximum sequence length (prefill + decode).
max_sequence_length = max_decodes_length + max_prefill_length
# Maximum number of concurrent decoding requests to process in a batch.
max_concurrent_decodes = 256

# --- TPU Environment Configuration ---

# Environment variables to be set on the remote TPU workers.
# These are necessary for authentication, caching, and library configuration.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",  # Flag to enable EasyDeL's automatic features.
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # Use in-memory cache for speed.
    "HF_HOME": "/dev/shm/huggingface",
    "HF_DATASETS_OFFLINE": "0",  # Allow downloading datasets.
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}

# Additional Python packages to be installed on the TPU workers.
TPU_PIP_PACKAGES = ["lm_eval"]

# Print the environment variables for verification.
pprint.pprint(TPU_EXECUTION_ENV_VARS)

# Configuration for the TPU accelerator using `eformer`.
# This specifies the TPU type and the runtime environment (env vars and pip packages).
tpu_config = TpuAcceleratorConfig(
    "v4-32",  # TPU type (e.g., v4 pod with 16 chips).
    execution_env={
        "env_vars": TPU_EXECUTION_ENV_VARS,
        "pip": TPU_PIP_PACKAGES,
    },
)


@execute(tpu_config)
@ray.remote
def main():
    """Sets up and runs LM evaluation on a remote Ray worker (TPU).

    This function is decorated with `@ray.remote` and `@execute(tpu_config)`
    to be executed on a TPU Pod slice provisioned by the `eformer` library.

    The function performs the following steps:
    1.  Initializes the EasyDeL environment and tokenizer.
    2.  Loads the specified language model (`MODEL_ID`) with sharding
        configured for TPU execution.
    3.  Initializes `vSurge`, EasyDeL's serving engine, to handle batching
        and inference requests efficiently.
    4.  Creates a `vSurgeLMEvalAdapter`, a wrapper that makes `vSurge`
        compatible with the `lm-eval` library.
    5.  Runs the evaluation on the specified benchmarks using the adapter.
    6.  Saves the results to a JSON file and prints a summary.
    """
    # Imports are done inside the remote function as they are needed on the worker.
    import jax
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    import easydel as ed  # The EasyDeL library.

    logger = ed.utils.get_logger("vSurge-EasyDeL")
    logger.info(f"Starting main execution on Ray worker with JAX backend: {jax.default_backend()}")

    # Load the tokenizer for the specified model.
    processor = AutoTokenizer.from_pretrained(MODEL_ID)

    # Set pad token if it's not already set, which is required for batching.
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {processor.pad_token_id}")

    logger.info(f"Loading model: {MODEL_ID}")

    # Load the model using EasyDeL's auto-model class.
    # Configuration is tailored for sharded execution on TPUs.
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(-1, 1, 1, 8, 1),  # Sharding configuration for model parallelism.
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
    )
    logger.info("Model loaded successfully.")

    # Initialize vSurge, the high-throughput serving engine.
    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        max_concurrent_decodes=max_concurrent_decodes,
        max_concurrent_prefill=1,
        max_prefill_length=max_prefill_length,
        max_length=max_sequence_length,
        verbose=True,
        interleaved_mode=False,
        bytecode_decode=False,
    )
    # Compile the vSurge engine for optimal performance.
    surge.compile()
    # Start the background vSurge process.
    surge.start()

    # Create an adapter to use vSurge with the lm-eval harness.
    eval_runner = ed.vSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=max_sequence_length,
        max_new_tokens=max_decodes_length,
        top_p=0.95,
        temperature=0.1,
    )
    logger.info("Starting evaluation...")

    # Run the evaluation using the lm-eval harness.
    results = eval_runner.simple_evaluate(
        model=eval_runner,
        tasks=BENCHMARKS,
        num_fewshot=num_fewshot,
        batch_size=max_concurrent_decodes,
        device="cpu",  # The 'device' arg is for lm-eval, data loading happens on CPU.
    )

    # Save the results to a file.
    output = "out.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output}")

    # Print a summary of the results.
    logger.info("Summary of results:")
    for task, metrics in results["results"].items():
        logger.info(f"{task}: {metrics}")
    logger.info("evaluation finished.")


if __name__ == "__main__":
    # This triggers the execution of the remote function.
    # Ray and eformer will handle provisioning the TPUs and running the `main` function.
    main.remote()

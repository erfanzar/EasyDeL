
# Tutorial: High-Throughput Model Evaluation on TPUs with EasyDeL's vSurge

After fine-tuning a model, how do you measure its performance? This tutorial demonstrates how to perform **standardized, high-throughput model evaluation** on large language models using EasyDeL.

We will leverage **vSurge**, EasyDeL's powerful inference and serving engine, to run benchmarks from the popular `lm-evaluation-harness` library. This combination allows you to efficiently evaluate your models on a TPU cluster, getting reliable performance metrics quickly.

**Why is this important?**

* **Standardized Benchmarking:** Evaluate your model on established tasks like `gsm8k` (math reasoning) to compare it fairly against other models.
* **High Throughput:** `vSurge` is designed for performance. It uses techniques like continuous batching to process many requests simultaneously, making evaluation much faster than a simple one-by-one loop.
* **Scalability:** By running on a TPU pod via Ray, you can evaluate even the largest models that don't fit on a single accelerator.
* **Ease of Use:** EasyDeL provides a simple `vSurgeLMEvalAdapter` that seamlessly connects the powerful `vSurge` engine to the familiar `lm-eval` framework.

**Key Technologies Used:**

* **EasyDeL:** A JAX-based library for training and serving LLMs.
* **vSurge:** EasyDeL's high-performance inference engine.
* **lm-evaluation-harness:** The industry-standard library for LLM evaluation.
* **Ray:** An open-source framework for distributed computing on our TPU cluster.
* **JAX:** A high-performance numerical computing library, ideal for TPUs.

---

## Prerequisites

1. **Google Cloud TPU:** Access to a TPU environment (e.g., a TPU VM). This script is configured for a `v4-64` slice.
2. **Google Cloud Account & Project:** Properly configured for TPU usage.
3. **Basic Python & ML Knowledge:** Familiarity with Python and machine learning concepts.

---

## Step 1: Environment Setup

The setup process is identical to previous tutorials.

1. **SSH into your TPU VM.**
2. **Run the EasyDeL setup script:**

    ```bash
    bash <(curl -sL https://raw.githubusercontent.com/erfanzar/EasyDeL/refs/heads/main/tpu_setup.sh)
    ```

3. **Set Environment Variables:**
    Ensure your Hugging Face and (optional) Weights & Biases environment variables are set.

    ```bash
    export HF_TOKEN_FOR_EASYDEL="hf_YOUR_HUGGINGFACE_TOKEN_HERE"
    export WANDB_API_KEY_FOR_EASYDEL="YOUR_WANDB_API_KEY_HERE"
    export WANDB_ENTITY="your_wandb_username"
    # Add to ~/.bashrc for persistence
    ```

---

## Step 2: Understanding the Evaluation Script

Save the provided Python script as `evaluate_model.py`. This script is not for training; its sole purpose is to load a model and run it against benchmark tasks. Let's break down its components.

```python
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# ... (license header) ...

import json
import os
import pprint

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize the Ray cluster.
ray.init()

# --- Evaluation Configuration ---

# The Hugging Face model ID to be evaluated. This can be a base model
# or a fine-tuned model you've pushed to the Hub.
MODEL_ID = "Qwen/Qwen3-14B"
# A list of benchmark tasks to run. These are names recognized by the
# `lm-eval` library. You can add more, e.g., ["gsm8k", "arc_challenge", "hellaswag"].
BENCHMARKS = ["gsm8k"]
# Optional: Your Weights & Biases entity for logging.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Model and Inference Parameters ---

num_fewshot = 3  # Number of examples to include in the prompt (in-context learning).
max_decodes_length = 4096  # Max tokens to generate for each answer.
max_prefill_length = 4096  # Max tokens in the input prompt.
max_sequence_length = max_decodes_length + max_prefill_length # Total sequence length.
# The maximum number of requests `vSurge` will process in a single batch.
max_concurrent_decodes = 256

# --- TPU Environment Configuration (Standard for EasyDeL on Ray/TPU) ---
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1", "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset", "HF_HOME": "/dev/shm/huggingface",
    "HF_DATASETS_OFFLINE": "0", "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),
}
# We must ensure `lm_eval` is installed on the remote TPU workers.
TPU_PIP_PACKAGES = ["lm_eval"]
tpu_config = TpuAcceleratorConfig("v4-64", execution_env={"env_vars": TPU_EXECUTION_ENV_VARS, "pip": TPU_PIP_PACKAGES})


@execute(tpu_config)
@ray.remote
def main():
    """
    This function sets up and runs the LM evaluation on a remote Ray worker (TPU).
    """
    import easydel as ed
    import jax
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    # --- Model and Tokenizer Loading ---
    # The setup is similar to training, but the model is loaded purely for inference.
    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        auto_shard_model=True,
        # Note the sharding config might be different for inference optimization.
        sharding_axis_dims=(-1, 1, 1, 8, 1),
        # ... other config options ...
    )

    # --- Initialize vSurge: The High-Throughput Inference Engine ---
    # `vSurge` is the core component for fast inference. It manages a request queue
    # and batches them intelligently to maximize TPU utilization.
    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        max_concurrent_decodes=max_concurrent_decodes,
        max_prefill_length=max_prefill_length,
        max_length=max_sequence_length,
        # ... other engine parameters ...
    )
    # The `compile()` step optimizes the vSurge engine with JAX's JIT compiler.
    surge.compile()
    # `start()` runs the engine in a background thread to continuously process requests.
    surge.start()

    # --- Create the Evaluation Adapter ---
    # The `vSurgeLMEvalAdapter` is a brilliant bridge. It takes the `vSurge` engine
    # and gives it a simple interface that the `lm-eval` library can understand and use.
    eval_runner = ed.vSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=max_sequence_length,
        max_new_tokens=max_decodes_length,
        # You can specify generation parameters like top_p and temperature here.
        top_p=0.95,
        temperature=0.1,
    )

    # --- Run the Evaluation ---
    # `simple_evaluate` is a helper function that calls the `lm-eval` harness.
    # It will automatically download the benchmark tasks, format the prompts,
    # send them to our `eval_runner` (which uses vSurge), and score the results.
    results = eval_runner.simple_evaluate(
        model=eval_runner,  # We pass our adapter as the "model".
        tasks=BENCHMARKS,
        num_fewshot=num_fewshot,
        batch_size=max_concurrent_decodes, # Match the vSurge batch size.
        device="cpu",  # This arg is for lm-eval; data loading/processing happens on CPU.
                       # The actual model inference is happening on the TPU via vSurge.
    )

    # --- Save and Display Results ---
    output = "out.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output}")
    logger.info("Summary of results:")
    for task, metrics in results["results"].items():
        logger.info(f"{task}: {metrics}")


if __name__ == "__main__":
    # Note the call to `.remote()`. This tells Ray to execute the `main` function
    # on a remote worker, not locally on the head node.
    main.remote()
```

---

## Step 3: Running the Evaluation

1. **Save the Code:** Ensure the Python script is saved as `evaluate_model.py` on your TPU VM.
2. **Customize Your Evaluation:**
    * Change `MODEL_ID` to point to your own fine-tuned model (e.g., `"your-username/your-finetuned-model"`).
    * Modify the `BENCHMARKS` list to include the tasks you care about (e.g., `["mmlu", "truthfulqa", "winogrande"]`). A full list can be found in the `lm-evaluation-harness` documentation.
    * Adjust `num_fewshot` based on the standard practice for the benchmarks you are running.
3. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python evaluate_model.py
    ```

    The script will:
    1. Provision the TPU resources via Ray and `eformer`.
    2. Install `lm_eval` on the workers.
    3. Load your model and start the `vSurge` engine.
    4. Download the benchmark datasets.
    5. Run the evaluation, showing a progress bar from `lm-eval`.
    6. When finished, it will print a summary of the scores and save the detailed results to `out.json`.

---

## Key Points and Customization

* **From Fine-Tuning to Evaluation:** The typical workflow is: fine-tune a model using a script like `sft_finetune.py`, push the final checkpoint to the Hugging Face Hub, and then use this `evaluate_model.py` script by setting `MODEL_ID` to your newly pushed model ID.
* **Throughput is Key:** The `max_concurrent_decodes` parameter is crucial. It tells `vSurge` how many evaluation prompts to batch together. A higher number increases throughput but also uses more memory. `256` is a good starting point for a large TPU pod slice.
* **Understanding `lm-eval`:** The `lm-evaluation-harness` is a powerful library. It handles all the complexity of prompt formatting, few-shot example selection, and metric calculation (e.g., accuracy, F1-score). EasyDeL's adapter lets you tap into this power without leaving the JAX/TPU ecosystem.
* **Interpreting Results:** The output will give you standardized metrics for each task. You can use these to track your model's progress, compare different fine-tuning methods, and see how your model stacks up against others on public leaderboards.

This tutorial provides a robust and scalable framework for the critical final step of the model development lifecycle: rigorous evaluation. By using `vSurge`, you can get the answers you need about your model's performance quickly and efficiently.

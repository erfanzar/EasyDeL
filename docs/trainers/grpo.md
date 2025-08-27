# Group Relative Policy Optimization (GRPO) Trainer

Group Relative Policy Optimization (GRPO) is an advanced reinforcement learning technique for fine-tuning language models using multiple generations and reward functions. This tutorial demonstrates how to use EasyDeL's GRPOTrainer.

## Overview

The GRPO trainer generates multiple completions for each prompt, evaluates them with reward functions, and optimizes the policy to favor higher-reward generations. This is particularly useful for tasks like mathematical reasoning, where sampling multiple solutions and selecting the best ones can significantly improve performance.

## Configuration

The GRPOTrainer is configured using the `GRPOConfig` class with the following key parameters:

```python
from easydel.trainers import GRPOConfig

grpo_config = GRPOConfig(
    # Model and training basics
    model_name="GRPOTrainer",    # Name of the model
    learning_rate=1e-6,          # Learning rate for optimization

    # Sequence parameters
    max_prompt_length=512,       # Maximum length for prompts
    max_completion_length=256,   # Maximum length for completions

    # GRPO specific parameters
    beta=0.04,                   # Controls policy deviation penalty

    # Reference model parameters
    sync_ref_model=False,        # Whether to sync the reference model periodically
    ref_model_mixup_alpha=0.9,   # Alpha parameter for reference model mixing
    ref_model_sync_steps=64,     # Steps between reference model syncs

    # Dataset processing
    dataset_num_proc=None,       # Processes for dataset processing
    skip_apply_chat_template=False,  # Skip chat template extraction

    # Batch and training parameters
    total_batch_size=16,         # Total batch size
    num_train_epochs=3,          # Number of training epochs
)
```

## Basic Usage

Here's how to set up and use the GRPOTrainer for fine-tuning a language model:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jnp.bfloat16
)

# Load dataset (must have a "prompt" field)
dataset = load_dataset("gsm8k", "main", split="train[:10%]")

# Define a reward function - must return values between 0 and 1
def math_problem_reward_function(completion, reference_answer=None):
    """Example reward function that extracts and verifies an answer."""
    try:
        # Extract answer number from completion
        import re
        answer_match = re.search(r"The answer is (\d+)", completion)
        if answer_match:
            extracted_answer = int(answer_match.group(1))

            # Compare with reference if available
            if reference_answer is not None:
                return 1.0 if extracted_answer == reference_answer else 0.0

            # Some other scoring logic when reference not available
            return min(1.0, max(0.0, 0.5))  # Default score
        return 0.0
    except:
        return 0.0

# Initialize vInference for generation
inference = ed.vInference(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_return_sequences=4,  # Generate 4 different completions
    temperature=0.7,
    top_p=0.95
)

# Create GRPO config
config = ed.GRPOConfig(
    model_name="grpo_math_solver",
    save_directory="grpo_checkpoints",
    beta=0.04,
    max_prompt_length=512,
    max_completion_length=256,
    total_batch_size=16,
    learning_rate=1e-6,
    num_train_epochs=3,
    use_wandb=True,
)

# Initialize trainer
trainer = ed.GRPOTrainer(
    arguments=config,
    vinference=inference,
    model=model,
    reward_funcs=math_problem_reward_function,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Start training
trainer.train()
```

## Command Line Training

You can run GRPO training directly from the command line for math problem solving:

```bash
python -m easydel.scripts.finetune.gsm8k_grpo \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --attn_mechanism vanilla \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --beta 0.04 \
  --top_p 0.95 \
  --top_k 50 \
  --num_return_sequences 4 \
  --xml_reward 0.125 \
  --correctness_reward 2.0 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --num_train_epochs 3 \
  --use_wandb
```

Or for NuminaMath fine-tuning:

```bash
python -m easydel.scripts.finetune.numinamath_grpo \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --attn_mechanism vanilla \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --beta 0.04 \
  --top_p 0.95 \
  --num_return_sequences 4 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --num_train_epochs 3 \
  --use_wandb
```

## Dataset Format

For GRPO, the dataset must include a "prompt" field that contains the text prompt:

```json
{
  "prompt": "Solve this math problem step-by-step: If John has 5 apples and buys 3 more, how many does he have in total?",
  "answer": "8"  # Optional reference answer for reward calculation
}
```

## Advanced Usage

### Multiple Reward Functions

You can use multiple reward functions to shape the model's behavior:

```python
def correctness_reward(completion, reference=None):
    # Check if the answer matches the reference
    return 1.0 if reference in completion else 0.0

def reasoning_reward(completion):
    # Score the quality of reasoning in the completion
    if "step 1" in completion.lower() and "therefore" in completion.lower():
        return 0.8  # Good reasoning pattern
    return 0.3  # Lacking clear reasoning

# Use multiple reward functions
trainer = ed.GRPOTrainer(
    # Other parameters...
    reward_funcs=[correctness_reward, reasoning_reward],
    # If you need specific tokenizers for different reward functions:
    reward_processing_classes=[tokenizer, tokenizer],
)
```

### Custom Generation Parameters

You can customize the generation parameters in vInference:

```python
inference = ed.vInference(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_return_sequences=8,      # More generations for better exploration
    temperature=0.9,             # Higher temperature for more diversity
    top_p=0.92,
    top_k=100,
    do_sample=True,
    repetition_penalty=1.1,
)
```

## Tips for Effective GRPO Training

1. **Reward function design**: Create reward functions that accurately measure desired qualities in generations
2. **Number of generations**: More generations (4-8) typically provide better training signal, but use more memory
3. **Beta tuning**: Start with lower values (0.01-0.05) and adjust based on training stability
4. **Diversity vs. quality**: Balance generation parameters to get diverse but high-quality samples
5. **Dataset selection**: GRPO works best on datasets with clear evaluation criteria (math, coding, reasoning)

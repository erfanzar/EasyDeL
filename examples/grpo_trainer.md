# GRPO Training with EasyDeL: A Complete Tutorial

## Overview

This tutorial demonstrates how to use EasyDeL's GRPO (Generative Reinforcement Learning from Policy Optimization) implementation to fine-tune language models with custom reward functions. We'll train a model to solve math problems while following a specific XML response format.

## What is GRPO?

GRPO is a reinforcement learning technique that optimizes language models by:

1. Generating multiple completions for each prompt
2. Evaluating completions using reward functions
3. Updating the model to increase the likelihood of high-reward responses

## Hardware Flexibility

This tutorial is designed to work efficiently on small consumer GPUs (8-16GB VRAM), but EasyDeL's architecture allows seamless scaling to larger hardware configurations including TPUs, multi-GPU setups, and cloud accelerators. The same code can be adapted with minimal changes to leverage more powerful hardware.

## Core Components Explained

### 1. Configuration Setup

```python
arguments = ed.GRPOConfig(
    save_directory="grpotrainer",
    tx_mu_dtype=jnp.bfloat16,  # Mixed precision for efficiency
    num_train_epochs=1,
    total_batch_size=1,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_return_sequences=4,  # Generate 4 completions per prompt
    top_k=10,                # Sampling parameters
    top_p=0.95,
    temperature=0.7,
    learning_rate=9e-7,
    optimizer=ed.EasyDeLOptimizers.ADAMW,
    scheduler=ed.EasyDeLSchedulers.COSINE,
    clip_grad=1.0,
)
```

Key parameters:

- `num_return_sequences`: Number of completions to generate for reward comparison
- `temperature`, `top_k`, `top_p`: Control generation diversity
- `clip_grad`: Prevents gradient explosion

### 2. Model Loading and LoRA Application

```python
# Load base model
max_sequence_length = prompt_length + completion_length

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    repo_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    auto_shard_model=True,
    sharding_axis_dims=(1, -1, 1, 1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        freq_max_position_embeddings=max_sequence_length,
        mask_max_position_embeddings=max_sequence_length,
        attn_dtype=jnp.bfloat16,
        attn_softmax_dtype=jnp.bfloat16,
        kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        attn_mechanism=ed.AttentionMechanisms.SDPA,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE, # change this if u go OOM
    ),
    quantization_method=ed.EasyDeLQuantizationMethods.NONE,
    precision=jax.lax.Precision.DEFAULT,
    partition_axis=ed.PartitionAxis(),
)

# Apply LoRA to specific layers
model = model.apply_lora_to_layers(
    rank=32,  # LoRA rank
    target_modules=".*(gate_proj|up_proj|lm_head|q_proj).*"  # Regex pattern
)
```

LoRA benefits:

- Reduces trainable parameters from millions to thousands
- Maintains model quality while improving training efficiency
- Easy to merge back into base model

### 3. Reward Functions

The power of GRPO lies in custom reward functions. Here's how they work:

```python
def correctness_reward_func(prompts, completions, batch, **kwargs) -> list[float]:
    """Rewards correct mathematical answers"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    answer = processor.batch_decode(batch["answer_ids"]) * arguments.num_return_sequences
    return [correctness_reward if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
```

Each reward function receives:

- `prompts`: Original prompts
- `completions`: Generated completions
- `batch`: Original batch data
- Returns: List of float rewards (one per completion)

### 4. Multiple Reward Functions

You can combine multiple objectives:

```python
trainer = ed.GRPOTrainer(
    model=model,
    reward_funcs=[
        xmlcount_reward_func,      # Partial XML format (0.25 reward)
        soft_format_reward_func,   # Basic XML structure (0.5 reward)
        strict_format_reward_func, # Perfect XML format (0.5 reward)
        int_reward_func,          # Integer answer (0.5 reward)
        correctness_reward_func,  # Correct answer (2.0 reward)
    ],
    # ... other parameters
)
```

Total reward = sum of all reward functions

## Complete Example: Math Problem Solver

Here's a simplified example focusing on the key concepts:

```python
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

# 1. Define your reward structure
CORRECT_ANSWER_REWARD = 2.0
FORMAT_REWARD = 0.5

# 2. Create reward functions
def format_reward(completions, **kwargs):
    """Reward proper XML formatting"""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        has_think_tags = "<think>" in text and "</think>" in text
        rewards.append(FORMAT_REWARD if has_think_tags else 0.0)
    return rewards

def correctness_reward(prompts, completions, batch, **kwargs):
    """Reward correct answers"""
    rewards = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        # Extract answer from response
        if "####" in response:
            predicted = response.split("####")[1].strip()
            correct = batch["answer"][i % len(batch["answer"])]
            rewards.append(CORRECT_ANSWER_REWARD if predicted == correct else 0.0)
        else:
            rewards.append(0.0)
    return rewards

# 3. Setup configuration
config = ed.GRPOConfig(
    save_directory="math-solver",
    num_train_epochs=3,
    total_batch_size=4,
    num_return_sequences=4,
    learning_rate=5e-7,
    max_prompt_length=512,
    max_completion_length=512,
)

# 4. Load model with LoRA
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=jnp.bfloat16,
    auto_shard_model=True,
)
model = model.apply_lora_to_layers(rank=16, target_modules=".*q_proj.*")

# 5. Prepare data
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
dataset = load_dataset("openai/gsm8k", "main")["train"]

def tokenize_function(batch, tokenizer, tools):
    # Format prompts
    prompts = []
    for question in batch["question"]:
        prompts.append([
            {"role": "system", "content": "Solve step by step. Use <think> tags."},
            {"role": "user", "content": question}
        ])

    # Tokenize
    return tokenizer(
        prompts,
        return_tensors="np",
        padding="max_length",
        max_length=config.max_prompt_length,
        truncation=True,
    )

# 6. Train
trainer = ed.GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, correctness_reward],
    processing_class=tokenizer,
    train_dataset=dataset,
    arguments=config,
    data_tokenize_fn=tokenize_function,
)

trainer.train()
```

## Advanced Tips

### 1. Balancing Rewards

```python
# Adjust reward magnitudes to balance objectives
FORMAT_REWARD = 0.5      # Encourage format
CORRECTNESS_REWARD = 2.0 # Strongly reward correctness
PARTIAL_CREDIT = 0.1     # Small rewards for progress
```

### 2. Progressive Training

```python
# Start with format rewards, then add correctness
initial_rewards = [format_reward_func]
final_rewards = [format_reward_func, correctness_reward_func]
```

### 3. Debugging Rewards

```python
def debug_reward_func(completions, **kwargs):
    """Print completions for debugging"""
    for i, comp in enumerate(completions[:2]):  # First 2 only
        print(f"Completion {i}: {comp[0]['content'][:100]}...")
    return [0.0] * len(completions)  # No reward, just debug
```

### 4. Custom Sharding for Large Models

```python
# For multi-GPU setups
sharding_axis_dims = (1, -1, 1, 1, 1)  # (batch, seq, hidden, heads, kv)
partition_axis = ed.PartitionAxis(
    batch_axis=0,
    sequence_axis=1,
    hidden_axis=2,
)
```

## Monitoring Training

Use Weights & Biases integration:

```python
arguments = ed.GRPOConfig(
    use_wandb=True,
    wandb_entity="your-entity",
    wandb_project="grpo-math",
    log_steps=10,
    weight_distribution_log_steps=100,  # Log weight distributions
)
```

## Common Issues and Solutions

1. **Low rewards across all completions**
   - Reduce temperature for more focused generation
   - Simplify reward functions initially

2. **Model not improving**
   - Check if rewards are too sparse
   - Increase `num_return_sequences` for more exploration
   - Verify reward function logic

3. **Memory issues**
   - Reduce `max_completion_length`
   - Use gradient checkpointing
   - Decrease LoRA rank

## Next Steps

1. Experiment with different reward combinations
2. Try curriculum learning (gradually increase task difficulty)
3. Implement custom generation strategies
4. Fine-tune sampling parameters for your use case

This GRPO implementation provides a flexible framework for training models with complex objectives that can't be captured by traditional supervised fine-tuning alone.

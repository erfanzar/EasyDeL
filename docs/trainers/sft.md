# Supervised Fine-Tuning (SFT) Trainer

Supervised Fine-Tuning (SFT) is the fundamental method for adapting language models to specific tasks and datasets. This tutorial demonstrates how to use EasyDeL's SFTTrainer.

## Overview

The SFTTrainer provides a straightforward way to fine-tune language models on supervised datasets. It supports various features like sequence packing, dataset preprocessing, and advanced optimization schedules to make the fine-tuning process more efficient and effective.

## Configuration

The SFTTrainer is configured using the `SFTConfig` class:

```python
from easydel.trainers import SFTConfig

sft_config = SFTConfig(
    # Model and training basics
    model_name="SFTTrainer",     # Name of the model
    learning_rate=2e-5,          # Learning rate for optimization

    # Dataset processing parameters
    dataset_text_field=None,     # Name of the text field in the dataset
    add_special_tokens=False,    # Whether to add special tokens
    packing=False,               # Controls whether sequences are packed

    # Packing parameters
    num_of_sequences=1024,       # Number of sequences for packing
    chars_per_token=3.6,         # Characters per token estimate

    # Dataset processing
    dataset_num_proc=None,       # Number of processes for dataset processing
    dataset_batch_size=1000,     # Batch size for dataset tokenization
    dataset_kwargs=None,         # Additional dataset creation arguments
    eval_packing=None,           # Whether to pack eval dataset

    # Batch and training parameters
    total_batch_size=16,         # Total batch size
    num_train_epochs=3,          # Number of training epochs
)
```

## Basic Usage

Here's a complete example of how to initialize and use the SFTTrainer:

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

# Load dataset
dataset = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="helpful-base",
    split="train[:10%]"  # Using a small subset for demonstration
)

# Load model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jnp.bfloat16
)

# Create SFT config
config = ed.SFTConfig(
    model_name="sft_example",
    save_directory="sft_checkpoints",
    dataset_text_field="chosen",  # Field containing the text to train on
    packing=True,                 # Enable sequence packing for efficiency
    learning_rate=2e-5,
    learning_rate_end=5e-6,
    total_batch_size=16,
    num_train_epochs=3,
    use_wandb=True,
    num_of_sequences=512,         # For packing
)

# Initialize trainer
trainer = ed.SFTTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Start training
trainer.train()
```

## Command Line Training

You can run SFT training directly from the command line:

```bash
python -m easydel.scripts.finetune.sft \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name trl-lib/Capybara \
  --dataset_split "train" \
  --dataset_text_field messages \
  --attn_mechanism vanilla \
  --max_sequence_length 2048 \
  --packing True \
  --total_batch_size 16 \
  --learning_rate 2e-5 \
  --learning_rate_end 5e-6 \
  --num_train_epochs 3 \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

## Dataset Formats

The SFTTrainer can work with different dataset formats:

### 1. Simple Text Dataset

When using `dataset_text_field`:

```json
{
  "text": "This is a training example for the language model."
}
```

### 2. Instruction Format

Instruction datasets with prompts and responses:

```json
{
  "instruction": "Write a short poem about machine learning.",
  "response": "Silicon minds learn and grow,\nPatterns found in data's flow.\nMathematical art so precise,\nLearning once, then twice, then thrice."
}
```

### 3. Conversation Format

For multi-turn conversation datasets, they are typically formatted with a list of messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
  ]
}
```

## Advanced Usage

### Packing for Efficient Training

Sequence packing combines multiple shorter examples into a single training batch for better throughput:

```python
config = ed.SFTConfig(
    packing=True,               # Enable packing
    num_of_sequences=1024,      # Number of sequences to pack
    chars_per_token=3.6,        # Estimate of characters per token
)
```

### Custom Evaluation Dataset

You can provide a separate evaluation dataset:

```python
# Load evaluation dataset
eval_dataset = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="helpful-base",
    split="validation[:10%]"
)

# Initialize trainer with eval dataset
trainer = ed.SFTTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
```

### Advanced Learning Rate Scheduling

Using a custom learning rate schedule:

```python
config = ed.SFTConfig(
    learning_rate=2e-5,            # Initial learning rate
    learning_rate_end=5e-6,        # Final learning rate
    scheduler=ed.EasyDeLSchedulers.COSINE,  # Cosine schedule
    warmup_steps=500,              # Steps for warmup phase
    optimizer=ed.EasyDeLOptimizers.ADAMW,   # AdamW optimizer
    weight_decay=0.01,             # Weight decay for regularization
)
```

## Tips for Effective SFT Training

1. **Data quality**: Clean, diverse, and high-quality data is crucial for good results
2. **Sequence packing**: Enable packing for datasets with many short examples to improve throughput
3. **Learning rate**: Start with 1e-5 to 5e-5 for most models, and use a decay schedule
4. **Batch size**: Use the largest batch size that fits in memory (16-64 is common)
5. **Epochs**: 2-5 epochs is typically sufficient; monitor for overfitting
6. **Tokenization**: Ensure proper tokenization, especially for special tokens and formats
7. **Gradual unfreezing**: For very large models, consider unfreezing layers gradually

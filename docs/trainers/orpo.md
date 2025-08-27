# Odds Ratio Preference Optimization (ORPO) Trainer

Odds Ratio Preference Optimization (ORPO) is an advanced preference alignment technique that offers improved performance over standard DPO by focusing on the odds ratio between chosen and rejected samples. This tutorial demonstrates how to use EasyDeL's ORPOTrainer.

## Overview

The ORPO trainer provides a more stable and efficient approach to preference-based fine-tuning, with particular advantages for tasks requiring nuanced preference learning. ORPO is especially effective when the quality gap between chosen and rejected samples is small.

## Configuration

The ORPOTrainer is configured using the `ORPOConfig` class:

```python
from easydel.trainers import ORPOConfig

orpo_config = ORPOConfig(
    # Model and training basics
    model_name="ORPOTrainer",    # Name of the model
    learning_rate=1e-6,          # Learning rate for optimization

    # Sequence parameters
    max_length=1024,             # Maximum total sequence length
    max_prompt_length=512,       # Maximum length for prompts
    max_completion_length=512,   # Maximum length for completions

    # ORPO specific parameters
    beta=0.1,                    # Controls deviation from reference policy

    # Training parameters
    disable_dropout=True,        # Disable dropout during training
    generate_during_eval=False,  # Whether to generate during evaluation

    # Tokenizer settings
    label_pad_token_id=-100,     # Padding token for labels
    padding_value=None,          # Optional custom padding value

    # Dataset processing
    dataset_num_proc=None,       # Number of processes for dataset processing
)
```

## Basic Usage

Here's a complete example of how to initialize and use the ORPOTrainer:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer and prepare it
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load dataset with preference pairs
dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="train[:5%]"  # Using a small subset for demonstration
)

# Load model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jnp.bfloat16
)

# Create ORPO config
config = ed.ORPOConfig(
    model_name="orpo_example",
    save_directory="orpo_checkpoints",
    beta=0.1,
    max_length=2048,
    max_prompt_length=1024,
    max_completion_length=1024,
    total_batch_size=16,
    learning_rate=1e-6,
    learning_rate_end=5e-7,
    num_train_epochs=3,
    use_wandb=True,
)

# Initialize trainer
trainer = ed.ORPOTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Start training
trainer.train()
```

## Command Line Training

You can run ORPO training directly from the command line:

```bash
python -m easydel.scripts.finetune.orpo \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --dataset_split "train" \
  --attn_mechanism vanilla \
  --beta 0.12 \
  --max_length 2048 \
  --max_prompt_length 1024 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --learning_rate_end 6e-7 \
  --num_train_epochs 3 \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

## Dataset Format

The ORPOTrainer expects a dataset with preference pairs in the following format:

```json
{
  "prompt": "What is the capital of Japan?",
  "chosen": "The capital of Japan is Tokyo, which is the most populous metropolitan area in the world.",
  "rejected": "The capital of Japan is Kyoto, an ancient city with many temples."
}
```

## Advanced Usage

### Custom Data Collator

You can use a custom data collator to handle specialized tokenization or data preparation:

```python
from easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_dataset import DPODataCollatorWithPadding

# Create a custom data collator
custom_collator = DPODataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=1024,
    label_pad_token_id=-100
)

# Use the custom collator with the trainer
trainer = ed.ORPOTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=custom_collator
)
```

### Evaluation Dataset

You can provide a separate evaluation dataset to monitor performance during training:

```python
# Load evaluation dataset
eval_dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="test[:10%]"
)

# Initialize trainer with eval dataset
trainer = ed.ORPOTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
```

### Mixed Precision Training

For larger models, you may want to use mixed precision training:

```python
# Create model with mixed precision
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    attn_dtype=jnp.bfloat16,
    compute_dtype=jnp.float32
)
```

## Comparing ORPO to DPO

ORPO offers several advantages over standard DPO:

1. **Improved stability**: ORPO tends to be more stable during training, especially with small quality differences
2. **Better preference alignment**: ORPO's focus on odds ratios can lead to better alignment with human preferences
3. **Sample efficiency**: ORPO often requires fewer training examples to achieve comparable results

## Tips for Effective ORPO Training

1. **Beta selection**: Start with values around 0.1-0.15 and adjust based on results
2. **Dataset quality**: The quality of preference pairs significantly impacts the final model
3. **Batch size**: Larger batch sizes (16-64) typically work better for preference learning
4. **Learning rate schedule**: A decaying learning rate (start: 1e-6, end: 5e-7) often works well
5. **Dataset curation**: Consider filtering or augmenting your preference dataset for better results

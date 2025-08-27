# Reward Model Training

Reward Model Training is a critical component in the RLHF (Reinforcement Learning from Human Feedback) pipeline. This tutorial demonstrates how to train a reward model using EasyDeL's RewardTrainer.

## Overview

The RewardTrainer helps you train a language model to predict preference scores for text samples. The resulting reward model can then be used in RLHF approaches like PPO, ORPO, or DPO to align language models with human preferences.

## Configuration

The RewardTrainer is configured using the `RewardConfig` class:

```python
from easydel.trainers import RewardConfig

reward_config = RewardConfig(
    # Model and training basics
    model_name="RewardTrainer",     # Name of the model
    learning_rate=1e-6,             # Learning rate for optimization

    # Sequence parameters
    max_sequence_length=1024,       # Maximum sequence length

    # Reward model parameters
    disable_dropout=True,           # Disable dropout during training
    center_rewards_coefficient=0.1, # Coefficient for centering reward outputs

    # Dataset processing
    dataset_num_proc=None,          # Processes for dataset processing
    remove_unused_columns=False,    # Whether to remove unused columns

    # Batch and training parameters
    total_batch_size=16,            # Total batch size
    num_train_epochs=3,             # Number of training epochs
)
```

## Basic Usage

Here's a complete example of how to initialize and use the RewardTrainer:

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

# Load dataset with preference pairs
dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="train[:5%]"  # Using a small subset for demonstration
)

# Load model
model = ed.AutoEasyDeLModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    num_labels=1,  # Reward model outputs a single score
    dtype=jnp.bfloat16
)

# Create Reward config
config = ed.RewardConfig(
    model_name="reward_model_example",
    save_directory="reward_checkpoints",
    max_sequence_length=2048,
    center_rewards_coefficient=0.1,
    total_batch_size=16,
    learning_rate=1e-6,
    learning_rate_end=5e-7,
    num_train_epochs=3,
    use_wandb=True,
)

# Initialize trainer
trainer = ed.RewardTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

## Command Line Training

You can run reward model training directly from the command line:

```bash
python -m easydel.scripts.finetune.reward \
  --repo_id meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --dataset_split "train" \
  --attn_mechanism vanilla \
  --max_sequence_length 2048 \
  --center_rewards_coefficient 0.1 \
  --total_batch_size 16 \
  --learning_rate 1e-6 \
  --learning_rate_end 6e-7 \
  --num_train_epochs 3 \
  --do_last_save \
  --save_steps 1000 \
  --use_wandb
```

## Dataset Format

For reward model training, the dataset should contain preference pairs:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris, which is located on the Seine River.",
  "rejected": "I think the capital of France is Lyon or maybe Marseille, not sure."
}
```

The model will be trained to assign higher scores to "chosen" completions compared to "rejected" ones.

## Advanced Usage

### Custom Dataset Preprocessing

You can customize the preprocessing for preference datasets:

```python
def preprocess_function(examples):
    # Custom preprocessing logic
    chosen_texts = [prompt + chosen for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
    rejected_texts = [prompt + rejected for prompt, rejected in zip(examples["prompt"], examples["rejected"])]

    # Tokenize both chosen and rejected
    chosen_inputs = tokenizer(chosen_texts, truncation=True, max_length=1024)
    rejected_inputs = tokenizer(rejected_texts, truncation=True, max_length=1024)

    return {
        "input_ids_chosen": chosen_inputs["input_ids"],
        "attention_mask_chosen": chosen_inputs["attention_mask"],
        "input_ids_rejected": rejected_inputs["input_ids"],
        "attention_mask_rejected": rejected_inputs["attention_mask"],
    }

# Process the dataset
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names
)

# Use the processed dataset with the trainer
trainer = ed.RewardTrainer(
    arguments=config,
    model=model,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
)
```

### Evaluation During Training

You can provide a separate evaluation dataset:

```python
# Load evaluation dataset
eval_dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="test[:10%]"
)

# Initialize trainer with eval dataset
trainer = ed.RewardTrainer(
    arguments=config,
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
```

### Reward Centering

The reward centering mechanism helps prevent the model from giving overly positive or negative rewards to all inputs:

```python
config = ed.RewardConfig(
    center_rewards_coefficient=0.1,  # Default value
    # Other parameters...
)
```

- Higher values encourage more strongly centered rewards (closer to zero mean)
- Lower values or zero disable centering

## Using the Trained Reward Model

After training, you can use the reward model as follows:

```python
# Load the trained reward model
reward_model = ed.AutoEasyDeLModelForSequenceClassification.from_pretrained(
    "path/to/reward_checkpoints",
    dtype=jnp.bfloat16
)

# Function to compute reward for a text
def compute_reward(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    outputs = reward_model(**inputs)
    return outputs.logits.item()  # Get the scalar reward value

# Test the reward model
sample_text = "This is a test response from the language model."
reward = compute_reward(sample_text)
print(f"Reward score: {reward}")
```

## Tips for Effective Reward Model Training

1. **Data quality**: Ensure clear preference distinctions in your dataset
2. **Model size**: Smaller models (1B-8B parameters) often work well as reward models
3. **Learning rate**: Use lower learning rates (1e-6 to 5e-6) than for standard classification
4. **Evaluation**: Validate your reward model's alignment with human preferences
5. **Integration**: Properly integrate with your RLHF pipeline (PPO, DPO, etc.)
6. **Reward centering**: Adjust center_rewards_coefficient based on your reward range needs

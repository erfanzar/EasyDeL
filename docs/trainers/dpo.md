# Direct Preference Optimization (DPO) Trainer

Direct Preference Optimization (DPO) is a technique for fine-tuning language models using human preferences without explicit reward modeling. This tutorial shows how to use EasyDeL's DPOTrainer.

## Overview

DPOTrainer helps you fine-tune language models to align with human preferences by optimizing the policy to prefer chosen responses over rejected ones. It avoids the separate reward model typically needed in RLHF.

## Configuration

The DPOTrainer is configured using the `DPOConfig` class which includes the following key parameters:

```python
from easydel.trainers import DPOConfig

dpo_config = DPOConfig(
    # Model and training basics
    model_name="DPOTrainer",  # Name of the model
    learning_rate=1e-6,       # Learning rate for optimization

    # Beta parameter controls how strongly to optimize preferences
    beta=0.1,                 # Temperature parameter for deviation from reference model

    # Loss function options
    loss_type="sigmoid",      # Loss type (sigmoid, hinge, ipo, etc.)
    label_smoothing=0.0,      # Smoothing factor for labels

    # Sequence length parameters
    max_length=512,           # Maximum total sequence length
    max_prompt_length=256,    # Maximum length for prompts
    max_completion_length=256,# Maximum length for completions

    # Reference model control
    reference_free=False,     # Whether to use reference-free variant
    sync_ref_model=False,     # Periodically sync reference model
    ref_model_sync_steps=64,  # Steps between reference model syncs

    # Training optimization
    disable_dropout=True,     # Disable dropout during training
    total_batch_size=16,      # Total batch size
    gradient_accumulation_steps=1  # Steps for gradient accumulation
)
```

## Basic Usage

Here's a simple example showing how to initialize and use the DPOTrainer:

```python
import easydel as ed
from transformers import AutoTokenizer
from datasets import load_dataset
import jax

# Load tokenizer and prepare dataset
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load dataset with preference pairs
dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="train[:10%]"  # Using a small subset for demonstration
)

# Load model and reference model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jax.numpy.bfloat16,
  # ... model loading configs
)
ref_model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jax.numpy.bfloat16,
  # ... model loading configs
)

# Create DPO config
config = ed.DPOConfig(
    model_name="dpo_example",
    save_directory="dpo_checkpoints",
    beta=0.1,
    loss_type="sigmoid",
    max_length=2048,
    max_prompt_length=1024,
    total_batch_size=16,
    learning_rate=1e-6,
    num_train_epochs=3,
    use_wandb=True,
)

# Initialize the trainer
trainer = ed.DPOTrainer(
    model=model,
    reference_model=ref_model,
    arguments=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Start training
trainer.train()
```

## Command Line Training

The legacy `easydel.scripts.finetune.*` entrypoints have been removed in favor of the unified YAML runner.

```bash
python -m easydel.scripts.elarge --config dpo.yaml
```

Example `dpo.yaml`:

```yaml
config:
  model:
    name_or_path: meta-llama/Llama-3.1-8B-Instruct
  reference_model:
    name_or_path: meta-llama/Llama-3.1-8B-Instruct
  mixture:
    informs:
      - type: hf
        data_files: trl-lib/ultrafeedback_binarized
        split: "train[:90%]"
  trainer:
    trainer_type: dpo
    beta: 0.08
    loss_type: sigmoid
    max_length: 2048
    max_prompt_length: 1024
    total_batch_size: 16
    learning_rate: 1e-6
    num_train_epochs: 3
actions:
  - train
```

## Dataset Format

DPOTrainer expects a dataset with the following format:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "The capital of France is London."
}
```

## Advanced Usage

### Custom Loss Functions

DPOTrainer supports various loss functions:

- `sigmoid`: Standard DPO loss (default)
- `hinge`: Hinge loss for more aggressive preference learning
- `ipo`: Implicit Preference Optimization loss
- Several others like `exo_pair`, `nca_pair`, `robust`, `aot`, etc.

Example with IPO loss:

```python
config = ed.DPOConfig(
    loss_type="ipo",
    beta=0.2,  # May need different beta for different loss types
    # Other parameters...
)
```

### Reference Model Syncing

To prevent policy drift during training, you can sync the reference model periodically:

```python
config = ed.DPOConfig(
    sync_ref_model=True,
    ref_model_sync_steps=128,  # Sync every 128 steps
    ref_model_mixup_alpha=0.9,  # Mixing parameter
    # Other parameters...
)
```

### Reference-Free Training

You can train without a separate reference model:

```python
config = ed.DPOConfig(
    reference_free=True,
    # Other parameters...
)
```

## Tips for Effective DPO Training

1. **Beta selection**: Start with values between 0.05-0.2 and tune based on results
2. **Learning rate**: Use smaller learning rates (1e-6 to 5e-6) than standard SFT
3. **Batch size**: Larger batch sizes often work better for preference learning
4. **Validation**: Monitor the preference accuracy on validation sets
5. **Dataset quality**: The quality of preference pairs greatly impacts results

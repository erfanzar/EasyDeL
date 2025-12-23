# Trainer Integration

EasyData integrates seamlessly with all EasyDeL trainers. This guide covers best practices for using data with SFT, DPO, GRPO, KTO, and other trainers.

## Quick Reference

| Trainer         | Dataset Format                 | Preprocessing                    |
| --------------- | ------------------------------ | -------------------------------- |
| `SFTTrainer`    | Text or chat messages          | Chat template applied            |
| `DPOTrainer`    | Chosen/rejected pairs          | Prompt extraction + tokenization |
| `ORPOTrainer`   | Chosen/rejected pairs          | Same as DPO                      |
| `KTOTrainer`    | Unpaired or paired preference  | Unpairs + tokenizes              |
| `GRPOTrainer`   | Prompts (with reward function) | Prompt extraction                |
| `RewardTrainer` | Chosen/rejected pairs          | Tokenization                     |
| `CPOTrainer`    | Chosen/rejected pairs          | Same as DPO                      |
| `BCOTrainer`    | Prompt/completion/label        | Tokenization                     |

## SFT Training

### Basic SFT

```python
from datasets import load_dataset
import easydel as ed

# Load conversational dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

trainer = ed.SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.SFTConfig(
        max_length=2048,
        # Chat template applied automatically
    ),
)
trainer.train()
```

### SFT with Mixed Datasets

```python
from easydel.data import block_mixture_interleave

# Mix instruction datasets
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

# Dict format for explicit mapping (recommended)
mixed = block_mixture_interleave(
    datasets={"alpaca": alpaca, "dolly": dolly},
    weights={"alpaca": 0.6, "dolly": 0.4},
    block_size=1000,
    seed=42,
    stop="restart",
)

# Equal weights
mixed = block_mixture_interleave(
    datasets={"alpaca": alpaca, "dolly": dolly},
    weights=None,  # Equal 50/50 mixing
)

trainer = ed.SFTTrainer(
    model=model,
    train_dataset=mixed,
    processing_class=tokenizer,
    arguments=ed.SFTConfig(...),
)
```

### SFT with Pre-tokenized Data

```python
from easydel.data import ParquetShardedSource

# Load pre-tokenized data
source = ParquetShardedSource("./tokenized_sft/*.parquet")

trainer = ed.SFTTrainer(
    model=model,
    train_dataset=source,  # Works directly
    processing_class=tokenizer,
    arguments=ed.SFTConfig(
        max_length=2048,
    ),
)
```

## DPO Training

### Basic DPO

```python
from datasets import load_dataset
import easydel as ed

# Load preference dataset (has chosen/rejected)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = ed.DPOTrainer(
    model=policy_model,
    reference_model=ref_model,  # Optional, deep-copied if None
    train_dataset=dataset,      # Raw preference data
    processing_class=tokenizer,
    arguments=ed.DPOConfig(
        max_prompt_length=512,
        max_completion_length=512,
        beta=0.1,
    ),
)
trainer.train()
```

### DPO Internal Preprocessing

DPO trainer automatically:

1. Extracts shared prompt from chosen/rejected
2. Applies chat template if conversational
3. Tokenizes with proper truncation
4. Handles padding and masking

```python
# Input format (conversational)
{
    "chosen": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
    "rejected": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Go away."},
    ],
}

# Or simpler format
{
    "prompt": "Hello",
    "chosen": "Hi there!",
    "rejected": "Go away.",
}
```

## GRPO Training

### Basic GRPO

```python
from datasets import load_dataset
import easydel as ed

# Load any preference dataset (prompts extracted automatically)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

def reward_fn(prompts, completions, **kwargs):
    # Your reward logic
    return [score_completion(c) for c in completions]

trainer = ed.GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.GRPOConfig(
        num_return_sequences=4,
        max_prompt_length=512,
        max_completion_length=512,
    ),
)
trainer.train()
```

### GRPO Internal Preprocessing

GRPO trainer automatically:

1. Extracts prompts from chosen/rejected conversations
2. Applies chat template with `add_generation_prompt=True`
3. Left-pads for efficient batch generation

### GRPO with Custom Prompts

```python
# Direct prompt format also works
dataset = Dataset.from_dict({
    "prompt": [
        "Write a poem about AI",
        "Explain quantum computing",
    ]
})

# Or conversational
dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "Write a poem about AI"}],
        [{"role": "user", "content": "Explain quantum computing"}],
    ]
})
```

## KTO Training

### Basic KTO

```python
from datasets import load_dataset
import easydel as ed

# KTO accepts paired preference data (will unpair internally)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = ed.KTOTrainer(
    model=policy_model,
    reference_model=ref_model,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.KTOConfig(
        max_prompt_length=512,
        max_completion_length=512,
        beta=0.1,
        loss_type="kto",
    ),
)
trainer.train()
```

### KTO Internal Preprocessing

KTO trainer automatically:

1. Extracts shared prompt
2. Unpairs preference data (1 pair → 2 examples with labels)
3. Applies chat template
4. Tokenizes with BCO collator

```python
# Input: paired preference data
{
    "chosen": [...],
    "rejected": [...],
}

# Internally converted to:
[
    {"prompt": "...", "completion": "...", "label": True},   # From chosen
    {"prompt": "...", "completion": "...", "label": False},  # From rejected
]
```

## ORPO Training

```python
from datasets import load_dataset
import easydel as ed

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = ed.ORPOTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.ORPOConfig(
        max_prompt_length=512,
        max_completion_length=512,
        beta=0.1,
    ),
)
trainer.train()
```

## Reward Training

```python
from datasets import load_dataset
import easydel as ed

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Use sequence classification model
trainer = ed.RewardTrainer(
    model=reward_model,  # AutoEasyDeLModelForSequenceClassification
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.RewardConfig(
        max_length=2048,
    ),
)
trainer.train()
```

## Using ShardedDataSource

All trainers accept `ShardedDataSource` directly:

```python
from easydel.data import ParquetShardedSource, JsonShardedSource

# From local files
source = ParquetShardedSource("data/*.parquet")

# From cloud
source = ParquetShardedSource(
    "gs://bucket/data/*.parquet",
    storage_options={"token": "cloud"},
)

# Use with any trainer
trainer = ed.SFTTrainer(
    model=model,
    train_dataset=source,  # Works directly
    processing_class=tokenizer,
    arguments=ed.SFTConfig(...),
)
```

## Mixed Datasets with Trainers

```python
from datasets import load_dataset
from easydel.data import block_mixture_interleave

# Load multiple datasets
ds1 = load_dataset("dataset1", split="train")
ds2 = load_dataset("dataset2", split="train")
ds3 = load_dataset("dataset3", split="train")

# Dict format for explicit mapping (recommended)
mixed = block_mixture_interleave(
    datasets={"ds1": ds1, "ds2": ds2, "ds3": ds3},
    weights={"ds1": 0.5, "ds2": 0.3, "ds3": 0.2},
    block_size=1000,
    seed=42,
    stop="restart",
)

# Works with any trainer
trainer = ed.DPOTrainer(
    model=model,
    train_dataset=mixed,
    processing_class=tokenizer,
    arguments=ed.DPOConfig(...),
)
```

## Dynamic Weight Scheduling

```python
from easydel.data import (
    MixedShardedSource,
    HFDatasetShardedSource,
    WeightScheduler,
    WeightSchedulePoint,
)

# Wrap datasets
source1 = HFDatasetShardedSource(dataset1)
source2 = HFDatasetShardedSource(dataset2)

# Define schedule
scheduler = WeightScheduler(
    schedule=[
        WeightSchedulePoint(step=0, weights={"easy": 0.9, "hard": 0.1}),
        WeightSchedulePoint(step=10000, weights={"easy": 0.5, "hard": 0.5}),
        WeightSchedulePoint(step=50000, weights={"easy": 0.1, "hard": 0.9}),
    ],
    interpolation="linear",
)

mixed = MixedShardedSource(
    sources={"easy": source1, "hard": source2},
    weight_scheduler=scheduler,
    block_size=1000,
)

trainer = ed.SFTTrainer(
    model=model,
    train_dataset=mixed,
    processing_class=tokenizer,
    arguments=ed.SFTConfig(...),
)
```

## Pre-tokenization for Trainers

### SFT Pre-tokenization

```python
from easydel.data import tokenize_and_save

# Tokenize once
tokenize_and_save(
    data_files="conversations/*.jsonl",
    tokenizer="meta-llama/Llama-2-7b-chat-hf",
    output_path="./sft_tokenized",
    max_length=2048,
)

# Use in training
from easydel.data import ParquetShardedSource

source = ParquetShardedSource("./sft_tokenized/*.parquet")
trainer = ed.SFTTrainer(train_dataset=source, ...)
```

### DPO Pre-tokenization

```python
from easydel.trainers.transforms import DPOPreprocessTransform
from easydel.data import JsonShardedSource, TransformedShardedSource

transform = DPOPreprocessTransform(
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_completion_length=512,
)

source = JsonShardedSource("preference_data/*.jsonl")
tokenized = TransformedShardedSource(source, transform=transform)

# Save for later
from easydel.data import save_iterator
save_iterator(
    tokenized.open_shard(tokenized.shard_names[0]),
    output_path="./dpo_tokenized",
    format="parquet",
)
```

## Trainer-Specific Transforms

Access trainer preprocessing logic directly:

```python
from easydel.trainers.transforms import (
    SFTPreprocessTransform,
    DPOPreprocessTransform,
    ORPOPreprocessTransform,
    KTOPreprocessTransform,
    GRPOPreprocessTransform,
    RewardPreprocessTransform,
    BCOPreprocessTransform,
    CPOPreprocessTransform,
)

# SFT transform
sft_transform = SFTPreprocessTransform(
    tokenizer=tokenizer,
    max_length=2048,
)

# DPO transform
dpo_transform = DPOPreprocessTransform(
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_completion_length=512,
)

# Apply to data
from easydel.data import TransformedShardedSource

source = JsonShardedSource("data/*.jsonl")
transformed = TransformedShardedSource(source, transform=dpo_transform)
```

## Streaming with Trainers

```python
from datasets import load_dataset

# Stream from HuggingFace
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
)

# Works with trainers
trainer = ed.SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    arguments=ed.SFTConfig(
        shuffle_train_dataset=False,  # Already streamed
        ...
    ),
)
```

## Best Practices

### 1. Match Data Format to Trainer

```python
# SFT: text or messages
{"text": "..."} or {"messages": [...]}

# DPO/ORPO/CPO: chosen/rejected
{"chosen": [...], "rejected": [...]}

# KTO: paired or unpaired
{"chosen": [...], "rejected": [...]}  # Paired (unpaired internally)
{"prompt": "...", "completion": "...", "label": True/False}  # Already unpaired

# GRPO: prompts
{"prompt": "..."} or {"chosen": [...], "rejected": [...]}

# Reward: chosen/rejected
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 2. Use Appropriate Sequence Lengths

```python
# DPO/KTO/ORPO: separate prompt and completion lengths
ed.DPOConfig(
    max_prompt_length=512,      # For prompt
    max_completion_length=512,  # For completion
)

# SFT: single length
ed.SFTConfig(
    max_length=2048,  # Total length
)

# GRPO: prompt length + generation length
ed.GRPOConfig(
    max_prompt_length=512,
    max_completion_length=512,  # For generation
)
```

### 3. Handle Pad Tokens

```python
# Most trainers handle this automatically
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# But you can specify explicitly
ed.DPOConfig(
    padding_value=tokenizer.pad_token_id,
    label_pad_token_id=-100,  # Ignore in loss
)
```

### 4. Monitor Data Processing

```python
# Check preprocessing
trainer = ed.DPOTrainer(...)

# Before training, verify a batch
batch = next(iter(trainer._training_batch_iterator()))
print(f"Batch keys: {batch.keys()}")
print(f"Input shape: {batch['input_ids'].shape}")
```

## Troubleshooting

### "Dataset must have column X"

```python
# Check your dataset columns
print(dataset.column_names)

# Map to expected format
dataset = dataset.map(lambda x: {"chosen": x["good"], "rejected": x["bad"]})
```

### Tokenization errors

```python
# Ensure tokenizer has required tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

### Out of memory

```python
# Use streaming
dataset = load_dataset(..., streaming=True)

# Or reduce batch size
ed.SFTConfig(total_batch_size=4)

# Or use gradient accumulation
ed.SFTConfig(
    total_batch_size=32,
    gradient_accumulation_steps=8,  # Effective batch = 4
)
```

## Next Steps

- [Quickstart](quickstart.md) - Get started quickly
- [Dataset Mixing](mixing.md) - Advanced mixing strategies
- [Pre-tokenization](pretokenization.md) - Offline tokenization

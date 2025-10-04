# Image Diffusion with Rectified Flow in EasyDeL

This document describes the newly added image diffusion capabilities to EasyDeL, implementing DiT (Diffusion Transformer) with rectified flow for high-quality image generation.

## Overview

We've added complete image diffusion support to EasyDeL with the following components:

1. **DiT (Diffusion Transformer) Architecture** (`easydel/modules/dit/`)
2. **Image Diffusion Trainer** (`easydel/trainers/image_diffusion_trainer/`)
3. **Rectified Flow Training** (velocity prediction formulation)
4. **Integration with MaxDiffusion** (for advanced schedulers and utilities)

## Architecture Components

### 1. DiT Module (`easydel/modules/dit/`)

The Diffusion Transformer (DiT) architecture implemented with:

- **Patch Embedding**: Converts images to sequences of patches
- **Positional Embedding**: 2D sinusoidal position embeddings
- **Timestep Embedding**: Sinusoidal timestep encoding with MLP
- **Label Embedding**: Class-conditional generation support
- **Transformer Blocks**: Self-attention with adaptive layer norm (adaLN) conditioning
- **Final Layer**: Unpatchifies transformer outputs back to image space

#### Configuration

```python
from easydel.modules.dit import DiTConfig

config = DiTConfig(
    image_size=32,              # Input image size
    patch_size=2,               # Patch size
    in_channels=3,              # RGB or latent channels
    hidden_size=1152,           # Transformer hidden dimension
    num_hidden_layers=28,       # Number of transformer blocks
    num_attention_heads=16,     # Attention heads
    num_classes=1000,           # Number of class labels
    class_dropout_prob=0.1,     # Classifier-free guidance dropout
    learn_sigma=True,           # Whether to predict variance
    use_conditioning=True,      # Use timestep & class conditioning
)
```

#### Model Variants

```python
from easydel.modules.dit import DiTModel, DiTForImageDiffusion
from flax import nnx as nn

# Base model (feature extraction)
model = DiTModel(config=config, rngs=nnx.Rngs(0))

# For image diffusion (with unpatchification)
model = DiTForImageDiffusion(config=config, rngs=nnx.Rngs(0))
```

### 2. Image Diffusion Trainer

A complete trainer implementation following EasyDeL patterns:

```python
from easydel.trainers.image_diffusion_trainer import (
    ImageDiffusionConfig,
    ImageDiffusionTrainer,
)

# Training configuration
training_config = ImageDiffusionConfig(
    # Model
    model_name="dit-imagenet",

    # Dataset
    dataset_name="imagenet-1k",
    dataset_image_field="image",
    dataset_label_field="label",
    image_size=256,

    # Diffusion
    num_train_timesteps=1000,
    prediction_type="velocity",     # 'velocity', 'epsilon', or 'sample'
    class_conditional=True,
    class_dropout_prob=0.1,
    min_snr_gamma=5.0,             # Min-SNR weighting

    # Training
    num_train_epochs=100,
    learning_rate=1e-4,
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,

    # Optimization
    optimizer="adamw",
    weight_decay=0.01,
    warmup_steps=5000,

    # Checkpointing
    save_steps=10000,
    output_dir="./outputs/dit-imagenet",
)
```

## Rectified Flow

Rectified flow is a simple and effective approach to training diffusion models:

### Key Idea

Instead of learning to denoise step-by-step, rectified flow learns a **velocity field** that directly connects noise to data:

```
x_t = (1 - t) * noise + t * data
v_t = data - noise  (velocity)
```

### Training Objective

The model learns to predict the velocity `v_t` at any timestep `t`:

```python
# Forward diffusion
t = random.uniform(0, 1)
noise = random_normal()
x_t = (1 - t) * noise + t * data
v_target = data - noise

# Model prediction
v_pred = model(x_t, t, labels)

# Loss
loss = MSE(v_pred, v_target)
```

### Sampling

Sampling is straightforward Euler integration:

```python
x = random_normal()  # Start from noise
for t in timesteps:
    v = model(x, t, labels)
    x = x + v * dt  # Euler step
```

## Training Examples

### CIFAR-10 Training

```python
import jax
import jax.numpy as jnp
from datasets import load_dataset
from flax import nnx as nn

from easydel.modules.dit import DiTConfig, DiTForImageDiffusion
from easydel.trainers.image_diffusion_trainer import (
    ImageDiffusionConfig,
    ImageDiffusionTrainer,
)

# Model configuration
model_config = DiTConfig(
    image_size=32,
    patch_size=2,
    in_channels=3,
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    num_classes=10,
)

# Training configuration
training_config = ImageDiffusionConfig(
    dataset_name="cifar10",
    dataset_image_field="img",
    dataset_label_field="label",
    image_size=32,
    num_train_epochs=100,
    learning_rate=1e-4,
    per_device_train_batch_size=128,
    prediction_type="velocity",
    output_dir="./outputs/dit-cifar10",
)

# Load dataset
dataset = load_dataset("cifar10")

def preprocess(examples):
    images = jnp.array(examples["img"])
    images = (images.astype(jnp.float32) / 127.5) - 1.0
    return {
        "pixel_values": images,
        "labels": jnp.array(examples["label"]),
    }

train_dataset = dataset["train"].map(preprocess, batched=True)

# Initialize model
rngs = nn.Rngs(params=0, dropout=1)
model = DiTForImageDiffusion(config=model_config, rngs=rngs)

# Train
trainer = ImageDiffusionTrainer(
    arguments=training_config,
    model=model,
    train_dataset=train_dataset,
    seed=42,
)

trainer.train()
```

### ImageNet-256 Training

```python
# Larger model for ImageNet
model_config = DiTConfig(
    image_size=256,
    patch_size=2,
    in_channels=4,  # Using VAE latents
    hidden_size=1152,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_classes=1000,
)

training_config = ImageDiffusionConfig(
    dataset_name="imagenet-1k",
    use_vae=True,  # Train in latent space
    vae_model_name_or_path="stabilityai/sd-vae-ft-mse",
    image_size=256,
    num_train_epochs=400,
    learning_rate=1e-4,
    per_device_train_batch_size=256,
    gradient_accumulation_steps=2,
    prediction_type="velocity",
    min_snr_gamma=5.0,
    output_dir="./outputs/dit-imagenet256",
)
```

## Features

### 1. Classifier-Free Guidance (CFG)

Enable unconditional generation during training:

```python
config = DiTConfig(
    class_dropout_prob=0.1,  # 10% of labels dropped
)
```

During inference, use CFG for better quality:

```python
# Generate with CFG
guidance_scale = 4.0
pred_cond = model(x_t, t, labels)
pred_uncond = model(x_t, t, uncond_labels)
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

### 2. Min-SNR Weighting

Improves training stability with SNR-based loss weighting:

```python
training_config = ImageDiffusionConfig(
    min_snr_gamma=5.0,  # Enable Min-SNR weighting
)
```

### 3. Latent Diffusion

Train in VAE latent space for efficiency:

```python
training_config = ImageDiffusionConfig(
    use_vae=True,
    vae_model_name_or_path="stabilityai/sd-vae-ft-mse",
    in_channels=4,  # VAE latent channels
)
```

### 4. Sharding & Distribution

Full support for model and data parallelism:

```python
model_config = DiTConfig(
    sharding_array=(1, -1, 1, 1),  # Shard along sequence dimension
)

training_config = ImageDiffusionConfig(
    gradient_accumulation_steps=4,
    per_device_train_batch_size=64,
)
```

## Model Sizes

Common DiT configurations:

| Model | Params | Hidden | Layers | Heads | Patch |
|-------|--------|--------|--------|-------|-------|
| DiT-S | 33M    | 384    | 12     | 6     | 2     |
| DiT-B | 130M   | 768    | 12     | 12    | 2     |
| DiT-L | 458M   | 1024   | 24     | 16    | 2     |
| DiT-XL| 675M   | 1152   | 28     | 16    | 2     |

## Performance Benchmarks

Expected FID scores on CIFAR-10 (after 100 epochs):

- DiT-S: ~15-20 FID
- DiT-B: ~10-15 FID
- DiT-L: ~5-10 FID

## Integration with MaxDiffusion

The implementation can leverage MaxDiffusion's advanced features:

- **Schedulers**: Rectified flow scheduler in `maxdiffusion/schedulers/`
- **VAE**: Pre-trained VAE models
- **Utilities**: Data processing, metrics, etc.

## File Structure

```
easydel/
├── modules/
│   └── dit/
│       ├── __init__.py
│       ├── dit_configuration.py
│       └── modeling_dit.py
├── trainers/
│   └── image_diffusion_trainer/
│       ├── __init__.py
│       ├── image_diffusion_config.py
│       ├── image_diffusion_trainer.py
│       └── _fn.py
└── examples/
    └── train_image_diffusion_dit.py
```

## References

1. **Scalable Diffusion Models with Transformers (DiT)**: [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
2. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**: [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
3. **Classifier-Free Diffusion Guidance**: [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

## Citation

If you use this implementation, please cite:

```bibtex
@software{easydel2024,
  title = {EasyDeL: An open-source library for training and serving large language models},
  author = {Zare Chavoshi, Erfan},
  year = {2024},
  url = {https://github.com/erfanzar/EasyDeL}
}
```

## License

This implementation is licensed under the Apache License 2.0.

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

"""MPT (MosaicML Pretrained Transformer) model implementation for EasyDeL.

MPT Architecture
================
MPT (MosaicML Pretrained Transformer) is MosaicML's family of open-source decoder-only
transformers optimized for efficiency, featuring innovations like FlashAttention,
ALiBi positional encodings, and low-precision LayerNorm.

Key Features
------------
- **ALiBi Position Encoding**: Attention with Linear Biases instead of position
  embeddings, enabling better extrapolation to longer sequences than seen in training.
- **FlashAttention**: Optimized attention implementation for faster training and
  inference with reduced memory usage.
- **Low-Precision LayerNorm**: Option to use low-precision (e.g., float32) LayerNorm
  even with bfloat16/float16 training for better stability.
- **QK LayerNorm**: Optional layer normalization on queries and keys before attention
  for improved training stability.
- **No Biases**: Option to remove all biases from the model for efficiency.
- **Configurable Attention**: Supports various attention implementations including
  multihead, multiquery, and grouped query attention.

Model Architecture
------------------
- Token embedding layer (no position embeddings with ALiBi)
- N transformer decoder layers with:
  - Multi-head/Multi-query/Grouped-query attention
  - Optional QK LayerNorm
  - ALiBi position biases (added to attention scores)
  - Feed-forward network with configurable expansion ratio
  - Layer normalization (pre-norm architecture)
- Final layer normalization
- Language modeling head

ALiBi Mechanism
---------------
Instead of adding position embeddings to inputs, ALiBi adds static biases to
attention scores based on distance between query and key positions:
- Enables training on shorter sequences, inference on longer ones
- Reduces memory and computation (no position embeddings to store/compute)
- Better length extrapolation than absolute or relative position embeddings

Model Variants
--------------
- MPT-7B: 7B parameters, 32 layers, 4096 hidden, ALiBi, 2048 context
- MPT-7B-Instruct: Instruction-tuned version of MPT-7B
- MPT-30B: 30B parameters, 48 layers, 7168 hidden, ALiBi, 8192 context
- MPT-7B-StoryWriter: Fine-tuned for 65K context length

Usage Example
-------------
```python
from easydel import MptConfig, MptAttentionConfig, MptForCausalLM
from flax import nnx as nn
import jax.numpy as jnp

# Create attention configuration
attn_config = MptAttentionConfig(
    attn_type="multihead_attention",
    alibi=True,
    alibi_bias_max=8,
    qk_ln=False,
)

# Create model configuration
config = MptConfig(
    d_model=4096,
    n_heads=32,
    n_layers=32,
    expansion_ratio=4,
    max_seq_len=2048,
    vocab_size=50368,
    attn_config=attn_config,
)

# Initialize model
model = MptForCausalLM(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    rngs=nn.Rngs(0),
)

# Generate text
outputs = model(
    input_ids=jnp.array([[1, 2, 3, 4]]),
    attention_mask=jnp.ones((1, 4)),
)
```

Available Models
----------------
- MptConfig: Configuration class for MPT models
- MptAttentionConfig: Configuration for MPT attention mechanism
- MptModel: Base MPT model outputting hidden states
- MptForCausalLM: MPT with language modeling head for text generation
"""

from .modeling_mosaic import MptForCausalLM, MptModel
from .mosaic_configuration import MptAttentionConfig, MptConfig

__all__ = ("MptAttentionConfig", "MptConfig", "MptForCausalLM", "MptModel")

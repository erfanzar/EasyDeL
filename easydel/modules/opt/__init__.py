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

"""OPT model implementation for EasyDeL.

OPT Architecture
================
OPT (Open Pre-trained Transformer) is Meta AI's family of open-source autoregressive
language models ranging from 125M to 175B parameters, designed as open alternatives
to GPT-3 with full reproducibility and transparency.

Key Features
------------
- **Pre-Norm Architecture**: Layer normalization applied before attention and FFN blocks
  (controlled by do_layer_norm_before flag, default True for most sizes).
- **Standard Transformer Decoder**: Uses classic transformer architecture without
  architectural innovations like parallel residuals or special attention patterns.
- **Learned Positional Embeddings**: Unlike sinusoidal embeddings, OPT learns positional
  embeddings with a configurable offset (default 2 for padding tokens).
- **Optional Word Embedding Projection**: Supports projecting between word embedding
  dimension and hidden size via word_embed_proj_dim parameter.
- **LayerDrop**: Optional stochastic layer dropping during training for efficiency.

Model Architecture
------------------
- Token embedding layer (vocab_size: 50272, with optional projection)
- Learned positional embedding layer with offset
- N transformer decoder layers with:
  - Pre-layer normalization (configurable)
  - Multi-head self-attention
  - Feed-forward network (FFN)
  - Residual connections with dropout
- Optional final layer normalization
- Language modeling head (can be tied to embeddings)

Model Variants
--------------
- OPT-125M: 12 layers, 768 hidden, 12 heads, 3072 FFN
- OPT-1.3B: 24 layers, 2048 hidden, 32 heads, 8192 FFN
- OPT-6.7B: 32 layers, 4096 hidden, 32 heads, 16384 FFN
- OPT-13B: 40 layers, 5120 hidden, 40 heads, 20480 FFN
- OPT-30B: 48 layers, 7168 hidden, 56 heads, 28672 FFN
- OPT-66B: 64 layers, 9216 hidden, 72 heads, 36864 FFN
- OPT-175B: 96 layers, 12288 hidden, 96 heads, 49152 FFN

Usage Example
-------------
```python
from easydel import OPTConfig, OPTForCausalLM
from flax import nnx as nn
import jax.numpy as jnp

# Create configuration (OPT-1.3B settings)
config = OPTConfig(
    vocab_size=50272,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=32,
    ffn_dim=8192,
    max_position_embeddings=2048,
    do_layer_norm_before=True,
)

# Initialize model
model = OPTForCausalLM(
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
- OPTConfig: Configuration class for OPT models
- OPTModel: Base OPT model outputting hidden states
- OPTForCausalLM: OPT model with language modeling head for text generation
"""

from .modeling_opt import OPTForCausalLM, OPTModel
from .opt_configuration import OPTConfig

__all__ = "OPTConfig", "OPTForCausalLM", "OPTModel"

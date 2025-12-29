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

"""GPT-J model implementation for EasyDeL.

GPT-J Architecture
==================
GPT-J is a 6-billion parameter autoregressive language model developed by EleutherAI.
It uses a transformer decoder architecture with several key innovations:

Key Features
------------
- **Parallel Attention and FFN**: Unlike standard transformers, GPT-J computes attention
  and feed-forward layers in parallel within each block, improving training efficiency.
- **Rotary Position Embeddings (RoPE)**: Uses partial rotary embeddings (rotary_dim=64)
  applied only to a subset of the attention head dimensions.
- **Dense Attention**: Full quadratic attention mechanism without sparsity patterns.
- **6B Parameters**: Trained on The Pile dataset with context length of 2048 tokens.

Model Architecture
------------------
- Embedding layer with vocabulary size of 50,400 tokens
- 28 transformer blocks with parallel attention and MLP
- Hidden size: 4096
- Attention heads: 16 (head_dim = 256)
- Intermediate size: 16,384 (4x hidden size)
- Activation: GELU (new approximation)

Usage Example
-------------
```python
from easydel import GPTJConfig, GPTJForCausalLM
from flax import nnx as nn

# Create configuration
config = GPTJConfig(
    vocab_size=50400,
    n_positions=2048,
    n_embd=4096,
    n_layer=28,
    n_head=16,
    rotary_dim=64,
)

# Initialize model
model = GPTJForCausalLM(
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
- GPTJConfig: Configuration class for GPT-J models
- GPTJModel: Base GPT-J model outputting hidden states
- GPTJForCausalLM: GPT-J model with language modeling head for text generation
"""

from .gpt_j_configuration import GPTJConfig
from .modeling_gpt_j import GPTJForCausalLM, GPTJModel

__all__ = "GPTJConfig", "GPTJForCausalLM", "GPTJModel"

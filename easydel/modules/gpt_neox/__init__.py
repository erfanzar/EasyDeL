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

"""GPT-NeoX model implementation for EasyDeL.

GPT-NeoX Architecture
=====================
GPT-NeoX is EleutherAI's family of autoregressive language models, with the flagship
GPT-NeoX-20B being a 20-billion parameter model trained on The Pile dataset.

Key Features
------------
- **Parallel Residual Connections**: Optionally computes attention and FFN in parallel
  (similar to GPT-J) for improved training efficiency via use_parallel_residual flag.
- **Rotary Position Embeddings (RoPE)**: Uses partial rotary embeddings with configurable
  rotary_pct (default 25% of head dimension).
- **Fused QKV Projection**: Single query_key_value projection matrix for efficiency.
- **Scalable Architecture**: Supports models from 125M to 20B+ parameters.
- **Flash Attention Compatible**: Optimized for modern attention implementations.

Model Architecture
------------------
- Token embedding layer with configurable vocabulary size
- N transformer blocks with optional parallel residual connections
- Each block contains:
  - Layer normalization (pre-attention)
  - Multi-head self-attention with partial RoPE
  - Layer normalization (post-attention, for MLP)
  - Feed-forward network (MLP)
- Final layer normalization
- Language modeling head (tied or untied embeddings)

Variants
--------
- GPT-NeoX-20B: 20B parameters, 44 layers, 6144 hidden size, 64 heads
- Pythia Suite: 70M to 12B parameters for scaling research
- Custom configurations supported via GPTNeoXConfig

Usage Example
-------------
```python
from easydel import GPTNeoXConfig, GPTNeoXForCausalLM
from flax import nnx as nn
import jax.numpy as jnp

# Create configuration (GPT-NeoX-20B settings)
config = GPTNeoXConfig(
    vocab_size=50432,
    hidden_size=6144,
    num_hidden_layers=44,
    num_attention_heads=64,
    intermediate_size=24576,
    rotary_pct=0.25,
    use_parallel_residual=True,
)

# Initialize model
model = GPTNeoXForCausalLM(
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
- GPTNeoXConfig: Configuration class for GPT-NeoX models
- GPTNeoXModel: Base GPT-NeoX model outputting hidden states
- GPTNeoXForCausalLM: GPT-NeoX with language modeling head for text generation
"""

from .gpt_neox_configuration import GPTNeoXConfig
from .modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXModel

__all__ = "GPTNeoXConfig", "GPTNeoXForCausalLM", "GPTNeoXModel"

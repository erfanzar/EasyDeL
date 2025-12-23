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

"""OpenELM model implementation for EasyDeL.

OpenELM Architecture
====================
OpenELM (Open Efficient Language Model) is Apple's family of open-source language
models designed for parameter efficiency and on-device deployment, featuring
layer-wise scaling of model dimensions.

Key Features
------------
- **Layer-wise Scaling**: Uses different numbers of attention heads and FFN
  dimensions per layer, increasing depth-wise for better parameter efficiency.
- **Grouped Query Attention (GQA)**: Uses GQA with configurable number of KV heads
  to reduce memory and computation costs.
- **RoPE Embeddings**: Rotary position embeddings for better position encoding.
- **RMSNorm**: Uses Root Mean Square Layer Normalization instead of LayerNorm.
- **Parameter Efficiency**: Achieves competitive performance with fewer parameters
  through architectural optimizations.

Model Architecture
------------------
- Token embedding layer
- N transformer decoder layers with:
  - Variable number of attention heads per layer (layer-wise scaling)
  - Grouped Query Attention (GQA) with shared KV heads
  - Variable FFN dimensions per layer
  - RMSNorm for normalization
  - RoPE position embeddings
  - SwiGLU activation function
- Final RMSNorm
- Language modeling head

Layer-wise Scaling
------------------
Unlike standard transformers with uniform layer dimensions, OpenELM scales:
- Number of attention heads increases in deeper layers
- FFN dimension increases in deeper layers
- Allows more efficient parameter allocation

Usage Example
-------------
```python
from easydel import OpenELMConfig, OpenELMForCausalLM
from flax import nnx as nn
import jax.numpy as jnp

# Create configuration
config = OpenELMConfig(
    vocab_size=32000,
    max_context_length=2048,
    num_transformer_layers=16,
    model_dim=1280,
    head_dim=64,
    num_gqa_groups=4,
)

# Initialize model
model = OpenELMForCausalLM(
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
- OpenELMConfig: Configuration class for OpenELM models
- OpenELMModel: Base OpenELM model outputting hidden states
- OpenELMForCausalLM: OpenELM with language modeling head for text generation
"""

from .modeling_openelm import OpenELMForCausalLM, OpenELMModel
from .openelm_configuration import OpenELMConfig

__all__ = "OpenELMConfig", "OpenELMForCausalLM", "OpenELMModel"

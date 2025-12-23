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

"""Phi-3 model implementation for EasyDeL.

Phi-3 is a family of small language models (SLMs) developed by Microsoft that deliver
impressive performance despite their compact size. The Phi-3 models are transformer-based
architectures optimized for efficiency and quality, using carefully curated training data
and advanced training techniques.

Architecture Overview
--------------------
Phi-3 is a decoder-only transformer architecture that incorporates several modern techniques
for improved efficiency and performance:

1. **Grouped Query Attention (GQA)**: Instead of using separate key-value heads for each
   query head, Phi-3 groups multiple query heads to share key-value heads. This reduces
   the memory footprint during inference while maintaining model quality.

2. **Rotary Position Embeddings (RoPE)**: Phi-3 uses RoPE to encode positional information
   directly into the attention mechanism. RoPE provides better extrapolation to longer
   sequences than learned positional embeddings.

3. **SwiGLU Activation**: The feed-forward layers use SwiGLU (Swish-Gated Linear Units),
   which combines gating with the Swish/SiLU activation function for improved expressiveness.

4. **RMS Normalization**: Layer normalization uses the Root Mean Square (RMS) variant,
   which is computationally simpler than standard LayerNorm while providing similar benefits.

5. **Sliding Window Attention (Optional)**: Some Phi-3 variants support sliding window
   attention, where each token only attends to a local window of surrounding tokens,
   reducing computational complexity for long sequences.

Key Components
-------------
- **Phi3Config**: Configuration class containing all hyperparameters including model size,
  number of layers, attention heads, GQA settings, RoPE configuration, and optional
  sliding window parameters.

- **Phi3Model**: The base transformer model consisting of token embeddings, stacked
  decoder layers with attention and MLP blocks, and final layer normalization. Outputs
  contextualized hidden states.

- **Phi3ForCausalLM**: Causal language modeling variant that adds a linear language
  modeling head on top of Phi3Model for next-token prediction and text generation.

Model Variants
-------------
Phi-3 comes in several sizes:
- **Phi-3-mini**: 3.8B parameters, optimized for edge devices and resource-constrained environments
- **Phi-3-small**: 7B parameters, balanced performance and efficiency
- **Phi-3-medium**: 14B parameters, higher capacity for complex tasks

All variants support context lengths up to 128K tokens with appropriate RoPE scaling.

Advantages
----------
- **Compact Size**: Achieves strong performance with fewer parameters than larger models
- **Efficient Inference**: GQA reduces KV cache memory requirements
- **Long Context**: Supports very long sequences (up to 128K tokens) via RoPE scaling
- **Quality Training Data**: Carefully filtered and curated training data improves quality
- **Versatile**: Suitable for chat, code generation, reasoning, and general NLP tasks

Usage Example
------------
```python
from easydel import Phi3Config, Phi3ForCausalLM
import jax.numpy as jnp
from flax import nnx as nn

# Initialize Phi-3-mini configuration
config = Phi3Config(
    vocab_size=32064,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,  # Set lower for GQA (e.g., 8)
    max_position_embeddings=4096,
    rope_theta=10000.0,
)

# Create model instance
model = Phi3ForCausalLM(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    rngs=nn.Rngs(0),
)

# Forward pass for training
input_ids = jnp.array([[1, 2, 3, 4, 5]])
outputs = model(input_ids=input_ids)
logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

# Autoregressive generation with caching
cache = None
for step in range(10):
    outputs = model(
        input_ids=input_ids[:, -1:],  # Only last token
        cache_view=cache,
    )
    cache = outputs.cache_view
    next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1)
    input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
```

References
---------
- Paper: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"
- Model Hub: https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3
"""

from .modeling_phi3 import Phi3ForCausalLM, Phi3Model
from .phi3_configuration import Phi3Config

__all__ = "Phi3Config", "Phi3ForCausalLM", "Phi3Model"

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

"""Mamba2 model implementation for EasyDeL.

Mamba2 is an advanced state-space model (SSM) that builds upon the original Mamba architecture
with significant improvements in efficiency and modeling capacity. It uses selective state spaces
with a grouped attention mechanism for efficient long-sequence modeling.

Architecture Overview
--------------------
Mamba2 is a decoder-only architecture that replaces traditional attention mechanisms with
selective state-space models. Key architectural components include:

1. **Selective State Space Model (SSM)**: Instead of quadratic attention, Mamba2 uses linear-time
   state-space models that selectively propagate information through sequences. This enables
   efficient processing of very long sequences (100k+ tokens) with linear complexity.

2. **Grouped SSM Heads**: Similar to multi-head attention, Mamba2 uses multiple SSM heads organized
   into groups (n_groups), where each group shares state transition matrices (B and C matrices).
   This improves model capacity while maintaining efficiency.

3. **Depthwise Convolution**: Each layer includes a causal 1D convolution that captures local
   dependencies before the SSM operation, similar to how CNNs capture local patterns.

4. **Gated RMSNorm**: A specialized normalization that gates the SSM output before normalization,
   improving training stability and model expressiveness.

Key Components
-------------
- **Mamba2Config**: Configuration class for all hyperparameters including state size, number of
  heads, chunk size for sequence processing, and time step parameters.

- **Mamba2Model**: The base model consisting of token embeddings, stacked Mamba2 blocks, and
  final layer normalization. Outputs hidden states without a language modeling head.

- **Mamba2ForCausalLM**: Causal language modeling variant that adds a linear projection head
  on top of Mamba2Model for next-token prediction tasks.

Advantages over Transformers
----------------------------
- **Linear Complexity**: O(n) time and memory complexity vs O(n^2) for attention
- **Constant Inference**: Single-token generation uses constant memory via recurrent cache
- **Long Context**: Efficiently handles sequences of 100k+ tokens
- **Hardware Efficient**: Better GPU utilization through custom CUDA kernels

Usage Example
------------
```python
from easydel import Mamba2Config, Mamba2ForCausalLM
import jax.numpy as jnp

# Initialize model configuration
config = Mamba2Config(
    vocab_size=50257,
    hidden_size=2048,
    num_hidden_layers=24,
    state_size=128,
    num_heads=64,
    n_groups=8,
)

# Create model instance
model = Mamba2ForCausalLM(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    rngs=nn.Rngs(0),
)

# Forward pass for training
input_ids = jnp.array([[1, 2, 3, 4, 5]])
outputs = model(input_ids=input_ids)
logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

# Inference with caching
cache_params = None
for token_id in input_ids[0]:
    outputs = model(
        input_ids=token_id[None, None],
        cache_params=cache_params,
    )
    cache_params = outputs.cache_params
```

References
---------
- Paper: "Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality"
- Original Implementation: https://github.com/state-spaces/mamba
"""

from .mamba2_configuration import Mamba2Config
from .modeling_mamba2 import Mamba2ForCausalLM, Mamba2Model

__all__ = ("Mamba2Config", "Mamba2ForCausalLM", "Mamba2Model")

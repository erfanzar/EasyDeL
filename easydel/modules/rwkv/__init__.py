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

"""RWKV model implementation for EasyDeL.

RWKV Architecture
=================
RWKV (Receptance Weighted Key Value) is a novel RNN-like architecture that combines
the efficient inference of RNNs with the parallelizable training of transformers,
achieving linear time and memory complexity.

Key Features
------------
- **Linear Attention**: O(d) time complexity per step (vs O(n*d) for transformers),
  enabling efficient processing of very long sequences.
- **RNN-style Recurrence**: Can be formulated as an RNN with hidden states, allowing
  O(1) inference complexity per token during generation.
- **Parallelizable Training**: Despite RNN formulation, training can be parallelized
  similar to transformers using time-mixing and channel-mixing blocks.
- **No Position Embeddings**: Uses time-decay mechanism instead of explicit position
  encoding, naturally handling arbitrary sequence lengths.
- **WKV Mechanism**: Novel "Weighted Key-Value" attention mechanism that maintains
  running statistics for efficient computation.

Model Architecture
------------------
- Token embedding layer
- N RWKV blocks, each containing:
  - Time-mixing block (linear attention with WKV mechanism)
  - Channel-mixing block (position-wise FFN)
  - Layer normalization
  - Time-decay and time-first parameters
- Final layer normalization
- Language modeling head

Time Complexity
---------------
- Training: O(B * T * d^2) - parallelizable across sequence
- Inference: O(d^2) per token - constant time, maintaining hidden state
- Memory: O(d) per layer - constant, independent of sequence length

Inference Modes
---------------
- Parallel mode: Process full sequences like transformers (training)
- Sequential mode: RNN-style one-token-at-a-time generation (inference)
- Can switch between modes seamlessly

Usage Example
-------------
```python
from easydel import RwkvConfig, RwkvForCausalLM
from flax import nnx as nn
import jax.numpy as jnp

# Create configuration
config = RwkvConfig(
    vocab_size=50277,
    hidden_size=768,
    num_hidden_layers=12,
    intermediate_size=None,  # Computed as 4 * hidden_size
)

# Initialize model
model = RwkvForCausalLM(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    rngs=nn.Rngs(0),
)

# Generate text with O(1) complexity per token
outputs = model(
    input_ids=jnp.array([[1, 2, 3, 4]]),
    attention_mask=jnp.ones((1, 4)),
)
```

Available Models
----------------
- RwkvConfig: Configuration class for RWKV models
- RwkvModel: Base RWKV model outputting hidden states
- RwkvForCausalLM: RWKV with language modeling head for efficient generation
"""

from .modeling_rwkv import RwkvForCausalLM, RwkvModel
from .rwkv_configuration import RwkvConfig

__all__ = "RwkvConfig", "RwkvForCausalLM", "RwkvModel"

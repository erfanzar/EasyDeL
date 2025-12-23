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

"""Falcon model implementation for EasyDeL.

Falcon is a family of open-source language models developed by TII (Technology Innovation
Institute) featuring unique architectural choices optimized for efficient training at scale.
Key architectural features include:

- ALiBi (Attention with Linear Biases): Optional positional encoding method that adds
  learned linear biases to attention scores instead of using positional embeddings,
  enabling better length extrapolation. Enabled via `alibi=True`.

- Multi-Query Attention (MQA): When `multi_query=True`, uses a single set of key/value
  heads shared across all query heads, dramatically reducing memory bandwidth and KV cache
  size during inference while maintaining quality.

- Parallel Attention and MLP: When `parallel_attn=True`, computes attention and
  feed-forward layers in parallel rather than sequentially, reducing layer depth and
  improving training throughput.

- New Decoder Architecture: The `new_decoder_architecture` flag enables architectural
  improvements including refined layer normalization placement with support for dual
  layer norms (`num_ln_in_parallel_attn=2`) for parallel attention/MLP paths.

- Flexible Position Encoding: Supports both ALiBi and RoPE (Rotary Position Embeddings),
  automatically selecting based on `alibi` configuration.

Usage Example:
    ```python
    from easydel.modules.falcon import FalconConfig, FalconForCausalLM
    import jax
    from flax import nnx as nn

    # Configure Falcon with parallel attention and MQA
    config = FalconConfig(
        vocab_size=65024,
        hidden_size=4544,
        num_hidden_layers=32,
        num_attention_heads=71,
        multi_query=True,
        parallel_attn=True,
        alibi=False,  # Use RoPE instead of ALiBi
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = FalconForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jax.numpy.array([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    ```
"""

from .falcon_configuration import FalconConfig
from .modeling_falcon import FalconForCausalLM, FalconModel

__all__ = "FalconConfig", "FalconForCausalLM", "FalconModel"

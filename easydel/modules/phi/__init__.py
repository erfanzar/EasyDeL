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

"""Microsoft Phi model implementation for EasyDeL.

Phi is a family of small language models (Phi-1, Phi-2, Phi-3) designed for efficiency
and strong performance on reasoning tasks despite their compact size. Key architectural
features include:

- Partial Rotary Position Embeddings (RoPE): Uses `partial_rotary_factor` (default 0.5)
  to apply rotary embeddings to only a portion of each attention head dimension, reducing
  computational overhead while maintaining positional awareness.

- Optional Q/K Layer Normalization: When `qk_layernorm=True`, applies LayerNorm to
  query and key projections before attention computation, improving training stability.

- Dense Feed-Forward Networks: Standard two-layer MLP with configurable activation
  (default: gelu_new).

- Grouped-Query Attention (GQA): Supports different numbers of key/value heads vs
  query heads via `num_key_value_heads` for memory efficiency.

Usage Example:
    ```python
    from easydel.modules.phi import PhiConfig, PhiForCausalLM
    import jax
    from flax import nnx as nn

    # Configure a Phi-2 style model
    config = PhiConfig(
        vocab_size=51200,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        partial_rotary_factor=0.5,
        qk_layernorm=True,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = PhiForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jax.numpy.array([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```
"""

from .modeling_phi import PhiForCausalLM, PhiModel
from .phi_configuration import PhiConfig

__all__ = "PhiConfig", "PhiForCausalLM", "PhiModel"

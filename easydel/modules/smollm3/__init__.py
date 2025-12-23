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

"""SmolLM3 decoder-only transformer model implementation for EasyDeL.

SmolLM3 is a compact, efficient language model featuring innovative architectural
choices designed for improved performance with reduced computational requirements.

Architecture:
    - Decoder-only transformer with pre-norm architecture
    - Conditional RoPE (NoPE): Selective positional encoding per layer
    - Grouped Query Attention (GQA) for efficiency
    - Optional sliding window attention for local context
    - SwiGLU activation in MLP layers

Key Features:
    - NoPE Layers: Some layers skip RoPE entirely (no_rope_layers config)
    - Layer-specific attention patterns (full vs sliding window)
    - Configurable via no_rope_layer_interval (default: every 4th layer uses NoPE)
    - Supports RoPE scaling for extended context lengths
    - Optional scan-based MLP for memory efficiency

Usage Example:
    ```python
    import jax
    from easydel import SmolLM3Config, SmolLM3ForCausalLM
    from flax import nnx as nn

    # Initialize configuration
    config = SmolLM3Config(
        vocab_size=128256,
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=16,  # GQA with same K/V heads
        intermediate_size=11008,
        no_rope_layer_interval=4,  # NoPE every 4th layer
        use_sliding_window=False,
        max_position_embeddings=32768,
    )

    # Create model for causal LM
    model = SmolLM3ForCausalLM(
        config=config,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        rngs=nn.Rngs(0),
    )

    # Prepare inputs
    input_ids = jax.numpy.array([[1, 2, 3, 4, 5]])

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    ```

Available Models:
    - HuggingFaceTB/SmolLM3-135M: 135M parameter model
    - HuggingFaceTB/SmolLM3-360M: 360M parameter model
    - HuggingFaceTB/SmolLM3-1.7B: 1.7B parameter model

Classes:
    - SmolLM3Config: Configuration class with NoPE and sliding window settings
    - SmolLM3Model: Base model without task-specific head
    - SmolLM3ForCausalLM: Model with language modeling head
    - SmolLM3ForSequenceClassification: Model with classification head
"""

from .modeling_smollm3 import (
    SmolLM3ForCausalLM,
    SmolLM3ForSequenceClassification,
    SmolLM3Model,
)
from .smollm3_configuration import SmolLM3Config

__all__ = (
    "SmolLM3Config",
    "SmolLM3ForCausalLM",
    "SmolLM3ForSequenceClassification",
    "SmolLM3Model",
)

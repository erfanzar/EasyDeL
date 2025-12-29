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

"""Falcon Mamba: Hybrid State-Space Transformer for EasyDeL.

This module implements the Falcon Mamba architecture, which combines the Mamba state-space
model (SSM) with transformer components to create an efficient hybrid architecture. Developed
by Technology Innovation Institute (TII), Falcon Mamba offers linear-time inference while
maintaining strong language modeling capabilities.

Architecture Overview:
    - **Hybrid SSM-Transformer**: Combines Mamba state-space layers with transformer blocks
      for optimal balance between efficiency and expressiveness
    - **Linear-time complexity**: State-space layers provide O(n) time complexity instead of
      O(n²) attention, enabling efficient processing of long sequences
    - **Selective state spaces**: Mamba's selective SSM mechanism allows dynamic filtering
      of inputs based on content
    - **Structured state**: Maintains a compressed hidden state that evolves over time,
      avoiding full attention computation
    - **Interleaved architecture**: Strategic mixing of SSM and attention layers for
      best-of-both-worlds performance

Key Components:
    - **Mamba layers**: State-space model layers with selective gating and structured states
    - **Attention layers**: Traditional multi-head attention for certain positions
    - **Layer mixing strategy**: Configurable pattern of SSM vs attention layers
    - **RMSNorm**: Root Mean Square Layer Normalization for stable training
    - **SiLU activation**: Swish/SiLU activation in feed-forward networks

Configuration:
    - **vocab_size** (default: varies): Size of vocabulary
    - **hidden_size**: Dimension of hidden states and embeddings
    - **num_hidden_layers**: Total number of layers (mix of SSM and attention)
    - **state_size**: Dimension of SSM hidden state
    - **conv_kernel**: Kernel size for causal convolution in Mamba layers
    - **expand_ratio**: Expansion factor for intermediate dimensions

Available Models:
    - **FalconMambaConfig**: Configuration class for model hyperparameters
    - **FalconMambaModel**: Base model without task-specific head
    - **FalconMambaForCausalLM**: Model with language modeling head for text generation

Example Usage:
    ```python
    from easydel import FalconMambaConfig, FalconMambaForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Create configuration
    config = FalconMambaConfig(
        vocab_size=50280,
        hidden_size=2048,
        num_hidden_layers=24,
        state_size=16,
        conv_kernel=4,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = FalconMambaForCausalLM(config, rngs=rngs)

    # Generate text with efficient linear-time inference
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```

Performance Benefits:
    - **Linear complexity**: O(n) instead of O(n²) for long sequences
    - **Efficient inference**: State-based generation without full KV caching
    - **Long context**: Can handle very long sequences efficiently
    - **Memory efficient**: Compressed state representation reduces memory usage

References:
    - Mamba paper: https://arxiv.org/abs/2312.00752
    - Falcon models: https://falconllm.tii.ae/
"""

from .falcon_mamba_configuration import FalconMambaConfig
from .modeling_falcon_mamba import FalconMambaForCausalLM, FalconMambaModel

__all__ = ("FalconMambaConfig", "FalconMambaForCausalLM", "FalconMambaModel")

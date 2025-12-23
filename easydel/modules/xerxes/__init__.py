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

"""Xerxes Model Implementation for EasyDeL.

This module provides the Xerxes architecture, a research-focused decoder-only transformer
model featuring optional Mixture-of-Experts (MoE), configurable normalization strategies,
and optimized attention mechanisms. Xerxes is designed for experimental language modeling
with flexible architectural choices.

Architecture Details:
    Xerxes is a decoder-only transformer with several configurable components:

    - Token Embeddings: Learned token embeddings (256K vocabulary)
    - Decoder Layers: Transformer decoder layers with:
        * Multi-head self-attention with RoPE (Rotary Position Embeddings)
        * Grouped Query Attention (GQA) with configurable head dimensions
        * **Optional Mixture-of-Experts**: 4 local experts per layer (configurable)
        * Optional Query/Key normalization (xe_kvnorm)
        * Optional MLP normalization (xe_mlpnorm)
        * Configurable softmax scaling for attention stability
        * RMSNorm for layer normalization
        * Residual connections
    - Final RMSNorm: Layer normalization after the last decoder layer
    - Language Model Head: Linear projection to vocabulary

Key Features:
    - **Flexible MoE**: Optional 4-expert MoE layers with configurable routing
    - **Advanced Normalization**: Optional Q/K norm and MLP normalization
    - Large Vocabulary: 256,128 tokens for extensive coverage
    - Configurable Head Dimensions: Independent head_dim parameter
    - Custom Softmax Scaling: Configurable attention scaling (default: ~14.97)
    - Grouped Query Attention: Efficient KV caching
    - Rotary Position Embeddings: RoPE for position encoding
    - Flash Attention Support: Optimized attention computation
    - Scan Layers: Optional layer scanning for memory efficiency
    - Swish Activation: Optional swish activation in FFN
    - Tensor Parallelism: Distributed training support
    - Gradient Checkpointing: Memory-efficient training

Default Configuration (Xerxes Base):
    - Vocabulary size: 256,128
    - Hidden size: 4096
    - Intermediate size: 16384
    - Number of layers: 32
    - Attention heads: 32
    - Key-value heads: 8 (GQA)
    - Head dimension: 144
    - Max position embeddings: 16384
    - RMSNorm epsilon: 1e-6
    - RoPE theta: 10000.0
    - Softmax scale: 14.9666295471
    - MoE: Optional (4 local experts)

Usage Example:
    ```python
    from easydel import XerxesConfig, XerxesForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration with MoE enabled
    config = XerxesConfig(
        vocab_size=256128,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=144,
        max_position_embeddings=16384,
        xe_moe=True,  # Enable MoE
        num_local_experts=4,
        xe_kvnorm=True,  # Enable Q/K normalization
        xe_mlpnorm=False,  # Disable MLP normalization
    )

    # Create causal language model
    rngs = nn.Rngs(0)
    model = XerxesForCausalLM(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=rngs,
    )

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    ```

Available Classes:
    - XerxesConfig: Configuration class for Xerxes models
    - XerxesModel: Base Xerxes transformer model
    - XerxesForCausalLM: Xerxes with causal language modeling head

Experimental Features:
    - MoE Support: Optional sparse expert routing
    - Query/Key Normalization: Improves training stability
    - MLP Normalization: Additional normalization in feed-forward networks
    - Custom Softmax Scaling: Fine-tuned attention scaling
    - Configurable Head Dimensions: Flexible attention head sizing

Note:
    Xerxes is a research model designed for experimentation with various
    architectural choices. The flexible configuration options allow testing
    different combinations of normalization strategies, MoE configurations,
    and attention mechanisms.
"""

from .modeling_xerxes import XerxesForCausalLM, XerxesModel
from .xerxes_configuration import XerxesConfig

__all__ = "XerxesConfig", "XerxesForCausalLM", "XerxesModel"

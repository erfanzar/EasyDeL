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

"""Xerxes2 Model Implementation for EasyDeL.

This module provides the Xerxes2 architecture, an enhanced version of Xerxes with
improved Mixture-of-Experts support, advanced RoPE configurations, and expert tensor
parallelism (EPxTP) for efficient distributed training. Xerxes2 represents the
next iteration of research-focused transformer models with state-of-the-art features.

Architecture Details:
    Xerxes2 is a decoder-only transformer building upon Xerxes with enhanced features:

    - Token Embeddings: Learned token embeddings (256K vocabulary)
    - Decoder Layers: Advanced transformer decoder layers with:
        * Multi-head self-attention with configurable RoPE
        * Grouped Query Attention (GQA) with flexible head dimensions
        * **Enhanced Mixture-of-Experts**: Improved MoE with better load balancing
        * Expert Tensor Parallelism (EPxTP): Specialized expert partitioning
        * Optional Query/Key normalization
        * Optional MLP normalization
        * Configurable softmax scaling
        * RMSNorm for layer normalization
        * Residual connections
    - Final RMSNorm: Layer normalization after the last decoder layer
    - Language Model Head: Linear projection to vocabulary

Key Features:
    - **Expert Tensor Parallelism (EPxTP)**: Dedicated sharding strategy for MoE
    - **Enhanced MoE**: Improved expert routing and load balancing
    - **Advanced RoPE Config**: Flexible rope configuration with RopeConfig
    - Large Vocabulary: 256,128 tokens
    - Configurable Head Dimensions: Independent head_dim parameter
    - Custom Softmax Scaling: Configurable attention scaling
    - Query/Key Normalization: Optional stability improvements
    - MLP Normalization: Optional feed-forward normalization
    - Scan Layers: Memory-efficient layer processing
    - Tensor Parallelism: Standard and expert-specific parallelism
    - Gradient Checkpointing: Memory-efficient training

Default Configuration (Xerxes2 Base):
    - Vocabulary size: 256,128
    - Hidden size: 4096
    - Intermediate size: 16384
    - Number of layers: 32
    - Attention heads: 32
    - Key-value heads: 8 (GQA)
    - Head dimension: 144
    - Max position embeddings: 16384
    - RMSNorm epsilon: 1e-6
    - RoPE: Configurable via RopeConfig
    - Softmax scale: 14.9666295471
    - MoE: Enhanced expert routing

Usage Example:
    ```python
    from easydel import Xerxes2Config, Xerxes2ForCausalLM
    from easydel.layers.rotary_embedding import RopeConfig
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration with enhanced MoE
    config = Xerxes2Config(
        vocab_size=256128,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=144,
        max_position_embeddings=16384,
        rope_config=RopeConfig(
            rope_type="default",
            rope_theta=10000.0,
        ),
        moe_config={
            "num_experts": 8,
            "top_k": 2,
            "load_balancing": True,
        },
    )

    # Create causal language model
    rngs = nn.Rngs(0)
    model = Xerxes2ForCausalLM(
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
    - Xerxes2Config: Configuration class for Xerxes2 models
    - Xerxes2Model: Base Xerxes2 transformer model
    - Xerxes2ForCausalLM: Xerxes2 with causal language modeling head
    - ExpertTensorParallel: Specialized sharding axes for MoE experts

Advanced Features:
    - Expert Tensor Parallelism: EPxTP sharding strategy for efficient expert distribution
    - RopeConfig Integration: Advanced RoPE configuration options
    - Enhanced Load Balancing: Improved expert utilization
    - Flexible Normalization: Multiple normalization strategies
    - Scan Support: Memory-efficient layer scanning

Note:
    Xerxes2 is the second generation research model with enhanced MoE capabilities
    and improved distributed training support. The Expert Tensor Parallelism (EPxTP)
    feature enables efficient scaling of MoE models across multiple devices.
"""

from .modeling_xerxes2 import Xerxes2ForCausalLM, Xerxes2Model
from .xerxes2_configuration import Xerxes2Config

__all__ = ("Xerxes2Config", "Xerxes2ForCausalLM", "Xerxes2Model")

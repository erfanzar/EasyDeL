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

"""Arctic model implementation for EasyDeL.

This module provides the Arctic model architecture, developed by Snowflake. Arctic
is a unique hybrid dense-MoE (Mixture-of-Experts) transformer that combines dense
transformer layers with sparse MoE layers in an interleaved pattern for optimal
efficiency and performance.

Architecture Highlights:
    - Hybrid Dense-MoE Architecture: Alternates between dense and MoE layers
    - MoE layer frequency: Every 2nd layer is MoE (configurable via moe_layer_frequency)
    - Sparse Expert Routing: 8 local experts with top-1 expert selection per token
    - Grouped Query Attention (GQA): Configurable query and key-value heads
    - Parallel Attention-MLP Residual: Optional parallel residual connections
    - RMSNorm: Root Mean Square Layer Normalization (epsilon=1e-5)
    - Rotary Position Embeddings (RoPE): With theta=1e6

Key Features:
    - Total experts per MoE layer: 8 (num_local_experts=8)
    - Active experts per token: 1 (num_experts_per_tok=1)
    - Hybrid architecture: Dense layers + periodic MoE layers
    - MoE layer frequency: Every 2nd layer by default (moe_layer_frequency=2)
    - Expert capacity: Configurable train/eval capacity factors
    - Token dropping: Optional token dropping for load balancing

MoE Configuration:
    - Router auxiliary loss: Weighted auxiliary loss (router_aux_loss_coef=0.001)
    - Capacity factors: Separate train (1.0) and eval (1.0) capacity factors
    - Token dropping: Enabled by default (moe_token_dropping=True)
    - Minimum capacity: Configurable minimum expert capacity (moe_min_capacity=0)
    - Expert tensor parallelism: Optional EPxTP sharding (enable_expert_tensor_parallelism)

Architecture Innovations:
    - Hybrid dense-sparse design: Reduces computational cost while maintaining quality
    - Interleaved MoE layers: Strategic placement for efficient computation
    - Top-1 routing: Single expert per token for maximum efficiency
    - Parallel residuals: Optional parallel attention and MLP paths

Technical Specifications:
    - Vocabulary size: 32,000 tokens (configurable)
    - Maximum sequence length: 4,096 tokens
    - Sliding window: Optional sliding window attention support
    - RoPE theta: 1,000,000 (optimized for long context)
    - RoPE scaling: Support for dynamic scaling strategies

Usage Example:
    ```python
    from easydel import ArcticConfig, ArcticForCausalLM
    from jax import random
    import jax.numpy as jnp

    # Initialize configuration
    config = ArcticConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        max_position_embeddings=4096,
        # MoE configuration
        num_local_experts=8,  # Total experts per MoE layer
        num_experts_per_tok=1,  # Top-1 routing
        moe_layer_frequency=2,  # Every 2nd layer is MoE
        router_aux_loss_coef=0.001,
        # Optional features
        parallel_attn_mlp_res=False,  # Parallel residuals
        sliding_window=None,  # Optional sliding window
    )

    # Create model for causal LM
    model = ArcticForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 512), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 512, 32000)

    # For sequence classification
    from easydel import ArcticForSequenceClassification

    classifier = ArcticForSequenceClassification(
        config=config,
        rngs=random.PRNGKey(0)
    )
    class_outputs = classifier(input_ids=input_ids)
    ```

Available Classes:
    - ArcticConfig: Configuration class for Arctic models
    - ArcticModel: Base transformer model without task-specific heads
    - ArcticForCausalLM: Arctic model with causal language modeling head
    - ArcticForSequenceClassification: Arctic model with classification head
"""

from .arctic_configuration import ArcticConfig
from .modeling_arctic import ArcticForCausalLM, ArcticForSequenceClassification, ArcticModel

__all__ = (
    "ArcticConfig",
    "ArcticForCausalLM",
    "ArcticForSequenceClassification",
    "ArcticModel",
)

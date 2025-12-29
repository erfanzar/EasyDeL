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

"""GLM-4-MoE: Large Language Model with Grouped Mixture of Experts.

This module implements the GLM-4-MoE architecture, a decoder-only transformer with grouped
mixture-of-experts (MoE) routing for efficient scaling. The model combines dense layers in
shallow positions with sparse MoE layers in deeper positions for optimal parameter efficiency.

Architecture Overview:
    GLM-4-MoE uses a hybrid dense-sparse architecture:

    1. Shallow Dense Layers:
       - The first `first_k_dense_replace` layers use standard dense FFN
       - Ensures stable gradient flow in early layers
       - Standard gate/up/down projection structure

    2. Deep MoE Layers:
       - Remaining layers use grouped mixture-of-experts routing
       - Each token routed to top-k experts from n_routed_experts pool
       - Shared experts process all tokens for common knowledge
       - Routed experts specialize for specific patterns

    3. Grouped Expert Routing:
       - Experts organized into n_group groups
       - Router selects topk_group groups, then top-k experts within groups
       - Reduces routing computation and improves load balancing
       - Routed expert outputs scaled by routed_scaling_factor

Key Features:
    - Grouped Query Attention (GQA): Reduced KV heads (num_key_value_heads) for efficiency
    - Partial RoPE: Only partial_rotary_factor (default 0.5) of head dimensions use RoPE
    - Query-Key Normalization: Optional use_qk_norm for training stability
    - Shared + Routed Experts: n_shared_experts always active, n_routed_experts conditionally routed
    - Top-K Probability Normalization: norm_topk_prob ensures routing weights sum to 1

MoE Configuration:
    - num_experts_per_tok: Number of routed experts selected per token (default: 8)
    - n_routed_experts: Total number of routed experts (default: 128)
    - n_shared_experts: Number of shared experts always active (default: 1)
    - n_group: Number of expert groups for hierarchical routing (default: 1)
    - topk_group: Number of groups to select (default: 1)
    - moe_intermediate_size: Hidden dimension in expert FFNs (default: 1408)
    - routed_scaling_factor: Scale factor for routed expert outputs (default: 1.0)

Usage Example:
    ```python
    from easydel import Glm4MoeForCausalLM, Glm4MoeConfig
    import jax.numpy as jnp

    # Initialize model with MoE configuration
    config = Glm4MoeConfig(
        hidden_size=4096,
        num_hidden_layers=46,
        num_attention_heads=96,
        num_key_value_heads=8,  # GQA
        n_routed_experts=128,
        num_experts_per_tok=8,
        n_shared_experts=1,
        first_k_dense_replace=1,  # First layer is dense
    )

    model = Glm4MoeForCausalLM(config, rngs=nnx.Rngs(0))

    # Forward pass
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids, output_router_logits=True)

    logits = outputs.logits  # (batch, seq_len, vocab_size)
    router_logits = outputs.router_logits  # For auxiliary load balancing loss
    ```

Configuration:
    - Glm4MoeConfig: Main configuration with MoE and attention parameters

Models:
    - Glm4MoeModel: Base transformer model with MoE layers
    - Glm4MoeForCausalLM: Causal language modeling with LM head
    - Glm4MoeForSequenceClassification: Sequence classification head
"""

from .glm4_moe_configuration import Glm4MoeConfig
from .modeling_glm4_moe import Glm4MoeForCausalLM, Glm4MoeForSequenceClassification, Glm4MoeModel

__all__ = ("Glm4MoeConfig", "Glm4MoeForCausalLM", "Glm4MoeForSequenceClassification", "Glm4MoeModel")

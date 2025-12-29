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

"""Phi-MoE (Mixture of Experts) model implementation for EasyDeL.

Phi-MoE is a sparse mixture-of-experts transformer that achieves excellent performance
while maintaining computational efficiency through conditional computation. Each layer
contains multiple expert networks, with only a subset activated per token.

Architecture:
    - Decoder-only transformer with RMSNorm
    - Sparse Mixture-of-Experts (MoE) layers
    - Grouped Query Attention (GQA) for efficiency
    - Rotary Position Embeddings (RoPE) with LongRoPE scaling
    - Optional sliding window attention

Key Features:
    - Sparse Expert Routing: Only top-k experts activated per token (default k=2)
    - Load Balancing: Auxiliary loss ensures even expert utilization
    - Router Jitter Noise: Random noise for exploration during training
    - LongRoPE Scaling: Extended context through rope_scaling configuration
    - Efficient Inference: Sparse activation reduces compute vs dense models

Usage Example:
    ```python
    import jax
    from easydel import PhiMoeConfig, PhiMoeForCausalLM
    from flax import nnx as nn

    # Initialize configuration
    config = PhiMoeConfig(
        vocab_size=32064,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA for efficiency
        num_local_experts=16,   # Total number of experts
        num_experts_per_tok=2,  # Activate 2 experts per token
        intermediate_size=6400, # Per-expert MLP size
        router_aux_loss_coef=0.001,  # Load balancing coefficient
        max_position_embeddings=131072,
    )

    # Create model for causal LM
    model = PhiMoeForCausalLM(
        config=config,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Prepare inputs
    input_ids = jax.numpy.array([[1, 2, 3, 4, 5]])

    # Forward pass
    outputs = model(input_ids=input_ids, output_router_logits=True)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    router_logits = outputs.router_logits  # Expert routing decisions
    ```

MoE Concepts:
    - Expert Capacity: Maximum tokens each expert can process
    - Routing: Learned gating network selects top-k experts per token
    - Auxiliary Loss: Encourages balanced expert utilization
    - Jitter Noise: Adds randomness to routing for better exploration

Available Models:
    - microsoft/Phi-3.5-MoE-instruct: 16x3.8B MoE (6.6B active parameters)

Classes:
    - PhiMoeConfig: Configuration class with MoE-specific settings
    - PhiMoeModel: Base model without task-specific head
    - PhiMoeForCausalLM: Model with language modeling head
"""

from .modeling_phimoe import PhiMoeForCausalLM, PhiMoeModel
from .phimoe_configuration import PhiMoeConfig

__all__ = "PhiMoeConfig", "PhiMoeForCausalLM", "PhiMoeModel"

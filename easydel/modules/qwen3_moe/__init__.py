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

"""Qwen3-MoE: Advanced Sparse Mixture-of-Experts Transformer for EasyDeL.

This module implements the Qwen3-MoE architecture, an evolution of Qwen2-MoE with
enhanced sparse mixture-of-experts capabilities. Qwen3-MoE builds on the proven
MoE framework while introducing architectural improvements for better performance
and efficiency in conditional computation.

Architecture Overview:
    - **Decoder-only transformer**: Autoregressive language model with causal attention
    - **Sparse MoE layers**: Strategic placement of expert layers controlled by
      `decoder_sparse_step` for optimal compute-performance balance
    - **Scaled expert pool**: Supports up to 128 experts (default) with top-8 routing,
      enabling massive model capacity with controlled computation
    - **Enhanced routing**: Improved top-k routing mechanism with optional probability
      normalization for better expert utilization
    - **Sliding window attention**: Optional sliding window for efficient long-range modeling
    - **Grouped Query Attention (GQA)**: Efficient attention with separate key-value heads
    - **Q/K normalization**: Optional query and key normalization for training stability

MoE Configuration:
    - **num_experts** (default: 128): Total number of expert networks per MoE layer.
      Qwen3-MoE scales to larger expert pools compared to Qwen2-MoE (60 experts).
    - **num_experts_per_tok** (default: 8): Number of experts activated per token.
      Higher than Qwen2-MoE (4) for richer expert combinations.
    - **moe_intermediate_size** (default: 768): Intermediate dimension for expert MLPs.
      Smaller per-expert capacity balanced by more experts.
    - **decoder_sparse_step** (default: 1): Frequency of MoE layers in the model.
    - **norm_topk_prob** (default: False): Whether to normalize routing probabilities.
    - **router_aux_loss_coef** (default: 0.001): Coefficient for load balancing loss.

Key Improvements over Qwen2-MoE:
    - **Larger expert pools**: 128 vs 60 experts for greater model capacity
    - **Higher expert activation**: 8 vs 4 experts per token for richer representations
    - **Attention bias option**: Configurable attention bias for architectural flexibility
    - **Enhanced RoPE**: Improved rotary position encoding with scaling support
    - **Q/K norm support**: Optional query-key normalization for stable training

Available Models:
    - **Qwen3MoeConfig**: Configuration class with all hyperparameters
    - **Qwen3MoeModel**: Base transformer model without task-specific head
    - **Qwen3MoeForCausalLM**: Model with language modeling head for text generation
    - **Qwen3MoeForSequenceClassification**: Model with classification head

Example Usage:
    ```python
    from easydel import Qwen3MoeConfig, Qwen3MoeForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration with Qwen3-MoE settings
    config = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_experts=128,            # Larger expert pool
        num_experts_per_tok=8,      # More experts per token
        moe_intermediate_size=768,
        decoder_sparse_step=1,      # MoE every layer
        norm_topk_prob=False,
    )

    # Create model
    rngs = nn.Rngs(0)
    model = Qwen3MoeForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids, output_router_logits=True)

    logits = outputs.logits              # (batch, seq_len, vocab_size)
    router_logits = outputs.router_logits  # Routing information per layer
    aux_loss = outputs.aux_loss          # Load balancing loss
    ```

Performance Notes:
    - Qwen3-MoE achieves better parameter efficiency through larger expert pools
    - Higher expert activation (8 vs 4) provides richer token representations
    - Sparse activation maintains computational efficiency despite more experts
    - Auxiliary loss crucial for preventing expert collapse and ensuring load balance

References:
    - Qwen3 technical report: https://qwenlm.github.io/
"""

from .modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeForSequenceClassification, Qwen3MoeModel
from .qwen3_moe_configuration import Qwen3MoeConfig

__all__ = (
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForSequenceClassification",
    "Qwen3MoeModel",
)

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

"""GPT-OSS (Mixture of Experts) model implementation for EasyDeL.

GPT-OSS is a large-scale sparse Mixture of Experts language model that efficiently
scales to massive parameter counts while maintaining computational efficiency through
sparse expert routing and sliding window attention.

Architecture:
    - Decoder-only transformer with RMSNorm
    - Sparse Mixture of Experts (128 experts, activate 4 per token)
    - Alternating sliding window and full attention layers
    - Grouped Query Attention (GQA) for efficiency
    - Rotary Position Embeddings (RoPE) with YARN scaling

Key Features:
    - Massive Scale: 128 experts with sparse activation (activate 4/128)
    - Efficient Attention: Sliding window (128 tokens) alternates with full attention
    - YARN RoPE Scaling: Extended context via factor 32 scaling
    - Load Balancing: High coefficient (0.9) for even expert utilization
    - Long Context: Supports up to 131K tokens with RoPE scaling

Usage Example:
    ```python
    import jax
    from easydel import GptOssConfig, GptOssForCausalLM
    from flax import nnx as nn

    # Initialize configuration
    config = GptOssConfig(
        vocab_size=201088,
        hidden_size=2880,
        num_hidden_layers=36,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA
        num_local_experts=128,  # Total experts
        num_experts_per_tok=4,  # Sparse activation
        sliding_window=128,     # Local attention window
        max_position_embeddings=131072,
        rope_scaling={  # YARN scaling
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )

    # Create model for causal LM
    model = GptOssForCausalLM(
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

MoE Architecture Details:
    - 128 expert networks per MoE layer
    - Top-4 expert routing per token
    - Auxiliary loss coefficient: 0.9 (strong load balancing)
    - Alternating layer pattern: sliding window / full attention

Attention Pattern:
    - Even layers: Sliding window attention (128 token window)
    - Odd layers: Full attention
    - This pattern balances efficiency with global context

Available Models:
    - GPT-OSS variants optimized for sparse computation

Classes:
    - GptOssConfig: Configuration class with MoE and attention settings
    - GptOssModel: Base model without task-specific head
    - GptOssForCausalLM: Model with language modeling head
    - GptOssForSequenceClassification: Model with classification head
"""

from .gpt_oss_configuration import GptOssConfig
from .modeling_gpt_oss import GptOssForCausalLM, GptOssForSequenceClassification, GptOssModel

__all__ = ("GptOssConfig", "GptOssForCausalLM", "GptOssForSequenceClassification", "GptOssModel")

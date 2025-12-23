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

"""Mixtral Mixture-of-Experts (MoE) model implementation for EasyDeL.

Mixtral is a sparse mixture-of-experts decoder-only language model that achieves strong
performance while maintaining inference efficiency through conditional computation. Each
token is dynamically routed to a subset of expert feed-forward networks, allowing the
model to have a large parameter count while keeping FLOPs per token manageable.

Key architectural features:

- Sparse Mixture-of-Experts (SMoE): Each Mixtral decoder layer contains `num_local_experts`
  (default 8) independent feed-forward expert networks. A learned router network selects
  `num_experts_per_tok` (default 2) experts per token, enabling sparse activation while
  maintaining model capacity.

- Top-K Gating: The router produces logits for all experts, then applies top-k selection
  to choose the most relevant experts for each token. Selected experts' outputs are
  combined via weighted sum using softmax-normalized router scores.

- Router Auxiliary Loss: To encourage balanced expert utilization and prevent expert
  collapse, an auxiliary load-balancing loss (scaled by `router_aux_loss_coef`) penalizes
  uneven expert selection. Controlled via `output_router_logits`.

- Sliding Window Attention: Uses `sliding_window` (default 4096) for local attention
  in lower layers while retaining full attention capability, balancing long-range context
  with computational efficiency.

- Grouped-Query Attention (GQA): Employs fewer key/value heads (`num_key_value_heads=8`)
  than query heads (`num_attention_heads=32`) to reduce memory bandwidth and KV cache size.

- RMSNorm: Uses Root Mean Square normalization for better training stability.

- Expert Tensor Parallelism (EPxTP): Supports sharding experts across devices for
  distributed training and inference.

Usage Example:
    ```python
    from easydel.modules.mixtral import MixtralConfig, MixtralForCausalLM
    import jax
    from flax import nnx as nn

    # Configure Mixtral-8x7B style model
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_local_experts=8,      # 8 experts per layer
        num_experts_per_tok=2,    # Top-2 routing
        sliding_window=4096,
        output_router_logits=True,  # Enable load balancing loss
        router_aux_loss_coef=0.001,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = MixtralForCausalLM(config, rngs=rngs)

    # Generate text with MoE routing
    input_ids = jax.numpy.array([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # router_logits available if output_router_logits=True
    ```
"""

from .mixtral_configuration import MixtralConfig
from .modeling_mixtral import MixtralForCausalLM, MixtralForSequenceClassification, MixtralModel

__all__ = (
    "MixtralConfig",
    "MixtralForCausalLM",
    "MixtralForSequenceClassification",
    "MixtralModel",
)

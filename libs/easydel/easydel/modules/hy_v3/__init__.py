# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""HYV3: Tencent Hunyuan V3 mixture-of-experts transformer for EasyDeL.

HYV3 (HF ``model_type="hy_v3"``) is a decoder-only transformer combining:

- **Qwen3-style attention**: grouped-query attention (64 query / 8 KV heads
  by default, explicit ``head_dim`` 128) with per-head RMSNorm applied to
  queries and keys before full-dim rotary embeddings; rotary base defaults
  to ``11_158_840.0``.
- **Per-layer MLP schedule** (``mlp_layer_types``): the first layer is a
  dense SwiGLU MLP (``intermediate_size`` 13312) and the remaining layers
  are sparse MoE by default.
- **Sigmoid MoE routing**: 192 routed experts, top-8 per token. Experts are
  selected on fp32 ``sigmoid(logits) + e_score_correction_bias`` (a
  checkpoint-loaded per-expert fp32 bias), weighted by the raw sigmoid
  scores normalized to sum 1 and scaled by ``router_scaling_factor``
  (2.826). One always-on shared SwiGLU expert
  (``moe_intermediate_size * num_shared_experts``) is added — in float32
  when ``enable_moe_fp32_combine`` (default) is set.

Available classes:
    - **HyV3Config**: Configuration class with all hyperparameters.
    - **HyV3Model**: Base transformer model without a task-specific head.
    - **HyV3ForCausalLM**: Model with language modeling head for generation.

Example usage:
    ```python
    import easydel as ed
    import jax.numpy as jnp
    import spectrax as spx

    config = ed.HyV3Config(
        vocab_size=120832,
        hidden_size=4096,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_experts=192,
        num_experts_per_tok=8,
    )
    model = ed.HyV3ForCausalLM(config, rngs=spx.Rngs(0))
    outputs = model(input_ids=jnp.array([[1, 2, 3, 4, 5]]))
    ```

References:
    - Hunyuan: https://hunyuan.tencent.com/
"""

from .hy_v3_configuration import HyV3Config
from .modeling_hy_v3 import HyV3ForCausalLM, HyV3Model

__all__ = (
    "HyV3Config",
    "HyV3ForCausalLM",
    "HyV3Model",
)

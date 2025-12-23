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

"""Cohere Command model implementation for EasyDeL.

Cohere Command is a family of instruction-following language models designed for chat,
retrieval-augmented generation (RAG), and tool use. The architecture incorporates several
unique design choices optimized for commercial applications:

- Logit Scaling: Applies a learnable `logit_scale` parameter (default 0.0625) to final
  layer outputs before computing next-token probabilities, providing finer control over
  prediction sharpness and calibration.

- RMSNorm (Root Mean Square Normalization): Uses RMS normalization instead of LayerNorm
  throughout the model, normalizing activations by their RMS without mean centering,
  which is more stable and efficient for large-scale training.

- Optional Q/K Normalization: When `use_qk_norm=True`, applies RMSNorm to query and key
  projections before attention computation, improving training stability and model quality
  especially at larger scales.

- Gated Feed-Forward Networks: Uses SwiGLU-style gated activations in the MLP layers
  (gate_proj, up_proj, down_proj) with SiLU activation, providing better gradient flow
  and representation capacity than standard FFNs.

- Grouped-Query Attention (GQA): Supports independent configuration of query and key/value
  head counts via `num_key_value_heads` for memory-efficient inference.

- Standard RoPE: Uses full rotary position embeddings (unlike Phi's partial RoPE) with
  configurable `rope_theta` for position encoding.

Usage Example:
    ```python
    from easydel.modules.cohere import CohereConfig, CohereForCausalLM
    import jax
    from flax import nnx as nn

    # Configure Cohere Command-R style model
    config = CohereConfig(
        vocab_size=256000,
        hidden_size=8192,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA for efficiency
        use_qk_norm=True,
        logit_scale=0.0625,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = CohereForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jax.numpy.array([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Already scaled by logit_scale
    ```
"""

from .cohere_configuration import CohereConfig
from .modeling_cohere import CohereForCausalLM, CohereForSequenceClassification, CohereModel

__all__ = (
    "CohereConfig",
    "CohereForCausalLM",
    "CohereForSequenceClassification",
    "CohereModel",
)

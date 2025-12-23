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

"""OLMo model implementation for EasyDeL.

OLMo (Open Language Model) is a fully open-source large language model developed
by the Allen Institute for AI (AI2). Unlike many other LLMs, OLMo provides complete
transparency with open access to training data, code, model weights, and evaluation
tools, making it a valuable resource for research and development.

Architecture Overview
--------------------
OLMo is a decoder-only transformer architecture with several distinctive design choices:

1. **Non-Parametric Layer Norm**: Uses a simplified layer normalization without learnable
   affine parameters (no weight/bias), which reduces parameters while maintaining
   performance. This is applied both before attention and before the MLP.

2. **SwiGLU Activation**: The feed-forward layers use SwiGLU (Swish-Gated Linear Units),
   combining element-wise gating with the Swish activation function for improved
   expressiveness.

3. **Rotary Position Embeddings (RoPE)**: Encodes positional information directly in
   the attention mechanism rather than using additive position embeddings, enabling
   better length extrapolation.

4. **Grouped Query Attention (Optional)**: Supports GQA to reduce memory requirements
   during inference by sharing key-value heads across multiple query heads.

5. **No Bias Terms**: Following modern practices, OLMo omits bias terms in linear
   layers to reduce parameters and improve training efficiency.

Key Components
-------------
- **OlmoConfig**: Configuration class specifying all model hyperparameters including
  hidden size, number of layers, attention heads, activation functions, and
  normalization settings.

- **OlmoModel**: The base transformer model outputting contextualized hidden states.
  Consists of token embeddings, stacked transformer blocks, and final layer norm.

- **OlmoForCausalLM**: Causal language modeling variant with a linear language modeling
  head for next-token prediction and text generation.

- **OlmoForSequenceClassification**: Sequence classification variant with a
  classification head for tasks like sentiment analysis or text categorization.

Design Philosophy
----------------
OLMo emphasizes transparency and reproducibility:

- **Open Training Data**: Complete training dataset (Dolma) is publicly available
- **Open Training Code**: Full training infrastructure and scripts released
- **Open Evaluation**: Comprehensive evaluation suite and results shared
- **Intermediate Checkpoints**: Training checkpoints at various stages available
- **Ablation Studies**: Detailed design choice experiments documented

This openness enables researchers to understand model behavior, reproduce results,
and build upon the foundation.

Model Variants
-------------
OLMo comes in multiple sizes:
- **OLMo-1B**: 1.2 billion parameters
- **OLMo-7B**: 7 billion parameters
- **OLMo-13B**: 13 billion parameters (in development)

All variants share the same architecture but differ in dimensions and layer counts.

Usage Example
------------
```python
from easydel import OlmoConfig, OlmoForCausalLM
import jax.numpy as jnp
from flax import nnx as nn

# Initialize OLMo-7B configuration
config = OlmoConfig(
    vocab_size=50280,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=2048,
    rope_theta=10000.0,
)

# Create model instance
model = OlmoForCausalLM(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    rngs=nn.Rngs(0),
)

# Forward pass for training
input_ids = jnp.array([[1, 2, 3, 4, 5]])
outputs = model(input_ids=input_ids)
logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

# Generate text with caching
cache = None
for step in range(20):
    outputs = model(
        input_ids=input_ids[:, -1:],
        past_key_values=cache,
    )
    cache = outputs.past_key_values
    next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1)
    input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
```

Training Details
---------------
- **Data**: Trained on Dolma, a diverse 3T token dataset
- **Context**: 2048 token context length (extendable with RoPE)
- **Precision**: Mixed precision training (bf16)
- **Optimizer**: AdamW with specific learning rate schedules
- **Efficiency**: Optimized for training on multi-GPU clusters

References
---------
- Paper: "OLMo: Accelerating the Science of Language Models" (AI2, 2024)
- GitHub: https://github.com/allenai/OLMo
- Model Hub: https://huggingface.co/allenai/OLMo-7B
- Dataset: https://huggingface.co/datasets/allenai/dolma
"""

from .modeling_olmo import OlmoForCausalLM, OlmoForSequenceClassification, OlmoModel
from .olmo_configuration import OlmoConfig

__all__ = (
    "OlmoConfig",
    "OlmoForCausalLM",
    "OlmoForSequenceClassification",
    "OlmoModel",
)

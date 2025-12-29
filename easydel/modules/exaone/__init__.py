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

"""EXAONE model implementation for EasyDeL.

EXAONE is a series of large language models developed by LG AI Research, designed
to provide strong multilingual capabilities with particular emphasis on Korean and
English. The models combine advanced transformer architectures with efficient training
techniques to achieve competitive performance across various language understanding
and generation tasks.

Architecture Overview
--------------------
EXAONE follows a decoder-only transformer architecture with modern design choices:

1. **Standard Transformer Blocks**: Uses self-attention and feed-forward layers with
   residual connections and layer normalization. The architecture follows proven
   patterns from GPT-style models.

2. **Rotary Position Embeddings (RoPE)**: Encodes positional information through
   rotary embeddings applied to query and key projections, enabling better length
   generalization than absolute position embeddings.

3. **Multi-Head Attention**: Standard multi-head attention mechanism with optional
   Grouped Query Attention (GQA) support for improved memory efficiency during
   inference by reducing the number of key-value heads.

4. **Feed-Forward Networks**: Uses gated linear units (GLU) or SwiGLU activations
   in the MLP layers for improved expressiveness and performance.

5. **Layer Normalization**: Applies RMSNorm (Root Mean Square normalization) which
   is computationally simpler than standard LayerNorm while providing similar benefits.

Key Components
-------------
- **ExaoneConfig**: Configuration class that defines all model hyperparameters including
  dimensions, layer counts, attention settings, and architectural choices.

- **ExaoneModel**: The base transformer model that outputs contextualized hidden states.
  Includes token embeddings, stacked transformer blocks, and final normalization.

- **ExaoneForCausalLM**: Causal language modeling variant with a language modeling head
  for autoregressive text generation and next-token prediction tasks.

- **ExaoneForSequenceClassification**: Sequence classification variant with a pooling
  and classification head for tasks like sentiment analysis, topic classification, or
  text categorization.

Model Characteristics
--------------------
EXAONE models are distinguished by:

1. **Multilingual Focus**: Trained on diverse multilingual data with strong emphasis
   on Korean-English bilingual capabilities, making them particularly effective for
   cross-lingual tasks.

2. **Efficient Architecture**: Optimized for both training efficiency and inference
   speed through careful architectural choices and training techniques.

3. **Domain Adaptation**: Can be effectively fine-tuned for specific domains including
   technical, business, and conversational applications.

4. **Instruction Following**: Later versions include instruction-tuning for improved
   alignment with user intentions and task-specific performance.

Model Variants
-------------
EXAONE comes in multiple sizes to suit different use cases:
- **EXAONE-3.0 2.4B**: Compact model for resource-constrained deployments
- **EXAONE-3.0 7.8B**: Medium-sized model balancing performance and efficiency
- **EXAONE-3.0 32B**: Large model for maximum capability

Usage Example
------------
```python
from easydel import ExaoneConfig, ExaoneForCausalLM
import jax.numpy as jnp
from flax import nnx as nn

# Initialize EXAONE configuration
config = ExaoneConfig(
    vocab_size=102400,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,  # GQA for efficiency
    max_position_embeddings=4096,
    rope_theta=1000000.0,
)

# Create model instance
model = ExaoneForCausalLM(
    config=config,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    rngs=nn.Rngs(0),
)

# Forward pass for training
input_ids = jnp.array([[1, 2, 3, 4, 5]])
outputs = model(input_ids=input_ids)
logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

# Autoregressive generation
cache = None
generated_tokens = []
for step in range(50):
    outputs = model(
        input_ids=input_ids[:, -1:] if cache else input_ids,
        past_key_values=cache,
    )
    cache = outputs.past_key_values
    next_token = jnp.argmax(outputs.logits[:, -1, :], axis=-1)
    generated_tokens.append(next_token)
    input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
```

Training Approach
----------------
EXAONE models are trained using:
- **Large-Scale Pretraining**: Trained on diverse web text, books, and multilingual data
- **Bilingual Optimization**: Special focus on Korean-English parallel data
- **Instruction Tuning**: Fine-tuned on instruction-following datasets
- **Safety Alignment**: Additional training for safe and helpful responses

Applications
-----------
- **Multilingual NLP**: Korean-English translation and cross-lingual understanding
- **Text Generation**: Creative writing, content creation, code generation
- **Question Answering**: Information retrieval and knowledge-based QA
- **Conversational AI**: Chatbots and dialogue systems
- **Domain-Specific Tasks**: Finance, legal, medical text processing

References
---------
- LG AI Research: https://www.lgresearch.ai/
- Model Hub: https://huggingface.co/LGAI-EXAONE
"""

from .exaone_configuration import ExaoneConfig
from .modeling_exaone import ExaoneForCausalLM, ExaoneForSequenceClassification, ExaoneModel

__all__ = (
    "ExaoneConfig",
    "ExaoneForCausalLM",
    "ExaoneForSequenceClassification",
    "ExaoneModel",
)

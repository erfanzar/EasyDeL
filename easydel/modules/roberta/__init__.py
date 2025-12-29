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

"""RoBERTa model implementation for EasyDeL.

RoBERTa Architecture
====================
RoBERTa (Robustly Optimized BERT Pretraining Approach) is Meta AI's improved variant
of BERT that achieves state-of-the-art results through better training procedures
rather than architectural changes.

Key Features
------------
- **Dynamic Masking**: Unlike BERT's static masking, RoBERTa generates new masking
  patterns each time a sequence is fed to the model during training.
- **No Next Sentence Prediction (NSP)**: Removes BERT's NSP objective, using only
  masked language modeling (MLM) for pretraining.
- **Larger Batches and Learning Rate**: Trained with much larger batch sizes (8K)
  and higher learning rates than BERT.
- **Byte-Pair Encoding (BPE)**: Uses BPE tokenization instead of WordPiece.
- **Longer Training**: Trained on more data (160GB) for more steps than BERT.

Model Architecture
------------------
Identical to BERT's transformer encoder architecture:
- Token + position + (optional) token type embeddings
- N bidirectional transformer encoder layers with:
  - Multi-head self-attention
  - Feed-forward network
  - Layer normalization and residual connections
- Pooling layer for sequence-level tasks

Training Improvements
---------------------
- Dynamic masking during training (15% of tokens)
- No NSP objective (full sentences only)
- Larger mini-batches (8K examples)
- Byte-level BPE with 50K vocabulary
- Trained on 160GB of text (vs BERT's 16GB)

Usage Example
-------------
```python
from easydel import RobertaConfig, RobertaForSequenceClassification
from flax import nnx as nn
import jax.numpy as jnp

# Create configuration
config = RobertaConfig(
    vocab_size=50265,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=514,
)

# Initialize model for classification
model = RobertaForSequenceClassification(
    config=config,
    num_labels=2,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    rngs=nn.Rngs(0),
)

# Classify text
outputs = model(
    input_ids=jnp.array([[0, 1, 2, 3, 2]]),
    attention_mask=jnp.ones((1, 5)),
)
```

Available Models
----------------
- RobertaConfig: Configuration class for RoBERTa models
- RobertaModel: Base RoBERTa model outputting hidden states
- RobertaForCausalLM: RoBERTa for causal language modeling
- RobertaForSequenceClassification: RoBERTa for text classification
- RobertaForTokenClassification: RoBERTa for token-level tasks (NER, POS tagging)
- RobertaForQuestionAnswering: RoBERTa for extractive QA
- RobertaForMultipleChoice: RoBERTa for multiple choice tasks
"""

from .modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .roberta_configuration import RobertaConfig

__all__ = (
    "RobertaConfig",
    "RobertaForCausalLM",
    "RobertaForMultipleChoice",
    "RobertaForQuestionAnswering",
    "RobertaForSequenceClassification",
    "RobertaForTokenClassification",
    "RobertaModel",
)

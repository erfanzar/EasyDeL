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

"""GPT-2 (Generative Pre-trained Transformer 2) model implementation for EasyDeL.

GPT-2 is OpenAI's influential autoregressive language model that demonstrated the
effectiveness of unsupervised pre-training on large text corpora. It established many
architectural conventions that became standard in decoder-only language models.

Key architectural features:

- Absolute Learned Position Embeddings: Unlike modern models that use RoPE or ALiBi,
  GPT-2 uses traditional learned position embeddings that are added to token embeddings.
  Position embeddings are limited by `max_position_embeddings` (default 1024).

- Pre-Norm Architecture: Applies LayerNorm before attention and MLP blocks rather than
  after, which was an early design choice for training stability. This differs from the
  post-norm architecture used in the original Transformer.

- Standard Multi-Head Attention: Uses classic multi-head self-attention without grouped
  queries, multi-query, or sliding windows. All heads have independent key/value projections.

- GELU Activation: Uses the Gaussian Error Linear Unit (GELU) activation function in
  feed-forward layers, specifically the `gelu_new` approximation for efficiency.

- Embedding Tying: Optionally ties input token embeddings with output LM head weights
  via `tie_word_embeddings`, reducing parameters and improving low-resource performance.

- Byte-Pair Encoding (BPE): Originally designed for BPE tokenization with vocab_size
  of 50257, though this implementation supports custom vocabulary sizes.

Usage Example:
    ```python
    from easydel.modules.gpt2 import GPT2Config, GPT2LMHeadModel
    import jax
    from flax import nnx as nn

    # Configure GPT-2 Medium
    config = GPT2Config(
        vocab_size=50257,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=1024,
        use_cache=True,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = GPT2LMHeadModel(config, rngs=rngs)

    # Generate text
    input_ids = jax.numpy.array([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```
"""

from .gpt2_configuration import GPT2Config
from .modeling_gpt2 import GPT2LMHeadModel, GPT2Model

__all__ = "GPT2Config", "GPT2LMHeadModel", "GPT2Model"

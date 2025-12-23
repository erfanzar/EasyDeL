# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi) and @dvruette.
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

"""GIDD (Generative Infilling Diffusion Decoder) model implementation for EasyDeL.

GIDD is a novel diffusion-based language model that uses discrete diffusion processes
for text generation. Unlike autoregressive models, GIDD generates tokens through
iterative denoising, enabling parallel generation and improved sampling diversity.

Architecture:
    - Decoder-only transformer with RMSNorm
    - Query-Key normalization for stable attention
    - Squared ReLU activation in MLP (ReLU^2)
    - Rotary Position Embeddings (RoPE)
    - Diffusion-specific noise masking mechanism

Key Features:
    - Discrete Diffusion Process: Generates text through iterative denoising
    - Noise Masking: Special attention masks for handling noised tokens
    - QK Normalization: Optional learnable scale for query-key normalization
    - Residual Scaling: Scaled residual connections (default 4.0x)
    - Flexible Initialization: Separate scales for embeddings, layers, and head

Usage Example:
    ```python
    import jax
    import jax.numpy as jnp
    from easydel import GiddConfig, GiddForDiffusionLM
    from flax import nnx as nn

    # Initialize configuration
    config = GiddConfig(
        vocab_size=131072,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=1024,
        use_qk_norm=True,  # Enable query-key normalization
        resid_scale=4.0,   # Residual connection scaling
    )

    # Create model for diffusion LM
    model = GiddForDiffusionLM(
        config=config,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Prepare inputs for diffusion training
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    noise_mask = jnp.array([[1, 0, 1, 0, 1]])  # 1 = noised, 0 = clean

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        noise_mask=noise_mask,
    )
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    ```

Diffusion Process:
    - Training: Tokens are randomly masked (noised) and model predicts originals
    - Generation: Start with all masked tokens, iteratively denoise to final text
    - Noise Schedule: Controls which tokens are noised at each diffusion step

Available Models:
    - GIDD variants with diffusion-based generation
    - Optimized for infilling and parallel generation tasks

Classes:
    - GiddConfig: Configuration class with diffusion-specific settings
    - GiddModel: Base transformer model without task-specific head
    - GiddForDiffusionLM: Model with diffusion language modeling head
"""

from .gidd_configuration import GiddConfig
from .modeling_gidd import GiddForDiffusionLM, GiddModel

__all__ = ("GiddConfig", "GiddForDiffusionLM", "GiddModel")

"""FalconH1: Hybrid Architecture with Attention and State-Space Models.

This module implements the FalconH1 architecture, developed by Technology Innovation Institute (TII).
FalconH1 represents a hybrid approach combining multi-head attention mechanisms with state-space
models (SSMs) to achieve both strong performance and computational efficiency.

Architecture Overview:
    - **Hybrid Attention-SSM Design**: Strategically combines transformer attention layers with
      efficient state-space model layers for optimal performance-efficiency tradeoffs
    - **Multi-query Attention (MQA)**: Uses multiple query heads with shared key-value heads
      for efficient attention computation
    - **Rotary Position Embeddings (RoPE)**: Encodes position information through rotation
      in the complex plane for better length extrapolation
    - **Parallel block design**: Attention and feedforward layers can be computed in parallel
      in some configurations for improved throughput
    - **LayerNorm placement**: Strategic normalization for training stability

Key Components:
    - **Attention blocks**: Multi-head or multi-query attention with RoPE
    - **SSM blocks**: State-space model layers for linear-time sequence processing
    - **Feedforward networks**: Dense layers with SiLU/GELU activation
    - **Hybrid layer mixing**: Configurable pattern of attention vs SSM layers
    - **Normalization**: LayerNorm or RMSNorm depending on configuration

Configuration:
    - **vocab_size**: Size of vocabulary
    - **hidden_size**: Dimension of hidden states and embeddings
    - **num_hidden_layers**: Total number of transformer layers
    - **num_attention_heads**: Number of attention heads
    - **num_key_value_heads**: Number of key-value heads for MQA/GQA
    - **intermediate_size**: Dimension of feedforward intermediate layer
    - **hidden_act**: Activation function (silu, gelu, etc.)
    - **max_position_embeddings**: Maximum sequence length
    - **rope_theta**: Base for RoPE frequency calculation

Available Models:
    - **FalconH1Config**: Configuration class for model hyperparameters
    - **FalconH1Model**: Base model without task-specific head
    - **FalconH1ForCausalLM**: Model with language modeling head for text generation

Example Usage:
    ```python
    from easydel import FalconH1Config, FalconH1ForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Create configuration
    config = FalconH1Config(
        vocab_size=65024,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Multi-query attention
        intermediate_size=16384,
        max_position_embeddings=8192,
    )

    # Initialize model
    rngs = nn.Rngs(0)
    model = FalconH1ForCausalLM(config, rngs=rngs)

    # Generate text
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    ```

Key Features:
    - **Efficient attention**: MQA reduces KV cache size for faster inference
    - **Hybrid architecture**: Combines benefits of attention and SSMs
    - **Long context support**: RoPE enables good length extrapolation
    - **Parallel blocks**: Optional parallel computation of attention and FFN
    - **Flexible configuration**: Adjustable attention/SSM layer ratios

Performance Characteristics:
    - **Training efficiency**: Hybrid design reduces training cost
    - **Inference speed**: MQA and SSM layers accelerate generation
    - **Memory usage**: Reduced KV cache from multi-query attention
    - **Scalability**: Efficient scaling to larger model sizes

HuggingFace References:
    - Configuration: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_h1/configuration_falcon_h1.py
    - Modeling: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_h1/modeling_falcon_h1.py

References:
    - Falcon models: https://falconllm.tii.ae/
    - Multi-Query Attention: https://arxiv.org/abs/1911.02150
"""

from .falcon_h1_configuration import FalconH1Config
from .modeling_falcon_h1 import FalconH1ForCausalLM, FalconH1Model

__all__ = ("FalconH1Config", "FalconH1ForCausalLM", "FalconH1Model")

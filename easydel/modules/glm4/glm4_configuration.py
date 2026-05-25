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

"""Configuration class for the GLM-4 decoder-only language model.

Defines :class:`Glm4Config`, the EasyDeL configuration object for THUDM's
GLM-4 family. The defaults match the GLM-4-9B style: 4096 hidden, 40 layers,
GQA with 32 query / 2 key-value heads, head dim 128, partial RoPE
(``partial_rotary_factor=0.5``), and a 151,552-token multilingual vocabulary
with a 131,072-token context window.
"""

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("glm4")
class Glm4Config(EasyDeLBaseConfig):
    """Configuration for the GLM-4 decoder-only language model.

    Inherits from :class:`EasyDeLBaseConfig`. Defaults reproduce the
    GLM-4-9B preset (THUDM/GLM-4-9B-0414): 4096 hidden, 40 layers, GQA with
    32 query / 2 KV heads, fixed head dim of 128, partial RoPE
    (``partial_rotary_factor=0.5``), 151,552-token multilingual
    vocabulary, and a 131,072-token context window.

    GLM-4 keeps the GLM attention layout (Q/K/V projections may carry
    biases while the output projection is always biasless) and adds dual
    post-normalization in the decoder layer (see :class:`Glm4DecoderLayer`).

    Attributes:
        vocab_size (int): Token vocabulary size.
        hidden_size (int): Residual/hidden stream dimension.
        intermediate_size (int): MLP inner width (the fused
            ``gate_up_proj`` is sized ``2 * intermediate_size``).
        num_hidden_layers (int): Number of decoder blocks.
        num_attention_heads (int): Query heads per attention layer.
        num_key_value_heads (int): KV heads for grouped-query attention.
        partial_rotary_factor (float): Fraction of each head dimension that
            receives RoPE; remaining channels are left unrotated.
        head_dim (int): Per-head attention dimension.
        hidden_act (str): Gate activation name (e.g. ``"silu"`` for SwiGLU).
        attention_dropout (float): Attention probability dropout.
        max_position_embeddings (int): Context-length bound used by RoPE
            tables and sequence-length asserts.
        initializer_range (float): Stddev for truncated-normal weight init.
        rms_norm_eps (float): Epsilon for every RMSNorm in the stack.
        use_cache (bool): Whether downstream code should return KV cache.
        tie_word_embeddings (bool): Tie input embeddings with the LM head.
        rope_theta (float): RoPE base frequency.
        pad_token_id (int): Padding token id.
        eos_token_id (int | list[int]): End-of-stream token id(s).
        bos_token_id (int | None): Beginning-of-stream token id.
        attention_bias (bool): Whether Q/K/V projections include bias.
        layer_types (list[str]): Per-layer attention types; defaults to
            ``["full_attention"] * num_hidden_layers``.
    """

    model_type = "glm4"

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        intermediate_size: int = 13696,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        partial_rotary_factor: float = 0.5,
        head_dim: int = 128,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 0.00000015625,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        pad_token_id: int = 151329,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        attention_bias: bool = True,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize a GLM-4 model configuration.

        Args:
            vocab_size: Token vocabulary size.
            hidden_size: Residual-stream / model dimension.
            intermediate_size: MLP inner width (the gated MLP uses
                ``2 * intermediate_size`` for the fused ``gate_up_proj``).
            num_hidden_layers: Number of stacked decoder blocks.
            num_attention_heads: Total query heads per attention layer.
            num_key_value_heads: KV heads for grouped-query attention.
            partial_rotary_factor: Fraction of each head dimension that
                receives RoPE; remaining channels are left unrotated.
            head_dim: Size of each attention head.
            hidden_act: Activation name applied to the gate half of the MLP.
            attention_dropout: Dropout on the attention probabilities.
            max_position_embeddings: Context-length bound used by RoPE
                frequency tables and sequence-length asserts.
            initializer_range: Stddev for truncated-normal weight init.
            rms_norm_eps: Epsilon used by every RMSNorm in the stack.
            use_cache: Whether downstream code should return KV cache.
            tie_word_embeddings: Tie input embeddings with the LM head.
            rope_theta: RoPE base frequency.
            pad_token_id: Padding token id.
            eos_token_id: End-of-stream token id(s); defaults to
                ``[151329, 151336, 151338]``.
            bos_token_id: Beginning-of-stream token id (optional).
            attention_bias: Whether Q/K/V projections carry biases (the
                modeling layer always uses a biasless output projection).
            layer_types: Optional per-layer attention type list; defaults
                to ``["full_attention"] * num_hidden_layers``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        if eos_token_id is None:
            eos_token_id = [151329, 151336, 151338]
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Glm4Config"]

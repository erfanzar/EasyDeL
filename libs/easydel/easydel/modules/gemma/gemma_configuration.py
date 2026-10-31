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

"""Configuration for the original Gemma decoder-only LLM.

Defines :class:`GemmaConfig` for Google DeepMind's first-generation Gemma
models. The config keeps the upstream HF parameter shape: large 256k SentencePiece
vocab, ``head_dim=256`` decoupled from ``hidden_size``, GeLU-tanh hidden
activation, RMSNorm with ``rms_norm_eps``, and tied input/output embeddings
by default. Position handling is plain RoPE (no scaling, no sliding window).
"""

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("gemma")
class GemmaConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key and value heads for each attention layer in the Transformer encoder.
        head_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the attention head.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of the padding token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 1):
            The index of the end of sequence token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 2):
            The index of the beginning of sequence token in the vocabulary.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use attention bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        scan_layers (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation of the layers.
        hidden_activation (`str`, *optional*):
            The hidden activation function to use.
    """

    model_type: str = "gemma"

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 3072,
        intermediate_size: int | None = 24576,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 256,
        hidden_act: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        scan_layers: bool = False,
        hidden_activation: str | None = "gelu_pytorch_tanh",
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize a :class:`GemmaConfig`.

        Args:
            vocab_size (int, optional): Token vocabulary size. Defaults to ``256000``.
            hidden_size (int, optional): Decoder hidden dimension. Defaults to ``3072``.
            intermediate_size (int | None, optional): SwiGLU MLP intermediate width.
                Defaults to ``24576``.
            num_hidden_layers (int, optional): Number of decoder layers. Defaults to ``28``.
            num_attention_heads (int, optional): Query heads per layer. Defaults to ``16``.
            num_key_value_heads (int, optional): KV heads (GQA when smaller than
                ``num_attention_heads``). Defaults to ``16``.
            head_dim (int, optional): Per-head attention dimension; decoupled from
                ``hidden_size``. Defaults to ``256``.
            hidden_act (str, optional): Name of the MLP activation. Defaults to
                ``"gelu_pytorch_tanh"`` (GeLU with PyTorch's tanh approximation).
            max_position_embeddings (int, optional): Maximum sequence length.
                Defaults to ``8192``.
            initializer_range (float, optional): Truncated-normal init stddev.
                Defaults to ``0.02``.
            rms_norm_eps (float, optional): RMSNorm epsilon. Defaults to ``1e-6``.
            use_cache (bool, optional): Return KV caches during forward. Defaults to ``True``.
            pad_token_id (int, optional): Padding token id. Defaults to ``0``.
            eos_token_id (int, optional): End-of-sequence id. Defaults to ``1``.
            bos_token_id (int, optional): Beginning-of-sequence id. Defaults to ``2``.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to ``True``.
            rope_theta (float, optional): RoPE base frequency. Defaults to ``10000.0``.
            attention_bias (bool, optional): Use bias on QKV/output projections.
                Defaults to ``False``.
            attention_dropout (float, optional): Attention probability dropout.
                Defaults to ``0.0``.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Checkpointing
                policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            bits (int | None, optional): Quantization bit-width. Defaults to ``None``.
            scan_layers (bool, optional): Use ``lax.scan`` to share decoder weights and
                shrink the trace. Defaults to ``False``.
            hidden_activation (str | None, optional): Optional explicit activation name
                taking precedence over ``hidden_act``. Defaults to ``"gelu_pytorch_tanh"``.
            layer_types (list[str] | None, optional): Per-layer attention types.
                ``None`` fills with ``"full_attention"``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        super().__init__(
            bos_token_id=bos_token_id,
            scan_layers=scan_layers,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

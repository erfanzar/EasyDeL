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

"""Configuration for Gemma 2 decoder-only LLMs.

Defines :class:`Gemma2Config` for Google DeepMind's Gemma 2 models. Compared
to the original Gemma (see :mod:`easydel.modules.gemma`), Gemma 2 introduces:

- Interleaved sliding-window attention (every other layer uses
  ``sliding_window``-bounded local attention; the rest use full attention).
- Final-logit and optional attention-logit softcapping
  (``final_logit_softcapping``, ``attn_logit_softcapping``) for stability.
- A learned ``query_pre_attn_scalar`` replacing the standard
  ``1 / sqrt(head_dim)`` scaling.
- A hybrid KV cache (``cache_implementation = "hybrid"``) holding both full
  and sliding views per layer.
"""

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("gemma2")
class Gemma2Config(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma2 model. Defines the number of different tokens that can be represented by the
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
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
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
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            The soft capping value for the final logits.
        query_pre_attn_scalar (`int`, *optional*, defaults to 224):
            The scalar value for the query pre-attention layer.
        sliding_window (`int`, *optional*, defaults to 4096):
            The sliding window size.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        scan_layers (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation of the layers.
    """

    model_type: str = "gemma2"

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 3072,
        intermediate_size: int | None = 24576,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
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
        final_logit_softcapping: float = 30.0,
        query_pre_attn_scalar: int = 224,
        sliding_window: int = 4096,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        layer_types: list[str] | None = None,
        bits: int | None = None,
        scan_layers: bool = False,
        attn_logit_softcapping: bool | None = None,
        **kwargs,
    ):
        """Initialize a :class:`Gemma2Config`.

        Args:
            vocab_size (int, optional): Token vocabulary size. Defaults to ``256000``.
            hidden_size (int, optional): Decoder hidden dimension. Defaults to ``3072``.
            intermediate_size (int | None, optional): MLP intermediate width.
                Defaults to ``24576``.
            num_hidden_layers (int, optional): Number of decoder layers. Defaults to ``28``.
            num_attention_heads (int, optional): Query heads per layer. Defaults to ``16``.
            num_key_value_heads (int, optional): KV heads (GQA). Defaults to ``16``.
            head_dim (int, optional): Per-head dimension. Defaults to ``256``.
            hidden_activation (str, optional): MLP activation. Defaults to
                ``"gelu_pytorch_tanh"``.
            max_position_embeddings (int, optional): Maximum sequence length.
                Defaults to ``8192``.
            initializer_range (float, optional): Truncated-normal init stddev.
                Defaults to ``0.02``.
            rms_norm_eps (float, optional): RMSNorm epsilon. Defaults to ``1e-6``.
            use_cache (bool, optional): Return KV caches. Defaults to ``True``.
            pad_token_id (int, optional): Padding id. Defaults to ``0``.
            eos_token_id (int, optional): End-of-sequence id. Defaults to ``1``.
            bos_token_id (int, optional): Beginning-of-sequence id. Defaults to ``2``.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to ``True``.
            rope_theta (float, optional): RoPE base frequency. Defaults to ``10000.0``.
            attention_bias (bool, optional): Use bias on attention projections.
                Defaults to ``False``.
            attention_dropout (float, optional): Attention dropout. Defaults to ``0.0``.
            final_logit_softcapping (float, optional): Tanh-softcap value applied to
                LM head logits. Defaults to ``30.0``.
            query_pre_attn_scalar (int, optional): Replaces the default
                ``1/sqrt(head_dim)`` query scaling. Defaults to ``224``.
            sliding_window (int, optional): Sliding-window length for the
                ``"sliding_attention"`` layer slots. Defaults to ``4096``.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional):
                Checkpointing policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            layer_types (list[str] | None, optional): Per-layer attention types
                (``"sliding_attention"`` or ``"full_attention"``). ``None`` defaults
                to alternating ``sliding`` / ``full`` starting from layer 0.
            bits (int | None, optional): Quantization bit-width. Defaults to ``None``.
            scan_layers (bool, optional): Use ``lax.scan`` for shared decoder weights.
                Defaults to ``False``.
            attn_logit_softcapping (bool | None, optional): Optional per-attention
                softcapping value (forwarded to the attention kernel).
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
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            bos_token_id=bos_token_id,
            scan_layers=scan_layers,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

        self.final_logit_softcapping = final_logit_softcapping
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.cache_implementation = "hybrid"
        self.attn_logit_softcapping = attn_logit_softcapping

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Retrieve attention mask details for each layer in the model.

        This method generates a dictionary mapping layer indices to their corresponding attention mask details.
        If a sliding window is defined, each layer is assigned a sliding window attention mask with the specified size.

        Returns:
            dict[int, AttnMaskDetail]: A dictionary where keys are layer indices (int) and values are AttnMaskDetail
            objects specifying the attention mask type and size for each layer.

        Notes:
            - If `self.sliding_window` is None, an empty dictionary is returned.
            - The method iterates over `self.num_hidden_layers` to assign mask details for each layer.
            - The attention mask type is set to `AttnMaskType.SLIDING` when a sliding window is defined.
        """
        mapping = {}
        if self.layer_types is not None:
            for layer_idx in range(self.num_hidden_layers):
                mapping[layer_idx] = AttnMaskDetail(
                    mask_type=AttnMaskType.from_hf(self.layer_types[layer_idx]),
                    size=self.sliding_window,
                )
        return mapping

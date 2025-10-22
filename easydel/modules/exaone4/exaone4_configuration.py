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


import typing

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("exaone4")
class Exaone4Config(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`Exaone4Model`]. It is used to
    instantiate a EXAONE 4.0 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs.

    Args:
            vocab_size (`int`, *optional*, defaults to 102400):
                    Vocabulary size of the EXAONE 4.0 model.
            hidden_size (`int`, *optional*, defaults to 4096):
                    Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 16384):
                    Dimensionality of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                    Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                    Number of attention heads for each attention layer.
            num_key_value_heads (`int`, *optional*):
                    Number of key_value heads for Grouped Query Attention. Defaults to `num_attention_heads`.
            hidden_act (`str`, *optional*, defaults to `"silu"`):
                    The non-linear activation function in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                    The maximum sequence length.
            initializer_range (`float`, *optional*, defaults to 0.02):
                    The standard deviation for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                    The epsilon used by the layer normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                    Whether the model should return the last key/values attentions.
            bos_token_id (`int`, *optional*, defaults to 0):
                    Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 2):
                    End of stream token id.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                    Whether to tie weight embeddings.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                    The base period of the RoPE embeddings.
            rope_scaling (`dict`, *optional*):
                    Dictionary containing the scaling configuration for the RoPE embeddings.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                    The dropout ratio for the attention probabilities.
            sliding_window (`int`, *optional*, defaults to 4096):
                    The size of the sliding window for sliding window attention.
            sliding_window_pattern (`int`, *optional*, defaults to 4):
                    Pattern for determining layer types. Every `sliding_window_pattern` layers uses full attention.
            layer_types (`list`, *optional*):
                    Attention pattern for each layer. Prioritized over `sliding_window_pattern`.
            gradient_checkpointing (`EasyDeLGradientCheckPointers`, *optional*):
                    Gradient checkpointing strategy.
            use_scan_mlp (`bool`, *optional*, defaults to False):
                    Whether to use scan for MLP layers.
            scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
                    Chunk size for scan MLP.
            bits (`int`, *optional*):
                    Quantization bits.
    """

    model_type = "exaone4"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        sliding_window=4096,
        sliding_window_pattern=4,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize Exaone4Config."""

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        # Set num_key_value_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        # Compute layer_types from sliding_window_pattern
        if self.sliding_window is None:
            sliding_window_pattern = 0

        if layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if ((i + 1) % sliding_window_pattern != 0 and i < num_hidden_layers)
                else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        self._validate_layer_types()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """Validate rope_scaling configuration."""
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                f"`rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _validate_layer_types(self):
        """Validate layer_types list."""
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` must have length equal to `num_hidden_layers` ({self.num_hidden_layers}), "
                f"got {len(self.layer_types)}"
            )

        valid_types = {"sliding_attention", "full_attention"}
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type not in valid_types:
                raise ValueError(f"`layer_types[{idx}]` must be one of {valid_types}, got '{layer_type}'")

    def get_partition_rules(self, *args, **kwargs):
        """Get the partition rules for the model."""
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"self_attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*/(post_attention_layernorm|post_feedforward_layernorm|norm)/kernel",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(post_attention_layernorm|post_feedforward_layernorm|norm)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(post_attention_layernorm|post_feedforward_layernorm|norm)/bias",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

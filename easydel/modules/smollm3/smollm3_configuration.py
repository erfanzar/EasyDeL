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


@register_config("smollm3")
class SmolLM3Config(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`SmolLM3Model`]. It is used to
    instantiate a SmolLM3 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs.

    Args:
            vocab_size (`int`, *optional*, defaults to 128256):
                    Vocabulary size of the SmolLM3 model.
            hidden_size (`int`, *optional*, defaults to 2048):
                    Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 11008):
                    Dimensionality of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 36):
                    Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 16):
                    Number of attention heads for each attention layer.
            num_key_value_heads (`int`, *optional*):
                    Number of key_value heads for Grouped Query Attention. Defaults to `num_attention_heads`.
            hidden_act (`str`, *optional*, defaults to `"silu"`):
                    The non-linear activation function in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 32768):
                    The maximum sequence length.
            initializer_range (`float`, *optional*, defaults to 0.02):
                    The standard deviation for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-06):
                    The epsilon used by the layer normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                    Whether the model should return the last key/values attentions.
            bos_token_id (`int`, *optional*, defaults to 128000):
                    Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 128001):
                    End of stream token id.
            pad_token_id (`int`, *optional*, defaults to 128004):
                    Padding token id.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                    Whether to tie weight embeddings.
            rope_theta (`float`, *optional*, defaults to 2000000.0):
                    The base period of the RoPE embeddings.
            rope_scaling (`dict`, *optional*):
                    Dictionary containing the scaling configuration for the RoPE embeddings.
            attention_bias (`bool`, *optional*, defaults to `False`):
                    Whether to use a bias in the query, key, value and output projection layers.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                    The dropout ratio for the attention probabilities.
            mlp_bias (`bool`, *optional*, defaults to `False`):
                    Whether to use bias in MLP layers.
            use_sliding_window (`bool`, *optional*, defaults to `False`):
                    Whether to use sliding window attention.
            sliding_window (`int`, *optional*):
                    Sliding window attention window size. If not specified, will default to `None`.
            no_rope_layers (`list[int]`, *optional*):
                    List indicating which layers use RoPE (1) or NoPE (0). Must have same length as num_hidden_layers.
            no_rope_layer_interval (`int`, *optional*, defaults to 4):
                    If `no_rope_layers` is `None`, create pattern with NoPE every `no_rope_layer_interval` layers.
            layer_types (`list`, *optional*):
                    Attention pattern for each layer. Automatically computed based on sliding window and NoPE settings.
            gradient_checkpointing (`EasyDeLGradientCheckPointers`, *optional*):
                    Gradient checkpointing strategy.
            use_scan_mlp (`bool`, *optional*, defaults to False):
                    Whether to use scan for MLP layers.
            scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
                    Chunk size for scan MLP.
            bits (`int`, *optional*):
                    Quantization bits.
    """

    model_type = "smollm3"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=128004,
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=False,
        rope_theta=2000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        no_rope_layers: list[int] | None = None,
        no_rope_layer_interval: int = 4,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
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
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.no_rope_layer_interval = no_rope_layer_interval
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        if no_rope_layers is None:
            self.no_rope_layers = [
                int((layer_idx + 1) % no_rope_layer_interval != 0) for layer_idx in range(num_hidden_layers)
            ]
        else:
            self.no_rope_layers = no_rope_layers

        if layer_types is None:
            layer_types = []
            for layer_idx in range(num_hidden_layers):
                has_rope = self.no_rope_layers[layer_idx]
                if use_sliding_window and sliding_window is not None and not has_rope:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")
            self.layer_types = layer_types
        else:
            self.layer_types = layer_types

        self._validate_no_rope_layers()
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

    def _validate_no_rope_layers(self):
        """Validate no_rope_layers list."""
        if len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"`no_rope_layers` must have length equal to `num_hidden_layers` ({self.num_hidden_layers}), "
                f"got {len(self.no_rope_layers)}"
            )

        for idx, use_rope in enumerate(self.no_rope_layers):
            if use_rope not in {0, 1}:
                raise ValueError(f"`no_rope_layers[{idx}]` must be 0 or 1, got {use_rope}")

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
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/kernel",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/bias",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

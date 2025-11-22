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

"""Configuration classes for Qwen3-VL model.

This module provides configuration classes that mirror the HuggingFace Qwen3-VL
implementation, with proper separation of vision and text configurations.
"""

import typing

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("qwen3_vl_vision")
class Qwen3VLVisionConfig(EasyDeLBaseConfig):
    """Configuration class for Qwen3-VL vision encoder.

    This configuration controls the vision transformer backbone that processes
    images and videos before they are integrated with the language model.

    Args:
        depth: Number of transformer layers in vision encoder. Defaults to 27.
        hidden_size: Dimensionality of vision encoder hidden states. Defaults to 1152.
        hidden_act: Activation function for vision MLP. Defaults to "gelu_pytorch_tanh".
        intermediate_size: Dimensionality of vision FFN. Defaults to 4304.
        num_heads: Number of attention heads. Defaults to 16.
        in_channels: Number of input image channels. Defaults to 3.
        patch_size: Size of image patches. Defaults to 16.
        spatial_merge_size: Spatial downsampling factor in patch merger. Defaults to 2.
        temporal_patch_size: Temporal patch size for video. Defaults to 2.
        out_hidden_size: Output dimension after patch merger. Defaults to 3584.
        num_position_embeddings: Maximum position embeddings. Defaults to 2304.
        deepstack_visual_indexes: Layer indices for deepstack injection. Defaults to [8, 16, 24].
        tokens_per_second: Temporal scaling for video. Defaults to 2.0.
        initializer_range: Weight initialization std. Defaults to 0.02.
    """

    model_type = "qwen3_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
        tokens_per_second: float = 2.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes or [8, 16, 24]
        self.tokens_per_second = tokens_per_second
        self.initializer_range = initializer_range

        # Computed attributes for compatibility
        self.embed_dim = hidden_size


@register_config("qwen3_vl_text")
class Qwen3VLTextConfig(EasyDeLBaseConfig):
    """Configuration class for Qwen3-VL text/language model.

    This configuration controls the language model decoder that processes
    text tokens and integrates vision features.

    Args:
        vocab_size: Size of vocabulary. Defaults to 151936.
        hidden_size: Dimensionality of hidden states. Defaults to 4096.
        intermediate_size: Dimensionality of MLP. Defaults to 22016.
        num_hidden_layers: Number of decoder layers. Defaults to 32.
        num_attention_heads: Number of attention heads. Defaults to 32.
        num_key_value_heads: Number of KV heads for GQA. Defaults to 32.
        head_dim: Dimension per attention head. Defaults to 128.
        hidden_act: Activation function. Defaults to "silu".
        max_position_embeddings: Maximum sequence length. Defaults to 128000.
        initializer_range: Weight initialization std. Defaults to 0.02.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-6.
        use_cache: Whether to use KV cache. Defaults to True.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
        rope_theta: RoPE base frequency. Defaults to 1000000.0.
        attention_bias: Whether to use attention bias. Defaults to False.
        attention_dropout: Attention dropout rate. Defaults to 0.0.
        rope_scaling: RoPE scaling configuration. Defaults to None.
        use_sliding_window: Whether to use sliding window attention. Defaults to False.
        sliding_window: Sliding window size. Defaults to 4096.
        max_window_layers: Layers using sliding window. Defaults to 80.
    """

    model_type = "qwen3_vl_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_scaling: dict | None = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # Handle rope_scaling type conversion for HF compatibility
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]


@register_config("qwen3_vl")
class Qwen3VLConfig(EasyDeLBaseConfig):
    """Main configuration class for Qwen3-VL multimodal model.

    This configuration combines vision and text configurations and provides
    the top-level parameters for the multimodal model.

    Args:
        vision_config: Vision encoder configuration dict or Qwen3VLVisionConfig.
        text_config: Text decoder configuration dict or Qwen3VLTextConfig.
        image_token_id: Token ID for image placeholders. Defaults to 151655.
        video_token_id: Token ID for video placeholders. Defaults to 151656.
        vision_start_token_id: Token ID for vision sequence start. Defaults to 151652.
        vision_end_token_id: Token ID for vision sequence end. Defaults to 151653.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
    """

    model_type = "qwen3_vl"
    sub_configs: typing.ClassVar = {
        "vision_config": Qwen3VLVisionConfig,
        "text_config": Qwen3VLTextConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vision_config: typing.Mapping[str, typing.Any] | Qwen3VLVisionConfig | None = None,
        text_config: typing.Mapping[str, typing.Any] | Qwen3VLTextConfig | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        # Allow passing text config fields directly for convenience
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str | None = None,
        max_position_embeddings: int | None = None,
        initializer_range: float | None = None,
        rms_norm_eps: float | None = None,
        use_cache: bool | None = None,
        rope_theta: float | None = None,
        attention_bias: bool | None = None,
        attention_dropout: float | None = None,
        rope_scaling: dict | None = None,
        use_sliding_window: bool | None = None,
        sliding_window: int | None = None,
        max_window_layers: int | None = None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Build vision config
        if vision_config is None:
            vision_config = {}
        if isinstance(vision_config, dict):
            self.vision_config = Qwen3VLVisionConfig(**vision_config)
        elif isinstance(vision_config, Qwen3VLVisionConfig):
            self.vision_config = vision_config
        else:
            raise ValueError("vision_config must be a dict or Qwen3VLVisionConfig.")

        # Build text config, allowing direct field overrides
        text_config_dict = {}
        if isinstance(text_config, dict):
            text_config_dict = text_config.copy()
        elif isinstance(text_config, Qwen3VLTextConfig):
            self.text_config = text_config
            text_config_dict = None

        if text_config_dict is not None:
            # Override with directly passed fields
            direct_fields = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "hidden_act": hidden_act,
                "max_position_embeddings": max_position_embeddings,
                "initializer_range": initializer_range,
                "rms_norm_eps": rms_norm_eps,
                "use_cache": use_cache,
                "rope_theta": rope_theta,
                "attention_bias": attention_bias,
                "attention_dropout": attention_dropout,
                "rope_scaling": rope_scaling,
                "use_sliding_window": use_sliding_window,
                "sliding_window": sliding_window,
                "max_window_layers": max_window_layers,
                "tie_word_embeddings": tie_word_embeddings,
            }
            for key, value in direct_fields.items():
                if value is not None:
                    text_config_dict[key] = value
            self.text_config = Qwen3VLTextConfig(**text_config_dict)

        # Token IDs
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Expose commonly used text config fields at top level for convenience
        self._sync_text_config_fields()

    def _sync_text_config_fields(self):
        """Sync commonly used text config fields to top level."""
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.intermediate_size = self.text_config.intermediate_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.hidden_act = self.text_config.hidden_act
        self.max_position_embeddings = self.text_config.max_position_embeddings
        self.initializer_range = self.text_config.initializer_range
        self.rms_norm_eps = self.text_config.rms_norm_eps
        self.use_cache = self.text_config.use_cache
        self.rope_theta = self.text_config.rope_theta
        self.attention_bias = self.text_config.attention_bias
        self.attention_dropout = self.text_config.attention_dropout
        self.rope_scaling = self.text_config.rope_scaling
        self.use_sliding_window = self.text_config.use_sliding_window
        self.sliding_window = self.text_config.sliding_window
        self.max_window_layers = self.text_config.max_window_layers

    def get_text_config(self, decoder: bool = True) -> Qwen3VLTextConfig:
        """Get the text configuration for the model.

        Args:
            decoder: Ignored, kept for HF API compatibility.

        Returns:
            The text configuration object.
        """
        return self.text_config

    def get_partition_rules(self, fully_sharded_data_parallel: bool = False):
        """Get partition rules for model parallelism."""
        pmag = self.partition_manager
        return (
            # Text model embeddings
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            # Text attention projections
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            # Text MLP
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            # Text norms
            (r".*(input_layernorm|post_attention_layernorm|norm)/kernel", pmag.resolve(Replicated)),
            # LM head
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            # Vision patch embed
            (r"visual/patch_embed/proj/kernel", pmag.resolve(ColumnWise)),
            # Vision attention
            (r"visual/.*/attn/qkv/kernel", pmag.resolve(ColumnWise)),
            (r"visual/.*/attn/qkv/bias", pmag.resolve(Replicated)),
            (r"visual/.*/attn/proj/kernel", pmag.resolve(RowWise)),
            (r"visual/.*/attn/proj/bias", pmag.resolve(Replicated)),
            # Vision MLP
            (r"visual/.*/mlp/fc1/kernel", pmag.resolve(ColumnWise)),
            (r"visual/.*/mlp/fc1/bias", pmag.resolve(Replicated)),
            (r"visual/.*/mlp/fc2/kernel", pmag.resolve(RowWise)),
            (r"visual/.*/mlp/fc2/bias", pmag.resolve(Replicated)),
            # Vision norms
            (r"visual/.*/norm(1|2)/scale", pmag.resolve(Replicated)),
            (r"visual/.*/norm(1|2)/bias", pmag.resolve(Replicated)),
            # Vision merger
            (r"visual/merger/ln_q/scale", pmag.resolve(Replicated)),
            (r"visual/merger/ln_q/bias", pmag.resolve(Replicated)),
            (r"visual/merger/mlp/0/kernel", pmag.resolve(ColumnWise)),
            (r"visual/merger/mlp/0/bias", pmag.resolve(Replicated)),
            (r"visual/merger/mlp/2/kernel", pmag.resolve(RowWise)),
            (r"visual/merger/mlp/2/bias", pmag.resolve(Replicated)),
            # Catch-all
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Get attention mask details for sliding window attention."""
        mapping = {}
        if self.sliding_window is not None and self.use_sliding_window:
            for layer_idx in range(self.num_hidden_layers):
                if layer_idx < self.max_window_layers:
                    mapping[layer_idx] = AttnMaskDetail(
                        mask_type=AttnMaskType.SLIDING,
                        size=self.sliding_window,
                    )
        return mapping

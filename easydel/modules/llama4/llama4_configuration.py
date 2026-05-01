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


import typing

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

logger = get_logger(__name__)


def _patch_hf_llama4_pooler_output() -> None:
    """HF compatibility: ensure Llama4 image features expose `pooler_output`."""
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.models.llama4 import modeling_llama4 as hf_llama4
    except Exception:
        return

    llama4_cls = getattr(hf_llama4, "Llama4ForConditionalGeneration", None)
    if llama4_cls is None:
        return

    original_get_image_features = getattr(llama4_cls, "get_image_features", None)
    if original_get_image_features is None or getattr(original_get_image_features, "_easydel_pooler_patch", False):
        return

    def _patched_get_image_features(self, *args, **kwargs):
        outputs = original_get_image_features(self, *args, **kwargs)
        if hasattr(outputs, "pooler_output"):
            return outputs

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None and isinstance(outputs, tuple) and len(outputs) > 0:
            last_hidden_state = outputs[0]
        if last_hidden_state is None:
            return outputs

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=last_hidden_state,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    _patched_get_image_features._easydel_pooler_patch = True  # type: ignore[attr-defined]
    llama4_cls.get_image_features = _patched_get_image_features


_patch_hf_llama4_pooler_output()


@register_config("llama4_vision_model")
class Llama4VisionConfig(EasyDeLBaseConfig):
    """Configuration for the Llama4 vision encoder and multi-modal projector.

    This stores all parameters for the vision transformer backbone and the linear
    projector that maps vision features into the text embedding space. Supports
    pixel-shuffle downsampling and configurable projector dimensions.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the vision encoder hidden layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in the vision encoder.
        num_hidden_layers (`int`, *optional*, defaults to 34):
            Number of transformer layers in the vision encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the vision encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels (3 for RGB).
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimensionality of the MLP intermediate layer in the vision encoder.
        vision_output_dim (`int`, *optional*, defaults to 7680):
            Output dimensionality of the vision encoder before projection.
        image_size (`int`, *optional*, defaults to 448):
            Input image resolution (height and width).
        patch_size (`int`, *optional*, defaults to 14):
            Size of each image patch for the vision transformer.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for layer normalization in the vision encoder.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            Which hidden layer's output to use as vision features (-1 for last).
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Strategy for selecting vision features from the encoder output.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        pixel_shuffle_ratio (`float`, *optional*, defaults to 0.5):
            Downsampling ratio applied via pixel shuffle to reduce spatial tokens.
        projector_input_dim (`int`, *optional*, defaults to 4096):
            Input dimensionality of the multi-modal projector.
        projector_output_dim (`int`, *optional*, defaults to 4096):
            Output dimensionality of the multi-modal projector (must match text hidden_size).
        multi_modal_projector_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the multi-modal projector linear layers.
        projector_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate in the multi-modal projector.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for attention weights in the vision encoder.
        rope_theta (`float`, *optional*, defaults to 10000):
            Base frequency for rotary position embeddings in the vision encoder.
    """

    model_type = "llama4_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 34,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5632,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "default",
        initializer_range: float = 0.02,
        pixel_shuffle_ratio: float = 0.5,
        projector_input_dim: int = 4096,
        projector_output_dim: int = 4096,
        multi_modal_projector_bias: bool = False,
        projector_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.vision_output_dim = vision_output_dim
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self.projector_input_dim = projector_input_dim
        self.projector_output_dim = projector_output_dim
        self.multi_modal_projector_bias = multi_modal_projector_bias
        self.projector_dropout = projector_dropout
        self.attention_dropout = attention_dropout
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.rope_theta = rope_theta
        super().__init__(**kwargs)


@register_config("llama4_text")
class Llama4TextConfig(EasyDeLBaseConfig):
    """Configuration for the Llama4 text decoder with interleaved MoE layers.

    Llama4 uses a decoder-only transformer with a mixture of dense and MoE layers
    interleaved at configurable intervals, chunked attention with temperature tuning,
    per-layer RoPE control (some layers skip RoPE entirely), and QK normalization.

    Args:
        vocab_size (`int`, *optional*, defaults to 202048):
            Vocabulary size of the Llama4 text model.
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the hidden layers.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the MoE expert MLP intermediate layer.
        intermediate_size_mlp (`int`, *optional*, defaults to 16384):
            Dimensionality of the dense MLP intermediate layer (non-MoE layers).
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for grouped query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function for MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values for caching.
        rope_theta (`float`, *optional*, defaults to 500000):
            Base frequency for rotary position embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for attention weights.
        num_experts_per_tok (`int`, *optional*, defaults to 1):
            Number of experts activated per token in MoE layers.
        num_local_experts (`int`, *optional*, defaults to 16):
            Total number of experts in each MoE layer.
        moe_layers (`list[int]`, *optional*):
            Explicit list of layer indices that use MoE. Computed from
            ``interleave_moe_layer_step`` if not provided.
        interleave_moe_layer_step (`int`, *optional*, defaults to 1):
            Interval for interleaving MoE layers (every N-th layer is MoE).
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply QK normalization in attention.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to output MoE router logits for auxiliary loss computation.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient for the router auxiliary loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Jitter noise added to router logits during training.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration dictionary (e.g., for extended context).
        no_rope_layers (`list[int]`, *optional*):
            Per-layer binary flags (1 = no RoPE, 0 = use RoPE). Computed from
            ``no_rope_layer_interval`` if not provided.
        no_rope_layer_interval (`int`, *optional*, defaults to 4):
            Every N-th layer uses RoPE; others skip it.
        attention_chunk_size (`int`, *optional*, defaults to 8192):
            Chunk size for chunked attention layers.
        attn_temperature_tuning (`int`, *optional*, defaults to 4):
            Temperature tuning factor for attention scaling.
        floor_scale (`int`, *optional*, defaults to 8192):
            Floor scale factor for attention temperature computation.
        attn_scale (`float`, *optional*, defaults to 0.1):
            Scaling factor for attention logits.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type (`"chunked_attention"` or `"full_attention"`).
            Auto-derived from ``no_rope_layers`` if not set.
    """

    model_type = "llama4_text"

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 5120,
        intermediate_size: int = 8192,
        intermediate_size_mlp: int = 16384,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 40,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096 * 32,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 1,
        num_local_experts: int = 16,
        moe_layers: list[int] | None = None,
        interleave_moe_layer_step: int = 1,
        use_qk_norm: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        rope_scaling: dict | None = None,
        no_rope_layers: list[int] | None = None,
        no_rope_layer_interval: int = 4,
        attention_chunk_size: int = 8192,
        attn_temperature_tuning: int = 4,
        floor_scale: int = 8192,
        attn_scale: float = 0.1,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.attn_temperature_tuning = attn_temperature_tuning
        self.attn_scale = attn_scale
        self.floor_scale = floor_scale
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rope_scaling = rope_scaling
        self.attention_bias = False

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.use_qk_norm = use_qk_norm

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts

        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        default_no_rope_layers = [
            int((layer_idx + 1) % no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
        ]

        self.no_rope_layers = no_rope_layers if no_rope_layers else default_no_rope_layers

        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.moe_layers = (
            moe_layers
            if moe_layers is not None
            else list(range(interleave_moe_layer_step - 1, num_hidden_layers, interleave_moe_layer_step))
        )
        self.attention_chunk_size = attention_chunk_size
        if layer_types is None:
            layer_types = ["chunked_attention" if no_rope else "full_attention" for no_rope in self.no_rope_layers]
        self.layer_types = layer_types


@register_config("llama4")
class Llama4Config(EasyDeLBaseConfig):
    """Top-level configuration for the Llama4 multimodal (vision-language) model.

    Combines ``Llama4VisionConfig`` and ``Llama4TextConfig`` sub-configurations
    and defines the special token indices used to delimit image regions in the
    input sequence.

    Args:
        vision_config (`dict` or `Llama4VisionConfig`, *optional*):
            Configuration for the vision encoder and projector. Defaults to
            ``Llama4VisionConfig()`` if not provided.
        text_config (`dict` or `Llama4TextConfig`, *optional*):
            Configuration for the text decoder. Defaults to ``Llama4TextConfig()``
            if not provided.
        boi_token_index (`int`, *optional*, defaults to 200080):
            Token index marking the beginning of an image region.
        eoi_token_index (`int`, *optional*, defaults to 200081):
            Token index marking the end of an image region.
        image_token_index (`int`, *optional*, defaults to 200092):
            Token index used as a placeholder for image patch embeddings.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output word embeddings.
    """

    model_type = "llama4"
    sub_configs: typing.ClassVar = {"text_config": Llama4TextConfig, "vision_config": Llama4VisionConfig}
    attribute_map: typing.ClassVar = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    def __init__(
        self,
        vision_config: dict | Llama4VisionConfig | None = None,
        text_config: dict | Llama4TextConfig | None = None,
        boi_token_index: int = 200080,
        eoi_token_index: int = 200081,
        image_token_index: int = 200092,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = Llama4VisionConfig()
            logger.info("vision_config is None, using default llama4 vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = Llama4VisionConfig(**self._fix_parent_kws(vision_config, kwargs))
        elif isinstance(vision_config, Llama4VisionConfig):
            self.vision_config = vision_config

        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = Llama4TextConfig()
            logger.info("text_config is None, using default llama4 text config")
        elif isinstance(text_config, dict):
            self.text_config = Llama4TextConfig(**self._fix_parent_kws(text_config, kwargs))
        elif isinstance(text_config, Llama4TextConfig):
            self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]

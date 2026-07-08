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

"""Configuration classes for the Tencent HunYuanVL multimodal model family.

This module defines three configurations consumed by HunYuanVL (the
dense-only, image-text variant used by e.g. ``tencent/HunyuanOCR``):

- :class:`HunYuanVLVisionConfig` (``hunyuan_vl_vision``): the SigLIP-style
  vision tower (conv patch embedding with an interpolated learned position
  grid, pre-LayerNorm blocks, and a convolutional patch merger that emits
  per-row newline tokens plus begin/end tokens).
- :class:`HunYuanVLTextConfig` (``hunyuan_vl_text``): the HunYuan dense
  decoder (GQA with per-head RMSNorm applied to queries/keys *after* RoPE,
  SwiGLU MLP) extended with HunYuan's 3-axis multimodal RoPE
  (``mrope_section`` over ``(width, height, image_index)``).
- :class:`HunYuanVLConfig` (``hunyuan_vl``): the composite configuration
  gluing both towers together plus the image-span token ids.
"""

import typing
from collections.abc import Mapping

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


def _default_mrope_section(head_dim: int) -> list[int]:
    """Derive a 3-axis ``mrope_section`` covering half of ``head_dim``.

    The released HunYuanVL checkpoints partition half of each attention head
    as roughly ``(1/4, 3/8, 3/8)`` across the ``(width, height, image_index)``
    axes — e.g. ``[16, 24, 24]`` for ``head_dim=128``. This helper reproduces
    that split for arbitrary head dimensions so default-constructed configs
    stay runnable.

    Args:
        head_dim: Per-head attention dimension.

    Returns:
        list[int]: Three section sizes summing to ``head_dim // 2``.
    """
    half = head_dim // 2
    first = half // 4
    second = (half - first) // 2
    return [first, second, half - first - second]


@register_config("hunyuan_vl_vision")
class HunYuanVLVisionConfig(EasyDeLBaseConfig):
    """Configuration for the HunYuanVL vision tower.

    The tower is a SigLIP-flavoured ViT: a strided ``Conv2d`` patch
    embedding with a learned position grid that is bilinearly interpolated
    to each image's patch grid, ``num_hidden_layers`` pre-LayerNorm encoder
    blocks with biased q/k/v/o projections, and a convolutional patch
    merger projecting merged patches into the text backbone's hidden size
    (adding a newline token per merged row plus begin/end tokens).

    Args:
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the vision MLP and the patch merger.
        hidden_size (`int`, *optional*, defaults to 1152):
            Vision transformer hidden dimension.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Vision MLP intermediate dimension.
        interpolate_mode (`str`, *optional*, defaults to `"bilinear"`):
            Interpolation mode used when resizing the learned patch
            positional grid to the current image grid.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon shared by the merger RMSNorms and the block LayerNorms.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate on attention weights.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads; defaults to ``num_attention_heads``.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of vision transformer blocks.
        out_hidden_size (`int`, *optional*, defaults to 4096):
            Declared output hidden size (kept for HF config parity).
        patch_size (`int`, *optional*, defaults to 16):
            Square spatial patch size in pixels.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Spatial merge factor used by the patch merger convolution.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            Temporal patch size (images only, so 1).
        img_max_token_num (`int`, *optional*, defaults to 4096):
            Maximum image token count expected by the vision stack.
        max_image_size (`int`, *optional*, defaults to 2048):
            Maximum supported image edge; defines the learned position grid
            edge (``max_image_size // patch_size``).
        min_image_size (`int`, *optional*, defaults to 512):
            Minimum supported image edge.
        max_vit_seq_len (`int`, *optional*, defaults to 16384):
            Maximum sequence length produced by the vision transformer.
        text_hidden_size (`int`, *optional*, defaults to 3072):
            Hidden size of the consuming text backbone (merger output dim).
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
    """

    model_type = "hunyuan_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_act: str = "gelu",
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        interpolate_mode: str = "bilinear",
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        num_channels: int = 3,
        num_hidden_layers: int = 27,
        out_hidden_size: int = 4096,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        img_max_token_num: int = 4096,
        max_image_size: int = 2048,
        min_image_size: int = 512,
        max_vit_seq_len: int = 16384,
        text_hidden_size: int = 3072,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize the HunYuanVL vision tower configuration.

        Args:
            hidden_act: Activation used in the vision MLP and merger.
            hidden_size: Vision transformer hidden dimension.
            intermediate_size: Vision MLP intermediate dimension.
            interpolate_mode: Position-grid interpolation mode.
            rms_norm_eps: Epsilon for merger RMSNorms and block LayerNorms.
            attention_dropout: Dropout rate on attention weights.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of key/value heads (defaults to
                ``num_attention_heads``).
            num_channels: Number of input image channels.
            num_hidden_layers: Number of vision transformer blocks.
            out_hidden_size: Declared output hidden size (HF parity field).
            patch_size: Square spatial patch size in pixels.
            spatial_merge_size: Merge factor of the patch merger.
            temporal_patch_size: Temporal patch size (1 for images).
            img_max_token_num: Maximum image token count.
            max_image_size: Maximum supported image edge in pixels.
            min_image_size: Minimum supported image edge in pixels.
            max_vit_seq_len: Maximum ViT sequence length.
            text_hidden_size: Hidden size of the consuming text backbone.
            initializer_range: Stddev for weight initialization.
            **kwargs: Additional arguments forwarded to
                :class:`EasyDeLBaseConfig`.
        """
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.img_max_token_num = img_max_token_num
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.text_hidden_size = text_hidden_size
        self.initializer_range = initializer_range

        # All q/k/v/o projections in the HF vision tower carry biases.
        self.attention_bias = True
        self.head_dim = hidden_size // num_attention_heads
        # HF duck-typing aliases: the reference implementation reads these
        # through `attribute_map`, which does not exist on EasyDeL configs.
        self.attention_heads = self.num_attention_heads
        self.layer_norm_eps = rms_norm_eps


@register_config("hunyuan_vl_text")
class HunYuanVLTextConfig(EasyDeLBaseConfig):
    """Configuration for the HunYuanVL dense text decoder.

    The decoder is a HunYuan dense V1 transformer: grouped-query attention
    where rotary embeddings are applied first and per-head RMSNorms
    (``query_layernorm`` / ``key_layernorm``) are applied to the rotated
    queries/keys, a SwiGLU MLP, and RMSNorm pre-normalization. Positional
    information uses HunYuan's 3-axis multimodal RoPE: ``mrope_section``
    partitions half of each head across ``(width, height, image_index)``
    axes and each doubled section is taken contiguously from its axis.

    Args:
        vocab_size (`int`, *optional*, defaults to 290943):
            Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 4096):
            Decoder hidden dimension.
        intermediate_size (`int`, *optional*, defaults to 11008):
            MLP intermediate dimension.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of query attention heads.
        num_key_value_heads (`int`, *optional*):
            Number of grouped-query KV heads; defaults to
            ``num_attention_heads``.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation used inside the MLP block.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMS normalization epsilon (also used by the QK norms).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return KV caches during generation.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning-of-sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token id.
        eod_token_id (`int`, *optional*, defaults to 3):
            End-of-document token id.
        sep_token_id (`int`, *optional*, defaults to 4):
            Separator token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the LM head to the input embedding.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for rotary position embeddings. When the rope
            scaling dict carries HunYuan's dynamic-NTK ``alpha``, the base
            is folded to ``rope_theta * alpha**(head_dim / (head_dim - 2))``
            (equivalent to the HF reference with ``attention_scaling=1``).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether q/k/v/o projections carry biases.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability on attention weights.
        head_dim (`int`, *optional*):
            Per-head dimension; defaults to
            ``hidden_size // num_attention_heads``.
        rope_scaling (`dict`, *optional*):
            RoPE configuration. Accepts HunYuan's legacy fields:
            ``xdrope_section`` is aliased onto ``mrope_section`` and
            ``rope_type="xdrope"`` is treated as ``"dynamic"``. When no
            ``mrope_section`` is given a default 3-axis split is derived.
        rope_parameters (`dict`, *optional*):
            Alias for ``rope_scaling`` (HF v5 compatibility).
    """

    model_type = "hunyuan_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 290943,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 2,
        eod_token_id: int | None = 3,
        sep_token_id: int | None = 4,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        head_dim: int | None = None,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        **kwargs,
    ):
        """Initialize the HunYuanVL text decoder configuration.

        Args:
            vocab_size: Vocabulary size.
            hidden_size: Decoder hidden dimension.
            intermediate_size: MLP intermediate dimension.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of grouped-query KV heads; defaults
                to ``num_attention_heads``.
            hidden_act: Activation used inside the MLP block.
            max_position_embeddings: Maximum sequence length supported.
            initializer_range: Stddev for weight initialization.
            rms_norm_eps: RMS normalization epsilon.
            use_cache: Whether to return KV caches during generation.
            pad_token_id: Padding token id.
            bos_token_id: Beginning-of-sequence token id.
            eos_token_id: End-of-sequence token id(s).
            eod_token_id: End-of-document token id.
            sep_token_id: Separator token id.
            tie_word_embeddings: Whether to tie the LM head to the embedding.
            rope_theta: Base frequency for rotary position embeddings.
            attention_bias: Whether q/k/v/o projections carry biases.
            attention_dropout: Dropout probability on attention weights.
            head_dim: Per-head dimension; defaults to
                ``hidden_size // num_attention_heads``.
            rope_scaling: RoPE configuration dictionary (HunYuan legacy
                fields are normalized; see class docstring).
            rope_parameters: Alias for ``rope_scaling``.
            **kwargs: Additional arguments forwarded to
                :class:`EasyDeLBaseConfig`.

        Raises:
            NotImplementedError: If a 4-axis ``mrope_section`` is supplied
                (only the 3-axis ``(width, height, image_index)`` scheme is
                supported).
            ValueError: If ``mrope_section`` does not sum to
                ``head_dim // 2``.
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.eod_token_id = eod_token_id
        self.sep_token_id = sep_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        rope = dict(rope_scaling or rope_parameters or {})

        # HunYuan legacy aliases.
        xdrope_section = rope.pop("xdrope_section", None)
        if xdrope_section is not None:
            mrope_section = rope.get("mrope_section")
            if mrope_section is not None and [int(s) for s in mrope_section] != [int(s) for s in xdrope_section]:
                raise ValueError(
                    f"rope scaling carries both mrope_section={mrope_section} and a differing "
                    f"legacy xdrope_section={xdrope_section}."
                )
            rope.setdefault("mrope_section", list(xdrope_section))

        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type == "xdrope":
            rope_type = "dynamic"

        theta = float(rope.pop("rope_theta", rope_theta))
        alpha = rope.pop("alpha", None)
        if rope_type == "dynamic" and alpha is not None:
            # HunYuan's DynamicNTKAlpha rope is a plain rope with a folded
            # base and attention_scaling == 1.0.
            theta = theta * float(alpha) ** (self.head_dim / (self.head_dim - 2))
            rope_type = "default"
            rope.pop("factor", None)
        self.rope_theta = theta

        mrope_section = rope.get("mrope_section")
        if mrope_section is None:
            mrope_section = _default_mrope_section(self.head_dim)
        mrope_section = [int(s) for s in mrope_section]
        if len(mrope_section) != 3:
            raise NotImplementedError(
                "EasyDeL HunYuanVL supports the 3-axis (width, height, image_index) multimodal RoPE; "
                f"got mrope_section={mrope_section}."
            )
        if sum(mrope_section) != self.head_dim // 2:
            raise ValueError(
                f"Illegal mrope partition: sections must sum to head_dim // 2 = {self.head_dim // 2}; "
                f"got {mrope_section}."
            )

        self.rope_scaling = {
            "rope_type": rope_type,
            "type": rope_type,
            "mrope_section": mrope_section,
            # HunYuan splits the doubled sections into three contiguous
            # chunks (one per axis) — EasyDeL's chunked, non-repetition mRoPE.
            "mrope_interleaved": False,
            "repetition_style": False,
        }


@register_config("hunyuan_vl")
class HunYuanVLConfig(EasyDeLBaseConfig):
    """Composite configuration for HunYuanVL image-text models.

    Mirrors the Qwen2-VL family layout: the top-level config composes a
    :class:`HunYuanVLTextConfig` and a :class:`HunYuanVLVisionConfig` plus
    the token ids delimiting image spans in multimodal prompts. Every image
    contributes ``(h // m) * (w // m + 1) + 2`` placeholder tokens (grid
    tokens with a newline token per merged row, plus begin/end tokens
    emitted by the patch merger).

    Args:
        text_config (`HunYuanVLTextConfig` or `dict`, *optional*):
            Configuration of the text backbone.
        vision_config (`HunYuanVLVisionConfig` or `dict`, *optional*):
            Configuration of the vision tower.
        image_token_id (`int`, *optional*, defaults to 120120):
            Placeholder token id scattered with visual embeddings.
        im_start_id (`int`, *optional*, defaults to 120118):
            Token id marking the beginning of an image span.
        im_end_id (`int`, *optional*, defaults to 120119):
            Token id marking the end of an image span.
        im_newline_id (`int`, *optional*, defaults to 120121):
            Token id for newline-style separators inside image regions.
    """

    model_type = "hunyuan_vl"
    sub_configs: typing.ClassVar = {"vision_config": HunYuanVLVisionConfig, "text_config": HunYuanVLTextConfig}
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: Mapping[str, typing.Any] | HunYuanVLTextConfig | None = None,
        vision_config: Mapping[str, typing.Any] | HunYuanVLVisionConfig | None = None,
        image_token_id: int = 120120,
        im_start_id: int = 120118,
        im_end_id: int = 120119,
        im_newline_id: int = 120121,
        **kwargs,
    ):
        """Initialize the composite HunYuanVL configuration.

        Args:
            text_config: Text decoder configuration as a dict,
                :class:`HunYuanVLTextConfig`, or ``None`` to build from
                ``**kwargs`` (legacy flat checkpoints keep text fields at
                the top level).
            vision_config: Vision tower configuration as a dict,
                :class:`HunYuanVLVisionConfig`, or ``None`` for defaults.
            image_token_id: Placeholder token id for image content.
            im_start_id: Token id marking the beginning of an image span.
            im_end_id: Token id marking the end of an image span.
            im_newline_id: Token id for newline separators in image regions.
            **kwargs: Additional arguments forwarded to
                :class:`EasyDeLBaseConfig` and used as fallback when
                ``text_config`` is ``None``.
        """
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        # Keep the vision tower in sync with the consuming text backbone.
        self.vision_config.text_hidden_size = self.text_config.hidden_size

        self.image_token_id = image_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_newline_id = im_newline_id
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", True)

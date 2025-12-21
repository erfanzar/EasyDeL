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
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


class Qwen2VLVisionConfig(EasyDeLBaseConfig):
    """
    Configuration class for the vision component of Qwen2VL model.
    This class stores the configuration parameters for the vision encoder part of the Qwen2VL multimodal model.

    Args:
      depth (`int`, *optional*, defaults to 32):
        Number of layers in the vision transformer.
      embed_dim (`int`, *optional*, defaults to 1280):
        Dimensionality of the embeddings produced by the vision encoder.
      hidden_size (`int`, *optional*, defaults to 3584):
        Dimensionality of the intermediate representations in the vision transformer.
      hidden_act (`str`, *optional*, defaults to "quick_gelu"):
        The non-linear activation function used in the vision transformer.
      mlp_ratio (`int`, *optional*, defaults to 4):
        Ratio of the hidden size to the intermediate size in the MLP layers.
      num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads in the vision transformer.
      in_channels (`int`, *optional*, defaults to 3):
        Number of input channels for the image (typically 3 for RGB).
      patch_size (`int`, *optional*, defaults to 14):
        Size of the patches that the image is divided into.
      spatial_merge_size (`int`, *optional*, defaults to 2):
        The merge size for spatial dimensions in the vision transformer.
      temporal_patch_size (`int`, *optional*, defaults to 2):
        Size of the temporal patches when processing video input.
    """

    model_type = "qwen2_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range


class Qwen2VLTextConfig(EasyDeLBaseConfig):
    """Configuration for the Qwen2-VL text decoder stack."""

    model_type = "qwen2_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 8192,
        intermediate_size: int = 29568,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 80,
        attention_dropout: float = 0.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings

        self.rope_scaling = rope_scaling or rope_parameters
        if self.rope_scaling is not None:
            rtype = self.rope_scaling.get("type") or self.rope_scaling.get("rope_type", "default")
            if "mrope_section" in self.rope_scaling or rtype == "mrope":
                self.rope_scaling["type"] = "mrope"
                self.rope_scaling["rope_type"] = "mrope"
            elif "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.head_dim = hidden_size // num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


class Qwen2VLConfig(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2VLModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Args:
        text_config (`Union[Qwen2VLTextConfig, dict]`, *optional*):
            The config for the text decoder.
        vision_config (`Union[Qwen2VLVisionConfig, dict]`, *optional*):
            The config for the vision encoder.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode image prompts.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode video prompts.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.
    """

    model_type = "qwen2_vl"
    sub_configs: typing.ClassVar = {"vision_config": Qwen2VLVisionConfig, "text_config": Qwen2VLTextConfig}
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: typing.Mapping[str, typing.Any] | Qwen2VLTextConfig | None = None,
        vision_config: typing.Mapping[str, typing.Any] | Qwen2VLVisionConfig | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (r"(input_layernorm|post_attention_layernorm)/kernel", pmag.resolve(Replicated)),
            (r"norm/kernel", pmag.resolve(Replicated)),
            (r"visual/patch_embed/proj/kernel", pmag.resolve(ColumnWise)),
            (r"attn/qkv/kernel", pmag.resolve(ColumnWise)),
            (r"attn/qkv/bias", pmag.resolve(Replicated)),
            (r"attn/proj/kernel", pmag.resolve(RowWise)),
            (r"attn/proj/bias", pmag.resolve(Replicated)),
            (r"mlp/fc1/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/fc1/bias", pmag.resolve(Replicated)),
            (r"mlp/fc2/kernel", pmag.resolve(RowWise)),
            (r"mlp/fc2/bias", pmag.resolve(Replicated)),
            (r"norm(1|2)/scale", pmag.resolve(Replicated)),
            (r"norm(1|2)/bias", pmag.resolve(Replicated)),
            (r"visual/merger/ln_q/scale", pmag.resolve(Replicated)),
            (r"visual/merger/ln_q/bias", pmag.resolve(Replicated)),
            (r"visual/merger/mlp/0/kernel", pmag.resolve(ColumnWise)),
            (r"visual/merger/mlp/0/bias", pmag.resolve(Replicated)),
            (r"visual/merger/mlp/2/kernel", pmag.resolve(RowWise)),
            (r"visual/merger/mlp/2/bias", pmag.resolve(Replicated)),
            (r"multi_modal_projector/linear_1/kernel", pmag.resolve(ColumnWise)),
            (r"multi_modal_projector/linear_1/bias", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"lm_head/bias", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

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
        for layer_idx in range(self.num_hidden_layers):
            if self.sliding_window is not None and self.use_sliding_window:
                mapping[layer_idx] = AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=self.sliding_window)
        return mapping

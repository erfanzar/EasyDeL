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
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config

logger = get_logger(__name__)


def _get_partition_rules(self, *args, **kwargs):
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
        (r"self_attn/qk_norm/scale", pmag.resolve(Replicated)),
        (r"feed_forward/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
        (r"feed_forward/down_proj/kernel", pmag.resolve(RowWise)),
        (r"feed_forward/router/kernel", pmag.resolve(ColumnWise)),
        (r"feed_forward/experts/gate_up_proj", pmag.resolve(ColumnWise)),
        (r"feed_forward/experts/down_proj", pmag.resolve(RowWise)),
        (
            r"feed_forward/shared_expert/(gate_proj|up_proj)/kernel",
            pmag.resolve(ColumnWise),
        ),
        (r"feed_forward/shared_expert/down_proj/kernel", pmag.resolve(RowWise)),
        (
            r"(input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm|norm)/kernel",
            pmag.resolve(Replicated),
        ),
        (r"lm_head/kernel", pmag.resolve(ColumnWise)),
        (r"patch_embedding/linear/kernel", pmag.resolve(ColumnWise)),
        (r"class_embedding", pmag.resolve(Replicated)),
        (r"positional_embedding_vlm", pmag.resolve(ColumnWise)),
        (r"(layernorm_pre|layernorm_post)/scale", pmag.resolve(Replicated)),
        (r"(layernorm_pre|layernorm_post)/bias", pmag.resolve(Replicated)),
        (r"model/layers/\d+/self_attn/o_proj/kernel", pmag.resolve(RowWise)),
        (r"model/layers/\d+/self_attn/.*proj/bias", pmag.resolve(Replicated)),
        (r"model/layers/\d+/mlp/fc1/kernel", pmag.resolve(ColumnWise)),
        (r"model/layers/\d+/mlp/fc2/kernel", pmag.resolve(RowWise)),
        (r"model/layers/\d+/mlp/fc(1|2)/bias", pmag.resolve(Replicated)),
        (r"vision_adapter/mlp/fc1/kernel", pmag.resolve(ColumnWise)),
        (r"vision_adapter/mlp/fc2/kernel", pmag.resolve(RowWise)),
        (r"vision_adapter/mlp/fc(1|2)/bias", pmag.resolve(Replicated)),
        (r"multi_modal_projector/linear_1/kernel", pmag.resolve(ColumnWise)),
        (r"multi_modal_projector/linear_1/bias", pmag.resolve(Replicated)),
        (r"score/kernel", pmag.resolve(RowWise)),
        (r"score/bias", pmag.resolve(Replicated)),
        (r".*bias", pmag.resolve(Replicated)),
        (r".*", pmag.resolve(Replicated)),
    )


@register_config("llama4_vision_model")
class Llama4VisionConfig(EasyDeLBaseConfig):
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
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
        initializer_range: float = 0.02,
        pixel_shuffle_ratio=0.5,
        projector_input_dim=4096,
        projector_output_dim=4096,
        multi_modal_projector_bias=False,
        projector_dropout=0.0,
        attention_dropout=0.0,
        rope_theta=10000,
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

    get_partition_rules = _get_partition_rules


@register_config("llama4_text")
class Llama4TextConfig(EasyDeLBaseConfig):
    model_type = "llama4_text"

    def __init__(
        self,
        vocab_size=202048,
        hidden_size=5120,
        intermediate_size=8192,
        intermediate_size_mlp=16384,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_local_experts=16,
        moe_layers=None,
        interleave_moe_layer_step=1,
        use_qk_norm=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        rope_scaling=None,
        no_rope_layers=None,
        no_rope_layer_interval=4,
        attention_chunk_size=8192,
        attn_temperature_tuning=4,
        floor_scale=8192,
        attn_scale=0.1,
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

    get_partition_rules = _get_partition_rules


@register_config("llama4")
class Llama4Config(EasyDeLBaseConfig):
    model_type = "llama4"
    sub_configs: typing.ClassVar = {"text_config": Llama4TextConfig, "vision_config": Llama4VisionConfig}
    attribute_map: typing.ClassVar = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        boi_token_index=200080,
        eoi_token_index=200081,
        image_token_index=200092,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = Llama4VisionConfig()
            logger.info("vision_config is None, using default llama4 vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = Llama4VisionConfig(**vision_config)
        elif isinstance(vision_config, Llama4VisionConfig):
            self.vision_config = vision_config

        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = Llama4TextConfig()
            logger.info("text_config is None, using default llama4 text config")
        elif isinstance(text_config, dict):
            self.text_config = Llama4TextConfig(**text_config)
        elif isinstance(text_config, Llama4TextConfig):
            self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    get_partition_rules = _get_partition_rules


__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]

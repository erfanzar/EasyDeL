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

"""Configuration for Qwen3.5 text and multimodal models."""

import typing
from collections.abc import Mapping

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLVisionConfig


@register_config("qwen3_5_text")
class Qwen3_5TextConfig(Qwen3NextConfig):
    """
    Configuration objects inherit from [`Qwen3NextConfig`] and can be used to control the model outputs. Read
    the documentation from [`Qwen3NextConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Vocabulary size of the Qwen3.5 text model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimensionality of the dense MLP intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of query attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value heads (GQA).
        head_dim (`int`, *optional*, defaults to 256):
            Dimensionality of each attention head.
        hidden_act (`str`, *optional*, defaults to ``"silu"``):
            Activation function in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length the model supports.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalisation layers.
        rope_theta (`float`, *optional*):
            Base period for rotary position embeddings. Resolved from ``rope_scaling``/``rope_parameters``
            if not set directly; falls back to 10000.0.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Fraction of head dimension that uses rotary embeddings.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type schedule (``"full_attention"`` or ``"linear_attention"``).
        full_attention_interval (`int`, *optional*, defaults to 4):
            Interval between full-attention layers when using hybrid attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size for the convolutional component in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Key head dimension for linear attention layers.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Value head dimension for linear attention layers.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads in linear attention layers.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Sparse-step interval for MoE layers.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size for MoE expert MLPs.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size for shared expert MLPs.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts activated per token.
        num_experts (`int`, *optional*, defaults to 256):
            Total number of MoE experts.
        norm_topk_prob (`bool`, *optional*, defaults to ``True``):
            Whether to normalise top-k routing probabilities.
        output_router_logits (`bool`, *optional*, defaults to ``False``):
            Whether to output MoE router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient for the MoE auxiliary load-balancing loss.
        mlp_only_layers (`list[int]`, *optional*):
            Layer indices that use dense MLPs instead of MoE. Defaults to all layers
            (Qwen3.5 text is dense).
    """

    model_type = "qwen3_5_text"

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float | None = None,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 0.25,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        **kwargs,
    ):
        rope_scaling = rope_scaling or rope_parameters
        if rope_theta is None and isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000.0

        # Qwen3.5 text uses dense MLPs by default.
        if mlp_only_layers is None or len(mlp_only_layers) == 0:
            mlp_only_layers = list(range(num_hidden_layers))

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            partial_rotary_factor=partial_rotary_factor,
            layer_types=layer_types,
            full_attention_interval=full_attention_interval,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            decoder_sparse_step=decoder_sparse_step,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            num_experts=num_experts,
            norm_topk_prob=norm_topk_prob,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            mlp_only_layers=mlp_only_layers,
            **kwargs,
        )
        # Qwen3.5 linear attention uses split projections in HF:
        # in_proj_qkv, in_proj_z, in_proj_b, in_proj_a.
        self.linear_attention_separate_proj = True
        # Mirror HF naming for rope config interop.
        self.rope_parameters = rope_scaling

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Qwen3.5 text checkpoints are dense (no MoE FFN layers)."""
        return False

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Returns partition rules for model sharding."""
        return None


@register_config("qwen3_5_vision")
class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    """
    Configuration for the Qwen3.5 vision encoder, inheriting from [`Qwen3VLVisionConfig`].

    Qwen3.5 does not use deepstack mergers, so ``deepstack_visual_indexes`` defaults to
    an empty list.  A bootstrap index of ``[0]`` is passed to the parent constructor to
    satisfy its initialisation requirements.

    Args:
        deepstack_visual_indexes (`list[int]`, *optional*):
            Indices of vision transformer layers whose outputs are used as deepstack
            embeddings.  Defaults to an empty list for Qwen3.5.
        **kwargs: Forwarded to :class:`Qwen3VLVisionConfig`.
    """

    model_type = "qwen3_5"
    base_config_key = "vision_config"

    def __init__(
        self,
        deepstack_visual_indexes: list[int] | None = None,
        **kwargs,
    ):
        # Keep explicit empty list as empty (Qwen3.5 does not use deepstack mergers).
        requested_indexes = [] if deepstack_visual_indexes is None else list(deepstack_visual_indexes)
        bootstrap_indexes = requested_indexes if requested_indexes else [0]
        super().__init__(deepstack_visual_indexes=bootstrap_indexes, **kwargs)
        self.deepstack_visual_indexes = requested_indexes


@register_config("qwen3_5")
class Qwen3_5Config(EasyDeLBaseConfig):
    """
    Configuration for the Qwen3.5 multimodal (vision-language) model.

    Composes a :class:`Qwen3_5TextConfig` for the language backbone and a
    :class:`Qwen3_5VisionConfig` for the vision encoder.

    Args:
        text_config (`Mapping` or `Qwen3_5TextConfig`, *optional*):
            Text model sub-configuration. Instantiated from a dict if needed.
        vision_config (`Mapping` or `Qwen3_5VisionConfig`, *optional*):
            Vision encoder sub-configuration. Instantiated from a dict if needed.
        image_token_id (`int`, *optional*, defaults to 248056):
            Token id used to represent image placeholders in the input.
        video_token_id (`int`, *optional*, defaults to 248057):
            Token id used to represent video placeholders in the input.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            Token id marking the start of a vision span.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            Token id marking the end of a vision span.
        tie_word_embeddings (`bool`, *optional*, defaults to ``False``):
            Whether to tie input and output embedding weights.
    """

    model_type = "qwen3_5"
    sub_configs: typing.ClassVar = {
        "text_config": Qwen3_5TextConfig,
        "vision_config": Qwen3_5VisionConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        text_config: Mapping[str, typing.Any] | Qwen3_5TextConfig | None = None,
        vision_config: Mapping[str, typing.Any] | Qwen3_5VisionConfig | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self._fix_parent_kws(text_config, kwargs))
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self._fix_parent_kws(vision_config, kwargs))
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = True) -> Qwen3_5TextConfig:
        """Get the text configuration object."""
        return self.text_config  # pyright: ignore[reportReturnType]

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Returns partition rules for model sharding."""
        return None


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]

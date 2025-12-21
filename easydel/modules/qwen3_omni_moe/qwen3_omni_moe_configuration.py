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

"""Configuration classes for Qwen3OmniMoe multimodal model.

This module provides configuration classes for Qwen3OmniMoe, a multimodal
model that processes text, vision (images/videos), and audio inputs,
with optional speech synthesis capabilities.

The model consists of three main components:
- Thinker: Multimodal understanding (audio encoder, vision encoder, text decoder)
- Talker: Speech generation with MoE + shared experts
- Code2Wav: Codec-to-waveform vocoder
"""

import typing

from eformer.common_types import EMPTY, MODE_TRAIN, TP, ColumnWise, DynamicShardingAxes, Replicated, RowWise
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType
from easydel.layers.moe.utils import get_moe_partition_spec

logger = get_logger(__name__)


class ExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism (EPxTP) sharding axes."""

    axes: typing.ClassVar = [TP, EMPTY, EMPTY]
    mode: typing.ClassVar = MODE_TRAIN


@register_config("qwen3_omni_moe_audio_encoder")
class Qwen3OmniMoeAudioEncoderConfig(EasyDeLBaseConfig):
    """Configuration class for Qwen3OmniMoe audio encoder.

    The audio encoder processes mel-spectrogram inputs through:
    1. Convolutional feature extraction
    2. Transformer encoder blocks
    3. Output projection to match text decoder dimension

    Args:
        num_mel_bins: Number of mel frequency bins. Defaults to 128.
        encoder_layers: Number of transformer layers. Defaults to 32.
        encoder_attention_heads: Number of attention heads. Defaults to 20.
        encoder_ffn_dim: FFN intermediate dimension. Defaults to 5120.
        d_model: Hidden dimension. Defaults to 1280.
        dropout: Dropout probability. Defaults to 0.0.
        attention_dropout: Attention dropout. Defaults to 0.0.
        activation_function: Activation function name. Defaults to "gelu".
        activation_dropout: Activation dropout. Defaults to 0.0.
        scale_embedding: Whether to scale embeddings. Defaults to False.
        initializer_range: Weight initialization std. Defaults to 0.02.
        max_source_positions: Maximum source positions. Defaults to 1500.
        n_window: Window size for attention. Defaults to 100.
        output_dim: Output dimension (matches text decoder). Defaults to 3584.
        n_window_infer: Inference window size. Defaults to 400.
        conv_chunksize: Convolution chunk size. Defaults to 500.
        downsample_hidden_size: Downsampling hidden size. Defaults to 480.
    """

    model_type = "qwen3_omni_moe_audio_encoder"
    base_config_key = "audio_config"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        n_window_infer: int = 400,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


Qwen3OmniMoeAudioConfig = Qwen3OmniMoeAudioEncoderConfig


@register_config("qwen3_omni_moe_vision_encoder")
class Qwen3OmniMoeVisionEncoderConfig(EasyDeLBaseConfig):
    """Configuration class for Qwen3OmniMoe vision encoder.

    The vision encoder processes images and videos through:
    1. 3D patch embedding (spatial + temporal)
    2. Transformer encoder blocks
    3. Patch merger with spatial downsampling

    Args:
        depth: Number of transformer layers. Defaults to 27.
        hidden_size: Hidden dimension. Defaults to 1152.
        hidden_act: Activation function. Defaults to "gelu_pytorch_tanh".
        intermediate_size: FFN intermediate dimension. Defaults to 4304.
        num_heads: Number of attention heads. Defaults to 16.
        in_channels: Number of input channels. Defaults to 3.
        patch_size: Spatial patch size. Defaults to 16.
        spatial_merge_size: Spatial merge factor. Defaults to 2.
        temporal_patch_size: Temporal patch size for video. Defaults to 2.
        out_hidden_size: Output dimension (matches text decoder). Defaults to 3584.
        num_position_embeddings: Maximum position embeddings. Defaults to 2304.
        deepstack_visual_indexes: Layer indices for deepstack. Defaults to [8, 16, 24].
        tokens_per_second: Temporal tokens per second. Defaults to 2.0.
        initializer_range: Weight initialization std. Defaults to 0.02.
    """

    model_type = "qwen3_omni_moe_vision_encoder"
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
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [8, 16, 24]
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes


Qwen3OmniMoeVisionConfig = Qwen3OmniMoeVisionEncoderConfig


@register_config("qwen3_omni_moe_text")
class Qwen3OmniMoeTextConfig(EasyDeLBaseConfig):
    """Configuration class for Qwen3OmniMoe Thinker text decoder with MoE.

    The text decoder processes text tokens and multimodal embeddings,
    using Mixture of Experts layers for capacity.

    Args:
        vocab_size: Vocabulary size. Defaults to 151936.
        hidden_size: Hidden dimension. Defaults to 3584.
        intermediate_size: Dense MLP dimension. Defaults to 18944.
        num_hidden_layers: Number of decoder layers. Defaults to 28.
        num_attention_heads: Number of attention heads. Defaults to 28.
        num_key_value_heads: Number of KV heads for GQA. Defaults to 4.
        head_dim: Dimension per head. Auto-computed if None.
        hidden_act: Activation function. Defaults to "silu".
        max_position_embeddings: Maximum sequence length. Defaults to 32768.
        initializer_range: Weight initialization std. Defaults to 0.02.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-6.
        use_cache: Whether to use KV cache. Defaults to True.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
        rope_theta: RoPE base frequency. Defaults to 1000000.0.
        rope_scaling: RoPE scaling configuration. Defaults to None.
        attention_bias: Whether to use attention bias. Defaults to False.
        attention_dropout: Attention dropout rate. Defaults to 0.0.
        use_sliding_window: Whether to use sliding window. Defaults to False.
        sliding_window: Sliding window size. Defaults to None.
        max_window_layers: Layers using sliding window. Defaults to 28.
        decoder_sparse_step: MoE layer frequency. Defaults to 1.
        moe_intermediate_size: MoE expert hidden dimension. Defaults to 768.
        num_experts_per_tok: Active experts per token. Defaults to 8.
        num_experts: Total number of experts. Defaults to 128.
        norm_topk_prob: Normalize top-k probabilities. Defaults to True.
        output_router_logits: Return router logits. Defaults to False.
        router_aux_loss_coef: Router auxiliary loss coefficient. Defaults to 0.001.
        mlp_only_layers: Layers using dense MLP. Defaults to None.
        layer_types: Attention type per layer. Auto-computed if None.
    """

    model_type = "qwen3_omni_moe_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int = 28,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 768,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            rtype = self.rope_scaling["type"]
            if rtype == "mrope" or ("mrope_section" in self.rope_scaling and rtype == "default"):
                self.rope_scaling["type"] = "mrope"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Get attention mask details for sliding window attention."""
        mapping = {}
        if self.sliding_window is not None and self.use_sliding_window:
            for layer_idx in range(self.num_hidden_layers):
                if layer_idx >= self.max_window_layers:
                    mapping[layer_idx] = AttnMaskDetail(
                        mask_type=AttnMaskType.SLIDING,
                        size=self.sliding_window,
                    )
        return mapping


@register_config("qwen3_omni_moe_thinker")
class Qwen3OmniMoeThinkerConfig(EasyDeLBaseConfig):
    """Configuration class for Qwen3OmniMoe Thinker module.

    The Thinker module combines audio, vision, and text for multimodal
    understanding and reasoning.

    Args:
        audio_config: Audio encoder configuration.
        vision_config: Vision encoder configuration.
        text_config: Text decoder configuration.
        audio_token_id: Token ID for audio placeholders. Defaults to 151646.
        image_token_id: Token ID for image placeholders. Defaults to 151655.
        video_token_id: Token ID for video placeholders. Defaults to 151656.
        audio_start_token_id: Token ID for audio sequence start. Defaults to 151647.
        vision_start_token_id: Token ID for vision sequence start. Defaults to 151652.
        vision_end_token_id: Token ID for vision sequence end. Defaults to 151653.
        position_id_per_seconds: Position IDs per second for audio. Defaults to 25.
        user_token_id: User token ID. Defaults to 872.
        initializer_range: Weight initialization std. Defaults to 0.02.
    """

    model_type = "qwen3_omni_moe_thinker"
    sub_configs: typing.ClassVar = {
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
        "vision_config": Qwen3OmniMoeVisionEncoderConfig,
        "text_config": Qwen3OmniMoeTextConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_id: int = 151646,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        position_id_per_seconds: int = 25,
        audio_start_token_id: int = 151647,
        user_token_id: int = 872,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.initializer_range = initializer_range

        if isinstance(vision_config, dict):
            vision_config = Qwen3OmniMoeVisionEncoderConfig(**self._fix_parent_kws(vision_config, kwargs))
        elif vision_config is None:
            vision_config = Qwen3OmniMoeVisionEncoderConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**self._fix_parent_kws(audio_config, kwargs))
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3OmniMoeTextConfig(**self._fix_parent_kws(text_config, kwargs))
        elif text_config is None:
            text_config = Qwen3OmniMoeTextConfig(**kwargs)
        self.text_config = text_config

        if not hasattr(self, "rope_theta"):
            self.rope_theta = self.text_config.rope_theta
        if not hasattr(self, "rope_scaling"):
            self.rope_scaling = getattr(self.text_config, "rope_scaling", None)

        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for model parallelism."""
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
            (r"mlp/gate/kernel", pmag.resolve(Replicated if self.use_expert_tensor_mode else ColumnWise)),
            (r"mlp/gate/bias", pmag.resolve(Replicated)),
            (
                r"mlp/experts/(gate_proj|up_proj)/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="column",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (
                r"mlp/experts/down_proj/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="row",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (r"mlp/experts/.*bias", pmag.resolve(Replicated)),
            (r".*norm.*/kernel", pmag.resolve(Replicated)),
            (r"attn/(qkv|proj)/kernel", pmag.resolve(ColumnWise)),
            (r"attn/.*bias", pmag.resolve(Replicated)),
            (r"mlp/(fc1|fc2)/kernel", pmag.resolve(ColumnWise)),
            (r".*norm.*/kernel", pmag.resolve(Replicated)),
            (r"self_attn/(k_proj|v_proj|q_proj|out_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/.*bias", pmag.resolve(Replicated)),
            (r"fc(1|2)/kernel", pmag.resolve(ColumnWise)),
            (r".*norm.*/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*", pmag.resolve(Replicated)),
        )


@register_config("qwen3_omni_moe_talker_code_predictor")
class Qwen3OmniMoeTalkerCodePredictorConfig(EasyDeLBaseConfig):
    """Configuration for Talker code predictor (acoustic token predictor).

    This is a lightweight transformer for predicting acoustic codec tokens
    from talker hidden states. Uses non-MoE architecture.

    Args:
        vocab_size: Codebook vocabulary size. Defaults to 2048.
        hidden_size: Hidden dimension. Defaults to 1024.
        intermediate_size: FFN intermediate dimension. Defaults to 3072.
        num_hidden_layers: Number of decoder layers. Defaults to 5.
        num_attention_heads: Number of attention heads. Defaults to 16.
        num_key_value_heads: Number of KV heads. Defaults to 8.
        head_dim: Dimension per head. Defaults to 128.
        hidden_act: Activation function. Defaults to "silu".
        max_position_embeddings: Maximum sequence length. Defaults to 32768.
        initializer_range: Weight initialization std. Defaults to 0.02.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-6.
        use_cache: Whether to use KV cache. Defaults to True.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        rope_scaling: RoPE scaling configuration. Defaults to None.
        attention_bias: Whether to use attention bias. Defaults to False.
        attention_dropout: Attention dropout rate. Defaults to 0.0.
        sliding_window: Sliding window size. Defaults to None.
        layer_types: Attention type per layer. Auto-computed if None.
        num_code_groups: Number of residual codebook groups. Defaults to 32.
    """

    model_type = "qwen3_omni_moe_talker_code_predictor"
    base_config_key = "code_predictor_config"

    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        num_code_groups: int = 32,
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
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.num_code_groups = num_code_groups

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]


@register_config("qwen3_omni_moe_talker_text")
class Qwen3OmniMoeTalkerTextConfig(EasyDeLBaseConfig):
    """Configuration for Talker text model with MoE and shared experts.

    The Talker text model uses MoE layers with additional shared experts
    that are gated by a sigmoid function.

    Args:
        vocab_size: Vocabulary size. Defaults to 3072.
        hidden_size: Hidden dimension. Defaults to 1024.
        intermediate_size: Dense MLP dimension. Defaults to 2048.
        num_hidden_layers: Number of decoder layers. Defaults to 20.
        num_attention_heads: Number of attention heads. Defaults to 16.
        num_key_value_heads: Number of KV heads. Defaults to 2.
        head_dim: Dimension per head. Auto-computed if None.
        hidden_act: Activation function. Defaults to "silu".
        max_position_embeddings: Maximum sequence length. Defaults to 32768.
        initializer_range: Weight initialization std. Defaults to 0.02.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-6.
        use_cache: Whether to use KV cache. Defaults to True.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        rope_scaling: RoPE scaling configuration. Defaults to None.
        attention_bias: Whether to use attention bias. Defaults to False.
        attention_dropout: Attention dropout rate. Defaults to 0.0.
        sliding_window: Sliding window size. Defaults to None.
        decoder_sparse_step: MoE layer frequency. Defaults to 1.
        moe_intermediate_size: MoE expert hidden dimension. Defaults to 384.
        num_experts_per_tok: Active experts per token. Defaults to 8.
        num_experts: Total number of experts. Defaults to 128.
        norm_topk_prob: Normalize top-k probabilities. Defaults to False.
        output_router_logits: Return router logits. Defaults to False.
        router_aux_loss_coef: Router auxiliary loss coefficient. Defaults to 0.001.
        mlp_only_layers: Layers using dense MLP. Defaults to None.
        shared_expert_intermediate_size: Shared expert FFN size. Defaults to 2048.
    """

    model_type = "qwen3_omni_moe_talker_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 3072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 384,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        shared_expert_intermediate_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window

        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        self.shared_expert_intermediate_size = shared_expert_intermediate_size

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]


@register_config("qwen3_omni_moe_talker")
class Qwen3OmniMoeTalkerConfig(EasyDeLBaseConfig):
    """Configuration for Talker module (speech generation).

    The Talker module combines a code predictor and text model for
    generating acoustic tokens from semantic representations.

    Args:
        code_predictor_config: Code predictor configuration.
        text_config: Talker text model configuration.
        num_code_groups: Number of codebook groups. Defaults to 32.
        thinker_hidden_size: Hidden size from thinker. Defaults to 3584.
        accept_hidden_layer: Layer for accepting hidden states. Defaults to 18.
        codec_bos_token_id: Codec BOS token ID. Defaults to 4197.
        codec_eos_token_id: Codec EOS token ID. Defaults to 4198.
        codec_pad_token_id: Codec padding token ID. Defaults to 4196.
        codec_nothink_id: No-think token ID. Defaults to 4203.
        codec_think_bos_id: Think BOS token ID. Defaults to 4204.
        codec_think_eos_id: Think EOS token ID. Defaults to 4205.
        audio_token_id: Audio token ID. Defaults to 151646.
        image_token_id: Image token ID. Defaults to 151655.
        video_token_id: Video token ID. Defaults to 151656.
        vision_start_token_id: Vision start token ID. Defaults to 151652.
        position_id_per_seconds: Position IDs per second. Defaults to 25.
        audio_start_token_id: Audio start token ID. Defaults to 151669.
        speaker_id: Speaker ID mapping. Defaults to None.
    """

    model_type = "qwen3_omni_moe_talker"
    sub_configs: typing.ClassVar = {
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
        "text_config": Qwen3OmniMoeTalkerTextConfig,
    }

    def __init__(
        self,
        code_predictor_config=None,
        text_config=None,
        num_code_groups: int = 32,
        thinker_hidden_size: int = 2048,
        codec_eos_token_id: int = 4198,
        accept_hidden_layer: int = 18,
        codec_nothink_id: int = 4203,
        codec_think_bos_id: int = 4204,
        codec_think_eos_id: int = 4205,
        codec_pad_id: int = 4196,
        codec_bos_id: int = 4197,
        audio_token_id: int = 151646,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        position_id_per_seconds: int = 25,
        audio_start_token_id: int = 151669,
        speaker_id=None,
        spatial_merge_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if code_predictor_config is None:
            code_predictor_config = {}
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**kwargs)
        elif isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(
                **self._fix_parent_kws(code_predictor_config, kwargs)
            )

        if text_config is None:
            text_config = {}
            self.text_config = Qwen3OmniMoeTalkerTextConfig(**kwargs)
        elif isinstance(text_config, Qwen3OmniMoeTalkerTextConfig):
            self.text_config = text_config
        else:
            self.text_config = Qwen3OmniMoeTalkerTextConfig(**self._fix_parent_kws(text_config, kwargs))

        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.accept_hidden_layer = accept_hidden_layer
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.vision_start_token_id = vision_start_token_id
        self.speaker_id = speaker_id
        self.spatial_merge_size = spatial_merge_size

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for Talker model parallelism."""
        pmag = self.partition_manager
        return (
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"self_attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/gate/kernel", pmag.resolve(Replicated if self.use_expert_tensor_mode else ColumnWise)),
            (
                r"mlp/experts/(gate_proj|up_proj)/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="column",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (
                r"mlp/experts/down_proj/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="row",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (r"mlp/shared_expert/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/shared_expert/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/shared_expert_gate/kernel", pmag.resolve(Replicated)),
            (r".*norm.*/kernel", pmag.resolve(Replicated)),
            (r"codec_head/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


@register_config("qwen3_omni_moe_code2wav")
class Qwen3OmniMoeCode2WavConfig(EasyDeLBaseConfig):
    """Configuration for Code2Wav vocoder module.

    The vocoder converts discrete audio codes to waveforms using:
    1. Transformer decoder with sliding window attention
    2. ConvNeXt upsampling blocks

    Args:
        codebook_size: Size of each codebook. Defaults to 2048.
        hidden_size: Hidden dimension. Defaults to 1024.
        intermediate_size: FFN intermediate dimension. Defaults to 3072.
        num_hidden_layers: Number of transformer layers. Defaults to 8.
        num_attention_heads: Number of attention heads. Defaults to 16.
        num_key_value_heads: Number of KV heads. Defaults to 16.
        head_dim: Dimension per head. Auto-computed if None.
        hidden_act: Activation function. Defaults to "silu".
        max_position_embeddings: Maximum sequence length. Defaults to 8000.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        attention_bias: Whether to use attention bias. Defaults to False.
        attention_dropout: Attention dropout rate. Defaults to 0.0.
        sliding_window: Sliding window size. Defaults to 72.
        layer_scale_initial_scale: LayerScale init value. Defaults to 0.01.
        num_quantizers: Number of residual quantizers. Defaults to 16.
        upsample_rates: Upsample rates for ConvNeXt. Defaults to (8, 5, 4, 3).
        upsampling_ratios: Additional upsampling ratios. Defaults to (2, 2).
        decoder_dim: Decoder output dimension. Defaults to 1536.
    """

    model_type = "qwen3_omni_moe_code2wav"
    base_config_key = "code2wav_config"

    def __init__(
        self,
        codebook_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8000,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 72,
        layer_scale_initial_scale: float = 0.01,
        num_quantizers: int = 16,
        upsample_rates: tuple[int, ...] = (8, 5, 4, 3),
        upsampling_ratios: tuple[int, ...] = (2, 2),
        decoder_dim: int = 1536,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.num_quantizers = num_quantizers
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.decoder_dim = decoder_dim

    @property
    def layer_types(self) -> list[str]:
        """All layers use sliding attention."""
        return ["sliding_attention"] * self.num_hidden_layers

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for Code2Wav model parallelism."""
        pmag = self.partition_manager
        return (
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r".*conv.*/kernel", pmag.resolve(Replicated)),
            (r".*conv.*/bias", pmag.resolve(Replicated)),
            (r".*norm.*/kernel", pmag.resolve(Replicated)),
            (r".*layer_scale/scale", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


@register_config("qwen3_omni_moe")
class Qwen3OmniMoeConfig(EasyDeLBaseConfig):
    """Main configuration for complete Qwen3OmniMoe model.

    This configuration wraps the three main components:
    - Thinker: Multimodal understanding (audio, vision, text)
    - Talker: Speech generation with MoE + shared experts
    - Code2Wav: Codec-to-waveform vocoder

    Args:
        thinker_config: Thinker module configuration.
        talker_config: Talker module configuration.
        code2wav_config: Code2Wav vocoder configuration.
        enable_audio_output: Whether to load Talker/Code2Wav. Defaults to True.
        im_start_token_id: IM start token ID. Defaults to 151644.
        im_end_token_id: IM end token ID. Defaults to 151645.
        tts_pad_token_id: TTS padding token ID. Defaults to 151671.
        tts_bos_token_id: TTS BOS token ID. Defaults to 151672.
        tts_eos_token_id: TTS EOS token ID. Defaults to 151673.
        system_token_id: System token ID. Defaults to 8948.
        user_token_id: User token ID. Defaults to 872.
        assistant_token_id: Assistant token ID. Defaults to 77091.
        tie_word_embeddings: Whether to tie embeddings. Defaults to False.
    """

    model_type = "qwen3_omni_moe"
    sub_configs: typing.ClassVar = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
        "code2wav_config": Qwen3OmniMoeCode2WavConfig,
    }
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        code2wav_config=None,
        enable_audio_output: bool = True,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        tts_pad_token_id: int = 151671,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        system_token_id: int = 8948,
        user_token_id: int = 872,
        assistant_token_id: int = 77091,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if thinker_config is None:
            thinker_config = {}
        if talker_config is None:
            talker_config = {}
        if code2wav_config is None:
            code2wav_config = {}

        if isinstance(thinker_config, Qwen3OmniMoeThinkerConfig):
            self.thinker_config = thinker_config
        else:
            self.thinker_config = Qwen3OmniMoeThinkerConfig(**self._fix_parent_kws(thinker_config, kwargs))

        if isinstance(talker_config, Qwen3OmniMoeTalkerConfig):
            self.talker_config = talker_config
        else:
            self.talker_config = Qwen3OmniMoeTalkerConfig(**self._fix_parent_kws(talker_config, kwargs))

        if isinstance(code2wav_config, Qwen3OmniMoeCode2WavConfig):
            self.code2wav_config = code2wav_config
        else:
            self.code2wav_config = Qwen3OmniMoeCode2WavConfig(**self._fix_parent_kws(code2wav_config, kwargs))

        self.enable_audio_output = enable_audio_output
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id

    def get_text_config(self, decoder: bool = True) -> Qwen3OmniMoeTextConfig:
        """Get the text configuration from thinker."""
        return self.thinker_config.get_text_config(decoder)

    def get_partition_rules(self, *args, **kwargs):
        """Get partition rules for model parallelism.

        Combines partition rules from child configs (thinker, talker, code2wav)
        with appropriate prefixes for the full model hierarchy.
        """
        pmag = self.partition_manager

        # Get rules from child configs and prefix them
        thinker_rules = self.thinker_config.get_partition_rules(*args, **kwargs)
        talker_rules = self.talker_config.get_partition_rules(*args, **kwargs)
        code2wav_rules = self.code2wav_config.get_partition_rules(*args, **kwargs)

        # Combine all rules with top-level rules first, then child rules, then fallback
        return (
            # Top-level embeddings and heads
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            # Child config rules (prefixed)
            *thinker_rules,
            *talker_rules,
            *code2wav_rules,
            # Fallback rules
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )


__all__ = [
    "Qwen3OmniMoeAudioConfig",
    "Qwen3OmniMoeAudioEncoderConfig",
    "Qwen3OmniMoeCode2WavConfig",
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeTalkerCodePredictorConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeTalkerTextConfig",
    "Qwen3OmniMoeTextConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeVisionConfig",
    "Qwen3OmniMoeVisionEncoderConfig",
]

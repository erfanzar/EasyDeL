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
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("whisper")
class WhisperConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 51865):
            Vocabulary size of the Whisper model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~easydel.modules.WhisperModel`].
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel bins used by the feature extractor.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the decoder feed-forward network (FFN) layer.
        encoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder feed-forward network (FFN) layer.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556) for
            more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556) for
            more details.
        d_model (`int`, *optional*, defaults to 256):
            Dimensionality of the layers and the pooler layer.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to False):
            Scale embeddings by dividing by sqrt(d_model).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length allowed for the source text input to the model. tp.Any longer inputs will be
            truncated.
        max_target_positions (`int`, *optional*, defaults to 448):
            The maximum sequence length allowed for the target text input to the model. tp.Any longer inputs will be
            truncated.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        apply_spec_augment (`bool`, *optional*, defaults to `False`):
            Whether to apply SpecAugment data augmentation.
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Propability of each feature vector along the time axis to be chosen as the start of the vector span to
            be masked. Approximately `mask_time_prob * sequence_length // mask_time_length` feature vectors will
            be masked along the time axis. This is only relevant if `apply_spec_augment` is set to `True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2):
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time
            mask, the mask will be filled with floats sampled in (random_lower_bound, random_upper_bound).
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Propability of each feature vector along the feature axis to be chosen as the start of the vector span to
            be masked. Approximately `mask_time_prob * hidden_size // mask_feature_length` feature vectors will be
            masked along the time axis. This is only relevant if `apply_spec_augment` is set to `True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0):
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            mask, the mask will be filled with floats sampled in (random_lower_bound, random_upper_bound).
        median_filter_width (`int`, *optional*, defaults to 7):
            The width of the median filter applied to the mask.
        bits (`int`, *optional*):
            The number of bits to quantize the model to. If None, the model is not quantized.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            What to save during gradient checkpointing. Choose one of `"nothing_saveable"`, `"first_half_saveable"`,
            `"full_saveable"`.
    """

    model_type: str = "whisper"
    attribute_map: typing.ClassVar = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    def __init__(
        self,
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=4,
        encoder_attention_heads=6,
        decoder_layers=4,
        decoder_attention_heads=6,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50257,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=384,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],  # noqa: B006
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        apply_spec_augment=False,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        median_filter_width=7,
        bits: int | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ):
        """Initializes the WhisperConfig object.

        Args:
            vocab_size (int): Vocabulary size.
            num_mel_bins (int): Number of Mel frequency bins.
            encoder_layers (int): Number of encoder layers.
            encoder_attention_heads (int): Number of encoder attention heads.
            decoder_layers (int): Number of decoder layers.
            decoder_attention_heads (int): Number of decoder attention heads.
            decoder_ffn_dim (int): Dimensionality of the decoder FFN.
            encoder_ffn_dim (int): Dimensionality of the encoder FFN.
            encoder_layerdrop (float): Dropout probability for encoder layers.
            decoder_layerdrop (float): Dropout probability for decoder layers.
            decoder_start_token_id (int): Decoder start token ID.
            use_cache (bool): Whether to use KV cache.
            is_encoder_decoder (bool): Whether the model is an encoder-decoder.
            activation_function (str): Activation function name.
            d_model (int): Dimensionality of the model.
            dropout (float): Dropout probability.
            attention_dropout (float): Attention dropout probability.
            activation_dropout (float): Activation dropout probability.
            init_std (float): Standard deviation for initialization.
            scale_embedding (bool): Whether to scale embeddings.
            max_source_positions (int): Maximum source sequence length.
            max_target_positions (int): Maximum target sequence length.
            pad_token_id (int): Padding token ID.
            bos_token_id (int): Beginning of sequence token ID.
            eos_token_id (int): End of sequence token ID.
            suppress_tokens (list[int], optional): List of tokens to suppress.
            begin_suppress_tokens (list[int]): List of tokens to suppress at the beginning.
            use_weighted_layer_sum (bool): Whether to use weighted layer sum.
            classifier_proj_size (int): Projection size for classification.
            apply_spec_augment (bool): Whether to apply SpecAugment.
            mask_time_prob (float): Probability for time masking in SpecAugment.
            mask_time_length (int): Length for time masking in SpecAugment.
            mask_time_min_masks (int): Minimum number of time masks in SpecAugment.
            mask_feature_prob (float): Probability for feature masking in SpecAugment.
            mask_feature_length (int): Length for feature masking in SpecAugment.
            mask_feature_min_masks (int): Minimum number of feature masks in SpecAugment.
            median_filter_width (int): Width for median filtering.
            bits (tp.Optional[int]): Quantization bits.
            gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing strategy.
            **kwargs: Additional keyword arguments.
        """
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        # Audio Classification-specific parameters. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        self.median_filter_width = median_filter_width
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        self.max_position_embeddings = max(max_source_positions, max_target_positions)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs):
        """Returns the partition rules for the Whisper model. Arguments are ignored.

        Returns:
            tuple: Partition rules.
        """
        pmag = self.partition_manager
        return (
            ("model/encoder/conv[12]/kernel", PartitionSpec(None, "tp", ("fsdp", "sp"))),
            (r"encoder/embed_positions/embedding", pmag.resolve(Replicated)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/out_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"fc1/kernel", pmag.resolve(ColumnWise)),
            (r"fc2/kernel", pmag.resolve(RowWise)),
            (r"fc(1|2)/bias", pmag.resolve(Replicated)),
            (r"(self_attn_layer_norm|final_layer_norm)/scale", pmag.resolve(Replicated)),
            (r"(self_attn_layer_norm|final_layer_norm)/bias", pmag.resolve(Replicated)),
            (r"encoder/layer_norm/scale", pmag.resolve(Replicated)),
            (r"encoder/layer_norm/bias", pmag.resolve(Replicated)),
            (r"decoder/embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"decoder/embed_positions/embedding", pmag.resolve(Replicated)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/out_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"encoder_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"encoder_attn/out_proj/kernel", pmag.resolve(RowWise)),
            (r"encoder_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"fc1/kernel", pmag.resolve(ColumnWise)),
            (r"fc2/kernel", pmag.resolve(RowWise)),
            (r"fc(1|2)/bias", pmag.resolve(Replicated)),
            (
                r"(self_attn_layer_norm|encoder_attn_layer_norm|final_layer_norm)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r"(self_attn_layer_norm|encoder_attn_layer_norm|final_layer_norm)/bias",
                pmag.resolve(Replicated),
            ),
            (r"decoder/layer_norm/scale", pmag.resolve(Replicated)),
            (r"decoder/layer_norm/bias", pmag.resolve(Replicated)),
            (r"proj_out/kernel", pmag.resolve(ColumnWise)),
            (r"projector/kernel", pmag.resolve(ColumnWise)),
            (r"projector/bias", pmag.resolve(Replicated)),
            (r"classifier/kernel", pmag.resolve(RowWise)),
            (r"classifier/bias", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

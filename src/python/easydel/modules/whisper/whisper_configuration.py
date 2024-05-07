import math
from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDeLPretrainedConfig


class WhisperConfig(EasyDeLPretrainedConfig):
    model_type: str = "whisper"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model"
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
            begin_suppress_tokens=[220, 50256],
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
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
            **kwargs,
    ):
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

    def add_jax_args(
            self,
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
            **kwargs
    ):
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

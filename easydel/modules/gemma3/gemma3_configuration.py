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
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

from ..siglip.configuration_siglip import SiglipVisionConfig


@register_config("gemma3_text")
class Gemma3TextConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 262208):
            Vocabulary size of the Gemma3Text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Gemma3TextModel`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            Scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma3Text, every other layer uses sliding window
            attention. This is the size of the sliding window.
        final_logit_softcapping (`float`, *optional*):
            Scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*):
            Scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings used in gloabl attention.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        rope_local_base_freq (float, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings for local attention.
        sliding_window_pattern (`int`, *optional*, defaults to 6):
            Pattern for the sliding window attention.

        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        scan_layers (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation of the layers.
    """

    model_type: str = "gemma3_text"

    def __init__(
        self,
        vocab_size=262_208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        cache_implementation="hybrid",
        rope_scaling=None,
        rope_local_base_freq=10_000.0,
        sliding_window_pattern=6,
        layer_types: list[str] | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        scan_layers: bool = False,
        **kwargs,
    ):
        """The __init__ function is called when the class is instantiated.
        It sets up the attributes of an object, which are sometimes called fields or properties.
        The __init__ function can accept arguments, but self must be the first one.
        """

        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits
        self.scan_layers = scan_layers

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation

        self.rope_local_base_freq = rope_local_base_freq
        self.sliding_window_pattern = sliding_window_pattern
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self.sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.rope_scaling = rope_scaling

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
            (r"self_attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*(input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm|norm)/kernel",
                pmag.resolve(Replicated),
            ),
            (r"vision_tower/.*", pmag.resolve(Replicated)),
            (r"multi_modal_projector/mm_input_projection_weight", pmag.resolve(ColumnWise)),
            (r"multi_modal_projector/mm_soft_emb_norm/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"language_model/lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
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
        if self.layer_types is not None:
            for layer_idx in range(self.num_hidden_layers):
                mapping[layer_idx] = AttnMaskDetail(
                    mask_type=AttnMaskType.from_hf(self.layer_types[layer_idx]),
                    size=self.sliding_window,
                )
        return mapping


@register_config("gemma3")
class Gemma3Config(EasyDeLBaseConfig):
    r"""
    Args:
        text_config (`Union[Gemma3TextConfig, dict]`, *optional*):
            The config object of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom vision config or dict.
        mm_tokens_per_image (`int`, *optional*, defaults to 256):
            The number of tokens per image embedding.
        boi_token_index (`int`, *optional*, defaults to 255999):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_index (`int`, *optional*, defaults to 256000):
            The end-of-image token index to wrap the image prompt.
        image_token_id (`int`, *optional*, defaults to 262144):
            The image token index to encode the image prompt.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Example:

    ```python
    >>> from transformers import Gemma3ForConditionalGeneration, Gemma3Config, SiglipVisionConfig, Gemma3TextConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a Gemma3 Text config
    >>> text_config = Gemma3TextConfig()

    >>> # Initializing a Gemma3 gemma-3-4b style configuration
    >>> configuration = Gemma3Config(vision_config, text_config)

    >>> # Initializing a model from the gemma-3-4b style configuration
    >>> model = Gemma3TextConfig(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma3"
    sub_configs: typing.ClassVar = {"text_config": Gemma3TextConfig, "vision_config": SiglipVisionConfig}

    def __init__(
        self,
        text_config: Gemma3TextConfig | None = None,
        vision_config: SiglipVisionConfig | None = None,
        mm_tokens_per_image: int = 256,
        boi_token_index: int = 255_999,
        eoi_token_index: int = 256_000,
        image_token_id: int = 262_144,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize a Gemma3Config instance.

        Args:
          text_config (Optional[Gemma3TextConfig], optional): The configuration for the text model component.
              If None is provided, a default Gemma3TextConfig will be used. Defaults to None.
          vision_config (Optional[SiglipVisionConfig], optional): The configuration for the vision model component.
              If None is provided, a default SiglipVisionConfig will be used. Defaults to None.
          mm_tokens_per_image (int, optional): Number of tokens per image embedding. Defaults to 256.
          boi_token_index (int, optional): Begin-of-image token index to wrap the image prompt. Defaults to 255_999.
          eoi_token_index (int, optional): End-of-image token index to wrap the image prompt. Defaults to 256_000.
          image_token_id (int, optional): Image token index to encode the image prompt. Defaults to 262_144.
          initializer_range (float, optional): Standard deviation for weight initialization. Defaults to 0.02.
          **kwargs: Additional keyword arguments passed to the parent class constructor.
        """
        if text_config is None:
            text_config = Gemma3TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma3TextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        else:
            vision_config = SiglipVisionConfig()

        self.text_config = text_config
        self.vision_config = vision_config
        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_id = image_token_id
        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    def get_partition_rules(self, *args, **kwargs):
        """Get the partition rules for the model.

        Returns:
          Tuple[Tuple[str, PartitionSpec]]: A tuple of tuples, where each inner tuple contains a regex pattern
          matching parameter names and the corresponding PartitionSpec for sharding those parameters across devices.
        """
        text_partitions = self.text_config.get_partition_rules(*args, **kwargs)
        vision_partitions = self.vision_config.get_partition_rules(*args, **kwargs)
        return text_partitions + vision_partitions

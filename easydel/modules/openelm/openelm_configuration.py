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
from numbers import Number

from eformer.common_types import ColumnWise, Replicated, RowWise
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


def make_divisible(
    v: float | int,
    divisor: int | None = 8,
    min_value: float | int | None = None,
) -> float | int:
    """This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def compute_heads(model_dim: int, head_dim: int) -> int:
    """Compute the number of heads.
    Args:
        model_dim: Model dimension.
        head_dim: Head dimension.
    Returns:
        An integer denoting number of heads in multi-head attention is returned.
    Raises:
        ValueError: if model dimension is not divisible by head dimension.
    """
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    else:
        raise ValueError(f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}.")


@register_config("openelm")
class OpenELMConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the OpenELM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        max_context_length (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        num_transformer_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        model_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of the attention heads.
        qkv_multipliers (`float` or `list` of `float`, *optional*, defaults to 1.0):
            The multiplier for the query, key, and value projections.
        num_query_heads (`int`, *optional*):
            Number of query heads. If not provided, it will be calculated based on `model_dim` and `head_dim`.
        num_gqa_groups (`int`, *optional*, defaults to 1):
            Number of GQA (Grouped Query Attention) groups.
        ffn_multipliers (`float` or `list` of `float`, *optional*, defaults to 4.0):
            The multiplier for the feed-forward network.
        ffn_with_glu (`bool`, *optional*, defaults to `True`):
            Whether to use a gated linear unit (GLU) in the feed-forward network.
        ffn_dim_divisor (`int`, *optional*, defaults to 256):
            The divisor for the feed-forward network dimension.
        activation_fn_name (`str`, *optional*, defaults to `"swish"`):
            The activation function to use.
        normalization_layer_name (`str`, *optional*, defaults to `"rms_norm"`):
            The normalization layer to use.
        normalize_qk_projections (`bool`, *optional*, defaults to `False`):
            Whether to normalize the query and key projections.
        share_input_output_layers (`bool`, *optional*, defaults to `False`):
            Whether to share the input and output layers.
        rope_freq_constant (`int`, *optional*, defaults to 10000):
            The frequency constant for Rotary Position Embeddings (RoPE).
        rope_max_length (`int`, *optional*, defaults to 4096):
            The maximum length for RoPE.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The configuration for rope scaling.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        use_scan_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation for the MLP.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size to use when scanning the MLP.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
    """

    model_type: str = "openelm"
    attribute_map: typing.ClassVar = {"tie_word_embedding": "share_input_output_layers"}

    def __init__(
        self,
        vocab_size: int = 32000,
        max_context_length: int = 2048,
        num_transformer_layers: int = 12,
        model_dim: int = 2048,
        head_dim: int = 128,
        qkv_multipliers: Number | list[Number] = 1.0,
        num_query_heads: int | None = None,
        num_gqa_groups: int = 1,
        ffn_multipliers: Number | list[Number] = 4.0,
        ffn_with_glu: bool = True,
        ffn_dim_divisor: int = 256,
        activation_fn_name: str = "swish",
        normalization_layer_name: str = "rms_norm",
        normalize_qk_projections: bool = False,
        share_input_output_layers: bool = False,
        rope_freq_constant: int = 10000,
        rope_max_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_scaling: dict[str, str | float] | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        **kwargs,
    ):
        """The __init__ function is called when the class is instantiated.
        It allows the class to initialize the attributes of a class.
        The self parameter is a reference to the current instance of the class,
        and is used to access variables that belong to the class.

        Args:
            vocab_size (`int`, *optional*, defaults to 32000): Vocabulary size.
            max_context_length (`int`, *optional*, defaults to 2048): Maximum sequence length.
            num_transformer_layers (`int`, *optional*, defaults to 12): Number of transformer layers.
            model_dim (`int`, *optional*, defaults to 2048): Model dimension (embedding size).
            head_dim (`int`, *optional*, defaults to 128): Dimension of each attention head.
            qkv_multipliers (`float` or `list` of `float`, *optional*, defaults to 1.0):
                Multiplier(s) for QKV projection dimensions.
            num_query_heads (`int`, *optional*): Number of query heads. Calculated if None.
            num_gqa_groups (`int`, *optional*, defaults to 1): Number of GQA groups.
            ffn_multipliers (`float` or `list` of `float`, *optional*, defaults to 4.0):
                Multiplier(s) for FFN intermediate dimension.
            ffn_with_glu (`bool`, *optional*, defaults to `True`): Whether FFN uses GLU.
            ffn_dim_divisor (`int`, *optional*, defaults to 256): Divisor for FFN dimension calculation.
            activation_fn_name (`str`, *optional*, defaults to `"swish"`): Activation function name.
            normalization_layer_name (`str`, *optional*, defaults to `"rms_norm"`): Normalization layer name.
            normalize_qk_projections (`bool`, *optional*, defaults to `False`): Whether to normalize QK projections.
            share_input_output_layers (`bool`, *optional*, defaults to `False`): Whether to tie input/output embeddings.
            rope_freq_constant (`int`, *optional*, defaults to 10000): RoPE frequency constant.
            rope_max_length (`int`, *optional*, defaults to 4096): Maximum RoPE length.
            initializer_range (`float`, *optional*, defaults to 0.02): Initializer range.
            use_cache (`bool`, *optional*, defaults to `True`): Whether to use KV cache.
            bos_token_id (`int`, *optional*, defaults to 1): Beginning-of-sequence token ID.
            eos_token_id (`int`, *optional*, defaults to 2): End-of-sequence token ID.
            rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
                RoPE scaling configuration. Defaults to None.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            use_scan_mlp (bool, optional): Whether to use scan for MLP layers. Defaults to False.
            scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_gqa_groups = num_gqa_groups
        self.ffn_multipliers = ffn_multipliers
        self.ffn_with_glu = ffn_with_glu
        self.ffn_dim_divisor = ffn_dim_divisor
        self.activation_fn_name = activation_fn_name
        self.normalization_layer_name = normalization_layer_name
        self.normalize_qk_projections = normalize_qk_projections
        self.share_input_output_layers = share_input_output_layers
        self.rope_freq_constant = rope_freq_constant
        self.rope_max_length = rope_max_length
        self.num_query_heads = (
            compute_heads(model_dim=model_dim, head_dim=head_dim) if num_query_heads is None else num_query_heads
        )
        self.initializer_range = initializer_range
        self.bits = bits
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )
        self.__post_init__()

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"token_embeddings/embedding", pmag.resolve(ColumnWise)),
            (r"attn/qkv_proj/kernel", pmag.resolve(ColumnWise)),
            (r"attn/out_proj/kernel", pmag.resolve(RowWise)),
            (r"attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"ffn/proj_1/kernel", pmag.resolve(ColumnWise)),
            (r"ffn/proj_2/kernel", pmag.resolve(RowWise)),
            (r"ffn/.*proj/bias", pmag.resolve(Replicated)),
            (r".*/(attn_norm|ffn_norm|norm)/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"lm_head/bias", pmag.resolve(Replicated)),
            (r"classifier/kernel", pmag.resolve(ColumnWise)),
            (r"classifier/bias", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def __post_init__(self) -> None:
        """Performs post-initialization checks and calculations.

        This method validates the configuration and computes derived values like
        FFN dimensions and QKV dimensions based on the provided multipliers and divisors.
        It ensures that the configuration parameters are consistent and valid.

        Raises:
            ValueError: If the configuration parameters are invalid or inconsistent.
        """
        if self.num_gqa_groups is not None:
            head_multiple_of = self.num_gqa_groups
        else:
            head_multiple_of = 2

        if isinstance(self.qkv_multipliers, Number):
            # All attention layers have the same latent dimensions, resulting in uniform allocation of parameters.
            qkv_dim = make_divisible(
                self.model_dim * self.qkv_multipliers,  # type:ignore
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers

        elif isinstance(self.qkv_multipliers, tuple | list) and len(self.qkv_multipliers) == 2:
            # Each attention layer have different latent dimensions assuming qkv_multipliers[0] != qkv_multipliers[1].
            # This results in variable allocation of parameters in attention layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            qkv_multipliers = [
                round(v, 2)
                for v in jnp.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
            # Make sure that scaled model dimension is divisible by scaled head dimension.
            query_dims = [
                int(make_divisible(self.model_dim * m, divisor=self.head_dim * head_multiple_of))
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list containing exactly two numbers."
                f" Got: {qkv_multipliers}."
            )

        self.num_query_heads = [int(compute_heads(q_dim, self.head_dim)) for q_dim in query_dims]
        self.num_kv_heads = [q_heads // self.num_gqa_groups for q_heads in self.num_query_heads]

        # Feed-forward network (FFN) multipliers
        if isinstance(self.ffn_multipliers, Number):
            # All FFN layers have the same latent dimensions, resulting in uniform allocation of parameters.
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif isinstance(self.ffn_multipliers, tuple | list):
            # Each FFN layer have different latent dimensions assuming ffn_multipliers[0] != ffn_multipliers[1].
            # This results in variable allocation of parameters in FFN layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            if len(self.ffn_multipliers) == 2:
                self.ffn_multipliers = [
                    round(v, 2)
                    for v in jnp.linspace(
                        self.ffn_multipliers[0],
                        self.ffn_multipliers[1],
                        num=self.num_transformer_layers,
                        dtype=float,
                    )
                ]
            else:
                assert (
                    len(self.ffn_multipliers) == self.num_transformer_layers
                ), f"{len(self.ffn_multipliers)=}!={self.num_transformer_layers=}"
        else:
            raise NotImplementedError(
                f"FFN multipliers should be a single number or a list containing exactly two numbers. "
                f"Got: {qkv_multipliers}."
            )

        # check num_query_heads divisible by num_kv_heads for every layer
        for layer_idx in range(len(query_dims)):
            assert self.num_query_heads[layer_idx] % self.num_kv_heads[layer_idx] == 0

    @property
    def granted_freq_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size specifically for frequency-based position embeddings.

        If `freq_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
        `max_context_length`.

        Returns:
            int: The granted maximum position embedding size for frequency encoding.
        """
        return getattr(
            self,
            "freq_max_position_embeddings",
            self.max_context_length,
        )

    @property
    def granted_mask_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size specifically for mask-based position embeddings.

        If `mask_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
        `max_context_length`.

        Returns:
            int: The granted maximum position embedding size for mask encoding.
        """
        return getattr(
            self,
            "mask_max_position_embeddings",
            self.max_context_length,
        )

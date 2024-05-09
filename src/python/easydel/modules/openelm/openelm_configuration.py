from numbers import Number
from typing import Optional, Dict, Union, List
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from ..easydel_modelling_utils import EasyDeLPretrainedConfig


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62
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
        raise ValueError(
            f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}."
        )


class OpenELMConfig(EasyDeLPretrainedConfig):
    model_type: str = "openelm"

    def __init__(
            self,
            vocab_size: int = 32000,
            max_context_length: int = 2048,
            num_transformer_layers: int = 12,
            model_dim: int = 2048,
            head_dim: int = 128,
            qkv_multipliers: Union[Number, List[Number]] = 1.0,
            num_query_heads: Union[int, None] = None,
            num_gqa_groups: int = 1,
            ffn_multipliers: Union[Number, List[Number]] = 4.0,
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
            rope_scaling: Dict[str, Union[str, float]] = None,
            gradient_checkpointing: str = "nothing_saveable",
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            bits: Optional[int] = None,
            **kwargs,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It allows the class to initialize the attributes of a class.
        The self parameter is a reference to the current instance of the class, and is used to access variables that belong to the class.

        :param self: Represent the instance of the class
        :param vocab_size: Define the size of the vocabulary
        :param hidden_size: Determine the size of the embedding layers
        :param intermediate_size: Define the size of the intermediate layer in each transformer block
        :param num_hidden_layers: Determine the number of layers in the encoder and decoder
        :param num_attention_heads: Determine the number of attention heads in each layer
        :param num_key_value_heads: Specify the number of heads for key and value
        :param hidden_act: Specify the activation function used in the hidden layers
        :param max_position_embeddings: Set the maximum length of the sequence
        :param initializer_range: Initialize the weights of the model
        :param rms_norm_eps: Avoid division by zero in the rms normalization
        :param use_cache: Determine whether to use the cache in the decoder
        :param pad_token_id: Specify the token id of the padding token
        :param bos_token_id: Specify the beginning of sentence token id
        :param eos_token_id: Specify the end of sentence token
        :param tie_word_embeddings: Tie the word embeddings and the output layer
        :param rope_theta: Control the number of tokens in a rope
        :param sliding_window: Control the number of tokens that are processed in parallel
        :param gradient_checkpointing: str: Specify whether to use gradient checkpointing
        :param use_scan_mlp: bool: Determine whether or not to use the scan_mlp function
        :param scan_mlp_chunk_size: int: Specify the chunk size of the scan mlp
        :param number_rep_kv: int: Specify the number of times to repeat the key and value vectors
        :param attention_dropout: float: Set the dropout rate for the attention layer
        :param bits: Optional[int]: Specify the number of bits used for quantization
        :param axis_dims: Sequence[int]: Specify the dimension of each axis
        :param axis_names: Sequence[str]: Specify the names of each axis in the tensor
        :param &quot;mp&quot;): Define the maximum position embeddings
        :param attention_bias: bool: when ever to use attention_bias
        :param kwargs: Pass a variable number of keyword arguments to a function
        :param : Define the number of layers in the model
        :return: An instance of the class

        """
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_query_heads = num_query_heads
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
            compute_heads(model_dim=model_dim, head_dim=head_dim)
            if num_query_heads is None
            else num_query_heads
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

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        """
        The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
          1) A regex string that matches the name of one or more parameters in the model.
          2) A PartitionScheme object that defines how those parameters should be partitioned.

        :param fully_sharded_data_parallel: bool: Determine whether to use the fully_sharded_data_parallel partitioning scheme or not
        :return: A list of tuples

        """
        return (

            ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
            (".*", PartitionSpec(("fsdp", "sp"))),
        ) if not fully_sharded_data_parallel else (
            ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
            (".*", PartitionSpec(("fsdp", "sp"))),
        )

    def add_jax_args(
            self,
            gradient_checkpointing: str = "nothing_saveable",
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            bits: Optional[int] = None,
            rope_scaling: Dict[str, Union[str, float]] = None,
            **kwargs,
    ):
        """
        The add_jax_args function adds the following arguments to the model:

        :param self: Bind the attributes and methods of a class to an instance of that class
        :param gradient_checkpointing: str: Determine whether to use gradient checkpointing
        :param use_scan_mlp: bool: Determine whether to use the scan_mlp function or notn
        :param scan_mlp_chunk_size: int: Chunk the input to the mlp
        :param bits: Optional[int]: Specify the number of bits to use for quantization
        :param rope_scaling: Dict[str, Union[str, float]]: rope_scaling for rope
        :return: A tuple of the following:

        """

        self.rope_scaling = rope_scaling
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'

    def __post_init__(self) -> None:
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

        elif (
                isinstance(self.qkv_multipliers, (tuple, list))
                and len(self.qkv_multipliers) == 2
        ):
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
                int(
                    make_divisible(
                        self.model_dim * m, divisor=self.head_dim * head_multiple_of
                    )
                )
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # compute the number of query, key, and value heads
        # For multi-head and multi-query attention, the number of heads for query, key, and value are the same.
        # For group query attention, the number of key and value heads are the same.
        self.num_query_heads = [
            int(compute_heads(q_dim, self.head_dim)) for q_dim in query_dims
        ]
        self.num_kv_heads = [
            q_heads // self.num_gqa_groups for q_heads in self.num_query_heads
        ]

        # Feed-forward network (FFN) multipliers
        if isinstance(self.ffn_multipliers, Number):
            # All FFN layers have the same latent dimensions, resulting in uniform allocation of parameters.
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif isinstance(self.ffn_multipliers, (tuple, list)):
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
                f"FFN multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # check num_query_heads divisible by num_kv_heads for every layer
        for layer_idx in range(len(query_dims)):
            assert self.num_query_heads[layer_idx] % self.num_kv_heads[layer_idx] == 0

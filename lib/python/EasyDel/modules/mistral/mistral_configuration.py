from typing import Sequence, Optional, Dict, Union

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class MistralConfig(EasyDelPretrainedConfig):
    model_type = "mistral"
    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=4096 * 32,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling: Dict[str, Union[str, float]] = None,
            sliding_window=4096,
            gradient_checkpointing: str = 'nothing_saveable',
            use_pjit_attention_force: bool = False,
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            number_rep_kv: int = 1,
            attention_dropout: float = 0.0,
            bits: Optional[int] = None,
            attention_bias: bool = False,
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
        :param use_pjit_attention_force: bool: Force the use of pjit attention
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
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.bits = bits
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @staticmethod
    def get_partition_rules(fully_sharded_data_parallel: bool = True):
        """
        The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
          1) A regex string that matches the name of one or more parameters in the model.
          2) A PartitionScheme object that defines how those parameters should be partitioned.

        :param fully_sharded_data_parallel: bool: Determine whether to use the fully_sharded_data_parallel partitioning scheme or not
        :return: A list of tuples

        """
        return (

            ("model/embed_tokens/embedding", PartitionSpec("dp", "fsdp")),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec("fsdp", "dp")),
            ("self_attn/o_proj/kernel", PartitionSpec("dp", "fsdp")),

            ("mlp/gate_proj/kernel", PartitionSpec("fsdp", "dp")),
            ("mlp/down_proj/kernel", PartitionSpec("dp", "fsdp")),
            ("mlp/up_proj/kernel", PartitionSpec("fsdp", "dp")),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp", "dp")),
            (".*", PartitionSpec(None)),
        ) if not fully_sharded_data_parallel else (
            ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
            ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

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
            gradient_checkpointing: str = 'nothing_saveable',
            use_pjit_attention_force: bool = False,
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            number_rep_kv: int = 1,
            bits: Optional[int] = None,
            attention_dropout: float = 0.0,
            rope_scaling: Dict[str, Union[str, float]] = None,
            attention_bias: bool = False,
            **kwargs,
    ):
        """
        The add_jax_args function adds the following arguments to the model:

        :param self: Bind the attributes and methods of a class to an instance of that class
        :param gradient_checkpointing: str: Determine whether to use gradient checkpointing
        :param use_pjit_attention_force: bool: Determine whether to use the pjit_attention_force function
        :param use_scan_mlp: bool: Determine whether to use the scan_mlp function or notn
        :param scan_mlp_chunk_size: int: Chunk the input to the mlp
        :param number_rep_kv: int: Control the number of times that the key and value vectors are repeated
        :param bits: Optional[int]: Specify the number of bits to use for quantization
        :param attention_dropout: float: Set the dropout rate for the attention layer
        :param attention_bias: bool: when ever to use attention_bias
        :param rope_scaling: Dict[str, Union[str, float]]: rope_scaling for rope
        :return: A tuple of the following:

        """

        self.attention_bias = attention_bias
        self.rope_scaling = rope_scaling
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.attention_dropout = attention_dropout
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'

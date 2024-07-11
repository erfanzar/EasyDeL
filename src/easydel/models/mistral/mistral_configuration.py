from typing import Dict, Optional, Union

from jax.sharding import PartitionSpec

from easydel.models.modelling_utils import EDPretrainedConfig


class MistralConfig(EDPretrainedConfig):
    model_type: str = "mistral"

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
        gradient_checkpointing: str = "",
        number_rep_kv: int = 1,
        attention_dropout: float = 0.0,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: Optional[int] = None,
        attention_bias: bool = False,
        **kwargs,
    ):
        """The __init__ function is called when the class is instantiated.
        It allows the class to initialize the attributes of a class.
        The self parameter is a reference to the current instance of the class, and is used to access variables that belong to the class.

        Args:
            self: Represent the instance of the class
            vocab_size: Define the size of the vocabulary
            hidden_size: Determine the size of the embedding layers
            intermediate_size: Define the size of the intermediate layer
                in each transformer block
            num_hidden_layers: Determine the number of layers in the
                encoder and decoder
            num_attention_heads: Determine the number of attention heads
                in each layer
            num_key_value_heads: Specify the number of heads for key and
                value
            hidden_act: Specify the activation function used in the
                hidden layers
            max_position_embeddings: Set the maximum length of the
                sequence
            initializer_range: Initialize the weights of the model
            rms_norm_eps: Avoid division by zero in the rms
                normalization
            use_cache: Determine whether to use the cache in the decoder
            pad_token_id: Specify the token id of the padding token
            bos_token_id: Specify the beginning of sentence token id
            eos_token_id: Specify the end of sentence token
            tie_word_embeddings: Tie the word embeddings and the output
                layer
            rope_theta: Control the number of tokens in a rope
            sliding_window: Control the number of tokens that are
                processed in parallel
            gradient_checkpointing: str: Specify whether to use gradient
                checkpointing
            use_scan_mlp: bool: Determine whether or not to use the
                scan_mlp function
            scan_mlp_chunk_size: int: Specify the chunk size of the scan
                mlp
            number_rep_kv: int: Specify the number of times to repeat
                the key and value vectors
            attention_dropout: float: Set the dropout rate for the
                attention layer
            bits: Optional[int]: Specify the number of bits used for
                quantization
            axis_dims: Sequence[int]: Specify the dimension of each axis
            axis_names: Sequence[str]: Specify the names of each axis in
                the tensor
            &quot;mp&quot;): Define the maximum position embeddings
            attention_bias: bool: when ever to use attention_bias
            **kwargs: Pass a variable number of keyword arguments to a
                function
        :param : Define the number of layers in the model

        Returns:
            An instance of the class
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
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        """The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
          1) A regex string that matches the name of one or more parameters in the model.
          2) A PartitionScheme object that defines how those parameters should be partitioned.

        Args:
            fully_sharded_data_parallel: bool: Determine whether to use
                the fully_sharded_data_parallel partitioning scheme or
                not

        Returns:
            A list of tuples
        """
        return (
            (
                ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),
                (
                    "self_attn/(q_proj|k_proj|v_proj)/kernel",
                    PartitionSpec(("fsdp", "sp"), "tp"),
                ),
                ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
                ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
                ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
                ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),
                ("input_layernorm/kernel", PartitionSpec(None)),
                ("post_attention_layernorm/kernel", PartitionSpec(None)),
                ("model/norm/kernel", PartitionSpec(None)),
                ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
                (".*", PartitionSpec(("fsdp", "sp"))),
            )
            if not fully_sharded_data_parallel
            else (
                ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),
                (
                    "self_attn/(q_proj|k_proj|v_proj)/kernel",
                    PartitionSpec(("fsdp", "sp"), "tp"),
                ),
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
        )

    def add_jax_args(
        self,
        gradient_checkpointing: str = "",
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        number_rep_kv: int = 1,
        bits: Optional[int] = None,
        attention_dropout: float = 0.0,
        rope_scaling: Dict[str, Union[str, float]] = None,
        attention_bias: bool = False,
        **kwargs,
    ):
        """The add_jax_args function adds the following arguments to the model:

        Args:
            self: Bind the attributes and methods of a class to an
                instance of that class
            gradient_checkpointing: str: Determine whether to use
                gradient checkpointing
            use_scan_mlp: bool: Determine whether to use the scan_mlp
                function or notn
            scan_mlp_chunk_size: int: Chunk the input to the mlp
            number_rep_kv: int: Control the number of times that the key
                and value vectors are repeated
            bits: Optional[int]: Specify the number of bits to use for
                quantization
            attention_dropout: float: Set the dropout rate for the
                attention layer
            attention_bias: bool: when ever to use attention_bias
            rope_scaling: Dict[str, Union[str, float]]: rope_scaling for
                rope

        Returns:
            A tuple of the following:
        """

        self.attention_bias = attention_bias
        self.rope_scaling = rope_scaling
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.attention_dropout = attention_dropout
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return "params", "dropout", "fcm"

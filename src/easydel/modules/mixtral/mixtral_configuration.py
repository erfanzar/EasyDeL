from typing import Sequence, Optional, Dict, Union

from jax.sharding import PartitionSpec

from easydel.modules.easydel_modelling_utils import EasyDeLPretrainedConfig


class MixtralConfig(EasyDeLPretrainedConfig):
    model_type: str = "mixtral"

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
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=1e6,
            sliding_window=4096,
            attention_dropout=0.0,
            num_experts_per_tok=2,
            num_local_experts=8,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            gradient_checkpointing: str = "nothing_saveable",
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            number_rep_kv: int = 1,
            bits: Optional[int] = None,
            rope_scaling: Dict[str, Union[str, float]] = None,
            attention_bias: bool = False,
            initialization_of_moe: bool = False,
            router_jitter_noise=0.0,
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
            bits: Optional[int]: Specify the number of bits used for
                quantization
            axis_dims: Sequence[int]: Specify the dimension of each axis
            axis_names: Sequence[str]: Specify the names of each axis in
                the tensor
            &quot;mp&quot;): Define the maximum position embeddings
            **kwargs: Pass a variable number of keyword arguments to a
                function
            rope_scaling: Dict[str, Union[str, float]]: rope scaling
                information
            attention_dropout: float: Set the dropout rate for the
                attention layer
            initialization_of_moe: bool: initialization of moe needs to
                disable some dynamic part's this boolean variable will
                turn them off.
            attention_bias: bool: when ever to use attention_bias
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
        self.attention_dropout = attention_dropout
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.attention_bias = attention_bias
        # for backward compatibility
        self.rope_scaling = rope_scaling
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initialization_of_moe = initialization_of_moe
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.router_jitter_noise = router_jitter_noise
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

            ("model/embed_tokens/embedding", PartitionSpec("sp", "fsdp")),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),

            ("w1/kernel", PartitionSpec(("fsdp", "sp"))),
            ("w2/kernel", PartitionSpec(("fsdp", "sp"))),
            ("w3/kernel", PartitionSpec(("fsdp", "sp"))),
            ("gate/kernel", PartitionSpec(("fsdp", "sp"))),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp", "sp")),
            (".*", PartitionSpec(None)),
        ) if not fully_sharded_data_parallel else (
            ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),

            ("w1/kernel", PartitionSpec(("fsdp", "sp"))),
            ("w2/kernel", PartitionSpec(("fsdp", "sp"))),
            ("w3/kernel", PartitionSpec(("fsdp", "sp"))),
            ("gate/kernel", PartitionSpec(("fsdp", "sp"))),

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
            number_rep_kv: int = 1,
            bits: Optional[int] = None,
            attention_dropout: float = 0.0,
            rope_scaling: Dict[str, Union[str, float]] = None,
            attention_bias: bool = False,
            initialization_of_moe: bool = False,
            **kwargs,
    ):
        """The add_jax_args function adds the following arguments to the model:

        Args:
            self: Bind the attributes and methods of a class to an
                instance of that class
            gradient_checkpointing: str: Determine whether to use
                gradient checkpointing
            use_scan_mlp: bool: Determine whether to use the scan_mlp
                function or not
            scan_mlp_chunk_size: int: Chunk the input to the mlp
            number_rep_kv: int: Control the number of times that the key
                and value vectors are repeated
            bits: Optional[int]: Specify the number of bits to use for
                quantization
            attention_dropout: float: Set the dropout rate for the
                attention layer
            attention_bias: bool: when ever to use attention_bias
            initialization_of_moe: bool: initialization of moe needs to
                disable some dynamic part's this boolean variable will
                turn them off.
            rope_scaling: Dict[str, Union[str, float]]: rope_scaling for
                rope

        Returns:
            A tuple of the following:
        """
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rope_scaling = rope_scaling
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        self.initialization_of_moe = initialization_of_moe

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'

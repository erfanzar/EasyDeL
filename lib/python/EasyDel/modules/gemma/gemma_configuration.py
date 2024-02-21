from typing import Optional, Dict, Union

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class GemmaConfig(EasyDelPretrainedConfig):
    model_type = "gemma"

    def __init__(
            self,
            vocab_size=256000,
            hidden_size=3072,
            intermediate_size=24576,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=256,
            hidden_act="gelu",
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            gradient_checkpointing: str = "nothing_saveable",
            use_pjit_attention_force: bool = False,
            bits: Optional[int] = None,
            scan_layers: bool = False,
            **kwargs,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an object, which are sometimes called fields or properties.
        The __init__ function can accept arguments, but self must be the first one.
        """

        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.bits = bits
        self.scan_layers = scan_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        """
        The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
            1) A regex string that matches the name of one or more parameters in the model.
            2) A PartitionScheme object that defines how those parameters should be partitioned across devices.

        :param fully_sharded_data_parallel: bool: Determine whether to partition the model fully or not
        :return: A list of tuples

        """
        return (

            ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
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
            bits: Optional[int] = None,
            **kwargs,
    ):
        """
        The add_jax_args function adds the following arguments to the Transformer class:

        :param self: Refer to the current object
        :param gradient_checkpointing: str: Control the amount of memory used by jax
        :param use_pjit_attention_force: bool: Determine if the attention force is used
        :param bits: Optional[int]: Determine the number of bits used in the quantization
        """
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'

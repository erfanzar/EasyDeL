from ..easydel_modelling_utils import EasyDeLPretrainedConfig
from typing import Union, Optional
from jax.sharding import PartitionSpec


class JetMoEConfig(EasyDeLPretrainedConfig):
    model_type: str = "jetmoe"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=2048,
            num_hidden_layers=12,
            num_attention_heads=32,
            num_key_value_heads=16,
            kv_channels=128,
            ffn_hidden_size=5632,
            max_position_embeddings=4096,
            activation_function="silu",
            glu=True,
            moe_num_experts=8,
            moe_top_k=2,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            bias=True,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            initializer_range=0.01,
            gradient_checkpointing: str = "nothing_saveable",
            bits: Optional[int] = None,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.bias = bias
        self.glu = glu
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.activation_function = activation_function
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
            (".*", PartitionSpec(("fsdp", "sp"))),
        )

    def add_jax_args(
            self,
            tie_word_embeddings: bool = False,
            gradient_checkpointing: str = "nothing_saveable",
            bits: Optional[int] = None,
            **kwargs,
    ):
        """
        The add_jax_args function adds the following arguments to the Transformer class:

        :param self: Refer to the current object
        :param tie_word_embeddings: bool: Tie the word embeddings to the decoder
        :param gradient_checkpointing: str: Control the amount of memory used by jax
        :param bits: Optional[int]: Determine the number of bits used in the quantization
        """
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout'

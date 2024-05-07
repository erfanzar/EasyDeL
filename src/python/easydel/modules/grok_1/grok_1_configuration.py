from ..easydel_modelling_utils import EasyDeLPretrainedConfig
from typing import Union, Optional
from jax.sharding import PartitionSpec


class Grok1Config(EasyDeLPretrainedConfig):
    model_type: str = "grok-1"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=32768,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            attn_output_multiplier=1.0,
            max_attn_value=1.0,
            max_position_embeddings=4096,
            embedding_multiplier_scale: float = 1.0,
            output_multiplier_scale: float = 1.0,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            num_experts_per_tok=2,
            num_experts=8,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            gradient_checkpointing: str = "nothing_saveable",
            bits: Optional[int] = None,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.attn_output_multiplier = attn_output_multiplier
        self.max_attn_value = max_attn_value
        self.max_position_embeddings = max_position_embeddings
        self.embedding_multiplier_scale = embedding_multiplier_scale
        self.output_multiplier_scale = output_multiplier_scale
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits
        super().__init__(
            pad_token_id=pad_token_id,
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

            ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

            ("attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

            ("linear/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("linear_1/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
            ("linear_v/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("gate/kernel", PartitionSpec(("fsdp", "sp"))),

            ("post_attn_norm/kernel", PartitionSpec(None)),
            ("pre_attn_norm/kernel", PartitionSpec(None)),
            ("pre_moe_norm/kernel", PartitionSpec(None)),
            ("post_moe_norm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(None)),
        ) if not fully_sharded_data_parallel else (

            ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

            ("attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
            ("attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("linear/kernel", PartitionSpec(("fsdp", "sp"))),
            ("linear_1/kernel", PartitionSpec(("fsdp", "sp"))),
            ("linear_v/kernel", PartitionSpec(("fsdp", "sp"))),
            ("gate/kernel", PartitionSpec(("fsdp", "sp"))),

            ("post_attn_norm/kernel", PartitionSpec(("fsdp", "sp"))),
            ("pre_attn_norm/kernel", PartitionSpec(("fsdp", "sp"))),
            ("pre_moe_norm/kernel", PartitionSpec(("fsdp", "sp"))),
            ("post_moe_norm/kernel", PartitionSpec(("fsdp", "sp"))),

            ("model/norm/kernel", PartitionSpec(("fsdp", "sp"))),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
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

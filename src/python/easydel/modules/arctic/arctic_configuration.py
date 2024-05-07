from transformers.utils import logging
from typing import Sequence, Optional, Dict, Union
from ..easydel_modelling_utils import EasyDeLPretrainedConfig
from jax.sharding import PartitionSpec

logger = logging.get_logger(__name__)


class ArcticConfig(EasyDeLPretrainedConfig):
    model_type: str = "arctic"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=1e6,
            sliding_window=None,
            attention_dropout=0.0,
            num_experts_per_tok=1,
            num_local_experts=8,
            router_aux_loss_coef=0.001,
            moe_layer_frequency=2,
            parallel_attn_mlp_res=False,
            moe_train_capacity_factor=1,
            moe_eval_capacity_factor=1,
            enable_expert_tensor_parallelism=False,
            moe_min_capacity=0,
            moe_token_dropping=True,
            quantization=None,
            gradient_checkpointing: str = "nothing_saveable",
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            bits: Optional[int] = None,
            rope_scaling: Dict[str, Union[str, float]] = None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_layer_frequency = moe_layer_frequency
        self.moe_train_capacity_factor = moe_train_capacity_factor
        self.moe_eval_capacity_factor = moe_eval_capacity_factor
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.moe_min_capacity = moe_min_capacity
        self.moe_token_dropping = moe_token_dropping
        self.parallel_attn_mlp_res = parallel_attn_mlp_res
        self.quantization = quantization

        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        self.rope_scaling = rope_scaling

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
          2) A PartitionScheme object that defines how those parameters should be partitioned.

        :param fully_sharded_data_parallel: bool: Determine whether to use the fully_sharded_data_parallel partitioning
         scheme or not
        :return: A list of tuples

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
            bits: Optional[int] = None,
            rope_scaling: Dict[str, Union[str, float]] = None,
            **kwargs,
    ):
        """
        The add_jax_args function adds the following arguments to the model:

        :param self: Bind the attributes and methods of a class to an instance of that class
        :param gradient_checkpointing: str: Determine whether to use gradient checkpointing
        :param use_scan_mlp: bool: Determine whether to use the scan_mlp function or not
        :param scan_mlp_chunk_size: int: Chunk the input to the mlp
        :param bits: Optional[int]: Specify the number of bits to use for quantization
         variable will turn them off.
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

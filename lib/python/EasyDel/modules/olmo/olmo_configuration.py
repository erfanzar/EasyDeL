from enum import StrEnum
from typing import Optional
from jax.sharding import PartitionSpec
from ..easydel_modelling_utils import EasyDelPretrainedConfig


class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"
    amd_compatible = "amd_compatible"


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"
    llama = "llama"


class OLMoConfig(EasyDelPretrainedConfig):
    """
    OLMo (model) configuration.
    """

    def __init__(
            self,
            d_model: int = 768,
            n_heads: int = 12,
            n_layers: int = 12,
            mlp_ratio: int = 4,
            mlp_hidden_size: Optional[int] = None,
            activation_type: ActivationType = ActivationType.swiglu,
            block_type: BlockType = BlockType.sequential,
            block_group_size: int = 1,
            alibi: bool = False,
            alibi_bias_max: float = 8.0,
            rope: bool = False,
            rope_full_precision: bool = True,
            flash_attention: bool = False,
            attention_dropout: float = 0.1,
            multi_query_attention: bool = False,
            attention_layer_norm: bool = False,
            residual_dropout: float = 0.1,
            embedding_dropout: float = 0.1,
            layer_norm_type: LayerNormType = LayerNormType.default,
            layer_norm_with_affine: bool = True,
            attention_layer_norm_with_affine: bool = True,
            max_sequence_length: int = 1024,
            include_bias: bool = True,
            bias_for_layer_norm: Optional[bool] = None,
            scale_logits: bool = False,
            vocab_size: int = 50257,
            embedding_size: Optional[int] = 50304,
            weight_tying: bool = True,
            eos_token_id: int = 50256,
            pad_token_id: int = 50256,
            init_std: float = 0.02,
            init_cutoff_factor: Optional[float] = None,
            gradient_checkpointing: str = "nothing_saveable",
            **kwargs
    ):
        _ = kwargs.pop("precision", None)
        _ = kwargs.pop("init_fn", None)
        _ = kwargs.pop("init_device", None)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_size = mlp_hidden_size
        self.activation_type = activation_type
        self.block_type = block_type
        self.block_group_size = block_group_size
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max
        self.rope = rope
        self.rope_full_precision = rope_full_precision
        self.flash_attention = flash_attention
        self.attention_dropout = attention_dropout
        self.multi_query_attention = multi_query_attention
        self.attention_layer_norm = attention_layer_norm
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.layer_norm_type = layer_norm_type
        self.layer_norm_with_affine = layer_norm_with_affine
        self.attention_layer_norm_with_affine = attention_layer_norm_with_affine
        self.max_sequence_length = max_sequence_length
        self.include_bias = include_bias
        self.bias_for_layer_norm = bias_for_layer_norm
        self.scale_logits = scale_logits
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight_tying = weight_tying
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    def add_jax_args(
            self,
            gradient_checkpointing: str = 'nothing_saveable'
    ):
        if not hasattr(self, "gradient_checkpointing"):
            self.gradient_checkpointing = gradient_checkpointing

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

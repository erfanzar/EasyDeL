from typing import Dict, Optional, Union

from jax.sharding import PartitionSpec

from easydel.models.modelling_utils import EDPretrainedConfig


class ChatGLMConfig(EDPretrainedConfig):
    model_type: str = "chatglm"

    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        rope_ratio=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        gradient_checkpointing: str = "",
        rope_scaling: Dict[str, Union[str, float]] = None,
        scan_mlp_chunk_size: int = 1024,
        bits: Optional[int] = None,
        mlp_bias: bool = False,
        scan_layers: bool = False,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.rope_ratio = rope_ratio
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.gradient_checkpointing = gradient_checkpointing
        self.mlp_bias = mlp_bias
        self.rope_scaling = rope_scaling
        self.bits = bits
        self.scan_layers = scan_layers
        super().__init__(
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        """The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
            1) A regex string that matches the name of one or more parameters in the model.
            2) A PartitionScheme object that defines how those parameters should be partitioned across devices.

        Args:
            fully_sharded_data_parallel: bool: Determine whether to
                partition the model fully or not

        Returns:
            A list of tuples
        """
        return (
            (
                ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
                (
                    "self_attn/(q_proj|k_proj|v_proj)/kernel",
                    PartitionSpec(("fsdp", "sp"), "tp"),
                ),
                ("self_attn/o_proj/kernel", PartitionSpec("tp", ("sp", "fsdp"))),
                ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
                ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
                ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
                ("input_layernorm/kernel", PartitionSpec(None)),
                ("post_attention_layernorm/kernel", PartitionSpec(None)),
                ("model/norm/kernel", PartitionSpec(None)),
                ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
                (".*", PartitionSpec(None)),
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
        bits: Optional[int] = None,
        scan_layers: bool = False,
        **kwargs,
    ):
        self.scan_layers = scan_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return "params", "dropout", "fcm"

from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class FalconConfig(EasyDelPretrainedConfig):
    model_type = "falcon"
    attribute_map = {
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
    }

    def __init__(
            self,
            vocab_size: int = 65024,
            hidden_size: int = 4544,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 71,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            hidden_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_kv_heads=None,
            alibi: bool = False,
            new_decoder_architecture: bool = False,
            multi_query: bool = True,
            parallel_attn: bool = True,
            bias: bool = False,
            max_position_embeddings: int = 2048,
            rope_theta: float = 10000.0,
            rope_scaling=None,
            bos_token_id: int = 11,
            eos_token_id: int = 11,
            use_pjit_attention_force: bool = False,
            gradient_checkpointing: str = "",
            bits: Optional[int] = None,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.use_pjit_attention_force = use_pjit_attention_force
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.alibi = alibi
        self.bias = bias
        self.gradient_checkpointing = gradient_checkpointing
        self.parallel_attn = parallel_attn
        self.num_kv_heads = num_kv_heads
        self.new_decoder_architecture = new_decoder_architecture
        self.bits = bits
        self.from_pt = False

        super().__init__(
            axis_dims=axis_dims,
            axis_names=axis_names,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        return not self.alibi

    @staticmethod
    def get_partition_rules(fully_sharded_data_parallel: bool = False):
        return (
            ('word_embeddings/embedding', PartitionSpec("dp", ("fsdp", "sp"))),
            ('self_attention/query_key_value/(kernel)', PartitionSpec("dp", ("fsdp", "sp"))),
            ('self_attention/dense/(kernel)', PartitionSpec("dp", ("fsdp", "sp"))),
            ('mlp/dense_4h_to_h/(kernel)', PartitionSpec("dp", ("fsdp", "sp"))),
            ('mlp/dense_h_to_4h/(kernel)', PartitionSpec("dp", ("fsdp", "sp"))),
            ('lm_head/kernel', PartitionSpec("dp", ("fsdp", "sp"))),
            ('transformer/ln_f/bias', PartitionSpec(("fsdp", "sp"))),
            ('transformer/ln_f/scale', PartitionSpec(("fsdp", "sp"))),
            ('transformer/post_attention_layernorm/scale', PartitionSpec(("fsdp", "sp"))),
            ('transformer/post_attention_layernorm/bias', PartitionSpec(("fsdp", "sp"))),
            ('.*', PartitionSpec(("fsdp", "sp")))
        ) if not fully_sharded_data_parallel else (
            ('word_embeddings/embedding', PartitionSpec(("fsdp", "sp"))),
            ('self_attention/query_key_value/(kernel|bias)', PartitionSpec(("fsdp", "sp"))),
            ('self_attention/dense/(kernel|bias)', PartitionSpec(("fsdp", "sp"))),
            ('mlp/dense_4h_to_h/(kernel|bias)', PartitionSpec(("fsdp", "sp"))),
            ('mlp/dense_h_to_4h/(kernel|bias)', PartitionSpec(("fsdp", "sp"))),
            ('lm_head/kernel', PartitionSpec(("fsdp", "sp"))),
            ('transformer/ln_f/bias', PartitionSpec(("fsdp", "sp"))),
            ('transformer/ln_f/scale', PartitionSpec(("fsdp", "sp"))),
            ('transformer/post_attention_layernorm/scale', PartitionSpec(("fsdp", "sp"))),
            ('transformer/post_attention_layernorm/bias', PartitionSpec(("fsdp", "sp"))),
            ('.*', PartitionSpec(("fsdp", "sp")))
        )

    @staticmethod
    def get_mesh_names():
        return "dp", "fsdp", "tp", "sp"

    def add_jax_args(self,
                     vocab_size: int = 65024,
                     hidden_size: int = 4544,
                     num_hidden_layers: int = 32,
                     num_attention_heads: int = 71,
                     layer_norm_epsilon: float = 1e-5,
                     initializer_range: float = 0.02,
                     use_cache: bool = True,
                     hidden_dropout: float = 0.0,
                     attention_dropout: float = 0.0,
                     num_kv_heads=None,
                     alibi: bool = False,
                     new_decoder_architecture: bool = False,
                     multi_query: bool = True,
                     parallel_attn: bool = True,
                     bias: bool = False,
                     max_position_embeddings: int = 2048,
                     rope_theta: float = 10000.0,
                     rope_scaling=None,
                     bos_token_id: int = 11,
                     eos_token_id: int = 11,
                     use_pjit_attention_force: bool = False,
                     gradient_checkpointing: str = "",
                     bits: Optional[int] = None,
                     **kwargs,
                     ):
        basics = dict(
            bits=bits,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            rope_theta=rope_theta,
            initializer_range=initializer_range,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            num_kv_heads=num_kv_heads,
            eos_token_id=eos_token_id,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            multi_query=multi_query,
            alibi=alibi,
            bias=bias,
            parallel_attn=parallel_attn,
            rope_scaling=rope_scaling,
            use_pjit_attention_force=use_pjit_attention_force,
            gradient_checkpointing=gradient_checkpointing,
            new_decoder_architecture=new_decoder_architecture,
            **kwargs
        )
        for key_state, value_state in basics.items():
            if not hasattr(self, key_state):
                setattr(self, key_state, value_state)

        self.from_pt = False

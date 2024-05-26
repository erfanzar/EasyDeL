from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDeLPretrainedConfig


class FalconConfig(EasyDeLPretrainedConfig):
    model_type: str = "falcon"
    attribute_map = {
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
    }

    def __init__(
            self,
            vocab_size=65024,
            hidden_size=4544,
            num_hidden_layers=32,
            num_attention_heads=71,
            num_ln_in_parallel_attn=None,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            num_kv_heads=None,
            alibi=False,
            new_decoder_architecture=False,
            multi_query=True,
            parallel_attn=True,
            bias=False,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rope_scaling=None,
            bos_token_id=11,
            eos_token_id=11,
            ffn_hidden_size=None,
            ff_factor=None,
            activation="gelu",
            gradient_checkpointing: str = "",
            bits: Optional[int] = None,
            **kwargs
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        self.num_ln_in_parallel_attn = num_ln_in_parallel_attn
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
        self.activation = activation
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.alibi = alibi
        self.bias = bias
        self.gradient_checkpointing = gradient_checkpointing
        self.parallel_attn = parallel_attn
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.new_decoder_architecture = new_decoder_architecture
        self.bits = bits
        self.from_pt = False
        self.head_dim = self.hidden_size // self.num_attention_heads
        if ffn_hidden_size is None:
            ffn_hidden_size = hidden_size * 4
        self.ffn_hidden_size = ffn_hidden_size
        if ff_factor is None:
            ff_factor = ffn_hidden_size // hidden_size
        self.ff_factor = ff_factor
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            bits=bits,
            **kwargs
        )

    @property
    def rotary(self):
        return not self.alibi

    def get_partition_rules(self, fully_sharded_data_parallel: bool = False):
        return (
            ("word_embeddings/embedding", PartitionSpec("dp", ("fsdp", "sp"))),
            ("self_attention/query_key_value/(kernel)", PartitionSpec("dp", ("fsdp", "sp"))),
            ("self_attention/dense/(kernel)", PartitionSpec("dp", ("fsdp", "sp"))),
            ("mlp/dense_4h_to_h/(kernel)", PartitionSpec("dp", ("fsdp", "sp"))),
            ("mlp/dense_h_to_4h/(kernel)", PartitionSpec("dp", ("fsdp", "sp"))),
            ("lm_head/kernel", PartitionSpec("dp", ("fsdp", "sp"))),
            ("transformer/ln_f/bias", PartitionSpec(("fsdp", "sp"))),
            ("transformer/ln_f/scale", PartitionSpec(("fsdp", "sp"))),
            ("transformer/post_attention_layernorm/scale", PartitionSpec(("fsdp", "sp"))),
            ("transformer/post_attention_layernorm/bias", PartitionSpec(("fsdp", "sp"))),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
            (".*", PartitionSpec(("fsdp", "sp")))
        ) if not fully_sharded_data_parallel else (
            ("word_embeddings/embedding", PartitionSpec(("fsdp", "sp"))),
            ("self_attention/query_key_value/(kernel|bias)", PartitionSpec(("fsdp", "sp"))),
            ("self_attention/dense/(kernel|bias)", PartitionSpec(("fsdp", "sp"))),
            ("mlp/dense_4h_to_h/(kernel|bias)", PartitionSpec(("fsdp", "sp"))),
            ("mlp/dense_h_to_4h/(kernel|bias)", PartitionSpec(("fsdp", "sp"))),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
            ("transformer/ln_f/bias", PartitionSpec(("fsdp", "sp"))),
            ("transformer/ln_f/scale", PartitionSpec(("fsdp", "sp"))),
            ("transformer/post_attention_layernorm/scale", PartitionSpec(("fsdp", "sp"))),
            ("transformer/post_attention_layernorm/bias", PartitionSpec(("fsdp", "sp"))),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
            (".*", PartitionSpec(("fsdp", "sp")))
        )

    @staticmethod
    def get_mesh_names():
        return "dp", "fsdp", "tp", "sp"

    def add_jax_args(
            self,
            gradient_checkpointing: str = "",
            bits: Optional[int] = None,
            **kwargs,
    ):
        basics = dict(
            bits=bits,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )
        for key_states, value_states in basics.items():
            if not hasattr(self, key_states):
                setattr(self, key_states, value_states)

        self.from_pt = False

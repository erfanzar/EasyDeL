import math
from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class StableLmConfig(EasyDelPretrainedConfig):
    """Phi configuration."""

    model_type = "stablelm"

    def __init__(
            self,
            vocab_size=50304,
            intermediate_size=6912,
            hidden_size=2560,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            layer_norm_eps=1.0e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10_000,
            rope_scaling=None,
            use_qkv_bias=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            partial_rotary_factor=0.25,
            bos_token_id=0,
            eos_token_id=0,
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
            use_pjit_attention_force: bool = False,
            **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.use_qkv_bias = use_qkv_bias
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

    def add_jax_args(
            self,
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
            use_pjit_attention_force: bool = False,
            **kwargs
    ):
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        self.use_pjit_attention_force = use_pjit_attention_force
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
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

import math
from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class PhiConfig(EasyDelPretrainedConfig):
    """Phi configuration."""

    model_type = "phi"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }

    def __init__(
            self,
            vocab_size=51200,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=None,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attention_dropout=0.0,
            hidden_act="gelu_new",
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            partial_rotary_factor=0.5,
            qk_layernorm=False,
            bos_token_id=1,
            eos_token_id=2,
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
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm
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
            ("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),
            ("final_layernorm/(scale|bias)", PartitionSpec(None, )),
            ("final_layernorm/(scale|bias)", PartitionSpec(None, )),
            ("mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/fc1/bias", PartitionSpec("tp", )),
            ("mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
            ("mlp/fc2/bias", PartitionSpec(("fsdp", "sp"), )),
            ("self_attn/dense/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/dense/bias", PartitionSpec("tp")),
            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/(q_proj|k_proj|v_proj)/bias", PartitionSpec("tp", )),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("lm_head/bias", PartitionSpec("tp")),
            (".*", PartitionSpec(None, ))
        ) if fully_sharded_data_parallel else (
            ("embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"), )),
            ("final_layernorm/(scale|bias)", PartitionSpec(None, )),
            ("final_layernorm/(scale|bias)", PartitionSpec(None, )),
            ("mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/fc1/bias", PartitionSpec("tp", )),
            ("mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
            ("mlp/fc2/bias", PartitionSpec(("fsdp", "sp"), )),
            ("self_attn/dense/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/dense/bias", PartitionSpec("tp")),
            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/(q_proj|k_proj|v_proj)/bias", PartitionSpec("tp", )),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("lm_head/bias", PartitionSpec("tp")),
            (".*", PartitionSpec(None, ))
        )

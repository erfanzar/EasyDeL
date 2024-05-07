import math
from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDeLPretrainedConfig


class Phi3Config(EasyDeLPretrainedConfig):
    """Phi configuration."""

    model_type: str = "phi3"

    def __init__(
            self,
            vocab_size=32064,
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attention_dropout=0.0,
            hidden_act="silu",
            max_position_embeddings=4096,
            original_max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            bos_token_id=1,
            eos_token_id=32000,
            pad_token_id=32000,
            sliding_window=None,
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
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
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.sliding_window = sliding_window

        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs
        )

    def add_jax_args(
            self,
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
            **kwargs
    ):
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        return (
            ("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),

            ("norm/kernel", PartitionSpec(("fsdp", "sp"), )),
            ("post_attention_layernorm/kernel", PartitionSpec(("fsdp", "sp"), )),
            ("input_layernorm/kernel", PartitionSpec(("fsdp", "sp"),)),

            ("mlp/gate_up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),

            ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/qkv_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(None, ))

        ) if fully_sharded_data_parallel else (
            ("embed_tokens/embedding", PartitionSpec(("fsdp", "sp"), "tp")),

            ("norm/kernel", PartitionSpec(None, )),
            ("post_attention_layernorm/kernel", PartitionSpec(None, )),
            ("input_layernorm/kernel", PartitionSpec(None, )),

            ("mlp/gate_up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"), )),
            ("self_attn/qkv_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(None, ))
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["su", "yarn"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['su', 'yarn'], got {rope_scaling_type}")
        if not (
                isinstance(rope_scaling_short_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        if not len(rope_scaling_short_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
                isinstance(rope_scaling_long_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_long_factor)}"
            )

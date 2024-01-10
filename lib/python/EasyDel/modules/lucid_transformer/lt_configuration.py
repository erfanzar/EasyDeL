from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class FlaxLTConfig(EasyDelPretrainedConfig):
    def __init__(self,
                 initializer_range: float = 0.02,
                 hidden_size: int = 4096,
                 bos_token_id=2,
                 eos_token_id=1,
                 pad_token_id=0,
                 intermediate_size: int = 8192,
                 num_hidden_layers: int = 32,
                 vocab_size: int = 32000,
                 num_attention_heads: int = 32,
                 weight_decay: float = 0.02,
                 max_sequence_length: int = 2048,
                 softmax_scale: float = None,
                 alibi_bias_max: int = 8,
                 fsdp=False,
                 hidden_act="silu",
                 **kwargs
                 ):
        self.max_sequence_length = max_sequence_length
        self.weight_decay = weight_decay
        self.alibi_bias_max = alibi_bias_max
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.softmax_scale = softmax_scale
        self.fsdp = fsdp
        self.hidden_act = hidden_act

        super().__init__(
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )

    @staticmethod
    def get_partition_rules(fsdp: bool = True):
        return (
            # Emb
            ("model/wte/embedding", PartitionSpec("sp", "fsdp")),
            ("attn/(k_proj|v_proj|q_proj)/kernel", PartitionSpec("fsdp")),
            ("attn/o_proj/kernel", PartitionSpec("sp", "fsdp")),
            ("mlp/down/kernel", PartitionSpec("sp", "fsdp")),
            ("mlp/up/kernel", PartitionSpec("fsdp")),
            ("lm_head/kernel", PartitionSpec("fsdp", "sp")),
            (".*", PartitionSpec(("fsdp", "sp"))),
            ('ln/kernel', PartitionSpec(None)),
            ('ln1/kernel', PartitionSpec(None)),
            ('ln2/kernel', PartitionSpec(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'

    def add_jax_args(self, *args, **kwargs):
        ...

from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDeLPretrainedConfig


class RwkvConfig(EasyDeLPretrainedConfig):
    """RWKV configuration."""

    model_type: str = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length"}

    def __init__(
            self,
            vocab_size=50277,
            context_length=1024,
            hidden_size=4096,
            num_hidden_layers=32,
            attention_hidden_size=None,
            intermediate_size=None,
            layer_norm_epsilon=1e-5,
            bos_token_id=0,
            eos_token_id=0,
            rescale_every=6,
            tie_word_embeddings=False,
            use_cache=True,
            bits: Optional[int] = None,
            gradient_checkpointing: str = "nothing_saveable",
            **kwargs
    ) -> None:

        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
            (".*", PartitionSpec(("sp", "fsdp"))),
        ) if fully_sharded_data_parallel else (
            (".*", PartitionSpec(("sp", "fsdp"))),
        )

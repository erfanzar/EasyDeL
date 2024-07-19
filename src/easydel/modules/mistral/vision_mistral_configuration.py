from easydel.modules.mistral.mistral_configuration import MistralConfig as MistralConfig
from jax.sharding import PartitionSpec


class VisionMistralConfig(MistralConfig):
    def __init__(
            self,
            vision_vocab_size=8448,
            tie_vision_embeddings=False,
            sample_mode="all",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_vocab_size = vision_vocab_size
        self.tie_vision_embeddings = tie_vision_embeddings
        self.sample_mode = sample_mode

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        """
        Get the partition rules for the model.

        Args:
            fully_sharded_data_parallel (`bool`, *optional*, defaults to `True`):
                Whether to use fully sharded data parallelism.

        Returns:
            `Tuple[Tuple[str, PartitionSpec]]`: The partition rules.
        """
        return (

            ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
            ("model/embed_vision/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("vision_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(None)),
        ) if not fully_sharded_data_parallel else (

            ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),
            ("model/embed_vision/embedding", PartitionSpec(("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
            ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
            ("vision_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(("fsdp", "sp"))),
        )

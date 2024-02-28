from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class GPTNeoXConfig(EasyDelPretrainedConfig):
    model_type = "gpt_neox"

    def __init__(
            self,
            vocab_size=50432,
            hidden_size=6144,
            num_hidden_layers=44,
            num_attention_heads=64,
            intermediate_size=24576,
            hidden_act="gelu",
            rotary_pct=0.25,
            rotary_emb_base=10000,
            classifier_dropout=0.1,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=2,
            tie_word_embeddings=False,
            gradient_checkpointing='everything_saveable',
            use_parallel_residual=True,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing

        self.use_parallel_residual = use_parallel_residual
        self.from_pt = False
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    @staticmethod
    def get_partition_rules(fully_sharded_data_parallel: bool = False):
        return (
            ('wte/embedding', PartitionSpec("fsdp", "dp")),
            ('attention/w_qkv/(kernel|bias)', PartitionSpec("fsdp", "dp")),
            ('attention/wo/(kernel|bias)', PartitionSpec("fsdp", "dp")),
            ('mlp/dense_h_to_4h/(kernel|bias)', PartitionSpec("fsdp", "dp")),
            ('mlp/dense_4h_to_h/(kernel|bias)', PartitionSpec("dp", "fsdp")),

            ('post_attention_layernorm/(bias|scale)', PartitionSpec("fsdp", "dp")),
            ('input_layernorm/(bias|scale)', PartitionSpec("fsdp", "dp")),

            ('transformer/final_layer_norm/(scale|bias)', PartitionSpec("dp", "fsdp")),
            ('lm_head/kernel', PartitionSpec("dp", "fsdp")),
            (".*", PartitionSpec(None))
        ) if not fully_sharded_data_parallel else (

            ('embed_in/embedding', PartitionSpec("fsdp")),

            ('attention/w_qkv/(kernel|bias)', PartitionSpec("fsdp")),
            ('attention/wo/(kernel|bias)', PartitionSpec("fsdp")),
            ('mlp/dense_h_to_4h/(kernel|bias)', PartitionSpec("fsdp")),
            ('mlp/dense_4h_to_h/(kernel|bias)', PartitionSpec("fsdp")),

            ('post_attention_layernorm/(bias|scale)', PartitionSpec("fsdp")),
            ('input_layernorm/(bias|scale)', PartitionSpec("fsdp")),

            ('transformer/final_layer_norm/(scale|bias)', PartitionSpec("fsdp")),
            ('lm_head/kernel', PartitionSpec("fsdp")),
            (".*", PartitionSpec(("fsdp", "sp")))
        )

    @staticmethod
    def get_mesh_names():
        return "dp", "fsdp", "tp", "sp"

    def add_jax_args(
            self,
            **kwargs,
    ):
        self.from_pt = False

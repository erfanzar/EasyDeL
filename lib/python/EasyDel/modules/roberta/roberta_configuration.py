from jax.sharding import PartitionSpec
from ..easydel_modelling_utils import EasyDelPretrainedConfig


class RobertaConfig(EasyDelPretrainedConfig):
    model_type = "roberta"

    def __init__(
            self,
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            gradient_checkpointing="nothing_saveable",
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        return (
            ("embeddings/(position_embeddings|token_type_embeddings)/embedding", PartitionSpec()),
            ("embeddings/word_embeddings/embedding", PartitionSpec()),
            ("attention/self/(key|query|value)/kernel", PartitionSpec("fsdp", "tp")),
            ("attention/self/(key|query|value)/bias", PartitionSpec()),
            ("attention/output/dense/kernel", PartitionSpec("tp", "fsdp")),
            ("attention/output/dense/bias", PartitionSpec()),
            ("(LayerNorm|layer_norm)/(bias|scale)", PartitionSpec()),
            ("intermediate/dense/kernel", PartitionSpec("fsdp", "tp")),
            ("intermediate/dense/bias", PartitionSpec("tp")),
            ("output/dense/kernel", PartitionSpec("tp", "fsdp")),
            ("output/dense/bias", PartitionSpec()),
            ("lm_head/dense/kernel", PartitionSpec()),
            ("lm_head/dense/bias", PartitionSpec()),
            ("lm_head/decoder/kernel", PartitionSpec("fsdp", "tp")),
            ("lm_head/decoder/bias", PartitionSpec("tp")),
            (".*", PartitionSpec()),
        ) if not fully_sharded_data_parallel else (
            ("embeddings/(position_embeddings|token_type_embeddings)/embedding", PartitionSpec()),
            ("embeddings/word_embeddings/embedding", PartitionSpec()),
            ("attention/self/(key|query|value)/kernel", PartitionSpec(("fsdp", "sp"))),
            ("attention/self/(key|query|value)/bias", PartitionSpec()),
            ("attention/output/dense/kernel", PartitionSpec(("fsdp", "sp"))),
            ("attention/output/dense/bias", PartitionSpec()),
            ("(LayerNorm|layer_norm)/(bias|scale)", PartitionSpec()),
            ("intermediate/dense/kernel", PartitionSpec(("fsdp", "sp"))),
            ("intermediate/dense/bias", PartitionSpec("sp")),
            ("output/dense/kernel", PartitionSpec(("fsdp", "sp"))),
            ("output/dense/bias", PartitionSpec()),
            ("lm_head/dense/kernel", PartitionSpec()),
            ("lm_head/dense/bias", PartitionSpec()),
            ("lm_head/decoder/kernel", PartitionSpec(("fsdp", "sp"))),
            ("lm_head/decoder/bias", PartitionSpec("sp")),
            (".*", PartitionSpec()),
        )

    def add_jax_args(
            self,
            gradient_checkpointing="nothing_saveable",
            **kwargs
    ):
        self.gradient_checkpointing = gradient_checkpointing
        for k, v in kwargs.items():
            setattr(self, k, v)

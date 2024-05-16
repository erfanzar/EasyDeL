from typing import Sequence, Optional, Union

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDeLPretrainedConfig


class MptAttentionConfig(EasyDeLPretrainedConfig):
    def __init__(
            self,
            attn_type="multihead_attention",
            attn_pdrop=0,
            attn_impl="torch",
            clip_qkv=None,
            softmax_scale=None,
            prefix_lm=False,
            qk_ln=False,
            attn_uses_sequence_id=False,
            alibi=True,
            alibi_bias_max=8,
            **kwargs,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.attn_pdrop = attn_pdrop
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.softmax_scale = softmax_scale
        self.prefix_lm = prefix_lm
        self.attn_uses_sequence_id = attn_uses_sequence_id
        self.alibi = alibi
        self.qk_ln = qk_ln
        self.alibi_bias_max = alibi_bias_max

        if attn_type not in ["multihead_attention", "multiquery_attention"]:
            raise ValueError(
                f"`attn_type` has to be either `multihead_attention` or `multiquery_attention`. Received: {attn_type}"
            )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            **kwargs
    ) -> "EasyDeLPretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "mpt":
            config_dict = config_dict["attn_config"]
        return cls.from_dict(config_dict, **kwargs)


class MptConfig(EasyDeLPretrainedConfig):
    model_type = "mpt"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    def __init__(self,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 n_layers: int = 24,
                 expansion_ratio: int = 4,
                 max_seq_len: int = 2048,
                 vocab_size: int = 50368,
                 resid_prob_drop: float = 0.0,
                 layer_norm_epsilon: float = 1e-5,
                 emb_prob_drop: float = 0.0,
                 learned_pos_emb: bool = True,
                 attn_config: MptAttentionConfig = None,
                 init_device: str = "cpu",
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = True,
                 verbose: int = 0,
                 embedding_fraction: float = 1.0,
                 norm_type: str = "low_precision_layernorm",
                 use_cache: bool = False,
                 initializer_range=0.02,
                 alibi: bool = True,
                 use_bias: bool = False,
                 act_fn: str = "gelu",
                 qk_ln: bool = False,
                 use_lm_head: bool = False,
                 use_norm_bias: bool = False,
                 gradient_checkpointing: str = "nothing_saveable",
                 bits: Optional[int] = None,
                 **kwargs
                 ):

        self.d_model = d_model
        self.use_norm_bias = use_norm_bias
        self.use_lm_head = use_lm_head
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_prob_drop = resid_prob_drop
        self.use_bias = use_bias
        self.emb_prob_drop = emb_prob_drop
        self.gradient_checkpointing = gradient_checkpointing
        self.norm_type = norm_type
        self.learned_pos_emb = learned_pos_emb
        self.act_fn = act_fn
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.qk_ln = qk_ln
        self.alibi = alibi
        self.verbose = verbose
        self.initializer_range = initializer_range
        self.embedding_fraction = embedding_fraction
        self.init_device = init_device
        self.use_cache = use_cache
        self.bits = bits
        self.layer_norm_epsilon = layer_norm_epsilon
        self.from_pt = False
        self.attn_config = attn_config
        super().__init__(
            bits=bits,
            **kwargs
        )

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        return (

            ("transformer/wte/embedding", PartitionSpec("dp", "fsdp")),
            ("transformer/wpe/embedding", PartitionSpec("dp", "fsdp")),

            ("attn/w_qkv/kernel", PartitionSpec("fsdp", "dp")),
            ("attn/wo/kernel", PartitionSpec("dp", "fsdp")),
            ("attn/w_qkv/bias", PartitionSpec("fsdp", "dp")),
            ("attn/wo/bias", PartitionSpec("dp", "fsdp")),

            ("ffn/down/kernel", PartitionSpec("fsdp", "dp")),
            ("ffn/up/kernel", PartitionSpec("fsdp", "dp")),
            ("ffn/down/kernel", PartitionSpec("fsdp", "dp")),
            ("ffn/up/kernel", PartitionSpec("fsdp", "dp")),

            ("attention_norm/kernel", PartitionSpec(None)),
            ("norm_f/kernel", PartitionSpec(None)),
            ("norm_f/bias", PartitionSpec(None)),

            ("transformer/norm_f/kernel", PartitionSpec(None)),
            ("transformer/norm_f/bias", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp", "dp")),
            ("lm_head/bias", PartitionSpec("fsdp", "dp")),
            (".*", PartitionSpec(None)),
        ) if not fully_sharded_data_parallel else (

            ("transformer/wte/embedding", PartitionSpec("fsdp")),
            ("transformer/wpe/embedding", PartitionSpec("fsdp")),

            ("attn/w_qkv/kernel", PartitionSpec("fsdp")),
            ("attn/wo/kernel", PartitionSpec("fsdp")),
            ("attn/w_qkv/bias", PartitionSpec("fsdp")),
            ("attn/wo/bias", PartitionSpec("fsdp")),

            ("ffn/down/kernel", PartitionSpec("fsdp")),
            ("ffn/up/kernel", PartitionSpec("fsdp")),
            ("ffn/down/kernel", PartitionSpec("fsdp")),
            ("ffn/up/kernel", PartitionSpec("fsdp")),

            ("attention_norm/kernel", PartitionSpec(None)),
            ("norm_f/kernel", PartitionSpec(None)),
            ("norm_f/bias", PartitionSpec(None)),

            ("transformer/norm_f/kernel", PartitionSpec(None)),
            ("transformer/norm_f/bias", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec("fsdp")),
            ("lm_head/bias", PartitionSpec("fsdp")),
            (".*", PartitionSpec(("fsdp", "sp"))),
        )

    def add_jax_args(
            self,
            gradient_checkpointing: str = "nothing_saveable",
            bits: Optional[int] = None,
            **kwargs,
    ):
        if hasattr(self, "attn_config"):
            for k, v in self.attn_config.__dict__.items():
                setattr(self, k, v)
        basics = dict(bits=bits, gradient_checkpointing=gradient_checkpointing, **kwargs)
        for k, v in basics.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.from_pt = False

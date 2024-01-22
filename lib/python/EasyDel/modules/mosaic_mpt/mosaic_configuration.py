from typing import Sequence, Optional, Union

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class MptConfig(EasyDelPretrainedConfig):
    model_type = 'mpt'

    def __init__(self,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 n_layers: int = 24,
                 expansion_ratio: int = 4,
                 max_seq_len: int = 2048,
                 vocab_size: int = 50368,
                 resid_prob_drop: float = 0.0,
                 emb_prob_drop: float = 0.0,
                 alibi: bool = True,
                 use_bias: bool = False,
                 learned_pos_emb: bool = True,
                 act_fn: str = 'gelu',
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = False,
                 verbose: int = 0,
                 embedding_fraction: float = 1.0,
                 use_cache: bool = False,
                 qk_ln: bool = False,
                 use_lm_head: bool = False,
                 use_norm_bias: bool = False,
                 gradient_checkpointing: str = 'nothing_saveable',
                 use_pjit_attention_force: bool = False,
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
        self.use_pjit_attention_force = use_pjit_attention_force
        self.gradient_checkpointing = gradient_checkpointing
        self.learned_pos_emb = learned_pos_emb
        self.act_fn = act_fn
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.qk_ln = qk_ln
        self.alibi = alibi
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.use_cache = use_cache
        self.bits = bits

        self.from_pt = False
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        super().__init__(
            **kwargs
        )

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    @staticmethod
    def get_partition_rules(fully_sharded_data_parallel: bool = False):
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

    def add_jax_args(self,
                     d_model: int = 2048,
                     n_heads: int = 16,
                     n_layers: int = 24,
                     expansion_ratio: int = 4,
                     max_seq_len: int = 2048,
                     vocab_size: int = 50368,
                     resid_prob_drop: float = 0.0,
                     emb_prob_drop: float = 0.0,
                     alibi: bool = True,
                     use_bias: bool = True,
                     learned_pos_emb: bool = True,
                     act_fn: str = 'gelu',
                     logit_scale: Optional[Union[float, str]] = None,
                     no_bias: bool = False,
                     verbose: int = 0,
                     embedding_fraction: float = 1.0,
                     use_cache: bool = False,
                     qk_ln: bool = True,
                     use_lm_head: bool = False,
                     use_norm_bias: bool = False,
                     gradient_checkpointing: str = 'nothing_saveable',
                     use_pjit_attention_force: bool = False,
                     bits: Optional[int] = None,
                     **kwargs,
                     ):
        if hasattr(self, 'attn_config'):
            for k, v in self.attn_config.items():
                setattr(self, k, v)
        basics = dict(
            bits=bits,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            expansion_ratio=expansion_ratio,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            resid_prob_drop=resid_prob_drop,
            emb_prob_drop=emb_prob_drop,
            alibi=alibi,
            use_bias=use_bias,
            learned_pos_emb=learned_pos_emb,
            act_fn=act_fn,
            logit_scale=logit_scale,
            no_bias=no_bias,
            verbose=verbose,
            embedding_fraction=embedding_fraction,
            use_cache=use_cache,
            qk_ln=qk_ln,
            use_lm_head=use_lm_head,
            use_norm_bias=use_norm_bias,
            gradient_checkpointing=gradient_checkpointing,
            use_pjit_attention_force=use_pjit_attention_force,
            **kwargs
        )
        for k, v in basics.items():
            if not hasattr(self, k):
                print(f' Key {k} not found in loaded config setting that to default of {v}')
                setattr(self, k, v)

        self.from_pt = False

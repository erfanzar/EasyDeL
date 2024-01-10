from typing import Sequence, Optional

from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class OPTConfig(EasyDelPretrainedConfig):
    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size: int = 50272,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            ffn_dim: int = 3072,
            max_position_embeddings: int = 2048,
            do_layer_norm_before: bool = True,
            _remove_final_layer_norm: bool = False,
            word_embed_proj_dim: int = None,
            dropout: float = 0.1,
            attention_dropout: float = 0.0,
            num_attention_heads: int = 12,
            activation_function: str = "relu",
            layerdrop: float = 0.0,
            init_std: float = 0.02,
            use_cache: bool = True,
            pad_token_id: int = 1,
            bos_token_id: int = 2,
            eos_token_id: int = 2,
            enable_bias: bool = True,
            layer_norm_elementwise_affine: bool = True,
            gradient_checkpointing: str = 'nothing_saveable',
            use_pjit_attention_force: bool = False,
            **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.use_pjit_attention_force = use_pjit_attention_force
        self.gradient_checkpointing = gradient_checkpointing
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
        self._remove_final_layer_norm = _remove_final_layer_norm
        self.from_pt = False

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        if not fully_sharded_data_parallel:
            raise NotImplementedError
        else:
            return (
                (".*", PartitionSpec(("fsdp", "sp")))
            )

    def add_jax_args(
            self,
            vocab_size: int = 50272,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            ffn_dim: int = 3072,
            max_position_embeddings: int = 2048,
            do_layer_norm_before: bool = True,
            _remove_final_layer_norm: bool = False,
            word_embed_proj_dim: int = None,
            dropout: float = 0.1,
            attention_dropout: float = 0.0,
            num_attention_heads: int = 12,
            activation_function: str = "relu",
            layerdrop: float = 0.0,
            init_std: float = 0.02,
            use_cache: bool = True,
            pad_token_id: int = 1,
            bos_token_id: int = 2,
            eos_token_id: int = 2,
            enable_bias: bool = True,
            layer_norm_elementwise_affine: bool = True,
            gradient_checkpointing: str = 'nothing_saveable',
            use_pjit_attention_force: bool = False,
            **kwargs,
    ):
        basics = dict(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            max_position_embeddings=max_position_embeddings,
            do_layer_norm_before=do_layer_norm_before,
            _remove_final_layer_norm=_remove_final_layer_norm,
            word_embed_proj_dim=word_embed_proj_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_attention_heads=num_attention_heads,
            activation_function=activation_function,
            layerdrop=layerdrop,
            init_std=init_std,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            enable_bias=enable_bias,
            layer_norm_elementwise_affine=layer_norm_elementwise_affine,
            gradient_checkpointing=gradient_checkpointing,
            use_pjit_attention_force=use_pjit_attention_force,
            **kwargs
        )
        for k, v in basics.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.from_pt = False

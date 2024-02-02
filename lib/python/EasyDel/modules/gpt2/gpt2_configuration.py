from typing import Optional

import jax

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class GPT2Config(EasyDelPretrainedConfig):
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            gradient_checkpointing: str = "nothing_saveable",
            use_pjit_attention_force: bool = False,
            tie_word_embeddings: bool = False,
            bits: Optional[int] = None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_pjit_attention_force = use_pjit_attention_force
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

    def add_jax_args(
            self,
            gradient_checkpointing: str = "nothing_saveable",
            use_pjit_attention_force: bool = False,
            bits: Optional[int] = None,
            **kwargs
    ):
        args = dict(
            use_pjit_attention_force=use_pjit_attention_force,
            gradient_checkpointing=gradient_checkpointing,
            bits=bits,
            **kwargs
        )
        for k, v in args.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        return (
            ("transformer/wte/embedding", jax.sharding.PartitionSpec("tp", ("fsdp", "sp"))),
            ("transformer/lm_head", jax.sharding.PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", jax.sharding.PartitionSpec(("fsdp", "sp"))),
        )

from typing import Optional

from jax.sharding import PartitionSpec

from easydel.modules.modeling_utils import EDPretrainedConfig


class GPTJConfig(EDPretrainedConfig):
    model_type: str = "gptj"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 50400,
        n_positions: int = 2048,
        n_embd: int = 4096,
        n_layer: int = 28,
        n_head: int = 16,
        rotary_dim: int = 64,
        n_inner: int = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: int = 0.02,
        use_cache: int = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        tie_word_embeddings: bool = False,
        gradient_checkpointing: str = "",
        bits: Optional[int] = None,
        **kwargs,
    ):
        self.bits = bits
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.from_pt = False
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

    @staticmethod
    def set_custom_partition(
        embedding_partition: PartitionSpec,
        kvq_partition: PartitionSpec,
        o_proj_partition: PartitionSpec,
        fc_out_partition: PartitionSpec,
        fc_in_partition: PartitionSpec,
        fc_lm_head_partition: PartitionSpec,
        rest_partitions: PartitionSpec = PartitionSpec(None),
    ):
        return (
            ("model/wte/embedding", embedding_partition),
            ("attn/(k_proj|v_proj|q_proj)/kernel", kvq_partition),
            ("attn/out_proj/kernel", o_proj_partition),
            ("mlp/fc_out/kernel", fc_out_partition),
            ("mlp/fc_out/bias", fc_out_partition),
            ("mlp/fc_in/kernel", fc_in_partition),
            ("mlp/fc_in/bias", fc_in_partition),
            ("lm_head/kernel", fc_lm_head_partition),
            ("lm_head/bias", fc_lm_head_partition),
            (".*", rest_partitions),
        )

    @staticmethod
    def get_partition_rules(just_fsdp: bool = True):
        if just_fsdp:
            rules = (
                (
                    "model/wte/embedding",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "attn/(k_proj|v_proj|q_proj)/kernel",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "attn/out_proj/kernel",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "mlp/fc_out/kernel",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "mlp/fc_out/bias",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "mlp/fc_in/kernel",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "mlp/fc_in/bias",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "lm_head/kernel",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (
                    "lm_head/bias",
                    PartitionSpec(
                        ("fsdp", "tp"),
                    ),
                ),
                (".*", PartitionSpec(None)),
            )
        else:
            rules = (
                ("model/wte/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
                (
                    "attn/(k_proj|v_proj|q_proj)/kernel",
                    PartitionSpec(("fsdp", "sp"), "tp"),
                ),
                (
                    "attn/out_proj/kernel",
                    PartitionSpec(
                        "tp",
                        ("fsdp", "sp"),
                    ),
                ),
                ("mlp/fc_out/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
                ("mlp/fc_out/bias", PartitionSpec(("fsdp", "sp"), "tp")),
                (
                    "mlp/fc_in/kernel",
                    PartitionSpec(
                        "tp",
                        ("fsdp", "sp"),
                    ),
                ),
                (
                    "mlp/fc_in/bias",
                    PartitionSpec(
                        "tp",
                        ("fsdp", "sp"),
                    ),
                ),
                (
                    "lm_head/kernel",
                    PartitionSpec(
                        "tp",
                        ("fsdp", "sp"),
                    ),
                ),
                (
                    "lm_head/bias",
                    PartitionSpec(
                        "tp",
                        ("fsdp", "sp"),
                    ),
                ),
                (".*", PartitionSpec(("fsdp", "sp"))),
            )
        return rules

    @staticmethod
    def get_mesh_names():
        return "dp", "fsdp", "tp", "sp"

    def add_jax_args(
        self,
        vocab_size: int = 50400,
        n_positions: int = 2048,
        n_embd: int = 4096,
        n_layer: int = 28,
        n_head: int = 16,
        rotary_dim: int = 64,
        n_inner: int = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: int = 0.02,
        use_cache: int = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        tie_word_embeddings: bool = False,
        bits: Optional[int] = None,
        gradient_checkpointing: str = "",
        **kwargs,
    ):
        basics = dict(
            bits=bits,
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            rotary_dim=rotary_dim,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=gradient_checkpointing,
        )

        for k, v in basics.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.from_pt = False
        return self


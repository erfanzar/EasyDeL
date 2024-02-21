from typing import Optional, Mapping
from jax.sharding import PartitionSpec

from ..easydel_modelling_utils import EasyDelPretrainedConfig


class Qwen2Config(EasyDelPretrainedConfig):
    model_type = "qwen2"

    def __init__(
            self,
            vocab_size=151936,
            hidden_size=4096,
            intermediate_size=22016,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            attention_dropout=0.0,
            resid_pdrop: float = 0.0,
            embd_pdrop: float = 0.0,
            gradient_checkpointing: str = "nothing_saveable",
            fcm_min_ratio: float = 0.0,
            fcm_max_ratio: float = 0.0,
            use_pjit_attention_force: bool = False,
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            number_rep_kv: int = 1,
            bits: Optional[int] = None,
            scan_layers: bool = True,
            rope_scaling: Optional[Mapping[str, str | float]] = None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.rope_scaling = rope_scaling
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.scan_layers = scan_layers
        self.embd_pdrop = embd_pdrop
        self.number_rep_kv = number_rep_kv
        self.resid_pdrop = resid_pdrop
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.use_pjit_attention_force = use_pjit_attention_force
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        """
        The get_partition_rules function is used to define the partitioning scheme for a model.
        It returns a list of tuples, where each tuple contains two elements:
            1) A regex string that matches the name of one or more parameters in the model.
            2) A PartitionScheme object that defines how those parameters should be partitioned across devices.

        :param fully_sharded_data_parallel: bool: Determine whether to partition the model fully or not
        :return: A list of tuples

        """
        return (

            ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(None)),
        ) if not fully_sharded_data_parallel else (

            ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

            ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
            ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
            ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

            ("input_layernorm/kernel", PartitionSpec(None)),
            ("post_attention_layernorm/kernel", PartitionSpec(None)),

            ("model/norm/kernel", PartitionSpec(None)),
            ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
            (".*", PartitionSpec(("fsdp", "sp"))),
        )

    def add_jax_args(
            self,
            resid_pdrop: float = 0.0,
            embd_pdrop: float = 0.0,
            attention_dropout: float = 0.0,
            tie_word_embeddings: bool = False,
            gradient_checkpointing: str = "nothing_saveable",
            fcm_min_ratio: float = 0.0,
            fcm_max_ratio: float = 0.0,
            use_pjit_attention_force: bool = False,
            use_scan_mlp: bool = False,
            scan_mlp_chunk_size: int = 1024,
            number_rep_kv: int = 1,
            bits: Optional[int] = None,
            rope_theta: float = 10000.,
            hidden_act: str = "silu",
            scan_layers: bool = True,
            rope_scaling: Optional[Mapping[str, str | float]] = None,
            **kwargs,
    ):
        """
        The add_jax_args function adds the following arguments to the Transformer class:

        :param self: Refer to the current object
        :param resid_pdrop: float: Set the dropout rate for residual connections
        :param embd_pdrop: float: Set the probability of dropping an embedding
        :param attention_dropout: float: Set the probability of dropping out the attention layer
        :param tie_word_embeddings: bool: Tie the word embeddings to the decoder
        :param gradient_checkpointing: str: Control the amount of memory used by jax
        :param fcm_min_ratio: float: Control the minimum ratio of the number of chunks to be used in flash-based computation
        :param fcm_max_ratio: float: Set the maximum ratio of the number of input tokens to output tokens
        :param use_pjit_attention_force: bool: Determine if the attention force is used
        :param use_scan_mlp: bool: Determine whether to use the scan_mlp function or not
        :param scan_mlp_chunk_size: int: Set the chunk size for scan_mlp
        :param number_rep_kv: int: Determine how many times the key and value vectors are repeated
        :param bits: Optional[int]: Determine the number of bits used in the quantization
        :param rope_theta: float : rope_theta for compute rope
        :param attention_bias: bool : whenever to use attention bias or no
        :param hidden_act: str : hidden_act for mlp
        :param scan_layers: bool: Determine whether to use scan layers or not
        :return: The following:

        """
        self.scan_layers = scan_layers
        self.embd_pdrop = embd_pdrop
        self.number_rep_kv = number_rep_kv
        self.resid_pdrop = resid_pdrop
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.use_pjit_attention_force = use_pjit_attention_force

        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return "params", "dropout", "fcm"

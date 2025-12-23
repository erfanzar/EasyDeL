# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("falcon")
class FalconConfig(EasyDeLBaseConfig):
    """

    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65024):
            Vocabulary size of the Falcon model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4544):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 71):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_ln_in_parallel_attn (`int`, *optional*):
            Number of separate layer normalizations when using parallel attention. When set to 2
            with `new_decoder_architecture=True`, uses independent LayerNorms for the attention
            and MLP parallel paths (ln_attn and ln_mlp). When 1 or 0, uses a single shared
            LayerNorm (input_layernorm).
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_kv_heads (`int`, *optional*):
            Number of key and value heads for each attention layer in the Transformer encoder. Will default to
            `num_attention_heads` if not set.
        alibi (`bool`, *optional*):
            Whether to use ALiBi (Attention with Linear Biases) for positional encoding instead
            of RoPE. ALiBi adds learned linear biases to attention scores based on token distances,
            enabling better extrapolation to longer sequences than seen during training. When False,
            uses standard Rotary Position Embeddings (RoPE).
        new_decoder_architecture (`bool`, *optional*):
            Whether to use the improved decoder architecture with refined layer normalization
            placement. When enabled, supports dual layer norms for parallel attention/MLP paths
            via `num_ln_in_parallel_attn`. This architecture was introduced in later Falcon variants
            for better training stability.
        multi_query (`bool`, *optional*, defaults to `True`):
            Whether to use Multi-Query Attention (MQA). When True, all query heads share a single
            set of key and value heads, dramatically reducing memory bandwidth and KV cache size
            during inference while maintaining quality. This is more aggressive than GQA (grouped-query).
        parallel_attn (`bool`, *optional*, defaults to `True`):
            Whether to compute attention and MLP layers in parallel rather than sequentially.
            When True, reduces the effective depth of each layer and improves training throughput,
            as attention and FFN can be computed simultaneously and their outputs summed. This is
            different from the standard sequential attention-then-FFN architecture.
        bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the linear layers.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The rope scaling configuration.
        bos_token_id (`int`, *optional*, defaults to 11):
            The index of the beginning of sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 11):
            The index of the end of sequence token in the vocabulary.
        ffn_hidden_size (`int`, *optional*):
            Dimensionality of the hidden layer in the FFN
        ff_factor (`int`, *optional*):
            The scaling factor of the FFN
        activation (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        gradient_checkpointing (`str`, *optional*, defaults to `""`):
            The gradient checkpointing configuration.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
    """

    model_type: str = "falcon"

    def __init__(
        self,
        vocab_size=65024,
        hidden_size=4544,
        num_hidden_layers=32,
        num_attention_heads=71,
        num_ln_in_parallel_attn=None,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        alibi=False,
        new_decoder_architecture=False,
        multi_query=True,
        parallel_attn=True,
        bias=False,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=11,
        eos_token_id=11,
        ffn_hidden_size=None,
        ff_factor=None,
        activation="gelu",
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        if num_ln_in_parallel_attn is None:
            num_ln_in_parallel_attn = 0
        self.num_ln_in_parallel_attn = num_ln_in_parallel_attn
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.activation = activation
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.alibi = alibi
        self.bias = bias
        self.gradient_checkpointing = gradient_checkpointing
        self.parallel_attn = parallel_attn
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.new_decoder_architecture = new_decoder_architecture
        self.bits = bits
        self.from_pt = False
        self.head_dim = self.hidden_size // self.num_attention_heads
        if ffn_hidden_size is None:
            ffn_hidden_size = hidden_size * 4
        self.ffn_hidden_size = ffn_hidden_size
        if ff_factor is None:
            ff_factor = ffn_hidden_size // hidden_size
        self.ff_factor = ff_factor
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, bits=bits, **kwargs)

    @property
    def rotary(self):
        return not self.alibi

    @property
    def num_key_value_heads(self):
        """Alias for num_kv_heads to match UnifiedAttention expectations."""
        return self.num_kv_heads

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"word_embeddings/embedding", pmag.resolve(ColumnWise)),
            (r"self_attention/query_key_value/kernel", pmag.resolve(ColumnWise)),
            (r"self_attention/dense/kernel", pmag.resolve(RowWise)),
            (r"mlp/dense_h_to_4h/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/dense_4h_to_h/kernel", pmag.resolve(RowWise)),
            (
                r".*/(ln_attn|ln_mlp|input_layernorm|post_attention_layernorm|ln_f)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(ln_attn|ln_mlp|input_layernorm|post_attention_layernorm|ln_f)/bias",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (
                r".*(query_key_value|dense|dense_h_to_4h|dense_4h_to_h|lm_head)/bias",
                pmag.resolve(Replicated),
            ),
            (r".*", pmag.resolve(Replicated)),
        )

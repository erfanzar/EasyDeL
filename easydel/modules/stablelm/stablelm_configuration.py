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


@register_config("stablelm")
class StableLmConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the StableLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~easydel.modules.StableLmModel`].
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6912):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `tp.Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"swish"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`int`, *optional*, defaults to 10000):
            The theta value for the rotary position embeddings.
        rope_scaling (`str`, *optional*):
            The scaling to use for the rotary position embeddings.
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to use layer normalization on the queries and keys in the attention layer.
        use_parallel_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a parallel residual connection in the attention layer.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            The factor to scale the partial rotary embeddings by.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id for the beginning of stream token.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id for the end of stream token.
        bits (`int`, *optional*):
            The number of bits to quantize the model to. If None, the model is not quantized.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            What to save during gradient checkpointing. Choose one of `"nothing_saveable"`, `"first_half_saveable"`,
            `"full_saveable"`.
    """

    model_type: str = "stablelm"

    def __init__(
        self,
        vocab_size=50304,
        intermediate_size=6912,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1.0e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10_000,
        rope_scaling=None,
        use_qkv_bias=False,
        qk_layernorm=False,
        use_parallel_residual=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.25,
        bos_token_id=0,
        eos_token_id=0,
        bits: int | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ) -> None:
        """Initializes the StableLmConfig object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 50304.
            intermediate_size (int, optional): Dimensionality of the intermediate layer in MLP. Defaults to 6912.
            hidden_size (int, optional): Dimensionality of the embeddings and hidden states. Defaults to 2560.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to 32.
            hidden_act (str, optional): Activation function name. Defaults to "silu".
            max_position_embeddings (int, optional): Maximum sequence length. Defaults to 4096.
            initializer_range (float, optional): Standard deviation for weight initialization. Defaults to 0.02.
            layer_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
            use_cache (bool, optional): Whether to use KV cache. Defaults to True.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            rope_theta (int, optional): Base value for RoPE. Defaults to 10000.
            rope_scaling (dict, optional): RoPE scaling configuration. Defaults to None.
            use_qkv_bias (bool, optional): Whether to use bias in QKV projections. Defaults to False.
            qk_layernorm (bool, optional): Whether to apply LayerNorm to query and key states. Defaults to False.
            use_parallel_residual (bool, optional): Whether to use parallel residual connections. Defaults to False.
            hidden_dropout (float, optional): Dropout probability for hidden layers. Defaults to 0.0.
            attention_dropout (float, optional): Dropout probability for attention scores. Defaults to 0.0.
            partial_rotary_factor (float, optional): Factor for partial rotary embeddings. Defaults to 0.25.
            bos_token_id (int, optional): Beginning of sequence token ID. Defaults to 0.
            eos_token_id (int, optional): End of sequence token ID. Defaults to 0.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.qk_layernorm = qk_layernorm
        self.use_parallel_residual = use_parallel_residual
        self.num_key_value_heads = num_key_value_heads
        self.use_qkv_bias = use_qkv_bias
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/(q_proj|k_proj|v_proj|o_proj)/bias", pmag.resolve(Replicated)),
            (
                r"self_attn/(q_layernorm|k_layernorm)/norms/\d+/scale",
                pmag.resolve(Replicated),
            ),
            (r"self_attn/(q_layernorm|k_layernorm)/norms/\d+/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/bias",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

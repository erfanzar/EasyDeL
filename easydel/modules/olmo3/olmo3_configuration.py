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


import typing

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("olmo3")
class Olmo3Config(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`Olmo3Model`]. It is used to instantiate an OLMo3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/OLMo-3-0725-1B](https://huggingface.co/allenai/OLMo-3-0725-1B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
            vocab_size (`int`, *optional*, defaults to 50304):
                    Vocabulary size of the Olmo3 model. Defines the number of different tokens that can be represented by the
                    `inputs_ids` passed when calling [`Olmo3Model`]
            hidden_size (`int`, *optional*, defaults to 4096):
                    Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 11008):
                    Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                    Number of hidden layers in the Transformer decoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                    Number of attention heads for each attention layer in the Transformer decoder.
            num_key_value_heads (`int`, *optional*):
                    This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                    `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                    `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                    converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                    by meanpooling all the original heads within that group. For more details, check out [this
                    paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
                    `num_attention_heads`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                    The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                    The maximum sequence length that this model might ever be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                    The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            use_cache (`bool`, *optional*, defaults to `True`):
                    Whether or not the model should return the last key/values attentions (not used by all models). Only
                    relevant if `config.is_decoder=True`.
            pad_token_id (`int`, *optional*, defaults to 1):
                    Padding token id.
            bos_token_id (`int`, *optional*):
                    Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 50279):
                    End of stream token id.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                    Whether to tie weight embeddings
            rope_theta (`float`, *optional*, defaults to 10000.0):
                    The base period of the RoPE embeddings.
            rope_scaling (`tp.Dict`, *optional*):
                    Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
                    strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
                    `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
                    `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
                    these scaling strategies behave:
                    https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
                    experimental feature, subject to breaking API changes in future versions.
            attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
                    Whether to use a bias in the query, key, value and output projection layers during self-attention.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                    The dropout ratio for the attention probabilities.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                    The epsilon used by the rms normalization layers.
            sliding_window (`int`, *optional*, defaults to 4096):
                    Size of the sliding window for sliding window attention.
            layer_types (`list`, *optional*):
                    Attention pattern for each layer. Defaults to sliding window attention
                    for 3 out of 4 layers, and full attention for every 4th layer.

    >>> from transformers import Olmo3Model, Olmo3Config

    >>> # Initializing a Olmo3 7B style configuration
    >>> configuration = Olmo3Config()

    >>> # Initializing a model from the Olmo3 7B style configuration
    >>> model = Olmo3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo3"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        sliding_window=4096,
        layer_types: list[str] | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        **kwargs,
    ):
        """Initializes an Olmo3Config object.

        Args:
                vocab_size (int, optional): Vocabulary size. Defaults to 50304.
                hidden_size (int, optional): Hidden size. Defaults to 4096.
                intermediate_size (int, optional): Intermediate size of the feed-forward network. Defaults to 11008.
                num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
                num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
                num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to `num_attention_heads`.
                hidden_act (str, optional): Activation function. Defaults to "silu".
                max_position_embeddings (int, optional): Maximum sequence length. Defaults to 2048.
                initializer_range (float, optional): Initializer range. Defaults to 0.02.
                use_cache (bool, optional): Whether to use KV cache. Defaults to True.
                pad_token_id (int, optional): Padding token ID. Defaults to 1.
                bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to None.
                eos_token_id (int, optional): End-of-sequence token ID. Defaults to 50279.
                tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
                rope_theta (float, optional): Base value for RoPE. Defaults to 10000.0.
                rope_scaling (dict, optional): RoPE scaling configuration. Defaults to None.
                attention_bias (bool, optional): Whether to use bias in attention layers. Defaults to False.
                attention_dropout (float, optional): Dropout probability for attention. Defaults to 0.0.
                rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-5.
                sliding_window (int, optional): Size of sliding window for sliding window attention. Defaults to 4096.
                layer_types (list[str], optional): List of attention types per layer ("sliding_attention" or "full_attention").
                        If None, defaults to sliding window for 3/4 layers and full attention every 4th layer.
                gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                        Defaults to EasyDeLGradientCheckPointers.NONE.
                use_scan_mlp (bool, optional): Whether to use scan for MLP layers. Defaults to False.
                scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
                bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
                **kwargs: Additional keyword arguments.
        """
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.head_dim = hidden_size // num_attention_heads

        # Set layer types: default pattern is sliding for 3/4 layers, full for every 4th layer
        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types
        self._validate_layer_types()

    def _rope_scaling_validation(self):
        """
        Validates the `rope_scaling` configuration dictionary to ensure it meets the expected format and values.
        Raises:
                ValueError: If `rope_scaling` is not a dictionary with the correct fields (`type`, `factor`)
                        or if the values are invalid (type not 'linear' or 'dynamic', factor not a float > 1.0).
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                f"`rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _validate_layer_types(self):
        """
        Validates the `layer_types` list to ensure it has correct length and valid values.

        Raises:
                ValueError: If `layer_types` length doesn't match `num_hidden_layers` or contains invalid values.
        """
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` must have length equal to `num_hidden_layers` ({self.num_hidden_layers}), "
                f"got {len(self.layer_types)}"
            )

        valid_types = {"sliding_attention", "full_attention"}
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type not in valid_types:
                raise ValueError(f"`layer_types[{idx}]` must be one of {valid_types}, got '{layer_type}'")

    def get_partition_rules(self, *args, **kwargs):
        """Get the partition rules for distributed training of the OLMo3 model.

        This method defines how model parameters should be partitioned across devices
        for tensor parallelism, following the same patterns as OLMo2.

        Args:
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            tuple: A tuple of tuples, where each inner tuple contains:
                - str: Regular expression pattern matching parameter names
                - PartitionSpec: Specification for how to partition the parameter

        Note:
            The partition rules are identical to OLMo2, as sliding window attention
            doesn't change the parameter structure, only the attention computation pattern.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"self_attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*/(post_attention_layernorm|post_feedforward_layernorm|norm)/kernel",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(post_attention_layernorm|post_feedforward_layernorm|norm)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(post_attention_layernorm|post_feedforward_layernorm|norm)/bias",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("glm4")
class Glm4Config(EasyDeLBaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4Model`]. It is used to instantiate an Glm4
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Glm4-4-9b-chat.
    e.g. [THUDM/GLM-4-9B-0414](https://huggingface.co/THUDM/GLM-4-9B-0414)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the Glm4 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5): The factor of the partial rotary position.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The legacy activation function. It is overwritten by the `hidden_activation`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1.5625e-07):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        pad_token_id (`int`, *optional*, defaults to 151329):
            Padding token id.
        eos_token_id (`int` | `list`, *optional*, defaults to `[151329, 151336, 151338]`):
            End of stream token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
    ```python
    >>> from transformers import Glm4Model, Glm4Config
    >>> # Initializing a Glm4 glm4-4-9b-chat style configuration
    >>> configuration = Glm4Config()
    >>> # Initializing a model from the glm4-4-9b-chat style configuration
    >>> model = Glm4Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4"

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        intermediate_size: int = 13696,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        partial_rotary_factor: float = 0.5,
        head_dim: int = 128,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 0.00000015625,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        pad_token_id: int = 151329,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        attention_bias: bool = True,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        if eos_token_id is None:
            eos_token_id = [151329, 151336, 151338]
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Returns partition rules for model sharding.

        Providing explicit partition rules is preferred over automatic sharding resolution,
        as it gives full control over parameter distribution across the device mesh.
        Returns ``None`` by default, which triggers automatic sharding via
        module-level ``craft_sharding`` hooks.

        Returns:
            Partition rules as ``tuple[tuple[str, PartitionSpec], ...] | None``.
        """
        return None


__all__ = ["Glm4Config"]

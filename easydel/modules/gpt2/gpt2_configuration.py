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


import typing

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("gpt2")
class GPT2Config(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers.
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`str`, *optional*, defaults to `"cls_index"`):
            The summary type to use. Possible values are `"cls_index"` (equivalent to the output of the last token
            of the first sentence in a sequence) and `"last"` (equivalent to the output of the last token in
            the sequence).
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Whether to use a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            The activation to use for the summary.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Whether to project the summary to the labels.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 50256):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 50256):
            The id of the *end-of-sequence* token.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to reorder and upcast attention.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
    """

    model_type: str = "gpt2"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]
    attribute_map: typing.ClassVar = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int | None = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        summary_type: str = "cls_index",
        summary_use_proj: bool = True,
        summary_activation: str | None = None,
        summary_proj_to_labels: bool = True,
        summary_first_dropout: float = 0.1,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        tie_word_embeddings: bool = False,
        bits: int | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initializes a GPT2Config object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 50257.
            n_positions (int, optional): Maximum sequence length. Defaults to 1024.
            n_embd (int, optional): Hidden size. Defaults to 768.
            n_layer (int, optional): Number of hidden layers. Defaults to 12.
            n_head (int, optional): Number of attention heads. Defaults to 12.
            n_inner (int, optional): Inner dimension of FFN. Defaults to None.
            activation_function (str, optional): Activation function. Defaults to "gelu_new".
            resid_pdrop (float, optional): Residual dropout probability. Defaults to 0.1.
            embd_pdrop (float, optional): Embedding dropout probability. Defaults to 0.1.
            attn_pdrop (float, optional): Attention dropout probability. Defaults to 0.1.
            layer_norm_epsilon (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
            initializer_range (float, optional): Initializer range. Defaults to 0.02.
            summary_type (str, optional): Type of summary. Defaults to "cls_index".
            summary_use_proj (bool, optional): Whether to use projection in summary. Defaults to True.
            summary_activation (str, optional): Activation for summary. Defaults to None.
            summary_proj_to_labels (bool, optional): Whether to project summary to labels. Defaults to True.
            summary_first_dropout (float, optional): Dropout after summary projection. Defaults to 0.1.
            scale_attn_weights (bool, optional): Whether to scale attention weights. Defaults to True.
            use_cache (bool, optional): Whether to use KV cache. Defaults to True.
            bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 50256.
            eos_token_id (int, optional): End-of-sequence token ID. Defaults to 50256.
            scale_attn_by_inverse_layer_idx (bool, optional):
                Whether to scale attention by inverse layer index. Defaults to False.
            reorder_and_upcast_attn (bool, optional): Whether to reorder and upcast attention. Defaults to False.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional):
                Gradient checkpointing strategy. Defaults to EasyDeLGradientCheckPointers.NONE.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
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
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.n_layer
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
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

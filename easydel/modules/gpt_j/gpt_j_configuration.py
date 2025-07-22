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


@register_config("gptj")
class GPTJConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50400):
            Vocabulary size of the GPT-J model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        n_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        n_embd (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        rotary_dim (`int`, *optional*, defaults to 64):
            The dimension of the rotary position embedding.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers.
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 50256):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 50256):
            The id of the *end-of-sequence* token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        gradient_checkpointing (`str`, *optional*, defaults to `""`):
            The gradient checkpointing configuration.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
    """

    model_type: str = "gptj"
    attribute_map: typing.ClassVar = {
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
        n_inner: int | None = None,
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
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        **kwargs,
    ):
        """Initializes a GPTJConfig object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 50400.
            n_positions (int, optional): Maximum sequence length. Defaults to 2048.
            n_embd (int, optional): Hidden size. Defaults to 4096.
            n_layer (int, optional): Number of hidden layers. Defaults to 28.
            n_head (int, optional): Number of attention heads. Defaults to 16.
            rotary_dim (int, optional): Dimension for rotary position embeddings. Defaults to 64.
            n_inner (int, optional): Inner dimension of FFN. Defaults to None.
            activation_function (str, optional): Activation function. Defaults to "gelu_new".
            resid_pdrop (float, optional): Residual dropout probability. Defaults to 0.0.
            embd_pdrop (float, optional): Embedding dropout probability. Defaults to 0.0.
            attn_pdrop (float, optional): Attention dropout probability. Defaults to 0.0.
            layer_norm_epsilon (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
            initializer_range (int, optional): Initializer range. Defaults to 0.02.
            use_cache (int, optional): Whether to use KV cache. Defaults to True.
            bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 50256.
            eos_token_id (int, optional): End-of-sequence token ID. Defaults to 50256.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
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

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"wte/embedding", pmag.resolve(ColumnWise)),
            (r"attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"attn/out_proj/kernel", pmag.resolve(RowWise)),
            (r"attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/fc_in/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/fc_out/kernel", pmag.resolve(RowWise)),
            (r"mlp/fc_in/bias", pmag.resolve(Replicated)),
            (r"mlp/fc_out/bias", pmag.resolve(Replicated)),
            (r".*/(ln_1|ln_f)/scale", pmag.resolve(Replicated)),
            (r".*/(ln_1|ln_f)/bias", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"lm_head/bias", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

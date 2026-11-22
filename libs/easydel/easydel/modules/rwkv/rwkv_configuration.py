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

"""Configuration class for the RWKV model family.

Defines :class:`RwkvConfig`, the EasyDeL configuration object for the
RWKV recurrent-attention architecture (linear-time time-mix + channel-mix
blocks parameterized by per-channel decay/key/receptance vectors). The
configuration captures embedding dimension, number of layers, intermediate
size, attention/feed-forward hidden sizes, and rescaling/init factors.
"""

import typing

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("rwkv")
class RwkvConfig(EasyDeLBaseConfig):
    """Configuration for the RWKV recurrent-attention language model.

    Inherits from :class:`EasyDeLBaseConfig` and adds the RWKV-specific
    hyperparameters: number of hidden layers, attention/intermediate
    sizes, per-layer rescaling cadence, and recurrent-cache toggles.
    The defaults mirror the public ``RWKV-4-Pile-7B`` checkpoint.

    Attributes:
        vocab_size (int): Vocabulary size. Defines the number of distinct
            tokens representable by ``input_ids``. Defaults to 50277.
        context_length (int): Maximum sequence length supported by the
            model (aliased as ``max_position_embeddings``). Defaults to 1024.
        hidden_size (int): Dimensionality of the encoder layers and the
            pooler layer. Defaults to 4096.
        num_hidden_layers (int): Number of hidden RWKV blocks. Defaults to 32.
        attention_hidden_size (int | None): Dimensionality of the QKV-like
            projections of the RWKV time-mix block. Falls back to
            ``hidden_size`` when ``None``.
        intermediate_size (int | None): Dimensionality of the channel-mix
            ("feed-forward") layer. Falls back to ``4 * hidden_size`` when
            ``None``.
        layer_norm_epsilon (float): Epsilon for the LayerNorm modules.
            Defaults to 1e-5.
        rescale_every (int): Cadence (in layers) at which the attention
            output is rescaled by 1/2 during inference to stabilise the
            recurrence; 0 disables the rescale. Defaults to 6.
        bos_token_id (int): Beginning-of-stream token id. Defaults to 0.
        eos_token_id (int): End-of-stream token id. Defaults to 0.
        tie_word_embeddings (bool): Whether to tie the input embedding
            and the LM head weights. Defaults to False.
        use_cache (bool): Whether the model should return the recurrent
            state for stepwise generation. Defaults to True.
        bits (int | None): Optional quantisation bit-width. ``None`` keeps
            the model in full precision.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient
            checkpointing policy. Defaults to
            ``EasyDeLGradientCheckPointers.NONE``.
    """

    model_type: str = "rwkv"
    attribute_map: typing.ClassVar = {"max_position_embeddings": "context_length"}

    def __init__(
        self,
        vocab_size=50277,
        context_length=1024,
        hidden_size=4096,
        num_hidden_layers=32,
        attention_hidden_size=None,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        rescale_every=6,
        tie_word_embeddings=False,
        use_cache=True,
        bits: int | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ) -> None:
        """Initialize RwkvConfig.

        See the class docstring for parameter semantics. Note that
        ``attention_hidden_size`` defaults to ``hidden_size`` and
        ``intermediate_size`` defaults to ``4 * hidden_size`` when left
        unspecified. ``**kwargs`` are forwarded to
        :class:`EasyDeLBaseConfig`.
        """
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            bits=bits,
            **kwargs,
        )

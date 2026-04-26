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


from eformer.loggings import get_logger
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

logger = get_logger(__name__)


@register_config("qwen3")
class Qwen3Config(EasyDeLBaseConfig):
    """Configuration for the Qwen3 decoder-only transformer architecture.

    Qwen3 is a decoder-only transformer featuring grouped query attention (GQA),
    SiLU-gated MLP, RMS normalization, and optional sliding window attention.
    Supports RoPE scaling for extended context lengths.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden layers.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimensionality of the MLP intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of transformer decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads for grouped query attention. Falls back to
            ``num_attention_heads`` (MHA) if ``None``.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function used in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length this model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values for caching during generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output word embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for rotary position embeddings.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration (e.g., ``{"type": "yarn", "factor": 4.0}``).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in QKV and output projection layers.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to enable sliding window attention for lower layers.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size (only used when ``use_sliding_window=True``).
        max_window_layers (`int`, *optional*, defaults to 28):
            Layers at or above this index use sliding window attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for attention weights.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type (``"full_attention"`` or ``"sliding_attention"``).
            Auto-derived from sliding window settings if not provided.
    """

    model_type = "qwen3"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int | None = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize Qwen3Config with model architecture hyperparameters.

        See class docstring for detailed parameter descriptions.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Returns partition rules for model sharding.

        Providing explicit partition rules is preferred over automatic sharding resolution,
        as it gives full control over parameter distribution across the device mesh.
        Returns ``None`` by default, which triggers automatic sharding via
        spectrax parameter metadata.

        Returns:
            Partition rules as ``tuple[tuple[str, PartitionSpec], ...] | None``.
        """
        return None

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Retrieve attention mask details for each layer in the model.

        This method generates a dictionary mapping layer indices to their corresponding attention mask details.
        If a sliding window is defined, each layer is assigned a sliding window attention mask with the specified size.

        Returns:
            dict[int, AttnMaskDetail]: A dictionary where keys are layer indices (int) and values are AttnMaskDetail
            objects specifying the attention mask type and size for each layer.

        Notes:
            - If `self.sliding_window` is None, an empty dictionary is returned.
            - The method iterates over `self.num_hidden_layers` to assign mask details for each layer.
            - The attention mask type is set to `AttnMaskType.SLIDING` when a sliding window is defined.
        """
        mapping = {}
        if self.layer_types is not None:
            for layer_idx in range(self.num_hidden_layers):
                mapping[layer_idx] = AttnMaskDetail(
                    mask_type=AttnMaskType.from_hf(self.layer_types[layer_idx]),
                    size=self.sliding_window,
                )
        return mapping


__all__ = ["Qwen3Config"]

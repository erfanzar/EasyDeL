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


import typing as tp

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("qwen2")
class Qwen2Config(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read
    the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key and value heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use a sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            The sliding window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            The maximum number of layers to use for the sliding window attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        fcm_min_ratio (`float`, *optional*, defaults to 0.0):
            The minimum ratio for Flash Attention.
        fcm_max_ratio (`float`, *optional*, defaults to 0.0):
            The maximum ratio for Flash Attention.
        use_scan_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation for the MLP.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size to use when scanning the MLP.
        number_rep_kv (`int`, *optional*, defaults to 1):
            Number of repetitions for the key and value vectors.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        scan_layers (`bool`, *optional*, defaults to `True`):
            Whether to use the scan implementation for the layers.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The configuration for rope scaling.
    """

    model_type: str = "qwen2"

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
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        fcm_min_ratio: float = 0.0,
        fcm_max_ratio: float = 0.0,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        number_rep_kv: int = 1,
        bits: int | None = None,
        scan_layers: bool = True,
        layer_types: list[str] | None = None,
        rope_scaling: tp.Mapping[str, str | float] | None = None,
        **kwargs,
    ):
        """Initializes a Qwen2Config object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 151936.
            hidden_size (int, optional): Dimensionality of the embeddings and hidden states. Defaults to 4096.
            intermediate_size (int, optional): Dimensionality of the intermediate layer in MLP. Defaults to 22016.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to 32.
            hidden_act (str, optional): Activation function name. Defaults to "silu".
            max_position_embeddings (int, optional): Maximum sequence length. Defaults to 32768.
            initializer_range (float, optional): Standard deviation for weight initialization. Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-6.
            use_cache (bool, optional): Whether to use KV cache. Defaults to True.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            rope_theta (float, optional): Base value for RoPE. Defaults to 10000.0.
            use_sliding_window (bool, optional): Whether to use sliding window attention. Defaults to False.
            sliding_window (int, optional): Sliding window size. Defaults to 4096.
            max_window_layers (int, optional): Maximum number of layers for sliding window attention. Defaults to 28.
            attention_dropout (float, optional): Dropout probability for attention scores. Defaults to 0.0.
            resid_pdrop (float, optional): Dropout probability for residual connections. Defaults to 0.0.
            embd_pdrop (float, optional): Dropout probability for embeddings. Defaults to 0.0.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            fcm_min_ratio (float, optional): Minimum ratio for Flash Attention. Defaults to 0.0.
            fcm_max_ratio (float, optional): Maximum ratio for Flash Attention. Defaults to 0.0.
            use_scan_mlp (bool, optional): Whether to use scan for MLP layers. Defaults to False.
            scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
            number_rep_kv (int, optional): Number of repetitions for key/value vectors. Defaults to 1.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            scan_layers (bool, optional): Whether to use scan for transformer layers. Defaults to True.
            rope_scaling (tp.Optional[tp.Mapping[str, str | float]], optional):
                RoPE scaling configuration. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
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
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        self.head_dim = hidden_size // num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
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
            (r"self_attn/(q_proj|k_proj|v_proj)/bias", pmag.resolve(Replicated)),
            (r"self_attn/o_proj/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/kernel",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

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

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


@register_config("internlm2")
class InternLM2Config(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the InternLM2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            Number of key and value heads for each attention layer in the Transformer encoder. Will default to
            `number_rep_kv * num_attention_heads` if not set.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the *pad* token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        bias (`bool`, *optional*, defaults to `False`):
            Whether to use attention bias.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        fcm_min_ratio (`float`, *optional*, defaults to -1):
            The minimum ratio for Flash Attention.
        fcm_max_ratio (`float`, *optional*, defaults to -1):
            The maximum ratio for Flash Attention.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The configuration for rope scaling.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size to use when scanning the MLP.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The hidden activation function to use.
        pretraining_tp (`int`, *optional*, defaults to 1):
            The tensor parallelism degree used during pretraining.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP.
        scan_layers (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation for the layers.
    """

    model_type: str = "internlm2"

    def __init__(
        self,
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        bias=True,
        rope_theta=10000,
        rope_scaling=None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        fcm_min_ratio: float = -1,
        fcm_max_ratio: float = -1,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        scan_layers: bool = False,
        **kwargs,
    ):
        """Initializes an InternLM2Config object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 103168.
            hidden_size (int, optional): Hidden size. Defaults to 4096.
            intermediate_size (int, optional): Intermediate size of the feed-forward network. Defaults to 11008.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional):
                Number of key/value heads (for GQA). Defaults to None (uses num_attention_heads).
            hidden_act (str, optional): Activation function. Defaults to "silu".
            max_position_embeddings (int, optional): Maximum sequence length. Defaults to 2048.
            initializer_range (float, optional): Initializer range. Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-6.
            use_cache (bool, optional): Whether to use KV cache. Defaults to True.
            pad_token_id (int, optional): Padding token ID. Defaults to 0.
            bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 1.
            eos_token_id (int, optional): End-of-sequence token ID. Defaults to 2.
            pretraining_tp (int, optional): Tensor parallelism degree during pretraining. Defaults to 1.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
            rope_theta (float, optional): Base value for RoPE. Defaults to 10000.
            rope_scaling (dict, optional): RoPE scaling configuration. Defaults to None.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            fcm_min_ratio (float, optional): Minimum ratio for Flash Attention. Defaults to -1.
            fcm_max_ratio (float, optional): Maximum ratio for Flash Attention. Defaults to -1.
            scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            scan_layers (bool, optional): Whether to use scan for layers. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        num_key_value_heads = num_key_value_heads or num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.bias = bias
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = fcm_min_ratio
        self.hidden_act = hidden_act
        self.fcm_max_ratio = fcm_max_ratio
        self.rope_scaling = rope_scaling
        self.bits = bits
        self.scan_layers = scan_layers
        self.attn_implementation = "eager"
        # HF: AttributeError: 'InternLM2Config' object has no attribute 'attn_implementation'.
        # Did you mean: '_attn_implementation'?
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
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
            (r"tok_embeddings/embedding", pmag.resolve(ColumnWise)),
            (r"attention/wqkv/kernel", pmag.resolve(ColumnWise)),
            (r"attention/wo/kernel", pmag.resolve(RowWise)),
            (r"feed_forward/(w1|w3)/kernel", pmag.resolve(ColumnWise)),
            (r"feed_forward/w2/kernel", pmag.resolve(RowWise)),
            (r".*/(attention_norm|ffn_norm|norm)/kernel", pmag.resolve(Replicated)),
            (r"output/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*/(wqkv|wo|w1|w3|w2|output|score)/bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    @property
    def granted_freq_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size specifically for frequency-based position embeddings.

        If `freq_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
        `max_position_embeddings`.

        Returns:
            int: The granted maximum position embedding size for frequency encoding.
        """
        return getattr(
            self,
            "freq_max_position_embeddings",
            self.max_position_embeddings,
        )

    @property
    def granted_mask_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size specifically for mask-based position embeddings.

        If `mask_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
        `max_position_embeddings`.

        Returns:
            int: The granted maximum position embedding size for mask encoding.
        """
        return getattr(
            self,
            "mask_max_position_embeddings",
            self.max_position_embeddings,
        )

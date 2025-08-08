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


from eformer.common_types import ColumnWise, ExpertColumnWiseAlt, ExpertRowWiseAlt, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("mixtral")
class MixtralConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Mixtral model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key and value heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 4096 * 32):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`float`, *optional*, defaults to 1e6):
            The theta value to use for rotary position embeddings.
        sliding_window (`int`, *optional*, defaults to 4096):
            The sliding window size.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts per token.
        num_local_experts (`int`, *optional*, defaults to 8):
            The number of local experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to output router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The router auxiliary loss coefficient.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        use_scan_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation for the MLP.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size to use when scanning the MLP.
        number_rep_kv (`int`, *optional*, defaults to 1):
            Number of repetitions for the key and value vectors.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The configuration for rope scaling.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        initialization_of_moe (`bool`, *optional*, defaults to `False`):
            Whether to initialize the MoE layers.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            The jitter noise for the router.
    """

    model_type: str = "mixtral"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=4096,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        number_rep_kv: int = 1,
        bits: int | None = None,
        rope_scaling: dict[str, str | float] | None = None,
        attention_bias: bool = False,
        initialization_of_moe: bool = False,
        router_jitter_noise=0.0,
        head_dim: int | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initializes a MixtralConfig object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 32000.
            hidden_size (int, optional): Hidden size. Defaults to 4096.
            intermediate_size (int, optional): Intermediate size of the feed-forward network. Defaults to 14336.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to 8.
            hidden_act (str, optional): Activation function. Defaults to "silu".
            max_position_embeddings (int, optional): Maximum sequence length. Defaults to 4096 * 32.
            initializer_range (float, optional): Initializer range. Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-5.
            use_cache (bool, optional): Whether to use KV cache. Defaults to True.
            pad_token_id (int, optional): Padding token ID. Defaults to None.
            bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 1.
            eos_token_id (int, optional): End-of-sequence token ID. Defaults to 2.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            rope_theta (float, optional): Base value for RoPE. Defaults to 1e6.
            sliding_window (int, optional): Sliding window size for attention. Defaults to 4096.
            attention_dropout (float, optional): Dropout probability for attention. Defaults to 0.0.
            num_experts_per_tok (int, optional): Number of experts to route per token. Defaults to 2.
            num_local_experts (int, optional): Total number of local experts. Defaults to 8.
            output_router_logits (bool, optional): Whether to output router logits. Defaults to False.
            router_aux_loss_coef (float, optional): Coefficient for router auxiliary loss. Defaults to 0.001.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            use_scan_mlp (bool, optional): Whether to use scan for MLP layers. Defaults to False.
            scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
            number_rep_kv (int, optional): Number of repetitions for key/value heads. Defaults to 1.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            rope_scaling (tp.Dict[str, tp.Union[str, float]], optional): RoPE scaling configuration. Defaults to None.
            attention_bias (bool, optional): Whether to use bias in attention layers. Defaults to False.
            initialization_of_moe (bool, optional): Whether MoE layers are being initialized. Defaults to False.
            router_jitter_noise (float, optional): Jitter noise for router gates. Defaults to 0.0.
            **kwargs: Additional keyword arguments.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.bits = bits
        self.attention_dropout = attention_dropout
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.attention_bias = attention_bias
        # for backward compatibility
        self.rope_scaling = rope_scaling
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initialization_of_moe = initialization_of_moe
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.number_rep_kv = number_rep_kv
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.router_jitter_noise = router_jitter_noise
        self.layer_types = layer_types
        self.head_dim = head_dim or hidden_size // num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"block_sparse_moe/gate/kernel", pmag.resolve(ColumnWise)),
            (r"block_sparse_moe/gate/bias", pmag.resolve(Replicated)),
            (r"block_sparse_moe/experts/(w1|w3)/kernel", pmag.resolve(ExpertColumnWiseAlt)),
            (r"block_sparse_moe/experts/w2/kernel", pmag.resolve(ExpertRowWiseAlt)),
            (r"block_sparse_moe/experts/.*bias", pmag.resolve(Replicated)),
            (r".*/(input_layernorm|post_attention_layernorm|norm)/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
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
        return getattr(self, "freq_max_position_embeddings", self.max_position_embeddings)

    @property
    def granted_mask_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size specifically for mask-based position embeddings.

        If `mask_max_position_embeddings` is set, it returns that value. Otherwise, it falls back to
        `max_position_embeddings`.

        Returns:
            int: The granted maximum position embedding size for mask encoding.
        """
        return getattr(self, "mask_max_position_embeddings", self.max_position_embeddings)

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

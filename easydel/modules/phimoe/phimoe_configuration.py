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
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("phimoe")
class PhiMoeConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32064):
            Vocabulary size of the PhiMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`PhiMoEModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6400):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Mixtral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            The scaling strategy for the RoPE embeddings. If `None`, no scaling is applied. If a dictionary, it must
            contain the following keys: `type`, `short_factor`, `long_factor`, `short_mscale`, `long_mscale` and
            `original_max_position_embeddings`. The `type` must be `longrope`, the `short_mscale` and `long_scale` must
            be numbers, the `short_factor` and `long_factor` must be lists of numbers with the same length as half of
            the attention head size and the `original_max_position_embeddings` must be an integer.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `262144`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_local_experts (`int`, *optional*, defaults to 16):
            Number of experts per Sparse MLP layer.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabeling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.0):
            The aux loss factor for the total loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.01):
            Amount of noise to add to the router.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
    """

    model_type: str = "phimoe"

    def __init__(
        self,
        vocab_size=32064,
        hidden_size=4096,
        intermediate_size=6400,
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
        rope_scaling=None,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=16,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.01,
        input_jitter_noise=0.0,
        attention_bias=False,
        embd_pdrop: float = 0.0,
        lm_head_bias=False,
        bits: int | None = None,
        layer_types: list[str] | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ) -> None:
        """Initializes a PhiMoeConfig object.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 32064.
            hidden_size (int, optional): Dimensionality of the embeddings and hidden states. Defaults to 4096.
            intermediate_size (int, optional): Dimensionality of the intermediate layer in MLP. Defaults to 6400.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to 8.
            hidden_act (str, optional): Activation function name. Defaults to "silu".
            max_position_embeddings (int, optional): Maximum sequence length. Defaults to 4096 * 32.
            initializer_range (float, optional): Standard deviation for weight initialization. Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-5.
            use_cache (bool, optional): Whether to use KV cache. Defaults to True.
            pad_token_id (int, optional): Padding token ID. Defaults to None.
            bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 1.
            eos_token_id (int, optional): End-of-sequence token ID. Defaults to 2.
            tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
            rope_theta (float, optional): Base value for RoPE. Defaults to 1e6.
            rope_scaling (dict, optional): RoPE scaling configuration. Defaults to None.
            sliding_window (int, optional): Sliding window size for attention. Defaults to None.
            attention_dropout (float, optional): Dropout probability for attention scores. Defaults to 0.0.
            num_experts_per_tok (int, optional): Number of experts to route per token. Defaults to 2.
            num_local_experts (int, optional): Total number of local experts. Defaults to 16.
            output_router_logits (bool, optional): Whether to output router logits. Defaults to False.
            router_aux_loss_coef (float, optional): Coefficient for router auxiliary loss. Defaults to 0.001.
            router_jitter_noise (float, optional): Jitter noise for router gates. Defaults to 0.01.
            input_jitter_noise (float, optional): Jitter noise for input tokens (not typically used). Defaults to 0.0.
            attention_bias (bool, optional): Whether to use bias in attention projections. Defaults to False.
            embd_pdrop (float, optional): Dropout probability for embeddings. Defaults to 0.0.
            lm_head_bias (bool, optional): Whether to use bias in the LM head. Defaults to False.
            bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                Defaults to EasyDeLGradientCheckPointers.NONE.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise
        self.embd_pdrop = embd_pdrop
        self.rope_scaling = rope_scaling or {}
        self._rope_scaling_validation()
        self.bits = bits
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_types = layer_types
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
            (r"block_sparse_moe/experts/(w1|w3)/kernel", pmag.resolve(ColumnWise)),
            (r"block_sparse_moe/experts/w2/kernel", pmag.resolve(RowWise)),
            (r"block_sparse_moe/experts/.*bias", pmag.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/scale",
                pmag.resolve(Replicated),
            ),
            (
                r".*/(input_layernorm|post_attention_layernorm|norm)/bias",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"lm_head/bias", pmag.resolve(Replicated)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        """Validates the `rope_scaling` configuration dictionary.

    Ensures that `rope_scaling` is a dictionary with the correct keys and value types
    for the 'longrope' scaling type.

    Raises:
        ValueError: If `rope_scaling` is not a dictionary, is missing keys,
            or has invalid values/types for the 'longrope' configuration.
    """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 6:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor`, `long_factor`, "
                f"`short_mscale`, `long_mscale` and `original_max_position_embeddings`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        rope_scaling_short_mscale = self.rope_scaling.get("short_mscale", None)
        rope_scaling_long_mscale = self.rope_scaling.get("long_mscale", None)
        original_max_position_embeddings = self.rope_scaling.get("original_max_position_embeddings", None)
        if rope_scaling_type is None or rope_scaling_type not in ["longrope"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}")
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, int | float) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        if not len(rope_scaling_short_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length "
                f"{self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, int | float) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length "
                f"{self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_long_factor)}"
            )
        if not isinstance(rope_scaling_short_mscale, int | float):
            raise ValueError(f"`rope_scaling`'s short_mscale field must be a number, got {rope_scaling_short_mscale}")
        if not isinstance(rope_scaling_long_mscale, int | float):
            raise ValueError(f"`rope_scaling`'s long_mscale field must be a number, got {rope_scaling_long_mscale}")
        if not isinstance(original_max_position_embeddings, int):
            raise ValueError(
                f"`rope_scaling`'s original_max_position_embeddings field must be an integer, "
                f"got {original_max_position_embeddings}"
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

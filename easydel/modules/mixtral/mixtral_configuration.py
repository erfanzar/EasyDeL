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


from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("mixtral")
class MixtralConfig(EasyDeLBaseConfig):
    """Configuration for Mistral AI's Mixtral sparse-MoE decoder family.

    Mixtral is a LLaMA-style transformer where every FFN is replaced by a
    Top-``num_experts_per_tok``-of-``num_local_experts`` Mixture-of-Experts.
    The reference 8x7B variant uses 8 experts and routes each token to 2,
    giving an 8x7B-parameter model with only ~14B active parameters per
    token. Routing is *softmax-based*: the router produces a distribution
    over experts via softmax, the top-k weights are kept, and the surviving
    weights are renormalized to sum to one before scaling expert outputs.
    The auxiliary load-balancing loss (``router_aux_loss_coef``) is the
    standard Switch-Transformer objective minimizing
    :math:`N_e \\cdot \\sum_i f_i p_i` over experts (fraction of tokens
    routed and mean router probability).

    Other notable features mirrored in this config:
        * **Sliding-window attention** (``sliding_window``) restricts each
          token to attend to only the most recent ``W`` tokens, dropping
          attention cost from O(L²) to O(L·W); long-range mixing is
          recovered by stacking layers.
        * **Grouped-query attention** with ``num_key_value_heads = 8``
          shrinks the KV cache footprint relative to full MHA.
        * **Long RoPE base** (``rope_theta = 1e6``) enables the
          ``max_position_embeddings ≈ 130k`` context.

    Attributes:
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
            Size of the sliding window for local attention in lower layers. Tokens can only
            attend to tokens within this window distance, reducing computational complexity
            from O(n²) to O(n*window) while maintaining long-range modeling via stacked layers.
            Set to None or a very large value to use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of expert networks to route each token to (top-k selection). The router selects
            this many experts with the highest gating scores, and the token's representation is
            computed as a weighted sum of these experts' outputs. Mixtral-8x7B uses k=2 (top-2),
            activating only 2 out of 8 experts per token for sparse computation.
        num_local_experts (`int`, *optional*, defaults to 8):
            Total number of expert feed-forward networks per MoE layer. Each layer contains this
            many independent expert FFNs, and each token is routed to `num_experts_per_tok` of them.
            Mixtral-8x7B uses 8 experts per layer, creating an 8x7B parameter model (8 experts of
            ~7B params each) while only activating ~2x7B=14B params per token.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits in model outputs. When True, includes the raw gating
            scores from the router network for each layer, which can be used to compute the
            auxiliary load-balancing loss during training or analyze expert selection patterns.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient for the auxiliary load-balancing loss added to the main training loss.
            This loss encourages balanced expert utilization by penalizing uneven token routing,
            preventing expert collapse where some experts are overused and others underused.
            Typical values are 0.001-0.01.
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
            Amount of uniform noise to add to router logits during training to improve exploration
            and prevent premature expert specialization. The noise is sampled from Uniform(-noise, +noise)
            and added to gating scores before top-k selection. Set to 0.0 to disable noise injection.
    """

    model_type: str = "mixtral"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096 * 32,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1e6,
        sliding_window: int | None = 4096,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        number_rep_kv: int = 1,
        bits: int | None = None,
        rope_scaling: dict[str, str | float] | None = None,
        attention_bias: bool = False,
        initialization_of_moe: bool = False,
        router_jitter_noise: float = 0.0,
        head_dim: int | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the Mixtral configuration.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 32000.
            hidden_size (int, optional): Hidden dimension. Defaults to 4096.
            intermediate_size (int, optional): Per-expert MLP intermediate dimension.
                Defaults to 14336.
            num_hidden_layers (int, optional): Number of decoder layers. Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
            num_key_value_heads (int | None, optional): Number of key/value heads for
                grouped-query attention. Defaults to 8.
            hidden_act (str, optional): MLP activation. Defaults to "silu".
            max_position_embeddings (int, optional): Maximum sequence length.
                Defaults to ``4096 * 32``.
            initializer_range (float, optional): Initializer standard deviation.
                Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon for RMSNorm. Defaults to 1e-5.
            use_cache (bool, optional): Whether to enable KV caching. Defaults to True.
            pad_token_id (int | None, optional): Padding token id. Defaults to None.
            bos_token_id (int, optional): Beginning-of-sequence token id. Defaults to 1.
            eos_token_id (int, optional): End-of-sequence token id. Defaults to 2.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to False.
            rope_theta (float, optional): RoPE base period. Defaults to 1e6.
            sliding_window (int | None, optional): Sliding-window size for sliding
                attention layers. Defaults to 4096.
            attention_dropout (float, optional): Attention dropout. Defaults to 0.0.
            num_experts_per_tok (int, optional): MoE top-k routing parameter.
                Defaults to 2.
            num_local_experts (int, optional): Total experts per MoE layer. Defaults to 8.
            output_router_logits (bool, optional): Whether to output router logits.
                Defaults to False.
            router_aux_loss_coef (float, optional): Auxiliary load-balancing loss
                coefficient. Defaults to 0.001.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient
                checkpointing policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            use_scan_mlp (bool, optional): Whether to use the scan implementation for
                the MLP. Defaults to False.
            scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
            number_rep_kv (int, optional): Number of repetitions for key/value heads.
                Defaults to 1.
            bits (int | None, optional): Quantization bits. Defaults to None.
            rope_scaling (dict[str, str | float] | None, optional): RoPE scaling
                configuration. Defaults to None.
            attention_bias (bool, optional): Whether attention projections use bias.
                Defaults to False.
            initialization_of_moe (bool, optional): Whether to initialize MoE layers
                explicitly. Defaults to False.
            router_jitter_noise (float, optional): Uniform noise amplitude added to
                router logits during training. Defaults to 0.0.
            head_dim (int | None, optional): Per-head dimension; defaults to
                ``hidden_size // num_attention_heads`` when ``None``. Defaults to None.
            layer_types (list[str] | None, optional): Per-layer attention type. If
                ``None``, defaults to ``"sliding_attention"`` (or ``"full_attention"``
                if ``sliding_window`` is ``None``) for every layer. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to ``EasyDeLBaseConfig``.
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
                for _ in range(self.num_hidden_layers)
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

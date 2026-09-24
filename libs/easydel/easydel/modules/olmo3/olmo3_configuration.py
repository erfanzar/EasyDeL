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

"""Configuration class for the OLMo-3 (Open Language Model v3) family.

Defines :class:`Olmo3Config`, registered as ``model_type="olmo3"``. OLMo-3
extends OLMo-2 with a *hybrid sliding-window + full-attention schedule*: the
``layer_types`` list assigns each layer either ``"sliding_attention"`` or
``"full_attention"``, mirroring the local/global pattern used by Llama 4 /
Mistral-Nemo. QK normalization and post-norm residual layout are inherited
from OLMo-2.
"""

import copy
import typing

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("olmo3")
class Olmo3Config(EasyDeLBaseConfig):
    r"""Configuration for AI2's OLMo-3 family.

    OLMo-3 is the third iteration of AI2's fully-open OLMo decoder LMs.
    Compared to OLMo-2 it adopts a **hybrid sliding-window / full-attention
    layer schedule**: most layers use sliding-window attention of size
    ``sliding_window``, with a small number of full-attention layers
    sprinkled in (controlled by ``layer_types`` — a list whose entries
    are either ``"sliding_attention"`` or ``"full_attention"``). This
    is the same idea behind Mistral-Nemo / Llama-4 NoPE: cheap local
    attention on most layers, dense long-range mixing on a few. OLMo-3
    keeps OLMo-2's QK-norm (RMSNorm on Q and K) and post-normalization
    architecture for training stability. Default sizing mirrors the
    public ``allenai/OLMo-3-0725-1B`` checkpoint.

    Attributes:
            vocab_size (`int`, *optional*, defaults to 50304):
                    Vocabulary size of the Olmo3 model. Defines the number of different tokens that can be represented by the
                    `inputs_ids` passed when calling [`Olmo3Model`]
            hidden_size (`int`, *optional*, defaults to 4096):
                    Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 11008):
                    Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                    Number of hidden layers in the Transformer decoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                    Number of attention heads for each attention layer in the Transformer decoder.
            num_key_value_heads (`int`, *optional*):
                    This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                    `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                    `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                    converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                    by meanpooling all the original heads within that group. For more details, check out [this
                    paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
                    `num_attention_heads`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                    The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                    The maximum sequence length that this model might ever be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                    The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            use_cache (`bool`, *optional*, defaults to `True`):
                    Whether or not the model should return the last key/values attentions (not used by all models). Only
                    relevant if `config.is_decoder=True`.
            pad_token_id (`int`, *optional*, defaults to 1):
                    Padding token id.
            bos_token_id (`int`, *optional*):
                    Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 50279):
                    End of stream token id.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                    Whether to tie weight embeddings
            rope_theta (`float`, *optional*, defaults to 10000.0):
                    The base period of the RoPE embeddings.
            rope_scaling (`tp.Dict`, *optional*):
                    Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
                    strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
                    `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
                    `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
                    these scaling strategies behave:
                    https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
                    experimental feature, subject to breaking API changes in future versions.
            attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
                    Whether to use a bias in the query, key, value and output projection layers during self-attention.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                    The dropout ratio for the attention probabilities.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                    The epsilon used by the rms normalization layers.
            sliding_window (`int`, *optional*, defaults to 4096):
                    Size of the sliding window for sliding window attention.
            layer_types (`list`, *optional*):
                    Attention pattern for each layer. Defaults to sliding window attention
                    for 3 out of 4 layers, and full attention for every 4th layer.

    Example:
        >>> from easydel import Olmo3Config, Olmo3Model
        >>> configuration = Olmo3Config()
        >>> model = Olmo3Model(configuration)
        >>> configuration = model.config
    """

    model_type = "olmo3"
    keys_to_ignore_at_inference: typing.ClassVar = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: int | None = None,
        eos_token_id: int = 50279,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        sliding_window: int = 4096,
        layer_types: list[str] | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        rope_parameters: dict | None = None,
        rope_theta_is_shared: bool | None = None,
        **kwargs,
    ):
        """Initializes an Olmo3Config object.

        Args:
                vocab_size (int, optional): Vocabulary size. Defaults to 50304.
                hidden_size (int, optional): Hidden size. Defaults to 4096.
                intermediate_size (int, optional): Intermediate size of the feed-forward network. Defaults to 11008.
                num_hidden_layers (int, optional): Number of hidden layers. Defaults to 32.
                num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
                num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to `num_attention_heads`.
                hidden_act (str, optional): Activation function. Defaults to "silu".
                max_position_embeddings (int, optional): Maximum sequence length. Defaults to 2048.
                initializer_range (float, optional): Initializer range. Defaults to 0.02.
                use_cache (bool, optional): Whether to use KV cache. Defaults to True.
                pad_token_id (int, optional): Padding token ID. Defaults to 1.
                bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to None.
                eos_token_id (int, optional): End-of-sequence token ID. Defaults to 50279.
                tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
                rope_theta (float, optional): Base value for RoPE. Defaults to 10000.0.
                rope_scaling (dict, optional): RoPE scaling configuration. Defaults to None.
                attention_bias (bool, optional): Whether to use bias in attention layers. Defaults to False.
                attention_dropout (float, optional): Dropout probability for attention. Defaults to 0.0.
                rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-5.
                sliding_window (int, optional): Size of sliding window for sliding window attention. Defaults to 4096.
                layer_types (list[str], optional): List of attention types per layer ("sliding_attention" or "full_attention").
                        If None, defaults to sliding window for 3/4 layers and full attention every 4th layer.
                gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient checkpointing strategy.
                        Defaults to EasyDeLGradientCheckPointers.NONE.
                use_scan_mlp (bool, optional): Whether to use scan for MLP layers. Defaults to False.
                scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
                bits (tp.Optional[int], optional): Quantization bits. Defaults to None.
                rope_parameters (dict, optional): Canonical RoPE settings keyed by attention type.
                        Explicit per-type settings take precedence over legacy shared rope fields.
                        A flat mapping retains the legacy shared local/global RoPE behavior.
                rope_theta_is_shared (bool, optional): Persisted provenance for legacy shared theta.
                        Inferred from the input format when omitted: legacy/flat settings follow late
                        rope_theta overrides, while explicit per-type mappings retain their own theta.
                **kwargs: Additional keyword arguments.
        """
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        # HF 5.13 aliases rope_scaling to rope_parameters. Validate the raw
        # legacy input before any assignment can expand it into a nested map.
        rope_scaling = self._rope_scaling_validation(rope_scaling)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.head_dim = hidden_size // num_attention_heads

        # Set layer types: default pattern is sliding for 3/4 layers, full for every 4th layer
        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types
        self._validate_layer_types()

        # HF OLMo3 reads RoPE by attention type, even when both types share
        # the same parameters. Keep old EasyDeL checkpoints' shared math and
        # theta (10000 by default), rather than substituting HF's newer default.
        shared_rope = rope_scaling or {"rope_type": "default", "rope_theta": rope_theta}
        if rope_parameters is not None and not any(
            key in rope_parameters for key in ("sliding_attention", "full_attention")
        ):
            # Older saved configs could carry a stale flat default payload
            # alongside the actual scaling used by EasyDeL's rotary builders.
            # Only an explicitly supplied legacy argument can override it;
            # self.rope_scaling is the canonical alias, not that argument.
            if rope_scaling is not None:
                rope_parameters = shared_rope
        self.rope_parameters = shared_rope if rope_parameters is None else rope_parameters
        # Do not infer provenance from numerical equality: an explicit nested
        # map can deliberately use the same theta for both attention types.
        self.rope_theta_is_shared = (
            rope_parameters is None or not any(key in rope_parameters for key in ("sliding_attention", "full_attention"))
            if rope_theta_is_shared is None
            else rope_theta_is_shared
        )

    def __setattr__(self, key, value):
        """Propagate late legacy theta overrides without overwriting explicit maps."""
        if key == "rope_parameters" and hasattr(self, "rope_theta_is_shared"):
            # Replacing the canonical map explicitly establishes new provenance.
            # Internal normalization/backfill bypasses this assignment hook.
            self.rope_theta_is_shared = isinstance(value, dict) and not any(
                layer_type in value for layer_type in ("sliding_attention", "full_attention")
            )
        super().__setattr__(key, value)
        if key == "rope_theta" and getattr(self, "rope_theta_is_shared", False):
            parameters = getattr(self, "rope_parameters", None)
            if isinstance(parameters, dict):
                # Use the base setter: this is synchronization of a legacy map,
                # not a new explicit per-type assignment.
                super().__setattr__(
                    "rope_parameters",
                    {
                        layer_type: {**layer_parameters, "rope_theta": value}
                        for layer_type, layer_parameters in parameters.items()
                    },
                )

    def to_diff_dict(self) -> dict:
        """Persist shared-theta provenance even when it equals the class default."""
        result = super().to_diff_dict()
        result["rope_theta_is_shared"] = self.rope_theta_is_shared
        return result

    def _normalize_rope_assignment(self, rope_parameters: dict) -> dict:
        """Normalize per-type mappings independently of the active layer schedule.

        Both entries must survive serialization even for an all-local or
        all-global reduced model. Legacy flat mappings remain flat here because
        ``rope_scaling`` is also normalized through this hook.
        """
        layer_types = {"sliding_attention", "full_attention"}
        if rope_parameters and set(rope_parameters).issubset(layer_types):
            return {
                layer_type: self._normalize_rope_parameters_dict(
                    rope_parameters.get(layer_type) or {},
                    rope_theta=getattr(self, "rope_theta", 10000.0),
                )
                for layer_type in ("sliding_attention", "full_attention")
            }
        return super()._normalize_rope_assignment(rope_parameters)

    def _backfill_rope_parameters(self) -> None:
        """Keep the canonical HF mapping nested after construction and mutation."""
        super()._backfill_rope_parameters()
        parameters = getattr(self, "rope_parameters", None)
        if isinstance(parameters, dict) and "rope_type" in parameters:
            # Bypass our assignment hook to avoid recursively backfilling.
            object.__setattr__(
                self,
                "rope_parameters",
                {layer_type: dict(parameters) for layer_type in ("sliding_attention", "full_attention")},
            )

    def get_layer_rope_config(self, layer_type: str) -> "Olmo3Config":
        """Return a shallow config view for the existing flat RoPE builders.

        The original config and its persisted per-type parameters are untouched.
        """
        config = copy.copy(self)
        parameters = self.rope_parameters[layer_type]
        # This detached builder view must stay flat. Attribute assignment would
        # run OLMo3's canonical backfill, including through HF's rope_scaling
        # property setter. Populate both spellings directly so old independent
        # fields and the new HF alias expose the same selected-layer settings.
        config.__dict__.update(
            rope_theta=parameters["rope_theta"],
            rope_parameters=dict(parameters),
            rope_scaling=dict(parameters),
        )
        return config

    @staticmethod
    def _rope_scaling_validation(rope_scaling: dict | None) -> dict | None:
        """Validate and copy the raw legacy shared RoPE scaling argument.

        Args:
            rope_scaling: Legacy constructor input, not the HF property alias.

        Returns:
            A copied scaling dictionary, or None for default/unscaled RoPE.

        Raises:
            ValueError: If the input is not a dictionary or its scaling type
                or factor is invalid.
        """
        if rope_scaling is None:
            return None

        if not isinstance(rope_scaling, dict):
            raise ValueError(
                f"`rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {rope_scaling}"
            )

        rope_scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        # Base config compatibility can inject a default rope payload; treat it as no scaling.
        if rope_scaling_type in (None, "default"):
            return None

        rope_scaling_factor = rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
        return dict(rope_scaling)

    def _validate_layer_types(self):
        """
        Validates the `layer_types` list to ensure it has correct length and valid values.

        Raises:
                ValueError: If `layer_types` length doesn't match `num_hidden_layers` or contains invalid values.
        """
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` must have length equal to `num_hidden_layers` ({self.num_hidden_layers}), "
                f"got {len(self.layer_types)}"
            )

        valid_types = {"sliding_attention", "full_attention"}
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type not in valid_types:
                raise ValueError(f"`layer_types[{idx}]` must be one of {valid_types}, got '{layer_type}'")

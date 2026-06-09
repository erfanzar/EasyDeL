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


"""Configuration classes for the DBRX model.

DBRX is Databricks' fine-grained Mixture-of-Experts (MoE) decoder transformer.
A typical configuration uses 16 experts per MoE layer with top-4 routing,
multi-query attention (``kv_n_heads=1``), QKV clipping, and standard RoPE.

Exports:

- :class:`DbrxAttentionConfig` — attention sub-config (clip_qkv, kv_n_heads,
  rope_theta, attn_pdrop).
- :class:`DbrxFFNConfig` — MoE FFN sub-config (ffn_hidden_size, num/top_k
  experts, jitter eps, load-balancing loss weight, expert weight
  normalization).
- :class:`DbrxConfig` — top-level config that owns both sub-configs plus the
  decoder shape (``d_model``, ``n_heads``, ``n_layers``, ``max_seq_len``).
"""

import typing as tp
import warnings

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config

DBRX_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DbrxAttentionConfig(EasyDeLBaseConfig):
    """
    This is the configuration class to store the attention related configuration of a [`DbrxModel`].

    Args:
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output.
        clip_qkv (`float`, *optional*, defaults to 8.0):
            The clip value applied to the query, key, and value tensors.
        kv_n_heads (`int`, *optional*, defaults to 1):
            The number of attention heads for the key and value tensors.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value for the rotary position embedding.
    """

    def __init__(
        self,
        attn_pdrop: float = 0,
        clip_qkv: float | None = 8,
        kv_n_heads: int = 1,
        rope_theta: float = 10000.0,
        **kwargs: tp.Any,
    ):
        """Initialize the DBRX attention sub-configuration.

        Args:
            attn_pdrop (float, optional): Dropout probability on attention
                outputs. Defaults to ``0``.
            clip_qkv (float | None, optional): Symmetric clip value applied to
                the Q/K/V projections (``None`` disables). Defaults to ``8``.
            kv_n_heads (int, optional): Number of key/value heads. Set to ``1``
                for multi-query attention; larger values for grouped-query.
                Defaults to ``1``.
            rope_theta (float, optional): Base frequency for rotary position
                embeddings. Defaults to ``10000.0``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`. ``model_type``
                is silently dropped; any other unknown key raises.

        Raises:
            ValueError: If ``kwargs`` contains unknown keys other than
                ``model_type``.
        """
        super().__init__(**kwargs)
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta

        for k in ["model_type"]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: tp.Any) -> "PretrainedConfig":  # type: ignore[misc] # noqa: F821
        """Load a DBRX attention sub-config from a pretrained checkpoint.

        Looks at the resolved config JSON; when ``model_type == "dbrx"`` the
        nested ``attn_config`` dict is unwrapped before construction. Mismatched
        ``model_type`` values trigger a warning but still proceed.

        Args:
            pretrained_model_name_or_path (str): HuggingFace Hub identifier or
                local path containing a ``config.json``.
            **kwargs: Forwarded to :meth:`get_config_dict` and to the final
                :meth:`from_dict` call.

        Returns:
            PretrainedConfig: A populated :class:`DbrxAttentionConfig`.
        """
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "dbrx":
            config_dict = config_dict["attn_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            warnings.warn(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.",
                stacklevel=1,
            )

        return cls.from_dict(config_dict, **kwargs)


class DbrxFFNConfig(EasyDeLBaseConfig):
    """
    This is the configuration class to store the feed forward related configuration of a [`DbrxModel`].

    Args:
        ffn_act_fn (`dict`, *optional*):
            The activation function configuration for the feed-forward network.
        ffn_hidden_size (`int`, *optional*, defaults to 3584):
            The hidden size of the feed-forward network.
        moe_num_experts (`int`, *optional*, defaults to 4):
            The number of experts in the Mixture-of-Experts (MoE) layer.
        moe_top_k (`int`, *optional*, defaults to 1):
            The number of top experts to use in the MoE layer.
        moe_jitter_eps (`float`, *optional*):
            The jitter epsilon value for the MoE layer.
        moe_loss_weight (`float`, *optional*, defaults to 0.01):
            The loss weight for the MoE auxiliary loss.
        moe_normalize_expert_weights (`float`, *optional*, defaults to 1.0):
            The normalization factor for the expert weights in the MoE layer.
        uniform_expert_assignment (`bool`, *optional*, defaults to `False`):
            Whether to use uniform expert assignment in the MoE layer.
    """

    def __init__(
        self,
        ffn_act_fn: dict | None = None,
        ffn_hidden_size: int = 3584,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_jitter_eps: float | None = None,
        moe_loss_weight: float = 0.01,
        moe_normalize_expert_weights: float | None = 1,
        uniform_expert_assignment: bool = False,
        **kwargs: tp.Any,
    ):
        """Initialize the DBRX MoE feed-forward sub-configuration.

        Args:
            ffn_act_fn (dict | None, optional): Activation spec dict (e.g.
                ``{"name": "silu"}``). Defaults to ``{"name": "silu"}``.
            ffn_hidden_size (int, optional): Per-expert FFN intermediate
                width. Defaults to ``3584``.
            moe_num_experts (int, optional): Total number of experts per MoE
                layer. Defaults to ``4``.
            moe_top_k (int, optional): Active experts per token. Defaults to
                ``1``.
            moe_jitter_eps (float | None, optional): Multiplicative jitter on
                router logits during training (``None`` disables). Defaults to
                ``None``.
            moe_loss_weight (float, optional): Weight on the MoE auxiliary
                load-balancing loss. Defaults to ``0.01``.
            moe_normalize_expert_weights (float | None, optional): Lp-norm
                exponent for normalizing expert outputs. Defaults to ``1``.
            uniform_expert_assignment (bool, optional): Force uniform
                expert assignment (debug). Defaults to ``False``.
            **kwargs: Unknown keys (other than ``model_type``) raise
                ``ValueError``.

        Raises:
            ValueError: If ``kwargs`` contains unknown keys other than
                ``model_type``.
        """
        super().__init__()
        if ffn_act_fn is None:
            ffn_act_fn = {"name": "silu"}
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        for k in ["model_type"]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: tp.Any) -> "EasyDeLBaseConfig":
        """Load a DBRX FFN sub-config from a pretrained checkpoint.

        When the resolved JSON's ``model_type == "dbrx"``, the nested
        ``ffn_config`` dict is unwrapped before construction.

        Args:
            pretrained_model_name_or_path (str): HuggingFace Hub identifier or
                local path with a ``config.json``.
            **kwargs: Forwarded to :meth:`get_config_dict` and the final
                :meth:`from_dict` call.

        Returns:
            EasyDeLBaseConfig: A populated :class:`DbrxFFNConfig`.
        """
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "dbrx":
            config_dict = config_dict["ffn_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            warnings.warn(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.",
                stacklevel=1,
            )

        return cls.from_dict(config_dict, **kwargs)


@register_config("dbrx")
class DbrxConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        d_model (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        n_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        max_seq_len (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the DBRX model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_config ([`DbrxAttentionConfig`], *optional*):
            The configuration of the attention layer.
        ffn_config ([`DbrxFFNConfig`], *optional*):
            The configuration of the feed forward layer.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to output the router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.05):
            The coefficient of the router auxiliary loss.
    """

    model_type: str = "dbrx"
    attribute_map: tp.ClassVar = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "max_position_embeddings": "max_seq_len",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        max_seq_len: int = 2048,
        vocab_size: int = 32000,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_config: DbrxAttentionConfig | None = None,
        ffn_config: DbrxFFNConfig | None = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.05,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        layer_types: list[str] | None = None,
        **kwargs: tp.Any,
    ):
        """Initialize the DbrxConfig with the specified parameters.

        Args:
                d_model (int, optional): Hidden size for the transformer. Defaults to 2048.
                n_heads (int, optional): Number of attention heads. Defaults to 16.
                n_layers (int, optional): Number of transformer layers. Defaults to 24.
                max_seq_len (int, optional): Maximum sequence length. Defaults to 2048.
                vocab_size (int, optional): Vocabulary size. Defaults to 32000.
                resid_pdrop (float, optional): Residual dropout probability. Defaults to 0.0.
                emb_pdrop (float, optional): Embedding dropout probability. Defaults to 0.0.
                attn_config (Optional[DbrxAttentionConfig], optional): Configuration for attention. Defaults to None.
                ffn_config (Optional[DbrxFFNConfig], optional): Configuration for feed-forward network. Defaults to None.
                use_cache (bool, optional): Whether to use KV cache for decoding. Defaults to True.
                initializer_range (float, optional): Range for weight initialization. Defaults to 0.02.
                output_router_logits (bool, optional): Whether to output router logits. Defaults to False.
                router_aux_loss_coef (float, optional): Coefficient for router auxiliary loss. Defaults to 0.05.
                gradient_checkpointing (EasyDeLGradientCheckPointers, optional):
                    Gradient checkpointing strategy. Defaults to EasyDeLGradientCheckPointers.NONE.
                **kwargs (Any): Additional arguments.
        """
        if attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        if ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**ffn_config)
        else:
            self.ffn_config = ffn_config

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.n_layers
        if getattr(self.ffn_config, "ffn_hidden_size", None) != d_model:
            self.ffn_config.ffn_hidden_size = d_model
        if not hasattr(self.ffn_config, "hidden_size"):
            self.ffn_config.hidden_size = d_model
        self.rope_theta = getattr(self.attn_config, "rope_theta", 10000.0)
        self.rope_parameters = {"rope_type": "default", "rope_theta": self.rope_theta}

        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        if tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported for Dbrx models.")

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def num_key_value_heads(self) -> int:
        """Get the number of key-value heads from the attention config."""
        return self.attn_config.kv_n_heads

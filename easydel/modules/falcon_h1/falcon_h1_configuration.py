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

"""Configuration for the FalconH1 hybrid attention/SSM model.

FalconH1 is TII's hybrid decoder where each layer runs a Mamba2-style
selective state-space mixer (SSM) and a RoPE-based grouped-query attention
mixer in parallel before the SwiGLU MLP. The HF parameter names are kept
verbatim so reference checkpoints round-trip cleanly.
"""

from __future__ import annotations

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.factory import register_config


@register_config("falcon_h1")
class FalconH1Config(EasyDeLBaseConfig):
    """Configuration for the FalconH1 architecture.

    FalconH1 is a decoder-only hybrid model that combines:
    - Mamba2-style selective state-space mixing (SSM)
    - RoPE-based grouped-query attention (GQA)
    - SwiGLU MLP blocks

    The config keeps the HuggingFace field names so that:
    - HuggingFace reference models can be instantiated from this config in tests.
    - EasyDeL state-dict conversion finds matching parameter paths.
    """

    model_type: str = "falcon_h1"

    def __init__(
        self,
        vocab_size: int = 128_000,
        tie_word_embeddings: bool = False,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        num_logits_to_keep: int | None = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        head_dim: int | None = None,
        # Mamba2 parameters (HF uses the `mamba_*` namespace)
        mamba_d_ssm: int | None = 1024,
        mamba_n_heads: int = 128,
        mamba_d_head: int | str = "auto",
        mamba_n_groups: int = 1,
        mamba_d_state: int = 256,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_chunk_size: int = 256,
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        mamba_norm_before_gate: bool = True,
        mamba_rms_norm: bool = False,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_limit: tuple[float, float] | None = (0.0, float("inf")),
        # Shared projection biases
        projectors_bias: bool = False,
        # RoPE
        rope_parameters: dict | None = None,
        rope_theta: float = 100_000.0,
        rope_scaling: dict[str, str | float] | None = None,
        # MuP multipliers and scaling knobs
        lm_head_multiplier: float = 1.0,
        embedding_multiplier: float = 1.0,
        mlp_multipliers: list[float] | None = None,
        key_multiplier: float | None = None,
        attention_out_multiplier: float | None = None,
        attention_in_multiplier: float | None = None,
        ssm_multipliers: list[float] | None = None,
        ssm_in_multiplier: float | None = None,
        ssm_out_multiplier: float | None = None,
        **kwargs,
    ):
        """Initialize a :class:`FalconH1Config`.

        Args:
            vocab_size (int, optional): Token vocabulary size. Defaults to ``128_000``.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to ``False``.
            hidden_size (int, optional): Decoder hidden dimension. Defaults to ``4096``.
            intermediate_size (int, optional): SwiGLU MLP intermediate width.
                Defaults to ``14336``.
            num_hidden_layers (int, optional): Number of hybrid decoder layers.
                Defaults to ``32``.
            num_attention_heads (int, optional): GQA query heads per layer.
                Defaults to ``32``.
            num_key_value_heads (int | None, optional): GQA KV heads. ``None`` falls
                back to ``num_attention_heads``. Defaults to ``8``.
            hidden_act (str, optional): MLP activation. Defaults to ``"silu"``.
            initializer_range (float, optional): Truncated-normal init stddev.
                Defaults to ``0.02``.
            rms_norm_eps (float, optional): RMSNorm epsilon. Defaults to ``1e-5``.
            use_cache (bool, optional): Return KV/SSM caches. Defaults to ``True``.
            num_logits_to_keep (int | None, optional): Slice the LM head to the last
                ``N`` positions. Defaults to ``1``.
            pad_token_id (int, optional): Padding token id. Defaults to ``0``.
            bos_token_id (int, optional): Beginning-of-sequence id. Defaults to ``1``.
            eos_token_id (int, optional): End-of-sequence id. Defaults to ``2``.
            max_position_embeddings (int, optional): Maximum sequence length.
                Defaults to ``8192``.
            attention_dropout (float, optional): Attention probability dropout.
                Defaults to ``0.0``.
            head_dim (int | None, optional): Per-head attention dimension. ``None``
                derives ``hidden_size // num_attention_heads``.
            mamba_d_ssm (int | None, optional): Inner SSM hidden width. ``None``
                derives ``mamba_expand * hidden_size``. Defaults to ``1024``.
            mamba_n_heads (int, optional): Number of SSM heads. Defaults to ``128``.
            mamba_d_head (int | str, optional): Per-head SSM dimension; ``"auto"``
                splits ``mamba_d_ssm`` evenly across heads. Defaults to ``"auto"``.
            mamba_n_groups (int, optional): SSM group count for B/C projections.
                Defaults to ``1``.
            mamba_d_state (int, optional): SSM state dimension. Defaults to ``256``.
            mamba_d_conv (int, optional): Causal conv kernel width. Defaults to ``4``.
            mamba_expand (int, optional): SSM expansion factor over ``hidden_size``.
                Defaults to ``2``.
            mamba_chunk_size (int, optional): SSM scan chunk size. Defaults to ``256``.
            mamba_conv_bias (bool, optional): Bias on the SSM causal conv.
                Defaults to ``True``.
            mamba_proj_bias (bool, optional): Bias on the SSM in/out projections.
                Defaults to ``False``.
            mamba_norm_before_gate (bool, optional): Apply norm before the SSM gate.
                Defaults to ``True``.
            mamba_rms_norm (bool, optional): Use RMSNorm inside the SSM mixer.
                Defaults to ``False``.
            time_step_min (float, optional): Minimum SSM time-step. Defaults to ``0.001``.
            time_step_max (float, optional): Maximum SSM time-step. Defaults to ``0.1``.
            time_step_limit (tuple[float, float] | None, optional): Lower/upper clamp
                on the discretized time-step. Defaults to ``(0.0, +inf)``.
            projectors_bias (bool, optional): Bias on the shared in/out projection
                bundle. Defaults to ``False``.
            rope_parameters (dict | None, optional): Extra RoPE parameters dict
                forwarded to the rotary builder.
            rope_theta (float, optional): RoPE base frequency. Defaults to ``100000.0``.
            rope_scaling (dict | None, optional): RoPE scaling spec. Defaults to ``None``.
            lm_head_multiplier (float, optional): MuP-style multiplier on the LM head
                logits. Defaults to ``1.0``.
            embedding_multiplier (float, optional): MuP multiplier on embeddings.
                Defaults to ``1.0``.
            mlp_multipliers (list[float] | None, optional): Per-projection MuP
                multipliers for the MLP block. ``None`` defaults to ``[1.0, 1.0]``.
            key_multiplier (float | None, optional): MuP multiplier on the key
                projection. ``None`` -> ``1.0``.
            attention_out_multiplier (float | None, optional): MuP multiplier on the
                attention output projection. ``None`` -> ``1.0``.
            attention_in_multiplier (float | None, optional): MuP multiplier on the
                attention input projections. ``None`` -> ``1.0``.
            ssm_multipliers (list[float] | None, optional): MuP multipliers for the
                SSM in/out/gate projections. ``None`` -> ``[1.0]*5``.
            ssm_in_multiplier (float | None, optional): MuP multiplier on the SSM
                input projection. ``None`` -> ``1.0``.
            ssm_out_multiplier (float | None, optional): MuP multiplier on the SSM
                output projection. ``None`` -> ``1.0``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.

        Raises:
            ValueError: If ``mamba_n_heads`` does not divide
                ``mamba_expand * hidden_size`` (or ``mamba_d_ssm`` when set), or if
                ``mamba_d_head * mamba_n_heads`` mismatches the SSM intermediate size.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        # HF compatibility fields
        self.attention_bias = False
        self.mlp_bias = False

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.projectors_bias = projectors_bias

        mamba_intermediate = mamba_expand * hidden_size if mamba_d_ssm is None else mamba_d_ssm
        if mamba_intermediate % mamba_n_heads != 0:
            raise ValueError("`mamba_n_heads` must divide `mamba_expand * hidden_size` (or `mamba_d_ssm`).")

        if mamba_d_head == "auto":
            mamba_d_head = mamba_intermediate // mamba_n_heads
        if int(mamba_d_head) * mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate size.")

        self.mamba_d_ssm = mamba_d_ssm
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = int(mamba_d_head)
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_norm_before_gate = mamba_norm_before_gate
        self.mamba_rms_norm = mamba_rms_norm
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_limit = tuple(time_step_limit) if time_step_limit is not None else None

        self.lm_head_multiplier = lm_head_multiplier
        self.embedding_multiplier = embedding_multiplier

        self.rope_parameters = rope_parameters
        self.mlp_multipliers = mlp_multipliers if mlp_multipliers is not None else [1.0, 1.0]
        self.attention_out_multiplier = attention_out_multiplier if attention_out_multiplier is not None else 1.0
        self.attention_in_multiplier = attention_in_multiplier if attention_in_multiplier is not None else 1.0
        self.key_multiplier = key_multiplier if key_multiplier is not None else 1.0

        self.ssm_multipliers = ssm_multipliers if ssm_multipliers is not None else [1.0, 1.0, 1.0, 1.0, 1.0]
        self.ssm_in_multiplier = ssm_in_multiplier if ssm_in_multiplier is not None else 1.0
        self.ssm_out_multiplier = ssm_out_multiplier if ssm_out_multiplier is not None else 1.0
        self.head_dim = self.hidden_size // self.num_attention_heads if head_dim is None else head_dim
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        """HF-compatible per-layer block-type list used by cache helpers.

        FalconH1's hybrid layers run attention and SSM in parallel, but
        HF's cache helpers key on ``"attention"`` for KV-cache shape inference,
        so this list is uniformly ``"attention"``.

        Returns:
            list[str]: ``["attention"] * num_hidden_layers``.
        """
        return ["attention" for _ in range(self.num_hidden_layers)]

    @property
    def layer_types(self) -> tuple[str, ...]:
        """Return layer types for hybrid cache configuration.

        FalconH1 uses parallel hybrid layers where both attention and SSM
        run in parallel within each layer. Use the HF-compatible public label
        ``"hybrid"`` here and normalize it to EasyDeL's internal
        ``"parallel_hybrid"`` cache type where needed.
        """
        return tuple("hybrid" for _ in range(self.num_hidden_layers))

    @property
    def mamba_intermediate_size(self) -> int:
        """Compute the SSM mixer's intermediate width.

        Returns:
            int: ``self.mamba_d_ssm`` if explicitly set, otherwise
            ``self.mamba_expand * self.hidden_size``.
        """
        if self.mamba_d_ssm is not None:
            return self.mamba_d_ssm
        return self.mamba_expand * self.hidden_size

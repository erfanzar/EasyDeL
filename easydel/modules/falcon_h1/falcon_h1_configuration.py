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

from __future__ import annotations

from eformer.common_types import ColumnWise, Replicated, RowWise

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
        # Shared projection biases
        projectors_bias: bool = False,
        # RoPE
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

        self.lm_head_multiplier = lm_head_multiplier
        self.embedding_multiplier = embedding_multiplier

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
        # HF uses this for cache helpers; keep for compatibility.
        return ["attention" for _ in range(self.num_hidden_layers)]

    @property
    def layer_types(self) -> tuple[str, ...]:
        """Return layer types for hybrid cache configuration.

        FalconH1 uses parallel hybrid layers where both attention and SSM
        run in parallel within each layer, so all layers are 'parallel_hybrid'.
        """
        return tuple("parallel_hybrid" for _ in range(self.num_hidden_layers))

    @property
    def mamba_intermediate_size(self) -> int:
        """Return the intermediate size for Mamba SSM."""
        if self.mamba_d_ssm is not None:
            return self.mamba_d_ssm
        return self.mamba_expand * self.hidden_size

    def get_partition_rules(self, *args, **kwargs):
        """Return regex-based parameter partition rules.

        The rules follow the standard EasyDeL linear sharding convention:
        - Column-wise sharding for expanding projections.
        - Row-wise sharding for contracting projections.
        - Biases, norms and non-matmul parameters replicated.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"self_attn/(q_norm|k_norm)/kernel", pmag.resolve(Replicated)),
            (r"feed_forward/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"feed_forward/down_proj/kernel", pmag.resolve(RowWise)),
            (r"feed_forward/.*proj/bias", pmag.resolve(Replicated)),
            (r"mamba/in_proj/kernel", pmag.resolve(ColumnWise)),
            (r"mamba/out_proj/kernel", pmag.resolve(RowWise)),
            (r"mamba/.*proj/bias", pmag.resolve(Replicated)),
            (r"mamba/(dt_bias|A_log|D)", pmag.resolve(Replicated)),
            (r"mamba/conv1d/(kernel|bias)", pmag.resolve(Replicated)),
            (r"mamba/norm/kernel", pmag.resolve(Replicated)),
            (r".*(input_layernorm|pre_ff_layernorm|final_layernorm)/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

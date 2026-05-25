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

"""Configuration for the GIDD (Generalised Interpolating Discrete Diffusion) model.

Defines :class:`GiddConfig`. GIDD is a non-autoregressive masked-diffusion
LLM: it shares the standard transformer backbone (RMSNorm, SwiGLU MLP,
RoPE, optional GQA) but expects bidirectional attention and is trained to
predict masked tokens. Notable knobs that depart from the usual decoder
recipe:

- ``resid_scale`` rescales the residual stream to match GIDD's recipe.
- ``init_scale``, ``emb_init_scale``, ``head_init_scale`` give per-stage
  weight-init scales used by the original implementation.
- ``use_qk_norm`` / ``qk_norm_eps`` toggle RMSNorm on Q/K for stability.
"""

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("gidd")
class GiddConfig(EasyDeLBaseConfig):
    """Configuration for the GIDD masked-diffusion decoder.

    Inherits from :class:`EasyDeLBaseConfig`. GIDD shares the standard
    decoder transformer skeleton (RMSNorm, gated/squared-ReLU MLP, RoPE,
    GQA-capable attention) but is trained as a non-autoregressive masked
    diffusion model. The configuration therefore exposes a handful of
    GIDD-specific knobs on top of the usual transformer fields:

    - ``resid_scale`` rescales the residual stream contribution at every
      sub-block, which the GIDD recipe relies on to keep activation
      variance under control as the diffusion chain refines noisy tokens.
    - ``use_qk_norm`` / ``qk_norm_eps`` toggle the Primer-style RMSNorm on
      the Q/K projections inside attention.
    - ``init_scale``, ``emb_init_scale``, and ``head_init_scale`` control
      the per-stage weight initialisation scales (block, embedding, and
      output head) used by the original GIDD implementation.

    Attributes:
        vocab_size (int): Token vocabulary size.
        hidden_size (int): Hidden/residual stream dimension.
        intermediate_size (int): MLP inner width.
        num_hidden_layers (int): Number of transformer blocks.
        num_attention_heads (int): Attention heads per layer.
        head_dim (int): Per-head dimension; derived from ``hidden_size /
            num_attention_heads`` when not set explicitly.
        max_position_embeddings (int): Maximum supported sequence length.
        resid_scale (float): Residual contribution rescale (default ``4.0``).
        rms_norm_eps (float): RMSNorm epsilon.
        use_qk_norm (bool): Whether attention applies RMSNorm to Q/K.
        qk_norm_eps (float): Epsilon for the QK norm.
        init_scale (float): Weight init scale for transformer blocks.
        emb_init_scale (float): Embedding init scale.
        head_init_scale (float): Output head init scale.
        rope_theta (float): RoPE base frequency.
        rope_scaling (dict | None): Optional RoPE scaling configuration.
        attention_bias (bool): Whether attention projections use bias.
        mlp_bias (bool): Whether MLP projections use bias.
        tie_word_embeddings (bool): Tie input embeddings with the LM head.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient
            checkpointing policy.
        scan_mlp_chunk_size (int): Chunk size used by the scan-MLP path.
        scan_layers (bool): Use ``lax.scan`` over decoder layers.
        bits (int | None): Optional quantization bit-width.
        layer_types (list[str]): Per-layer attention type list; defaults to
            ``["full_attention"] * num_hidden_layers``.
    """

    model_type: str = "gidd"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        head_dim: int | None = None,
        max_position_embeddings: int = 1024,
        resid_scale: float = 4.0,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        init_scale: float = 0.4,
        emb_init_scale: float = 0.1,
        head_init_scale: float = 0.0,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        rope_scaling: dict[str, str | float] | None = None,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        pretraining_tp: int = 1,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        scan_layers: bool = False,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize a :class:`GiddConfig`.

        Args:
            vocab_size (int, optional): Token vocabulary size. Defaults to ``131072``.
            hidden_size (int, optional): Hidden dimension. Defaults to ``768``.
            intermediate_size (int, optional): MLP intermediate width. Defaults to ``3072``.
            num_hidden_layers (int, optional): Number of transformer layers.
                Defaults to ``12``.
            num_attention_heads (int, optional): Attention heads per layer. Defaults to ``12``.
            head_dim (int | None, optional): Per-head dimension. ``None`` derives
                ``hidden_size // num_attention_heads``.
            max_position_embeddings (int, optional): Maximum sequence length.
                Defaults to ``1024``.
            resid_scale (float, optional): Residual stream rescaling factor used to
                stabilise GIDD's deeper diffusion training. Defaults to ``4.0``.
            rms_norm_eps (float, optional): RMSNorm epsilon. Defaults to ``1e-6``.
            use_qk_norm (bool, optional): Apply RMSNorm to Q/K projections.
                Defaults to ``True``.
            qk_norm_eps (float, optional): Epsilon for the Q/K RMSNorm. Defaults to ``1e-6``.
            init_scale (float, optional): Per-block weight init scale. Defaults to ``0.4``.
            emb_init_scale (float, optional): Embedding init scale. Defaults to ``0.1``.
            head_init_scale (float, optional): Output head init scale. Defaults to ``0.0``.
            bos_token_id (int, optional): Beginning-of-sequence id. Defaults to ``0``.
            eos_token_id (int, optional): End-of-sequence id. Defaults to ``1``.
            rope_theta (float, optional): RoPE base frequency. Defaults to ``10000.0``.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to ``False``.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Checkpointing
                policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            rope_scaling (dict | None, optional): RoPE scaling spec. Defaults to ``None``.
            scan_mlp_chunk_size (int, optional): Chunk size for scan-MLP.
                Defaults to ``1024``.
            bits (int | None, optional): Quantization bit-width. Defaults to ``None``.
            pretraining_tp (int, optional): Tensor-parallel degree used during
                pretraining (kept for HF compatibility). Defaults to ``1``.
            attention_bias (bool, optional): Use bias on attention projections.
                Defaults to ``False``.
            mlp_bias (bool, optional): Use bias on MLP projections. Defaults to ``False``.
            scan_layers (bool, optional): Use ``lax.scan`` over decoder layers.
                Defaults to ``False``.
            layer_types (list[str] | None, optional): Per-layer attention types.
                ``None`` fills with ``"full_attention"``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`.
        """
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.init_scale = init_scale
        self.emb_init_scale = emb_init_scale
        self.head_init_scale = head_init_scale
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.resid_scale = resid_scale
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rope_scaling = rope_scaling
        self.bits = bits
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        super().__init__(
            bos_token_id=bos_token_id,
            scan_layers=scan_layers,
            eos_token_id=eos_token_id,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def attach_custom_arguments(
        self,
        tie_word_embeddings: bool = False,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        scan_layers: bool = True,
        **kwargs,
    ):
        """Mutate the config in place with a small set of training-time toggles.

        Convenience hook that overwrites a handful of behavioural flags after
        the config has been constructed; useful when restoring a checkpoint
        and adjusting its quantization / checkpointing / RoPE settings without
        rebuilding the whole config.

        Args:
            tie_word_embeddings (bool, optional): Tie input embeddings with the
                LM head. Defaults to ``False``.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional):
                Checkpointing policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            bits (int | None, optional): Quantization bit-width. Defaults to ``None``.
            rope_theta (float, optional): RoPE base frequency. Defaults to ``10000.0``.
            attention_bias (bool, optional): Use bias on attention projections.
                Defaults to ``False``.
            mlp_bias (bool, optional): Use bias on MLP projections. Defaults to ``False``.
            scan_layers (bool, optional): Use ``lax.scan`` over decoder layers.
                Defaults to ``True``.
            **kwargs: Accepted and ignored for forward compatibility.
        """
        self.scan_layers = scan_layers
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits

    @staticmethod
    def get_weight_decay_exclusions():
        """Return the parameter-name patterns excluded from weight decay.

        Returns:
            tuple: Empty tuple (no exclusions for GIDD).
        """
        return tuple()

    @staticmethod
    def rng_keys():
        """Return the RNG stream key required by GIDD modules.

        Returns:
            str: The single RNG key name ``"parameters"``.
        """
        return "parameters"

    @property
    def granted_freq_max_position_embedding(self) -> int:
        """Return the maximum position used to precompute RoPE frequencies.

        Returns:
            int: ``freq_max_position_embeddings`` if set, otherwise
            ``max_position_embeddings``.
        """
        return getattr(
            self,
            "freq_max_position_embeddings",
            self.max_position_embeddings,
        )

    @property
    def granted_mask_max_position_embedding(self) -> int:
        """Return the maximum position used to construct attention masks.

        Returns:
            int: ``mask_max_position_embeddings`` if set, otherwise
            ``max_position_embeddings``.
        """
        return getattr(
            self,
            "mask_max_position_embeddings",
            self.max_position_embeddings,
        )

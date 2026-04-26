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

"""Configuration classes for the Gemma4 model family.

This module defines the three configuration classes that parameterise every
component of the Gemma4 architecture:

- ``Gemma4TextConfig``  — text decoder (attention, MLP, MoE, per-layer
  embeddings, KV sharing, RoPE, …).
- ``Gemma4VisionConfig`` — vision encoder (patch embedding, spatial pooling,
  2-D RoPE, clipped linears, …).
- ``Gemma4Config`` — top-level multimodal wrapper that bundles a text config,
  an optional vision config, and special-token IDs for image/video merging.

All three classes inherit from ``EasyDeLBaseConfig`` and are registered with
the EasyDeL factory so they can be instantiated via
``AutoConfig.from_pretrained``.
"""

import typing
from typing import Literal

from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("gemma4_text")
class Gemma4TextConfig(EasyDeLBaseConfig):
    """Configuration for the Gemma4 text decoder.

    Controls every architectural knob of the Gemma4 decoder stack including
    mixed sliding/global attention, per-layer input embeddings, KV sharing,
    Mixture-of-Experts, and per-layer-type RoPE parameters.

    Attention Layout:
        Gemma4 alternates between local sliding-window attention and global
        full attention with a configurable ratio (default 5:1).  The
        ``layer_types`` list is auto-generated from a ``sliding_window_pattern``
        of 6 — meaning every 6th layer is ``"full_attention"`` and all others
        are ``"sliding_attention"``.  The last layer is always forced to
        ``"full_attention"`` regardless of the pattern.

        Global layers may use a different head dimension (``global_head_dim``)
        and number of KV heads (``num_global_key_value_heads``) than sliding
        layers, and each layer type has its own RoPE parameterisation stored
        in ``rope_parameters``.

    Per-Layer Input Embeddings:
        When ``hidden_size_per_layer_input > 0``, each decoder layer receives
        an additional residual signal.  A separate embedding table of size
        ``vocab_size_per_layer_input × (num_hidden_layers × hidden_size_per_layer_input)``
        is looked up per token and sliced into per-layer vectors.  These are
        combined with a learned projection of the main embeddings and added
        as a residual inside each decoder layer.

    KV Sharing:
        The last ``num_kv_shared_layers`` layers reuse key-value projections
        from the most recent non-shared layer of the same attention type
        (sliding or global).  This reduces memory and parameter count while
        preserving per-layer query diversity.

    Mixture of Experts (MoE):
        When ``enable_moe_block=True``, every decoder layer adds a sparse
        MoE feed-forward block that runs in parallel with the dense MLP.
        The router selects ``top_k_experts`` out of ``num_experts`` for each
        token, and expert outputs are aggregated with normalised, per-expert-
        scaled routing weights.

    Key-Equals-Value (k_eq_v):
        When ``attention_k_eq_v=True``, global attention layers share the key
        projection as the value input (no separate ``v_proj``), relying on
        separate K-norm and V-norm to differentiate the two representations.

    Double-Wide MLP:
        When ``use_double_wide_mlp=True``, layers within the KV-sharing region
        double their MLP intermediate size to compensate for the reduced
        attention capacity.

    Args:
        vocab_size: Size of the token vocabulary. Defaults to 262 144.
        hidden_size: Dimension of the hidden representations in the decoder.
            Defaults to 2304.
        intermediate_size: Inner dimension of the gated MLP feed-forward
            network.  Defaults to 9216.
        num_hidden_layers: Total number of decoder layers.  Defaults to 30.
        num_attention_heads: Number of query attention heads per layer.
            Defaults to 8.
        num_key_value_heads: Number of key-value heads for sliding-window
            layers (Grouped Query Attention).  Defaults to 4.
        head_dim: Per-head dimension for sliding-window attention layers.
            Defaults to 256.
        hidden_activation: Activation function name used in the gated MLP and
            per-layer input gate.  Defaults to ``"gelu_pytorch_tanh"``.
        max_position_embeddings: Maximum sequence length the model can process.
            Defaults to 131 072.
        initializer_range: Standard deviation for weight initialisation.
            Defaults to 0.02.
        rms_norm_eps: Epsilon for all RMSNorm layers.  Defaults to 1e-6.
        use_cache: Whether to return KV cache from the forward pass.
            Defaults to ``True``.
        pad_token_id: Padding token index.  Defaults to 0.
        eos_token_id: End-of-sequence token index.  Defaults to 1.
        bos_token_id: Beginning-of-sequence token index.  Defaults to 2.
        tie_word_embeddings: Whether to tie the input and output embedding
            matrices.  Defaults to ``True``.
        attention_bias: Whether to include bias terms in Q/K/V/O projections.
            Defaults to ``False``.
        attention_dropout: Dropout probability applied to attention weights.
            Defaults to 0.0.
        sliding_window: Size of the causal sliding window for local attention
            layers.  When ``use_bidirectional_attention="all"``, this value is
            automatically halved (``(w // 2) + 1``).  Defaults to 512.
        layer_types: Explicit per-layer attention type list.  If ``None``
            (default), auto-generated with a 5:1 sliding/global ratio and the
            last layer forced to ``"full_attention"``.
        final_logit_softcapping: If set, logits are bounded to
            ``[-cap, cap]`` via ``cap * tanh(logits / cap)``.  ``None``
            disables capping.
        use_bidirectional_attention: Controls bidirectional masking.
            ``"vision"`` enables bidirectional attention for vision token
            blocks only.  ``"all"`` makes all tokens bidirectional and halves
            the sliding window.  ``None`` (default) uses standard causal
            masking.
        vocab_size_per_layer_input: Vocabulary size for the per-layer input
            embedding table.  Only used when
            ``hidden_size_per_layer_input > 0``.  Defaults to 262 144.
        hidden_size_per_layer_input: Dimension of each per-layer input
            embedding vector.  Set to 0 to disable per-layer inputs.
            Defaults to 256.
        num_global_key_value_heads: Number of KV heads for global (full)
            attention layers when ``attention_k_eq_v=True``.  If ``None``,
            falls back to ``num_key_value_heads``.
        global_head_dim: Per-head dimension for global attention layers.
            Can differ from ``head_dim`` to give global layers higher
            resolution.  Defaults to 512.
        attention_k_eq_v: When ``True``, global layers reuse the key
            projection as the value input instead of maintaining a separate
            ``v_proj``.  Defaults to ``False``.
        num_kv_shared_layers: Number of trailing decoder layers that share
            KV projections with the last non-shared layer of the same
            attention type.  0 disables sharing.  Defaults to 0.
        enable_moe_block: Whether to add a sparse MoE block in parallel with
            the dense MLP at each layer.  Defaults to ``False``.
        use_double_wide_mlp: Whether layers in the KV-sharing region double
            their MLP intermediate size.  Defaults to ``False``.
        num_experts: Total number of experts in each MoE block.  Only used
            when ``enable_moe_block=True``.
        top_k_experts: Number of experts activated per token.  Only used
            when ``enable_moe_block=True``.
        moe_intermediate_size: Inner dimension of each expert's feed-forward
            network.  Only used when ``enable_moe_block=True``.
        activations_in_float32: Whether to upcast dense and expert MLP
            pre-activation tensors to ``float32`` before applying the
            nonlinearity, for optional stability path.
            Defaults to ``True``.
        float32_gate_logits: Whether to run the MoE router pre-projection
            normalization/scaling path and gate projection in ``float32``.
            Only used when ``enable_moe_block=True``. Defaults to ``False``.
        rope_parameters: Per-layer-type RoPE configuration dictionary mapping
            ``"sliding_attention"`` and ``"full_attention"`` to their
            respective ``rope_type``, ``rope_theta``, and optional
            ``partial_rotary_factor``.  If ``None``, uses the Gemma4 defaults:
            sliding with theta=10 000 and global with proportional RoPE at
            theta=1 000 000 and ``partial_rotary_factor=0.25``.
        gradient_checkpointing: Gradient checkpointing strategy.  Defaults
            to ``NONE``.
        bits: Quantisation bit-width.  ``None`` disables quantisation.
        scan_layers: Whether to use ``jax.lax.scan`` over decoder layers.
            Defaults to ``False``.
    """

    model_type: str = "gemma4_text"

    def __init__(
        self,
        vocab_size: int = 262_144,
        hidden_size: int = 2304,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = None,
        use_bidirectional_attention: Literal["all", "vision"] | None = None,
        vocab_size_per_layer_input: int = 262_144,
        hidden_size_per_layer_input: int = 256,
        num_global_key_value_heads: int | None = None,
        global_head_dim: int = 512,
        attention_k_eq_v: bool = False,
        num_kv_shared_layers: int = 0,
        enable_moe_block: bool = False,
        use_double_wide_mlp: bool = False,
        num_experts: int | None = None,
        top_k_experts: int | None = None,
        moe_intermediate_size: int | None = None,
        activations_in_float32: bool = True,
        float32_gate_logits: bool = False,
        rope_parameters: dict | None = None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        scan_layers: bool = False,
        **kwargs,
    ):
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits

        super().__init__(
            bos_token_id=bos_token_id,
            scan_layers=scan_layers,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_global_key_value_heads = num_global_key_value_heads
        self.global_head_dim = global_head_dim
        self.attention_k_eq_v = attention_k_eq_v
        self.num_kv_shared_layers = num_kv_shared_layers
        self.enable_moe_block = enable_moe_block
        self.use_double_wide_mlp = use_double_wide_mlp
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.activations_in_float32 = activations_in_float32
        self.float32_gate_logits = float32_gate_logits

        if use_bidirectional_attention == "all":
            self.sliding_window = (self.sliding_window // 2) + 1

        if layer_types is None:
            sliding_window_pattern = 6
            layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if layer_types and layer_types[-1] != "full_attention":
            layer_types[-1] = "full_attention"
        self.layer_types = layer_types

        default_rope_params: dict[str, dict[str, typing.Any]] = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        }
        if rope_parameters is None:
            rope_parameters = default_rope_params
        self.rope_parameters = rope_parameters

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Return tensor-parallelism partition rules.

        Returns ``None`` to use the default partitioning strategy provided by
        the ``PartitionManager``.
        """
        return None

    def get_kv_shared_layer_mapping(self) -> dict[int, int]:
        """Return a mapping from KV-shared layer indices to their donor indices.

        Layers beyond the ``num_kv_shared_layers`` threshold reuse K/V from
        the last non-shared layer of the same attention type (sliding or full).

        Returns:
            Dict mapping ``shared_layer_idx -> donor_layer_idx``.  Empty when
            ``num_kv_shared_layers`` is 0 or ``layer_types`` is not set.
        """
        num_kv_shared = getattr(self, "num_kv_shared_layers", 0) or 0
        if num_kv_shared <= 0 or not self.layer_types:
            return {}

        first_kv_shared = self.num_hidden_layers - num_kv_shared
        if first_kv_shared <= 0:
            return {}

        prev_layers = self.layer_types[:first_kv_shared]
        mapping: dict[int, int] = {}
        for layer_idx in range(first_kv_shared, self.num_hidden_layers):
            layer_type = self.layer_types[layer_idx]
            if layer_type in prev_layers:
                donor = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
                mapping[layer_idx] = donor
        return mapping

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Return per-layer attention mask metadata for eSurge and cache setup.

        Maps each layer index to an ``AttnMaskDetail`` specifying whether the
        layer uses sliding-window or full causal attention, along with the
        window size.  This information is consumed by the inference engine to
        construct the correct KV cache layout (``SlidingWindowSpec`` vs
        ``FullAttentionSpec``) and attention masks.

        Returns:
            Dictionary mapping layer indices to their ``AttnMaskDetail``.
        """
        mapping = {}
        if self.layer_types is not None:
            for layer_idx in range(self.num_hidden_layers):
                mapping[layer_idx] = AttnMaskDetail(
                    mask_type=AttnMaskType.from_hf(self.layer_types[layer_idx]),
                    size=self.sliding_window,
                )
        return mapping


@register_config("gemma4_vision")
class Gemma4VisionConfig(EasyDeLBaseConfig):
    """Configuration for the Gemma4 vision encoder.

    Parameterises a ViT-style vision transformer with 2-D RoPE, spatial
    pooling after patch embedding, and optional weight-clipped linear layers
    for numerical stability.

    The vision encoder processes images by splitting them into non-overlapping
    patches of size ``patch_size × patch_size``, embedding each patch via a
    linear projection with learned 2-D positional embeddings, and passing the
    sequence of patch tokens through ``num_hidden_layers`` transformer encoder
    layers.  After encoding, a spatial pooling step (kernel size
    ``pooling_kernel_size``) reduces the spatial resolution before the features
    are projected into the language model's embedding space by the multimodal
    embedder.

    Args:
        hidden_size: Dimension of the patch embeddings and encoder hidden
            states.  Defaults to 768.
        intermediate_size: Inner dimension of the gated MLP in each encoder
            layer.  Defaults to 3072.
        num_hidden_layers: Number of transformer encoder layers.
            Defaults to 16.
        num_attention_heads: Number of attention heads per encoder layer.
            Defaults to 12.
        num_key_value_heads: Number of key-value heads (GQA).
            Defaults to 12 (i.e., MHA by default).
        head_dim: Per-head dimension for attention.  Defaults to 64.
        hidden_activation: Activation function in the encoder MLP.
            Defaults to ``"gelu_pytorch_tanh"``.
        rms_norm_eps: Epsilon for RMSNorm layers.  Defaults to 1e-6.
        max_position_embeddings: Maximum number of patch positions.
            Defaults to 131 072.
        attention_bias: Whether to include bias in attention projections.
            Defaults to ``False``.
        attention_dropout: Dropout probability for attention weights.
            Defaults to 0.0.
        rope_parameters: RoPE configuration dictionary.  If ``None``,
            defaults to standard RoPE with ``rope_theta=100.0`` (very low
            base frequency appropriate for spatial patch positions).
        pooling_kernel_size: Spatial average-pooling kernel size applied after
            patchification to reduce the number of vision tokens.
            Defaults to 3.
        patch_size: Size of each square image patch in pixels.
            Defaults to 16.
        position_embedding_size: Maximum number of entries in the learned 2-D
            position embedding table.  Defaults to 10 240.
        use_clipped_linears: Whether to clamp linear layer weights to a
            fixed range during the forward pass for numerical stability.
            Defaults to ``False``.
        standardize: Whether to apply a learned bias and scale to the pooled
            vision tokens before projection.  Defaults to ``False``.
        initializer_range: Standard deviation for weight initialisation.
            Defaults to 0.02.
        gradient_checkpointing: Gradient checkpointing strategy.
            Defaults to ``NONE``.
        bits: Quantisation bit-width.  ``None`` disables quantisation.
    """

    model_type: str = "gemma4_vision"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        head_dim: int = 64,
        hidden_activation: str = "gelu_pytorch_tanh",
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131_072,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_parameters: dict | None = None,
        pooling_kernel_size: int = 3,
        patch_size: int = 16,
        position_embedding_size: int = 10 * 1024,
        use_clipped_linears: bool = False,
        standardize: bool = False,
        initializer_range: float = 0.02,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        bits: int | None = None,
        **kwargs,
    ):
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits

        super().__init__(bits=bits, **kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size
        self.use_clipped_linears = use_clipped_linears
        self.standardize = standardize
        self.initializer_range = initializer_range

        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 100.0}
        self.rope_parameters = rope_parameters

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Return tensor-parallelism partition rules.

        Returns ``None`` to use the default partitioning strategy.
        """
        return None


@register_config("gemma4")
class Gemma4Config(EasyDeLBaseConfig):
    """Top-level multimodal configuration for Gemma4.

    Bundles a ``Gemma4TextConfig`` (the language decoder), an optional
    ``Gemma4VisionConfig`` (the vision encoder), an optional audio config blob
    kept for Hugging Face checkpoint/config compatibility, and special-token
    IDs used to locate image/video/audio placeholder positions in the token
    sequence during multimodal embedding merging.

    When ``vision_config`` is ``None``, the model is instantiated without a
    vision tower and can only process text inputs. Audio-specific config fields
    are accepted and preserved so upstream Gemma 4 configs can round-trip
    cleanly, even though EasyDeL's local Gemma 4 implementation does not yet
    expose an audio tower.

    Args:
        text_config: Configuration for the text decoder.  If ``None``, uses
            ``Gemma4TextConfig()`` defaults.  Can also be a dictionary that
            will be unpacked into ``Gemma4TextConfig(**dict)``.
        vision_config: Configuration for the vision encoder.  ``None`` disables
            vision.  Can be a dictionary unpacked into ``Gemma4VisionConfig``.
        audio_config: Optional audio tower configuration payload preserved for
            Hugging Face Gemma 4 config compatibility.
        boi_token_id: Begin-of-image sentinel token index.
            Defaults to 255 999.
        eoi_token_id: End-of-image sentinel token index.
            Defaults to 258 882.
        image_token_id: Placeholder token index that is replaced by vision
            soft tokens during embedding merging.  Defaults to 258 880.
        video_token_id: Placeholder token index for video frames.
            Defaults to 258 884.
        boa_token_id: Begin-of-audio sentinel token index.
            Defaults to 256 000.
        eoa_token_index: End-of-audio sentinel token index.
            Defaults to 258 883.
        audio_token_id: Placeholder token index for audio features.
            Defaults to 258 881.
        tie_word_embeddings: Whether top-level configs should advertise tied
            embeddings like upstream Gemma 4. Defaults to ``True``.
        initializer_range: Standard deviation for weight initialisation.
            Defaults to 0.02.
    """

    model_type = "gemma4"
    sub_configs: typing.ClassVar = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
    }

    def __init__(
        self,
        text_config: Gemma4TextConfig | dict | None = None,
        vision_config: Gemma4VisionConfig | dict | None = None,
        audio_config: dict | None = None,
        boi_token_id: int = 255_999,
        eoi_token_id: int = 258_882,
        image_token_id: int = 258_880,
        video_token_id: int = 258_884,
        boa_token_id: int = 256_000,
        eoa_token_index: int = 258_883,
        audio_token_id: int = 258_881,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = Gemma4VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = dict(audio_config) if isinstance(audio_config, dict) else audio_config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_index = eoa_token_index
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_partition_rules(self, *args, **kwargs) -> tuple[tuple[str, PartitionSpec], ...] | None:
        """Return tensor-parallelism partition rules.

        Returns ``None`` to use the default partitioning strategy.
        """
        return None

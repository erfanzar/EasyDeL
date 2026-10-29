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

"""Configuration for the DSpark block drafter.

DSpark trains a parallel block drafter from cached target-model hidden states.
For each sampled anchor position it predicts a block of future draft tokens,
optionally adds a Markov logit bias, and optionally predicts confidence logits
for accept/reject calibration.
"""

from __future__ import annotations

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.modules._drafting import validate_target_layer_ids


@register_config("dspark")
class DSparkConfig(EasyDeLBaseConfig):
    """Hyperparameters for a DeepSpec-compatible DSpark drafter.

    The transformer geometry mirrors the target model, while DSpark-specific
    fields control anchor sampling, block length, Markov heads, confidence
    prediction, and the DeepSpec CE/L1 loss mixture.
    """

    model_type = "dspark"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        target_hidden_size: int | None = None,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 5,
        num_draft_layers: int | None = None,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        target_layer_ids: tuple[int, ...] | list[int] = (1, 9, 17, 25, 33),
        target_num_hidden_layers: int | None = None,
        block_size: int = 7,
        mask_token_id: int | None = None,
        num_anchors: int = 512,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        confidence_head_alpha: float = 1.0,
        confidence_head_with_markov: bool = True,
        loss_decay_gamma: float = 4.0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        freeze_embeddings_and_lm_head: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize DSpark geometry and block-drafter controls.

        Args:
            vocab_size: Size of the vocabulary (token embedding count).
            hidden_size: Hidden dimension of the drafter transformer.
            target_hidden_size: Hidden dimension of the target model. ``None`` falls
                back to ``hidden_size``.
            intermediate_size: FFN intermediate dimension (typically ``2.5 * hidden_size``
                for SwiGLU).
            num_hidden_layers: Number of drafter transformer layers.
            num_draft_layers: Alias for ``num_hidden_layers``.  If provided, it overrides
                ``num_hidden_layers``.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of KV heads for GQA. ``None`` falls back to
                ``num_attention_heads``.
            head_dim: Dimension of each attention head.
            hidden_act: Activation function name for the MLP (e.g. ``"silu"``).
            max_position_embeddings: Maximum sequence length supported by RoPE.
            initializer_range: Standard deviation for weight initialization.
            rms_norm_eps: Epsilon for RMSNorm stability.
            attention_bias: Whether to add bias to attention Q/K/V/O projections.
            attention_dropout: Dropout rate applied to attention scores.
            rope_theta: RoPE base frequency.
            rope_scaling: Optional RoPE scaling configuration dict.
            target_layer_ids: Target hidden-state layer indices to concatenate as
                drafter context. ``-1`` means embedding output; non-negative ``n``
                maps to hidden-state output index ``n + 1``.
            target_num_hidden_layers: Decoder depth of the target model. Used to
                detect when ``target_layer_ids`` points at the final hidden state.
            block_size: Number of future positions predicted per anchor.
            mask_token_id: Token ID used for masked (non-anchor) draft positions.
                ``None`` defaults to ``vocab_size - 1``.
            num_anchors: Number of anchor blocks sampled per sequence.
            markov_rank: Low-rank Markov bias width. ``0`` disables the Markov head.
            markov_head_type: Markov head variant. One of ``"vanilla"``, ``"gated"``,
                or ``"rnn"``. Only checked when ``markov_rank > 0``.
            confidence_head_alpha: Positive values enable the confidence head.
            confidence_head_with_markov: Whether to concatenate Markov features into
                the confidence head input. Requires ``markov_rank > 0`` when
                ``confidence_head_alpha > 0.0``.
            loss_decay_gamma: Exponential decay factor for position-wise loss weights.
            ce_loss_alpha: Weight of the cross-entropy component in the total loss.
            l1_loss_alpha: Weight of the L1 component in the total loss.
            freeze_embeddings_and_lm_head: When ``True``, exclude the token
                embedding table and LM head from the drafter trainable state.
            tie_word_embeddings: Whether input and output embeddings share weights.
            **kwargs: Unpacked[EasyDeLBaseConfig]. Forwarded to
                ``EasyDeLBaseConfig.__init__``. Includes sharding, attention,
                quantization, and hardware fields such as ``sharding_axis_dims``,
                ``attn_mechanism``, ``gradient_checkpointing``, ``quantization_config``,
                and others.

        Returns:
            None.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
            ValueError: If ``num_attention_heads * head_dim`` does not equal
                ``hidden_size``.
            ValueError: If ``block_size < 1``.
            ValueError: If ``num_anchors < 1``.
            ValueError: If ``markov_rank < 0``.
            ValueError: If ``markov_rank > 0`` and ``markov_head_type`` is not one of
                ``"vanilla"``, ``"gated"``, or ``"rnn"``.
            ValueError: If ``confidence_head_with_markov=True`` and
                ``markov_rank <= 0`` while ``confidence_head_alpha > 0.0``.
        """
        if num_draft_layers is not None:
            num_hidden_layers = int(num_draft_layers)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError("`num_attention_heads * head_dim` must equal `hidden_size`.")
        if int(block_size) < 1:
            raise ValueError("`block_size` must be >= 1.")
        if int(num_anchors) < 1:
            raise ValueError("`num_anchors` must be >= 1.")
        if int(markov_rank) < 0:
            raise ValueError("`markov_rank` must be >= 0.")
        markov_head_type = str(markov_head_type).lower()
        if int(markov_rank) > 0 and markov_head_type not in {"vanilla", "gated", "rnn"}:
            raise ValueError("`markov_head_type` must be one of 'vanilla', 'gated', or 'rnn'.")
        if float(confidence_head_alpha) > 0.0 and confidence_head_with_markov and int(markov_rank) <= 0:
            raise ValueError("`confidence_head_with_markov=True` requires `markov_rank > 0`.")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.target_hidden_size = int(target_hidden_size or hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.hidden_act = hidden_act
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.target_layer_ids = validate_target_layer_ids(target_layer_ids, allow_embedding=True)
        self.target_num_hidden_layers = int(target_num_hidden_layers) if target_num_hidden_layers is not None else None
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id) if mask_token_id is not None else self.vocab_size - 1
        self.num_anchors = int(num_anchors)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = markov_head_type
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.enable_confidence_head = self.confidence_head_alpha > 0.0
        self.confidence_head_with_markov = bool(confidence_head_with_markov)
        self.loss_decay_gamma = float(loss_decay_gamma)
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.freeze_embeddings_and_lm_head = bool(freeze_embeddings_and_lm_head)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("DSparkConfig",)

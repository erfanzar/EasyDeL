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

"""Configuration for the EAGLE3 drafter architecture.

EAGLE3 trains a small autoregressive draft model from a fixed set of target
model hidden layers.  The target features are concatenated, projected to the
drafter width, mixed with token embeddings, and decoded by a shallow
Qwen-style stack.
"""

from __future__ import annotations

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.modules._drafting import validate_target_layer_ids


@register_config("eagle3")
class Eagle3Config(EasyDeLBaseConfig):
    """Hyperparameters for a DeepSpec-compatible EAGLE3 drafter.

    The geometry fields should normally mirror the target language model
    (`vocab_size`, hidden width, attention heads, RoPE settings).  EAGLE3 uses
    exactly five target decoder layers and stores the test-time-training knobs
    `ttt_length` and `step_loss_decay` so trainer code can reproduce the
    DeepSpec schedule.
    """

    model_type = "eagle3"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        target_hidden_size: int | None = None,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 1,
        draft_num_hidden_layers: int | None = None,
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
        ttt_length: int = 7,
        step_loss_decay: float = 0.8,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize EAGLE3 geometry and target-layer selection.

        Args:
            vocab_size: Size of the model vocabulary.
            hidden_size: Hidden dimension of the drafter model.
            target_hidden_size: Hidden dimension of the target model. If ``None``,
                defaults to ``hidden_size``.
            intermediate_size: FFN intermediate dimension (SwiGLU up-projection width).
            num_hidden_layers: Number of EAGLE3 refinement layers.
            draft_num_hidden_layers: Alias for ``num_hidden_layers``, matching
                the DeepSpec config naming.
            num_attention_heads: Number of attention heads for GQA.
            num_key_value_heads: Number of key/value heads for GQA. If ``None``,
                defaults to ``num_attention_heads``.
            head_dim: Dimension of each attention head.
            hidden_act: Activation function name for the MLP (e.g. ``"silu"``).
            max_position_embeddings: Maximum sequence length the drafter can
                attend over.
            initializer_range: Standard deviation for the normal weight initializer.
            rms_norm_eps: Epsilon used by RMSNorm layers.
            attention_bias: Whether to include bias terms in attention projections.
            attention_dropout: Dropout rate applied to attention probabilities.
            rope_theta: Base frequency for RoPE.
            rope_scaling: Optional RoPE scaling configuration dict.
            target_layer_ids: Five target decoder layer IDs. Layer ``n`` is read
                from hidden-state output index ``n + 1``; embedding output ``-1``
                is intentionally rejected for EAGLE3.
            ttt_length: Number of draft tokens used by EAGLE3 TTT training.
            step_loss_decay: Per-step loss decay used by EAGLE3 training.
            tie_word_embeddings: Whether input and output embeddings share weights.
            **kwargs: Forwarded to ``EasyDeLBaseConfig``. Common kwargs include
                ``sharding_axis_dims``, ``attn_mechanism``, ``gradient_checkpointing``,
                ``scan_layers``, ``backend``, ``platform``, ``easy_method``, ``bits``,
                ``quantization_config``, and ``kv_cache_quantization_config``.

        Returns:
            None.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
            ValueError: If ``num_attention_heads * head_dim`` does not equal
                ``hidden_size``.
            ValueError: If ``ttt_length`` is less than ``1``.
            ValueError: If ``step_loss_decay`` is not strictly positive.
        """
        if draft_num_hidden_layers is not None:
            num_hidden_layers = int(draft_num_hidden_layers)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError("`num_attention_heads * head_dim` must equal `hidden_size`.")
        if int(ttt_length) < 1:
            raise ValueError("`ttt_length` must be >= 1.")
        if float(step_loss_decay) <= 0.0:
            raise ValueError("`step_loss_decay` must be > 0.")

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
        self.target_layer_ids = validate_target_layer_ids(
            target_layer_ids,
            allow_embedding=False,
            expected_count=5,
        )
        self.ttt_length = int(ttt_length)
        self.step_loss_decay = float(step_loss_decay)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("Eagle3Config",)

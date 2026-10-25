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

"""Q/K RMSNorm over the whole projected width (all heads at once), applied before the head reshape."""

from __future__ import annotations

from ..norms._norms import RMSNorm

__all__ = ("FlattenedQKNormMixin",)


class FlattenedQKNormMixin:
    """Mixin for :class:`UnifiedAttention` subclasses whose Q/K norms span every head (OLMo-2 / OLMo-3).

    The default ``use_qk_norm`` path normalises each head separately (``head_dim`` wide). These families normalise
    the flattened projection instead: the query norm is ``num_attention_heads * head_dim`` wide, the key norm
    ``num_key_value_heads * head_dim`` wide, and both run in :meth:`_preprocess_qkv`, before the reshape to heads.
    Parameter names (``q_norm`` / ``k_norm``) are unchanged. List the mixin before ``UnifiedAttention`` and pass
    ``use_qk_norm=True``.
    """

    def _create_q_norm(self, config, dtype, param_dtype, rngs) -> RMSNorm:
        """Query RMSNorm over ``num_attention_heads * head_dim`` features."""
        return RMSNorm(
            dim=config.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config, dtype, param_dtype, rngs) -> RMSNorm:
        """Key RMSNorm over ``num_key_value_heads * head_dim`` features."""
        return RMSNorm(
            dim=config.num_key_value_heads * self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _preprocess_qkv(self, query_states, key_states, value_states):
        """Normalise the flattened query and key projections; values pass through."""
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states

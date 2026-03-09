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

"""Masking helpers shared by linear-attention modules.

Linear-attention layers zero out hidden states at padding positions before
the convolution and recurrence stages.  This module provides a single utility
for that operation so it is not duplicated across model implementations.
"""

from __future__ import annotations

from jaxtyping import Array


def apply_mask_to_padding_states(hidden_states: Array, attention_mask: Array | None) -> Array:
    """Zero out hidden states at padding positions.

    Applies an element-wise mask to ``hidden_states`` using the provided
    ``attention_mask``.  The mask is broadcast along the feature dimension
    so that entire token vectors are zeroed when the corresponding mask
    position is ``0`` (or ``False``).

    The function is a no-op (returns ``hidden_states`` unchanged) when:

    - ``attention_mask`` is ``None``.
    - The batch or sequence dimensions do not match.
    - The sequence length is 1 (single-token decode — padding is irrelevant).

    Args:
        hidden_states: Input tensor of shape ``[batch, seq_len, dim]``.
        attention_mask: Boolean or ``{0, 1}`` mask of shape
            ``[batch, seq_len]``, or ``None``.

    Returns:
        Masked hidden states with the same shape and dtype as the input.
    """
    if (
        attention_mask is not None
        and attention_mask.shape[0] == hidden_states.shape[0]
        and attention_mask.shape[1] == hidden_states.shape[1]
        and attention_mask.shape[1] > 1
    ):
        return (hidden_states * attention_mask[:, :, None]).astype(hidden_states.dtype)
    return hidden_states

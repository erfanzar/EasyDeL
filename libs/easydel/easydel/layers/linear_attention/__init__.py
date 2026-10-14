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

"""Shared utilities for linear-attention model implementations.

This package provides reusable building blocks that are common across different
linear-attention architectures (GatedDeltaNet, KDA, etc.):

    Convolution state management:
        - ``shift_conv_state_left``: Shift a causal conv cache and append a token.
        - ``apply_manual_depthwise_conv``: Apply cached depthwise conv at decode time.
        - ``apply_conv_with_state``: Unified train/decode conv with state bookkeeping.

    Masking:
        - ``apply_mask_to_padding_states``: Zero out padding positions in hidden states.

These are intentionally free functions (not classes) so that XLA can inline them
without vtable dispatch overhead on the TPU decode hot path.
"""

from ._conv_state import (
    apply_conv_with_state,
    apply_manual_depthwise_conv,
    shift_conv_state_left,
)
from ._masking import apply_mask_to_padding_states

__all__ = (
    "apply_conv_with_state",
    "apply_manual_depthwise_conv",
    "apply_mask_to_padding_states",
    "shift_conv_state_left",
)

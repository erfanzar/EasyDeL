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

    Kernel Delta Attention (every KDA model family: Kimi-Linear, GLM-5-Next):
        - ``KernelDeltaAttention``: The shared KDA layer, subclassed per model family.
        - ``kda_layer_spec`` / ``KDALayerSpec``: Layer (and cache) shapes from ``config.linear_attn_config``.
        - ``unnest_legacy_kda_forget_gate``: Native-checkpoint hook for saves with a nested ``forget_gate``.
"""

from ._conv_state import (
    apply_conv_with_state,
    apply_manual_depthwise_conv,
    shift_conv_state_left,
)
from ._depthwise_conv import ChannelMajorConv1D
from ._kda import KDALayerSpec, KernelDeltaAttention, kda_layer_spec, unnest_legacy_kda_forget_gate
from ._masking import apply_mask_to_padding_states

__all__ = (
    "ChannelMajorConv1D",
    "KDALayerSpec",
    "KernelDeltaAttention",
    "apply_conv_with_state",
    "apply_manual_depthwise_conv",
    "apply_mask_to_padding_states",
    "kda_layer_spec",
    "shift_conv_state_left",
    "unnest_legacy_kda_forget_gate",
)

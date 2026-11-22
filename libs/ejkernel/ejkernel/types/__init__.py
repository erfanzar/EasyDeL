# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Type definitions and mask utilities for ejkernel.

This module provides core type definitions and mask management utilities
for attention operations in ejkernel.

Key Components:
    MaskInfo: Comprehensive dataclass for managing attention masks and segment IDs.
        Supports various attention patterns including causal, sliding window,
        and chunked attention, with efficient conversions between representations.

    Debug Utilities:
        - set_debug_mode: Enable/disable debug tracing for mask operations
        - get_debug_mode: Check current debug mode status

Mask Representations:
    1. Attention Mask: 4D boolean/int array [batch, heads, q_len, kv_len]
       - True/1 = valid attention, False/0 = masked
    2. Segment IDs: 2D int32 arrays [batch, seq_len]
       - Non-negative = segment membership
       - -1 = padding tokens

Example:
    >>> from ejkernel.types import MaskInfo
    >>> import jax.numpy as jnp
    >>>
    >>> # Create mask from segment IDs
    >>> segment_ids = jnp.array([[1, 1, 2, 2]])
    >>> mask_info = MaskInfo.from_segments(segment_ids)
    >>>
    >>> # Apply causal masking
    >>> causal_mask = mask_info.apply_causal()
    >>>
    >>> # Get attention bias for softmax
    >>> bias = mask_info.bias

See Also:
    ejkernel.types.mask: Full mask module with all conversion functions
"""

from .mask import MaskInfo, get_debug_mode, set_debug_mode

__all__ = ("MaskInfo", "get_debug_mode", "set_debug_mode")

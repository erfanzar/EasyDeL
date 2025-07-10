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
import chex

from ._forward_pallas import inner_decode_tpu


def ragged_decode_tpu(
    query_tensor: chex.Array,
    key_tensor: chex.Array,
    value_tensor: chex.Array,
    sequence_start: chex.Array,
    sequence_end: chex.Array,
    softmax_scale: float | None = 1,
    block_size: int = 256,
) -> chex.Array:
    """Ragged MQA decoding entry point with TPU-accelerated Flash Attention.

    Args:
        query_tensor (chex.Array): Query tensor of shape [B, H, D] or [B, 1, H, D].
        key_tensor (chex.Array): Key tensor of shape [B, S, H, D].
        value_tensor (chex.Array): Value tensor of shape [B, S, H, D].
        sequence_start (chex.Array): int32 array of shape [B], start indices of sequences.
        sequence_end (chex.Array): int32 array of shape [B], end indices of sequences.
        softmax_scale (float | None): Optional scale for attention logits. Default is 1.
        block_size (int): Block size used for kernel tiling. Default is 256.

    Returns:
        chex.Array: Output tensor of shape [B, H, D] after attention decoding.
    """
    return inner_decode_tpu(
        query_tensor=query_tensor,
        key_tensor=key_tensor,
        value_tensor=value_tensor,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        block_size=block_size,
    )

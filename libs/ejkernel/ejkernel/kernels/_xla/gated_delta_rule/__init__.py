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


"""XLA backend for Gated Delta Rule (GDR) linear attention.

This submodule provides the XLA/JAX implementation of the Gated Delta Rule, a
linear attention mechanism used in hybrid transformer architectures such as
Qwen3Next.

Key Features:
    - O(N) complexity through recurrent state updates
    - Gated delta rule for selective memory management
    - Exact intra-chunk computation via triangular solve (replaces earlier
      Neumann-series approximation which diverged on padded batches)
    - Learnable per-token decay for temporal weighting
    - Support for chunked, recurrent, and single-step computation modes

Reference:
    Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
"""

from ._interface import gated_delta_rule

__all__ = ("gated_delta_rule",)

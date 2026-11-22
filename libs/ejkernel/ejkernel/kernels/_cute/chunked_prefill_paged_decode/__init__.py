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

"""CuTe backend for chunked prefill + paged decode attention.

This backend uses a CuTe DSL kernel to update the block-tabled KV cache and
then computes attention on the updated cache with the best available attention
implementation for the current runtime.
"""

from ._interface import chunked_prefill_paged_decode

__all__ = ("chunked_prefill_paged_decode",)

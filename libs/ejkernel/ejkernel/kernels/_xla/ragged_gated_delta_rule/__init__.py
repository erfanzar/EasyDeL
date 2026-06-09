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

"""XLA backend for Ragged Gated Delta Rule (packed continuous-batching GDR).

Provides chunked and decode-only forward implementations for processing
variable-length sequences packed into a flat token stream. Used by
continuous-batching inference engines to process multiple requests in a
single kernel call without padding overhead.
"""

from ._interface import ragged_gated_delta_rule

__all__ = ("ragged_gated_delta_rule",)

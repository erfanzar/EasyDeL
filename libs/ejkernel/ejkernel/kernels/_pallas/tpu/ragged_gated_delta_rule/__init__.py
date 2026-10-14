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

"""TPU Pallas backend for Ragged Gated Delta Rule.

Provides a Pallas TPU kernel for the decode path that achieves up to 3.7x
speedup over the XLA implementation by running per-token GDR updates in
parallel across TPU cores. The prefill path falls back to the XLA chunked
implementation.
"""

from ._interface import ragged_gated_delta_rule, ragged_gated_delta_rule_decode

__all__ = ("ragged_gated_delta_rule", "ragged_gated_delta_rule_decode")

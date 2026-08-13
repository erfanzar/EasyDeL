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

"""Pallas TPU one-shot all-reduce kernel.

Direct-exchange all-reduce optimised for the latency-bound small-message
regime (decode-step tensor-parallel partial sums).  Large inputs delegate to
``lax.psum``.

Public API:
    all_reduce: Registered under ``Platform.PALLAS / Backend.TPU``.
        Custom VJP passes the (replicated) output cotangent through unchanged.
"""

from ._interface import all_reduce

__all__ = ("all_reduce",)

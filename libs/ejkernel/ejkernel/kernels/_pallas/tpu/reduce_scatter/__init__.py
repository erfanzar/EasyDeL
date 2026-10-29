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

"""Pallas TPU one-shot reduce-scatter kernel.

Direct slice-exchange reduce-scatter optimised for the latency-bound
small-message regime.  Large or non-axis-0 requests delegate to
``lax.psum_scatter``.

Public API:
    reduce_scatter: Registered under ``Platform.PALLAS / Backend.TPU``.
        Custom VJP all-gathers the output cotangent via the sibling one-shot
        all-gather kernel.
"""

from ._interface import reduce_scatter

__all__ = ("reduce_scatter",)

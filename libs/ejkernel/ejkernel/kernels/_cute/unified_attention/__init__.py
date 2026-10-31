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

"""CuTe platform unified attention implementation.

Exposes :func:`unified_attention`, the registry entry point for the CuTe
platform.  Internally this delegates to the Triton unified-attention kernel
when Triton is available; it raises :class:`~ejkernel.errors.EjkernelRuntimeError`
if the Triton fast path is absent and no other CuTe fallback exists.
Only causal attention is supported; non-causal calls raise
:class:`NotImplementedError`.
"""

from ._interface import unified_attention

__all__ = ["unified_attention"]

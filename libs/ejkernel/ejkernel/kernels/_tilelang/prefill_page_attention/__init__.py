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

"""TileLang ``prefill_page_attention`` kernel package.

Exports the registered inference-only GPU kernel for chunked-prefill paged
attention over a single sequence.  See ``_interface.py`` for the public API
and ``_kernel.py`` for the TileLang ``@T.prim_func`` implementation.
"""

from ._interface import prefill_page_attention

__all__ = ["prefill_page_attention"]

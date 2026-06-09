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

"""TileLang ``multi_latent_ragged_page_attention_v2`` kernel package.

Exports the v2 registered GPU kernel for native MLA ragged paged attention.
Functionally identical to v1; registered under a distinct name so callers
that request the ``_v2`` variant via the kernel registry receive this backend.
"""

from ._interface import multi_latent_ragged_page_attention_v2

__all__ = ["multi_latent_ragged_page_attention_v2"]

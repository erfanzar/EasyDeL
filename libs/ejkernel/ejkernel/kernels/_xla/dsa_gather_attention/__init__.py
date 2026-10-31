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

"""Gather DSA attention: shared selection semantics and the XLA reference."""

from ._interface import dsa_gather_attention
from ._layout import default_positions, selection_layout
from ._xla_impl import dsa_gather_attention_xla

__all__ = ("default_positions", "dsa_gather_attention", "dsa_gather_attention_xla", "selection_layout")

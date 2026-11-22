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


"""XLA reference of the decode-size mHC read (gates + collapse) and residual write."""

from ._interface import mhc_decode_read, mhc_decode_write
from ._xla_impl import gate_table_layout, supports, table_gates

__all__ = ("gate_table_layout", "mhc_decode_read", "mhc_decode_write", "supports", "table_gates")

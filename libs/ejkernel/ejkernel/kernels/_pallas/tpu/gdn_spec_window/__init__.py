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

"""TPU Pallas backend package for the GDN speculative-window state scan.

Fuses the whole speculative verify window's gated-delta-rule recurrence into
one VMEM-resident kernel per layer (one state read, one write per candidate
row), replacing the per-step HBM round-trips of the XLA reference.
"""

from ._interface import gdn_spec_window_states

__all__ = ("gdn_spec_window_states",)

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


"""Pallas TPU prefill-size mHC projection, read (gates + collapse) and residual write. APPROXIMATE, opt-in."""

from ._interface import mhc_prefill_project, mhc_prefill_read, mhc_prefill_write
from ._pallas_impl import mhc_prefill_project_pallas, mhc_prefill_read_pallas, mhc_prefill_write_pallas

__all__ = (
    "mhc_prefill_project",
    "mhc_prefill_project_pallas",
    "mhc_prefill_read",
    "mhc_prefill_read_pallas",
    "mhc_prefill_write",
    "mhc_prefill_write_pallas",
)

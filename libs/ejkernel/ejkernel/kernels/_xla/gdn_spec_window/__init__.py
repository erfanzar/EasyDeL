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

"""XLA backend package for the GDN speculative-window state-scan kernel.

Computes the per-prefix recurrent states of a speculative-decoding verify
window for GDN (gated-delta-net) hybrid models in one call. The pure-JAX
numerics live in :mod:`._xla_impl_fwd`; :mod:`._interface` adds registry
registration (``Platform.XLA`` / ``Backend.ANY``) and runtime type checking.
"""

from ._interface import gdn_spec_window_states

__all__ = ("gdn_spec_window_states",)

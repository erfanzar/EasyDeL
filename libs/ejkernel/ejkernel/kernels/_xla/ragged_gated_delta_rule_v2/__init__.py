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

"""XLA implementation of the ragged Gated Delta Rule (GDN) v2 kernel.

Portable JAX/XLA Gated Delta Rule recurrence for the eSurge packed-inference
path, where many continuous-batching requests of heterogeneous lengths share a
single contiguous token buffer. This subpackage groups the public registry
entry with the forward-pass building blocks it dispatches to.

Public symbols:
    - ``ragged_gated_delta_rule_v2``: Registered (Platform.XLA, Backend.ANY)
      entry point that casts inputs to a runtime dtype and forwards to the
      unsharded packed-inference GDN kernel.
    - ``ragged_gated_delta_rule_decode_only``: Per-token recurrent update used
      when every active request consumes exactly one new token.
    - ``ragged_gated_delta_rule_mixed_prefill``: Chunked algorithm that pads
      each request to a multiple of ``chunk_size`` and propagates inter-chunk
      state via ``lax.scan``.
"""

from ._interface import ragged_gated_delta_rule_v2
from ._xla_impl_fwd import (
    ragged_gated_delta_rule_decode_only,
    ragged_gated_delta_rule_mixed_prefill,
)

__all__ = (
    "ragged_gated_delta_rule_decode_only",
    "ragged_gated_delta_rule_mixed_prefill",
    "ragged_gated_delta_rule_v2",
)

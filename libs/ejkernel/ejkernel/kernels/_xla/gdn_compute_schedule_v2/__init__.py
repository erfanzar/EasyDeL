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

"""XLA backend for ragged Gated Delta Rule (GDN) v2 schedule-table construction.

This package provides the pure-JAX/XLA implementation that precomputes the
per-grid-iteration work schedule consumed by the TPU Pallas ragged GDN v2
kernels. The schedule table flattens a mixed decode/prefill packed token stream
into a fixed-shape integer table whose rows describe, for each grid step,
whether decode or prefill work runs, which request and token range it covers,
and the sublane transition metadata required to stitch chunk-recurrent state
across misaligned request boundaries.

Modules:
    _xla_impl_fwd: The XLA implementation building the schedule table.
    _interface: Registry-registered, jaxtyping/beartype-checked entry point.

Exports:
    compute_schedule_table_v2: The registry-facing schedule-table builder.
"""

from ._interface import compute_schedule_table_v2

__all__ = ("compute_schedule_table_v2",)

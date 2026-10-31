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

"""Public eligibility checks for the MoE row-gather kernels.

The kernels are registry entries (``moe_row_gather`` / ``moe_row_gather_sum``; Pallas TPU with an XLA fallback). A
caller that must not fall back silently asks these helpers first: each returns why the Pallas kernel would not run for
the given shapes and dtypes, or ``None`` when it would. Only ``.shape`` / ``.dtype`` are read, so
``jax.ShapeDtypeStruct`` arguments work.
"""

from ejkernel.kernels._pallas.tpu.moe_row_gather import (
    row_gather_sum_unsupported_reason,
    row_gather_unsupported_reason,
)

__all__ = ("row_gather_sum_unsupported_reason", "row_gather_unsupported_reason")

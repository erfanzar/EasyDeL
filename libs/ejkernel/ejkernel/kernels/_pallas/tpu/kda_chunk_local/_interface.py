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

"""Registry entry for the TPU Pallas per-channel KDA chunk-local kernel."""

from __future__ import annotations

from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.kda_chunk_local._xla_impl import kda_chunk_local_xla
from ._pallas_impl import kda_chunk_local_pallas

_LANES = 128


@kernel_registry.register("kda_chunk_local", Platform.PALLAS, Backend.TPU, priority=10)
def kda_chunk_local(
    k_beta: Float[Array, "... chunk_size head_dim"],
    key: Float[Array, "... chunk_size head_dim"],
    query: Float[Array, "... chunk_size head_dim"],
    g_cumsum: Float[Array, "... chunk_size head_dim"],
    *,
    sub_block: int = 16,
) -> tuple[Float[Array, "... chunk_size chunk_size"], Float[Array, "... chunk_size chunk_size"]]:
    """Same contract as the XLA entry; Pallas forward + backward on TPU.

    Layouts the kernel does not tile (head_dim not a lane multiple, chunk not a multiple of
    ``sub_block`` or of 8, ``sub_block`` not a multiple of the 8-row sublane tile) route to
    the XLA implementation.

    Precondition: ``g_cumsum`` must be non-increasing along the chunk axis (a cumulative sum
    of non-positive log-decay). The kernel factors ``exp(g_i - g_j)`` through a per-block
    reference row, which is overflow-free only under that monotonicity; a non-monotone ``g``
    can overflow one factor (``inf * 0 = NaN``) where the XLA reference stays finite.
    """
    cs, kd = int(k_beta.shape[-2]), int(k_beta.shape[-1])
    sb = int(sub_block)
    if kd % _LANES or cs % 8 or sb <= 0 or sb % 8 or cs % sb:
        return kda_chunk_local_xla(k_beta, key, query, g_cumsum)
    return kda_chunk_local_pallas(k_beta, key, query, g_cumsum, sb)


__all__ = ("kda_chunk_local",)

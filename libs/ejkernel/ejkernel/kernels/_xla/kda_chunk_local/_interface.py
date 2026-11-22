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

"""XLA registry entry (priority 0 fallback) for the per-channel KDA chunk-local terms."""

from __future__ import annotations

from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import kda_chunk_local_xla


@kernel_registry.register("kda_chunk_local", Platform.XLA, Backend.ANY)
def kda_chunk_local(
    k_beta: Float[Array, "... chunk_size head_dim"],
    key: Float[Array, "... chunk_size head_dim"],
    query: Float[Array, "... chunk_size head_dim"],
    g_cumsum: Float[Array, "... chunk_size head_dim"],
    *,
    sub_block: int = 16,
) -> tuple[Float[Array, "... chunk_size chunk_size"], Float[Array, "... chunk_size chunk_size"]]:
    """Per-channel KDA intra-chunk matrices, differentiable.

    Args:
        k_beta: ``key * beta`` per chunk ``[..., cs, K]``.
        key: Keys ``[..., cs, K]``.
        query: Scaled queries ``[..., cs, K]``.
        g_cumsum: Within-chunk cumulative per-channel log-decay ``[..., cs, K]``. Callers are
            expected to pass a non-increasing sequence along ``cs`` (cumsum of non-positive
            log-decay); this implementation forms the pair difference before ``exp`` and is
            safe regardless, but the Pallas backend requires the monotone form.
        sub_block: Accepted for API parity with the Pallas backend (unused here).

    Returns:
        ``(attn, attn_qk)`` float32: ``attn`` strictly lower, ``-(k_beta_i . k_j) e^{g_i-g_j}``;
        ``attn_qk`` lower incl. diagonal, ``(q_i . k_j) e^{g_i-g_j}`` (decay per channel).
    """
    del sub_block
    return kda_chunk_local_xla(k_beta, key, query, g_cumsum)

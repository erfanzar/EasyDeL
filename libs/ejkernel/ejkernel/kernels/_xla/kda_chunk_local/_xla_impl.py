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

"""XLA reference / fallback for the per-channel KDA chunk-local terms.

For each chunk (last two axes ``[cs, K]``):

    attn[i, j]    = -(k_beta_i . k_j) * exp(g_i - g_j)   for j <  i, else 0
    attn_qk[i, j] =  (q_i . k_j)      * exp(g_i - g_j)   for j <= i, else 0

with the decay applied per channel. The ``(cs, cs, K)`` decay tensor is built one chunk at a
time (``lax.map`` over the leading axis) and rematerialized under autodiff, so backward keeps
only the ``[cs, K]`` inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

_HP = lax.Precision.HIGHEST


def _direct(k_beta, key, query, g_cumsum):
    cs = k_beta.shape[-2]
    tril = jnp.tril(jnp.ones((cs, cs), dtype=bool))
    upper_or_diag = jnp.triu(jnp.ones((cs, cs), dtype=bool), k=0)
    strict_upper = jnp.triu(jnp.ones((cs, cs), dtype=bool), k=1)
    g = g_cumsum.astype(jnp.float32)
    # Pair difference before exp, masked on both sides: exp(g_i) * exp(-g_j) overflows in f32
    # once a chunk's cumulative decay passes ~-88.
    g_diff = jnp.where(tril[:, :, None], g[..., :, None, :] - g[..., None, :, :], 0.0)
    decay_mask = jnp.where(tril[:, :, None], jnp.exp(g_diff), 0.0)
    kb, k, q = (x.astype(jnp.float32) for x in (k_beta, key, query))
    attn = -jnp.einsum("...ik,...jk,...ijk->...ij", kb, k, decay_mask, precision=_HP)
    attn_qk = jnp.einsum("...ik,...jk,...ijk->...ij", q, k, decay_mask, precision=_HP)
    return jnp.where(upper_or_diag, 0.0, attn), jnp.where(strict_upper, 0.0, attn_qk)


def kda_chunk_local_xla(k_beta, key, query, g_cumsum):
    """``[..., cs, K]`` x4 -> ``(attn, attn_qk)`` each ``[..., cs, cs]`` float32."""
    fn = jax.checkpoint(lambda a: _direct(*a), prevent_cse=False)
    if k_beta.ndim == 2:
        return fn((k_beta, key, query, g_cumsum))
    return lax.map(fn, (k_beta, key, query, g_cumsum))

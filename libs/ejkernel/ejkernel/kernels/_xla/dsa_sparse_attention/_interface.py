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

"""Pool-bitmap block-sparse DSA attention interface (XLA reference, mandatory fallback)."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._tables import validate_blocks, validate_options
from ._xla_impl import dsa_sparse_attention_xla


@kernel_registry.register("dsa_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def dsa_sparse_attention(
    query: Float[Array, "batch num_heads seq head_dim"],
    latent: Float[Array, "batch seq head_dim"],
    pool_mask: Bool[Array, "batch seq num_pools"] | Int[Array, "batch seq num_words"],
    tails: Int[Array, "batch seq num_tails"] | None = None,
    ste_pool: Float[Array, "batch seq num_pools"] | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    mode: str = "sparse",
    block_q: int = 256,
    block_kv: int = 512,
    backward: str = "fused",
    fwd_nt: bool = False,
    fwd_lean: bool = False,
    bwd_lean: bool = False,
    pool_sum: str = "reshape",
    stale_tau: float | None = None,
    fused_groups: int | None = None,
    interpret: bool | None = None,
) -> Float[Array, "batch num_heads seq head_dim"]:
    """Attention over the indexer's pool selection (dense XLA reference).

    Registered under ``"dsa_sparse_attention"`` for ``Platform.XLA`` / ``Backend.ANY``. Defines the value and
    gradient contract the Pallas TPU kernels must match. The per-pair cost options (``backward``, ``fwd_nt``,
    ``fwd_lean``, ``bwd_lean``, ``pool_sum``, ``stale_tau``, ``fused_groups``, ``interpret``) only shape the
    kernels; they are validated here and otherwise have no effect on this dense path.

    Args:
        query: ``[B, H, S, D]`` queries (absorbed-MLA queries for GLM-5-style DSA).
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        pool_mask: ``[B, S, P]`` bool selected pools; pool ``p`` = keys ``p*kpool .. p*kpool + kpool - 1``. Or the
            same selection as ``[B, S, W]`` int32 packed words (bit ``p % 32`` of word ``p // 32`` = pool ``p``).
        tails: ``[B, S, T]`` int32 always-selected tail keys (``-1`` = none, ``T <= 8``) or ``None``.
        ste_pool: ``[B, S, P]`` pooled straight-through term (value 0; its cotangent reaches the indexer).
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        mode: ``"sparse"``, ``"dense"`` (same numerics) or ``"budget:B"`` (numerics change; see ``_tables``).
        block_q: Query block (multiple of 128); used by ``budget`` tables.
        block_kv: Key block (multiple of 128 and ``32 * kpool``); used by ``budget`` tables.
        backward: Kernel backward variant, ``"fused"`` or ``"two_pass"``.
        fwd_nt: Kernel forward NT-dot layout.
        fwd_lean: Kernel forward lean variant.
        bwd_lean: Two-pass kernel backward lean variant.
        pool_sum: Kernel pool-cotangent reduction, ``"reshape"`` or ``"matmul"``.
        stale_tau: Kernel stale-max forward threshold (``None`` = off).
        fused_groups: Fused-backward core groups (``None`` = automatic).
        interpret: Pallas interpret mode (kernels only).

    Returns:
        ``[B, H, S, D]`` in ``latent``'s dtype.
    """
    mode = validate_options(
        mode=mode, backward=backward, pool_sum=pool_sum, stale_tau=stale_tau, fused_groups=fused_groups
    )
    validate_blocks(int(block_q), int(block_kv), int(kpool))
    _ = (fwd_nt, fwd_lean, bwd_lean, interpret)  # kernel-only options: accepted and ignored by the dense path
    return dsa_sparse_attention_xla(
        query,
        latent,
        pool_mask,
        tails,
        ste_pool,
        kpool=int(kpool),
        softmax_scale=softmax_scale,
        mode=mode,
        block_q=int(block_q),
        block_kv=int(block_kv),
    )

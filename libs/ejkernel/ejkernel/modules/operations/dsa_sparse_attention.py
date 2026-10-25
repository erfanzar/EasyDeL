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

"""Pool-bitmap block-sparse DSA attention module.

Attention of absorbed-MLA queries ``[B, H, S, D]`` over a shared ``K == V`` latent ``[B, S, D]`` restricted to the
DSA indexer's selection, given as a per-query POOL mask ``[B, S, P]`` (pool ``p`` = keys ``p*kpool ..
p*kpool + kpool - 1``) plus up to 8 always-selected TAIL keys per query. The Pallas TPU kernel skips every
(q-block, kv-block) pair the selection does not touch (no ``O(S^2)`` tensor) and is differentiable, including the
indexer's straight-through pool term ``ste_pool``. The XLA implementation is the dense reference and the mandatory
fallback.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import DsaSparseAttentionConfig

_PlatformName = Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"]


def _cfg_kwargs(cfg: DsaSparseAttentionConfig) -> dict:
    """Kernel keyword arguments carried by the config."""
    return dict(
        block_q=int(cfg.block_q),
        block_kv=int(cfg.block_kv),
        backward=str(cfg.backward),
        fwd_nt=bool(cfg.fwd_nt),
        fwd_lean=bool(cfg.fwd_lean),
        bwd_lean=bool(cfg.bwd_lean),
        pool_sum=str(cfg.pool_sum),
        stale_tau=cfg.stale_tau,
        fused_groups=cfg.fused_groups,
    )


class DsaSparseAttention(Kernel[DsaSparseAttentionConfig, Array]):
    """Pool-bitmap block-sparse DSA attention (Pallas TPU kernel, XLA reference).

    Both implementations are registered under ``"dsa_sparse_attention"``. The Pallas kernel carries its own
    ``custom_vjp`` (sparse forward, fused or two-pass sparse backward); the XLA reference differentiates natively.
    """

    def __init__(self):
        """Initialize the operation."""
        super().__init__(op_id="dsa_sparse_attention")

    def get_impl(self, cfg: DsaSparseAttentionConfig):
        """Resolve the registry implementation for ``cfg``'s platform.

        Args:
            cfg: Configuration selecting platform / backend.

        Returns:
            The registered implementation callable.
        """
        platform = detect_platform(self.op_id, cfg.platform, prefer_pallas=True)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_heads seq head_dim"],
        latent: Float[Array, "batch seq head_dim"],
        pool_mask: Bool[Array, "batch seq num_pools"],
        tails: Int[Array, "batch seq num_tails"] | None = None,
        ste_pool: Float[Array, "batch seq num_pools"] | None = None,
        *,
        kpool: int,
        softmax_scale: float | None = None,
        mode: str = "sparse",
        platform: _PlatformName | None = None,
        interpret: bool | None = None,
        cfg: DsaSparseAttentionConfig,
    ) -> Float[Array, "batch num_heads seq head_dim"]:
        """Execute the attention with the selected implementation.

        Args:
            query: ``[B, H, S, D]`` queries.
            latent: ``[B, S, D]`` shared ``K == V`` latent.
            pool_mask: ``[B, S, P]`` bool selected pools.
            tails: ``[B, S, T]`` int32 tail keys (``-1`` = none) or ``None``.
            ste_pool: ``[B, S, P]`` fp32 pooled straight-through term or ``None``.
            kpool: Keys per pool.
            softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
            mode: Visit mode (``"sparse"``, ``"dense"`` or ``"budget:B"``).
            platform: Optional platform override.
            interpret: Pallas interpret mode (``None``: off-TPU only).
            cfg: Kernel configuration.

        Returns:
            ``[B, H, S, D]`` in ``latent``'s dtype.
        """
        if platform is not None:
            cfg = DsaSparseAttentionConfig(
                **{k: v for k, v in cfg.to_dict().items() if k not in ("platform", "backend")},
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query,
            latent,
            pool_mask,
            tails,
            ste_pool,
            kpool=int(kpool),
            softmax_scale=softmax_scale,
            mode=mode,
            interpret=interpret,
            **_cfg_kwargs(cfg),
        )

    def heuristic_cfg(self, inv: Invocation[DsaSparseAttentionConfig, Array]) -> DsaSparseAttentionConfig:
        """Return the measured-best defaults (256 x 512 tiles, fused backward)."""
        return DsaSparseAttentionConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[DsaSparseAttentionConfig, Array]):
        """Return the deterministic default (the options are explicit cost choices, not autotuned)."""
        return [self.heuristic_cfg(inv)]

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_heads seq head_dim"],
        latent: Float[Array, "batch seq head_dim"],
        pool_mask: Bool[Array, "batch seq num_pools"],
        tails: Int[Array, "batch seq num_tails"] | None = None,
        ste_pool: Float[Array, "batch seq num_pools"] | None = None,
        *,
        kpool: int,
        softmax_scale: float | None = None,
        mode: str = "sparse",
        platform: _PlatformName | None = None,
        interpret: bool | None = None,
        cfg: DsaSparseAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a batch / head-parallel ``shard_map`` wrapper (the Mosaic call cannot be auto-partitioned).

        ``in_specs`` holds one spec per GIVEN array operand in the order ``(query, latent, pool_mask[, tails]
        [, ste_pool])``. The sequence axis must stay unsharded (the tables span the whole sequence).

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: When ``in_specs`` does not match the given operands.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        extras = [(name, value) for name, value in (("tails", tails), ("ste_pool", ste_pool)) if value is not None]
        if len(in_specs) != 3 + len(extras):
            raise ValueError(
                f"in_specs must have {3 + len(extras)} entries (optional operands given: {[n for n, _ in extras]})."
            )
        extra_names = tuple(name for name, _ in extras)
        chosen = cfg or self.heuristic_cfg(None)

        def _wrapper(query, latent, pool_mask, *rest):
            """Shard-map body delegating to ``run`` with the captured static options."""
            return self.run(
                query,
                latent,
                pool_mask,
                kpool=kpool,
                softmax_scale=softmax_scale,
                mode=mode,
                platform=platform,
                interpret=interpret,
                cfg=chosen,
                **dict(zip(extra_names, rest, strict=True)),
            )

        shard_map_fn = shard_map(_wrapper, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_vma)
        return shard_map_fn, (query, latent, pool_mask, *(value for _, value in extras))


_dsa_sparse_attention_executor: Executor[DsaSparseAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("dsa-sparse-attention", cfg_type=DsaSparseAttentionConfig),
    )
)


def dsa_sparse_attention(
    query: Float[Array, "batch num_heads seq head_dim"],
    latent: Float[Array, "batch seq head_dim"],
    pool_mask: Bool[Array, "batch seq num_pools"],
    tails: Int[Array, "batch seq num_tails"] | None = None,
    ste_pool: Float[Array, "batch seq num_pools"] | None = None,
    /,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    mode: str = "sparse",
    platform: _PlatformName | None = None,
    cfg: DsaSparseAttentionConfig | None = None,
    interpret: bool | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch num_heads seq head_dim"]:
    """Differentiable block-sparse attention over a DSA pool selection.

    Computes, per batch row ``b``, head ``h`` and query ``s``, softmax attention of ``query[b, h, s]`` over the
    open keys of ``latent[b]`` -- the keys of the selected pools plus the tail keys (set semantics) -- with
    ``ste_pool`` broadcast from pools to keys as an additive logit term. A query with no open key yields 0.
    ``jax.grad`` flows to ``query``, ``latent`` and ``ste_pool``.

    Args:
        query: ``[B, H, S, D]`` queries.
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        pool_mask: ``[B, S, P]`` bool selected pools (causal by the indexer's construction).
        tails: ``[B, S, T]`` int32 always-selected tail keys (``-1`` = none, ``T <= 8``) or ``None``.
        ste_pool: ``[B, S, P]`` fp32 pooled straight-through term (value 0) or ``None``.
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        mode: ``"sparse"`` (default), ``"dense"`` (every causal pair, bitwise ``"sparse"``) or ``"budget:B"``
            (the ``B`` heaviest kv-blocks per q-block; the rest are dropped -- a numerics change).
        platform: Optional platform override (``"pallas"`` / ``"xla"``).
        cfg: Optional :class:`~.configs.DsaSparseAttentionConfig` (tiles and per-pair cost variants).
        interpret: Pallas interpret mode (``None``: off-TPU only).
        mesh: Optional mesh for ``shard_map`` execution.
        in_specs: Specs for ``(query, latent, pool_mask[, tails][, ste_pool])`` (given operands only).
        out_specs: Output spec.

    Returns:
        ``[B, H, S, D]`` in ``latent``'s dtype.
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _dsa_sparse_attention_executor(
        DsaSparseAttention(),
        query,
        latent,
        pool_mask,
        tails,
        ste_pool,
        kpool=int(kpool),
        softmax_scale=softmax_scale,
        mode=mode,
        platform=platform,
        interpret=interpret,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

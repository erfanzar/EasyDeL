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

"""Per-query gather DSA attention module.

Attention of absorbed-MLA queries ``[B, N, H, D]`` (query-major) over a shared ``K == V`` latent ``[B, S, D]``, where
each query attends exactly to its selected pools (``indices [B, N, K]``, pool ``p`` = keys ``p*kpool ..
p*kpool + kpool - 1``) plus its own incomplete pool up to itself. The Pallas TPU kernel DMA-gathers only those pools, so
its cost is ``O(N * K * kpool * H * D)`` independent of the sequence length and of how the selection is spread; the
block-sparse ``dsa_sparse_attention`` instead visits every tile the selection touches. Queries may be a contiguous
shard of the sequence (``positions``), which is how a sequence-parallel caller splits the work.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import DsaGatherAttentionConfig

_PlatformName = Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"]


def _cfg_kwargs(cfg: DsaGatherAttentionConfig) -> dict:
    """Kernel keyword arguments carried by the config."""
    return dict(
        block_q=int(cfg.block_q),
        pools_per_step=None if cfg.pools_per_step is None else int(cfg.pools_per_step),
        num_buffers=None if cfg.num_buffers is None else int(cfg.num_buffers),
        num_cores=None if cfg.num_cores is None else int(cfg.num_cores),
        fused_backward=None if cfg.fused_backward is None else bool(cfg.fused_backward),
    )


class DsaGatherAttention(Kernel[DsaGatherAttentionConfig, Array]):
    """Per-query gather DSA attention (Pallas TPU kernel, XLA reference).

    Both implementations are registered under ``"dsa_gather_attention"``. The Pallas kernel carries its own
    ``custom_vjp``; the XLA reference differentiates natively.
    """

    def __init__(self):
        """Initialize the operation."""
        super().__init__(op_id="dsa_gather_attention")

    def get_impl(self, cfg: DsaGatherAttentionConfig):
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
        query: Float[Array, "batch seq num_heads head_dim"],
        latent: Float[Array, "batch kv_seq head_dim"],
        indices: Int[Array, "batch seq num_selected"],
        positions: Int[Array, "batch seq"] | None = None,
        *,
        kpool: int,
        softmax_scale: float | None = None,
        include_tail: bool = True,
        platform: _PlatformName | None = None,
        interpret: bool | None = None,
        cfg: DsaGatherAttentionConfig,
    ) -> Float[Array, "batch seq num_heads head_dim"]:
        """Execute the attention with the selected implementation.

        Args:
            query: ``[B, N, H, D]`` queries.
            latent: ``[B, S, D]`` shared ``K == V`` latent.
            indices: ``[B, N, K]`` selected pool ids (``-1`` = none).
            positions: ``[B, N]`` absolute query positions (negative = invalid) or ``None`` (``0 .. N - 1``).
            kpool: Keys per pool.
            softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
            include_tail: Open each query's own incomplete pool.
            platform: Optional platform override.
            interpret: Pallas interpret mode (``None``: off-TPU only).
            cfg: Kernel configuration.

        Returns:
            ``[B, N, H, D]`` in ``latent``'s dtype.
        """
        if platform is not None:
            cfg = DsaGatherAttentionConfig(
                **{k: v for k, v in cfg.to_dict().items() if k not in ("platform", "backend")},
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query,
            latent,
            indices,
            positions,
            kpool=int(kpool),
            softmax_scale=softmax_scale,
            include_tail=bool(include_tail),
            interpret=interpret,
            **_cfg_kwargs(cfg),
        )

    def heuristic_cfg(self, inv: Invocation[DsaGatherAttentionConfig, Array]) -> DsaGatherAttentionConfig:
        """Return the default config (every kernel option at its :class:`DsaGatherAttentionConfig` default)."""
        return DsaGatherAttentionConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[DsaGatherAttentionConfig, Array]):
        """Return the deterministic default (the options are explicit cost choices, not autotuned)."""
        return [self.heuristic_cfg(inv)]

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch seq num_heads head_dim"],
        latent: Float[Array, "batch kv_seq head_dim"],
        indices: Int[Array, "batch seq num_selected"],
        positions: Int[Array, "batch seq"] | None = None,
        *,
        kpool: int,
        softmax_scale: float | None = None,
        include_tail: bool = True,
        platform: _PlatformName | None = None,
        interpret: bool | None = None,
        cfg: DsaGatherAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper (the Mosaic call cannot be auto-partitioned).

        ``in_specs`` holds one spec per GIVEN array operand in the order ``(query, latent, indices[, positions])``.
        Queries may be split over batch, sequence and heads; the latent must hold every key a local query can select
        (in practice: the full sequence, replicated over the query-sequence axes). When the query sequence is split,
        pass ``positions`` (sharded like the queries) so each shard knows its absolute positions.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: When ``in_specs`` does not match the given operands.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        n_ops = 3 + (positions is not None)
        if len(in_specs) != n_ops:
            raise ValueError(f"in_specs must have {n_ops} entries (positions given: {positions is not None}).")
        chosen = cfg or self.heuristic_cfg(None)

        def _wrapper(query, latent, indices, *rest):
            """Shard-map body delegating to ``run`` with the captured static options."""
            return self.run(
                query,
                latent,
                indices,
                rest[0] if rest else None,
                kpool=kpool,
                softmax_scale=softmax_scale,
                include_tail=include_tail,
                platform=platform,
                interpret=interpret,
                cfg=chosen,
            )

        shard_map_fn = shard_map(_wrapper, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_vma)
        args = (query, latent, indices) if positions is None else (query, latent, indices, positions)
        return shard_map_fn, args


_dsa_gather_attention_executor: Executor[DsaGatherAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("dsa-gather-attention", cfg_type=DsaGatherAttentionConfig),
    )
)


def dsa_gather_attention(
    query: Float[Array, "batch seq num_heads head_dim"],
    latent: Float[Array, "batch kv_seq head_dim"],
    indices: Int[Array, "batch seq num_selected"],
    positions: Int[Array, "batch seq"] | None = None,
    /,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    include_tail: bool = True,
    platform: _PlatformName | None = None,
    cfg: DsaGatherAttentionConfig | None = None,
    interpret: bool | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch seq num_heads head_dim"]:
    """Differentiable attention of each query over its selected pools plus its own-pool tail.

    Query ``n`` (absolute position ``pos = positions[b, n]``) attends over ``latent[b]`` to every key of every
    selected pool ``p = indices[b, n, k]`` that is complete and causal (``p * kpool + kpool - 1 <= pos``) and, with
    ``include_tail``, to its own incomplete pool's keys ``kpool * (pos // kpool) .. pos``. A query without an open key
    (or with a negative position) yields 0. ``jax.grad`` flows to ``query`` and ``latent``.

    Args:
        query: ``[B, N, H, D]`` queries (query-major).
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        indices: ``[B, N, K]`` int selected pool ids (``-1`` = none); valid ids must be distinct.
        positions: ``[B, N]`` absolute query positions; ``None`` = ``0 .. N - 1``.
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        include_tail: Open each query's own incomplete pool.
        platform: Optional platform override (``"pallas"`` / ``"xla"``).
        cfg: Optional :class:`~.configs.DsaGatherAttentionConfig`.
        interpret: Pallas interpret mode (``None``: off-TPU only).
        mesh: Optional mesh for ``shard_map`` execution.
        in_specs: Specs for ``(query, latent, indices[, positions])`` (given operands only).
        out_specs: Output spec.

    Returns:
        ``[B, N, H, D]`` in ``latent``'s dtype.
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _dsa_gather_attention_executor(
        DsaGatherAttention(),
        query,
        latent,
        indices,
        positions,
        kpool=int(kpool),
        softmax_scale=softmax_scale,
        include_tail=bool(include_tail),
        platform=platform,
        interpret=interpret,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )

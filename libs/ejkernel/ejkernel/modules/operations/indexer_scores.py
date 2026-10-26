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

"""Fused Lightning-indexer scoring (``indexer_scores``).

``indexer_scores(query, index_k, weights)`` returns the head-summed weighted relu scores
``scores[b, s, t] = sum_h weights[b, h, s] * relu(query[b, h, s] . index_k[b, t])`` as fp32 ``[B, S, T]``. The
Pallas TPU kernel never materialises the ``[B, H, S, T]`` fp32 logits (written and re-read once in the forward and
again in the backward by the XLA expression); the XLA implementation is that reference and runs everywhere.

A Mosaic custom call cannot be GSPMD-partitioned: on a multi-device mesh pass ``mesh`` / ``in_specs`` /
``out_specs`` and the executor runs the kernel under ``jax.shard_map``. The output is summed over heads and the head
dim, so those reduced dims may not be sharded; every kept output dim must carry exactly the mesh axes of the input
dims it comes from. The contract is checked statically before tracing because ``check_vma`` cannot be used with a
Pallas body (``pallas_call`` outputs carry no manual-axis type).
"""

from __future__ import annotations

import typing as tp
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, FwdParams, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import IndexerScoresConfig

_CFG_FIELDS = frozenset(getattr(IndexerScoresConfig, "__dataclass_fields__", ()))
_PRECISION_NAMES = {
    "default": "default",
    "fastest": "default",
    "bfloat16": "default",
    "high": "high",
    "tensorfloat32": "high",
    "highest": "highest",
    "float32": "highest",
}


def _normalize_cfg(cfg: IndexerScoresConfig | dict | None) -> IndexerScoresConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return IndexerScoresConfig()
    if isinstance(cfg, dict):
        return IndexerScoresConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


def _precision_name(precision: tp.Any) -> str:
    """Static, hashable precision name (``default`` / ``high`` / ``highest``) for cache keys and the backends."""
    if precision is None:
        return "default"
    if isinstance(precision, (tuple, list)):
        names = {_precision_name(p) for p in precision}
        if len(names) != 1:
            raise ValueError(f"indexer_scores: per-operand precision {precision!r} is not supported")
        return names.pop()
    name = getattr(precision, "name", precision)
    if isinstance(name, str) and name.lower() in _PRECISION_NAMES:
        return _PRECISION_NAMES[name.lower()]
    raise ValueError(f"precision must be None, 'default', 'high', 'highest' or a jax.lax.Precision, got {precision!r}")


def _entry_axes(entry: tp.Any) -> tuple[str, ...]:
    """Mesh axis names of one partition-spec entry (``None`` -> none, a name -> itself, a tuple -> all, in order)."""
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)
    if isinstance(entry, (tuple, list)):
        if not all(isinstance(a, str) for a in entry):
            raise ValueError(f"indexer_scores: partition-spec tuple {entry!r} may only contain mesh axis names")
        return tuple(entry)
    raise ValueError(f"indexer_scores: unsupported partition-spec entry {entry!r} (only mesh axis names or None)")


def _dim_axes(spec: PartitionSpec | None, dim: int) -> tuple[str, ...]:
    """Ordered mesh axes of one dim (``('fsdp', 'tp')`` and ``('tp', 'fsdp')`` are different layouts)."""
    entries = tuple(spec or ())
    return _entry_axes(entries[dim]) if dim < len(entries) else ()


def validate_shard_specs(in_specs: tp.Sequence[PartitionSpec | None], out_specs: PartitionSpec | None) -> None:
    """Per-dimension sharding contract of ``scores[b, s, t] = sum_h w[b, h, s] relu(q[b, h, s] . k[b, t])``.

    Reduced dims (the heads of ``query`` / ``weights``, the head dim of ``query`` / ``index_k``, the trailing dim of
    ``weights``) may not be sharded: every device would hold a partial sum. Every kept output dim must carry exactly
    the mesh axes of the input dims it comes from (batch <- all three operands, q_len <- query / weights, entries <-
    index_k), in the same order; anything else yields duplicated, partial or permuted rows.

    Args:
        in_specs: Specs of ``(query, index_k, weights)``.
        out_specs: Spec of the ``[batch, q_len, num_entries]`` output.

    Raises:
        ValueError: When the specs violate the contract.
    """
    in_specs = tuple(in_specs)
    if len(in_specs) != 3:
        raise ValueError(f"indexer_scores: in_specs must hold (query, index_k, weights) specs; got {len(in_specs)}")
    q_spec, k_spec, w_spec = in_specs
    reduced = {
        "query heads (dim 1)": _dim_axes(q_spec, 1),
        "weights heads (dim 1)": _dim_axes(w_spec, 1),
        "query head_dim (dim 3)": _dim_axes(q_spec, 3),
        "index_k head_dim (dim 2)": _dim_axes(k_spec, 2),
        "weights trailing dim (dim 3)": _dim_axes(w_spec, 3),
    }
    sharded_reduced = {name: list(axes) for name, axes in reduced.items() if axes}
    if sharded_reduced:
        raise ValueError(
            f"indexer_scores: reduced dims are sharded {sharded_reduced}; the scores are summed over heads and the "
            "head dim, so every device would hold a partial sum. Replicate those dims."
        )
    kept = (
        ("batch", _dim_axes(out_specs, 0), (_dim_axes(q_spec, 0), _dim_axes(k_spec, 0), _dim_axes(w_spec, 0))),
        ("q_len", _dim_axes(out_specs, 1), (_dim_axes(q_spec, 2), _dim_axes(w_spec, 2))),
        ("entries", _dim_axes(out_specs, 2), (_dim_axes(k_spec, 1),)),
    )
    for name, out_axes, in_axes in kept:
        if any(axes != out_axes for axes in in_axes):
            raise ValueError(
                f"indexer_scores: output dim '{name}' is sharded over {list(out_axes)} but the input dims it comes "
                f"from are sharded over {[list(a) for a in in_axes]}; they must match exactly, in the same axis "
                "order (otherwise rows are duplicated, partial, or permuted)."
            )


class IndexerScores(Kernel[IndexerScoresConfig, Array]):
    """Head-summed weighted-relu indexer scoring (Pallas TPU forward + backward; XLA reference/fallback)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="indexer_scores"``."""
        super().__init__(op_id="indexer_scores")

    def get_impl(self, cfg: IndexerScoresConfig):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``indexer_scores`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform, prefer_pallas=True)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_heads q_len head_dim"],
        index_k: Float[Array, "batch num_entries head_dim"],
        weights: Float[Array, "batch num_heads q_len 1"],
        *,
        precision: str = "default",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: IndexerScoresConfig,
        **_,
    ) -> Float[Array, "batch q_len num_entries"]:
        """Execute the selected backend.

        Args:
            query: Index queries ``[B, H, S, d]``.
            index_k: Index keys ``[B, T, d]``.
            weights: Per-head weights ``[B, H, S, 1]``.
            precision: Normalised precision name.
            platform: Optional platform override.
            cfg: Operation config (tiles, platform, backend).

        Returns:
            fp32 scores ``[B, S, T]``.
        """
        cfg = _normalize_cfg(cfg)
        if platform is not None:
            cfg = IndexerScoresConfig(
                block_q=cfg.block_q,
                block_t=cfg.block_t,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        fwd_params = FwdParams(q_blocksize=int(cfg.block_q), kv_blocksize=int(cfg.block_t))
        return impl(query, index_k, weights, fwd_params=fwd_params, precision=precision)

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_heads q_len head_dim"],
        index_k: Float[Array, "batch num_entries head_dim"],
        weights: Float[Array, "batch num_heads q_len 1"],
        *,
        precision: str = "default",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: IndexerScoresConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
        **_,
    ):
        """Build a ``shard_map`` wrapper after checking the per-dimension sharding contract.

        ``check_vma`` stays off: a Pallas body cannot run under ``check_vma=True``; :func:`validate_shard_specs`
        rejects the specs ``check_vma`` would otherwise have caught.

        Args:
            query: Index queries ``[B, H, S, d]``.
            index_k: Index keys ``[B, T, d]``.
            weights: Per-head weights ``[B, H, S, 1]``.
            precision: Normalised precision name.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh.
            in_specs: Specs of ``(query, index_k, weights)``.
            out_specs: Spec of the ``[B, S, T]`` output.
            check_vma: Ignored (always off for the Pallas body).

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the mesh or specs are missing or violate the contract.
        """
        del check_vma
        if mesh is None or in_specs is None or out_specs is None:
            raise ValueError("indexer_scores: mesh, in_specs and out_specs are required for shard_map execution")
        validate_shard_specs(in_specs, out_specs)
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(q, k, w):
            return self.run(q, k, w, precision=precision, platform=platform, cfg=run_cfg)

        shard_map_fn = shard_map(_wrapped, mesh=mesh, in_specs=tuple(in_specs), out_specs=out_specs, check_vma=False)
        return shard_map_fn, (query, index_k, weights)

    def heuristic_cfg(self, inv: Invocation[IndexerScoresConfig, Array]) -> IndexerScoresConfig:
        """Return the default tiles (fitted to VMEM by the kernel); the platform resolves per backend."""
        del inv
        return IndexerScoresConfig(block_q=128, block_t=1024, platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[IndexerScoresConfig, Array]):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_indexer_scores_executor: Executor[IndexerScoresConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=False, cache_miss_fallback="heuristics", validate_backward=False),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("indexer-scores", cfg_type=IndexerScoresConfig),
    )
)


def indexer_scores(
    query: Float[Array, "batch num_heads q_len head_dim"],
    index_k: Float[Array, "batch num_entries head_dim"],
    weights: Float[Array, "batch num_heads q_len 1"],
    /,
    *,
    precision: tp.Any = None,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: IndexerScoresConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch q_len num_entries"]:
    """``scores[b, s, t] = sum_h weights[b, h, s] * relu(query[b, h, s] . index_k[b, t])``, fp32 ``[B, S, T]``.

    Args:
        query: Index queries ``[batch, num_heads, q_len, head_dim]``.
        index_k: Index keys ``[batch, num_entries, head_dim]`` (cast to the query dtype for the contraction).
        weights: Per-head, per-query weights ``[batch, num_heads, q_len, 1]`` (applied in fp32).
        precision: ``None`` / ``"default"`` contracts in the operand dtype with fp32 accumulation; ``"high"`` /
            ``"highest"`` or a ``jax.lax.Precision`` contracts fp32 operands (both platforms).
        platform: Platform override (``"pallas"`` on TPU, ``"xla"`` anywhere; auto by default).
        cfg: Optional :class:`IndexerScoresConfig` (tiles; requests rounded and fitted to VMEM).
        mesh: Optional device mesh; with ``in_specs`` / ``out_specs`` the op runs under ``jax.shard_map``.
        in_specs: Specs of ``(query, index_k, weights)``.
        out_specs: Spec of the ``[batch, q_len, num_entries]`` output.

    Returns:
        fp32 ``[batch, q_len, num_entries]``, differentiable in all three operands.

    Raises:
        ValueError: For an unknown precision, a mesh without both spec sets, or specs violating
            :func:`validate_shard_specs`.
    """
    name = _precision_name(precision)
    if mesh is not None and (in_specs is None or out_specs is None):
        raise ValueError("indexer_scores: in_specs and out_specs are required when a mesh is given")
    method = "shard_map" if mesh is not None else None
    if method is not None:
        validate_shard_specs(in_specs, out_specs)
    return _indexer_scores_executor(
        IndexerScores(),
        query,
        index_k,
        weights,
        precision=name,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=None if in_specs is None else tuple(in_specs),
        out_specs=out_specs,
        _cfg=cfg,
    )


__all__ = ("IndexerScores", "indexer_scores", "validate_shard_specs")

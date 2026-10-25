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

"""Row permutation by contiguous runs (``row_permute``).

A run table ``runs [R, 3] int32`` of ``(dst_start, src_start, length)`` describes ``out[dst + j] = x[src + j]``
(zero-length runs are padding; destination ranges must not overlap; uncovered rows take ``init`` or zero). The XLA
implementation is one gather (reference and fallback); the Pallas TPU implementation streams every run through VMEM
with tile-aligned DMAs and differentiates along the inverse run table.

A Mosaic custom call cannot be GSPMD-partitioned: on a multi-device mesh pass ``mesh`` / ``in_specs`` /
``out_specs`` so the executor runs the kernel under ``jax.shard_map``. Only the feature (column) axis may be sharded;
the row axis is what the permutation moves.
"""

from __future__ import annotations

from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RowPermuteConfig

_CFG_FIELDS = frozenset(getattr(RowPermuteConfig, "__dataclass_fields__", ()))
DEFAULT_PLATFORM: Literal["pallas", "xla", "auto"] = "auto"


def _normalize_cfg(cfg: RowPermuteConfig | dict | None) -> RowPermuteConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return RowPermuteConfig(platform=DEFAULT_PLATFORM)
    if isinstance(cfg, dict):
        return RowPermuteConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


class RowPermute(Kernel[RowPermuteConfig, Array]):
    """Run-table row permutation (Pallas TPU streaming copy; XLA gather reference/fallback)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="row_permute"``."""
        super().__init__(op_id="row_permute")

    def get_impl(self, cfg: RowPermuteConfig):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``row_permute`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Array,
        runs: Int[Array, "num_runs 3"],
        inverse_runs: Int[Array, "num_runs 3"] | None = None,
        init: Array | None = None,
        *,
        out_rows: int | None = None,
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RowPermuteConfig,
        **_,
    ) -> Array:
        """Execute the selected backend.

        Args:
            x: ``[rows, D]`` source rows.
            runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``.
            inverse_runs: Optional inverse table for the Pallas backward (default: ``runs`` with dst/src swapped).
            init: Optional ``[out_rows, D]`` values for uncovered rows.
            out_rows: Destination rows.
            platform: Optional platform override.
            cfg: Operation config (``chunk_rows``).

        Returns:
            ``[out_rows, D]`` in ``x.dtype``.
        """
        cfg = _normalize_cfg(cfg)
        if platform is not None:
            cfg = RowPermuteConfig(
                chunk_rows=cfg.chunk_rows,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(x, runs, inverse_runs, init, out_rows=out_rows, chunk_rows=int(cfg.chunk_rows))

    def create_shard_map_wrapper(
        self,
        x: Array,
        runs: Int[Array, "num_runs 3"],
        inverse_runs: Int[Array, "num_runs 3"] | None = None,
        init: Array | None = None,
        *,
        out_rows: int | None = None,
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RowPermuteConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper over the feature (column) axis.

        ``in_specs`` has one entry per positional array ``(x, runs, inverse_runs, init)``; entries for ``None``
        arrays are ignored. The row axis of ``x`` / ``init`` must be unsharded and the run tables replicated.

        Args:
            x: ``[rows, D]`` source rows.
            runs: ``[R, 3]`` run table.
            inverse_runs: Optional inverse table.
            init: Optional ``[out_rows, D]`` buffer.
            out_rows: Destination rows.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh.
            in_specs: Four input specs (see above).
            out_specs: Output spec.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the spec count is wrong, the row axis is sharded, or a run table is sharded.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if len(in_specs) != 4:
            raise ValueError(f"in_specs must have 4 entries, got {len(in_specs)}")
        for i in (0, 3):
            spec = tuple(in_specs[i] or ())
            if spec and spec[0] is not None:
                raise ValueError(f"row_permute shard_map requires an unsharded row axis, got spec {spec}")
        for i in (1, 2):
            if any(a is not None for a in tuple(in_specs[i] or ())):
                raise ValueError(f"row_permute shard_map requires replicated run tables, got spec {in_specs[i]}")
        out_spec = tuple(out_specs)
        if out_spec and out_spec[0] is not None:
            raise ValueError(f"row_permute shard_map requires an unsharded output row axis, got {out_spec}")
        arrays = (x, runs, inverse_runs, init)
        present = tuple(i for i, a in enumerate(arrays) if a is not None)
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(*local):
            full = [None] * 4
            for i, a in zip(present, local, strict=True):
                full[i] = a
            return self.run(*full, out_rows=out_rows, platform=platform, cfg=run_cfg)

        shard_map_fn = shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=tuple(in_specs[i] for i in present),
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, tuple(arrays[i] for i in present)

    def heuristic_cfg(self, inv: Invocation[RowPermuteConfig, Array]) -> RowPermuteConfig:
        """Return the measured default config (128-row windows; platform per :data:`DEFAULT_PLATFORM`).

        On TPU v5 at ``[262144, 4096]`` bf16 with 289 runs, the Pallas kernel at 128-row windows measured 7.11 ms
        per pass against 7.55 ms for the XLA gather, and 15.9 ms against 33.3 ms for forward + VJP, so ``"auto"``
        (Pallas on TPU) is the default.
        """
        return RowPermuteConfig(chunk_rows=128, platform=DEFAULT_PLATFORM, backend="any")

    def candidate_cfgs(self, inv: Invocation[RowPermuteConfig, Array]):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_row_permute_executor: Executor[RowPermuteConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback="heuristics",
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("row-permute", cfg_type=RowPermuteConfig),
    )
)


def row_permute(
    x: Array,
    runs: Int[Array, "num_runs 3"],
    inverse_runs: Int[Array, "num_runs 3"] | None = None,
    init: Array | None = None,
    /,
    *,
    out_rows: int | None = None,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RowPermuteConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Array:
    """``out[dst + j] = x[src + j]`` for every run ``(dst, src, length)`` through the ejkernel executor.

    Args:
        x: ``[rows, D]`` source rows. The Pallas kernel needs ``rows`` and ``out_rows`` multiples of 16 (at least
            32), ``D`` a multiple of 128 and a 4-byte or bfloat16 dtype; anything else runs the XLA gather.
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)`` in any order; zero-length runs are padding;
            destination ranges must not overlap.
        inverse_runs: Optional inverse table for the Pallas backward (default: ``runs`` with dst/src swapped). The
            Pallas VJP equals the gather's when no source row is read by two runs.
        init: Optional ``[out_rows, D]`` values for uncovered rows (zeros when ``None``).
        out_rows: Destination rows (default ``x.shape[0]`` or ``init.shape[0]``).
        platform: Optional platform override (``"pallas"`` / ``"xla"``).
        cfg: Optional :class:`RowPermuteConfig`.
        mesh: Optional mesh for ``shard_map`` execution (feature axis only).
        in_specs: Four input specs ``(x, runs, inverse_runs, init)``.
        out_specs: Output spec.

    Returns:
        ``[out_rows, D]`` in ``x.dtype``.
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _row_permute_executor(
        RowPermute(),
        x,
        runs,
        inverse_runs,
        init,
        out_rows=out_rows,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )


__all__ = ("RowPermute", "row_permute")

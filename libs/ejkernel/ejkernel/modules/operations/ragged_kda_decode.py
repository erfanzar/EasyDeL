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

"""Per-row KDA decode step over a recurrent-state pool (``ragged_kda_decode``).

Decode form of :mod:`.ragged_kda`: row ``r`` carries one token and owns pool slot ``state_indices[r]`` (slot ``r``
when ``state_indices`` is ``None``). Rows flagged ``valid`` advance their ``[H, K, V]`` state by one per-channel KDA
step::

    S = diag(exp(g)) S;   S += k (beta (v - k^T S))^T;   o = S^T q

(``q`` scaled by ``K**-0.5``, ``q`` / ``k`` optionally L2-normalised); every other row outputs zeros and every other
slot keeps its state bit for bit. On TPU a Pallas kernel reads and writes each row's slot once, in place; the XLA
implementation is the reference and the fallback. The step is independent per head, so on a mesh the executor runs
it under ``jax.shard_map`` over the head axis (no collectives); use :func:`ragged_kda_decode_head_sharded`.
"""

from __future__ import annotations

from typing import Literal

import jax
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.modules.base import mesh_to_jax_mesh
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedKDADecodeConfig
from .ragged_causal_conv1d import mesh_axis_size

_CFG_FIELDS = frozenset(getattr(RaggedKDADecodeConfig, "__dataclass_fields__", ()))
_NUM_ARRAYS = 9  # query, key, value, beta, decay, recurrent_state, state_indices, valid, has_initial_state


def _normalize_cfg(cfg: RaggedKDADecodeConfig | dict | None) -> RaggedKDADecodeConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return RaggedKDADecodeConfig()
    if isinstance(cfg, dict):
        return RaggedKDADecodeConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


class RaggedKDADecode(Kernel[RaggedKDADecodeConfig, tuple[Array, Array]]):
    """Per-row per-channel-decay KDA decode step over a recurrent-state pool (Pallas TPU; XLA reference)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="ragged_kda_decode"``."""
        super().__init__(op_id="ragged_kda_decode")

    def get_impl(self, cfg: RaggedKDADecodeConfig):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``ragged_kda_decode`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "num_rows num_heads head_dim"],
        key: Float[Array, "num_rows num_heads head_dim"],
        value: Float[Array, "num_rows num_heads v_dim"],
        beta: Float[Array, "num_rows num_heads"],
        decay: Float[Array, "num_rows num_heads head_dim"],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        state_indices: Int[Array, "num_rows"] | None,
        valid: Bool[Array, "num_rows"],
        has_initial_state: Bool[Array, "num_rows"] | None = None,
        *,
        use_qk_l2norm: bool = True,
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDADecodeConfig,
        **_,
    ) -> tuple[Array, Array]:
        """Execute the selected backend.

        Args:
            query: Post-conv queries ``[R, H, K]``.
            key: Post-conv keys ``[R, H, K]``.
            value: Post-conv values ``[R, H, V]``.
            beta: Write strengths ``[R, H]``.
            decay: Per-channel log-decay ``[R, H, K]``.
            recurrent_state: State pool ``[S, H, K, V]``.
            state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
            valid: ``[R]`` rows that advance.
            has_initial_state: Optional ``[R]`` flags (``False`` rows start from zero).
            use_qk_l2norm: L2-normalise queries and keys.
            platform: Optional platform override.
            cfg: Operation config (kernel tiling).

        Returns:
            ``(out [R, H, V] fp32, new_recurrent_state [S, H, K, V])``.
        """
        cfg = _normalize_cfg(cfg)
        if platform is not None:
            cfg = RaggedKDADecodeConfig(
                block_rows=cfg.block_rows,
                block_h=cfg.block_h,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            state_indices,
            valid,
            has_initial_state,
            use_qk_l2norm=use_qk_l2norm,
            block_rows=int(cfg.block_rows),
            block_h=int(cfg.block_h),
        )

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "num_rows num_heads head_dim"],
        key: Float[Array, "num_rows num_heads head_dim"],
        value: Float[Array, "num_rows num_heads v_dim"],
        beta: Float[Array, "num_rows num_heads"],
        decay: Float[Array, "num_rows num_heads head_dim"],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        state_indices: Int[Array, "num_rows"] | None,
        valid: Bool[Array, "num_rows"],
        has_initial_state: Bool[Array, "num_rows"] | None = None,
        *,
        use_qk_l2norm: bool = True,
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDADecodeConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper over the head axis.

        ``in_specs`` has one entry per positional array ``(query, key, value, beta, decay, recurrent_state,
        state_indices, valid, has_initial_state)``; the entry of a ``None`` array is ignored. Only the head axis may be
        sharded: every row metadata array must be replicated.

        Args:
            query: Post-conv queries ``[R, H, K]``.
            key: Post-conv keys ``[R, H, K]``.
            value: Post-conv values ``[R, H, V]``.
            beta: Write strengths ``[R, H]``.
            decay: Per-channel log-decay ``[R, H, K]``.
            recurrent_state: State pool ``[S, H, K, V]``.
            state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
            valid: ``[R]`` rows that advance.
            has_initial_state: Optional ``[R]`` flags.
            use_qk_l2norm: L2-normalise queries and keys.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh (a SpectraX mesh is unwrapped).
            in_specs: Nine input specs (see above).
            out_specs: Output specs ``(out, new_recurrent_state)``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the spec count is wrong or the row / feature axes of ``query`` are sharded.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if len(in_specs) != _NUM_ARRAYS:
            raise ValueError(f"in_specs must have {_NUM_ARRAYS} entries, got {len(in_specs)}")
        q_spec = tuple(in_specs[0]) + (None,) * (3 - len(tuple(in_specs[0])))
        if q_spec[0] is not None or q_spec[2] is not None:
            raise ValueError(f"ragged_kda_decode shard_map requires unsharded row/feature axes, got query spec {q_spec}")
        arrays = (query, key, value, beta, decay, recurrent_state, state_indices, valid, has_initial_state)
        present = tuple(i for i, a in enumerate(arrays) if a is not None)
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(*local):
            full = [None] * _NUM_ARRAYS
            for i, a in zip(present, local, strict=True):
                full[i] = a
            return self.run(*full, use_qk_l2norm=use_qk_l2norm, platform=platform, cfg=run_cfg)

        shard_map_fn = jax.shard_map(
            _wrapped,
            mesh=mesh_to_jax_mesh(mesh),
            in_specs=tuple(in_specs[i] for i in present),
            out_specs=tuple(out_specs),
            check_vma=check_vma,
        )
        return shard_map_fn, tuple(arrays[i] for i in present)

    def heuristic_cfg(self, inv: Invocation[RaggedKDADecodeConfig, tuple[Array, Array]]) -> RaggedKDADecodeConfig:
        """Return the default config: automatic kernel tiling and platform."""
        return RaggedKDADecodeConfig(block_rows=0, block_h=0, platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedKDADecodeConfig, tuple[Array, Array]]):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_ragged_kda_decode_executor: Executor[RaggedKDADecodeConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback="heuristics",
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("ragged-kda-decode", cfg_type=RaggedKDADecodeConfig),
    )
)


def ragged_kda_decode(
    query: Float[Array, "num_rows num_heads head_dim"],
    key: Float[Array, "num_rows num_heads head_dim"],
    value: Float[Array, "num_rows num_heads v_dim"],
    beta: Float[Array, "num_rows num_heads"],
    decay: Float[Array, "num_rows num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    state_indices: Int[Array, "num_rows"] | None,
    valid: Bool[Array, "num_rows"],
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    /,
    *,
    use_qk_l2norm: bool = True,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDADecodeConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> tuple[Array, Array]:
    """One per-channel KDA decode token per row over a recurrent-state pool, through the ejkernel executor.

    Row ``r`` owns pool slot ``state_indices[r]`` (slot ``r`` without ``state_indices``). Rows with ``valid[r]`` and an
    in-range slot advance by ``S = diag(exp(g)) S + k (beta (v - k^T diag(exp(g)) S))^T`` and output ``S^T q``
    (queries scaled by ``K**-0.5``, optionally L2-normalised with the keys); every other row outputs zeros and every
    other slot keeps its state bit for bit.

    Args:
        query: Post-conv queries ``[R, H, K]``.
        key: Post-conv keys ``[R, H, K]``.
        value: Post-conv values ``[R, H, V]``.
        beta: Write strengths ``[R, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[R, H, K]`` (``<= 0``), ideally fp32.
        recurrent_state: State pool ``[S, H, K, V]`` (the step always runs in fp32; the pool keeps its dtype).
        state_indices: Pool slot of every row ``[R]`` (out-of-range: the row does not advance), or ``None`` (row
            ``r`` owns slot ``r``, ``S >= R``; the TPU kernel then updates those slots in place).
        valid: ``[R]`` rows that advance (rows with one token this step).
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state instead of their slot.
        use_qk_l2norm: L2-normalise queries and keys first.
        platform: Optional platform override (``"xla"`` / ``"pallas"``).
        cfg: Optional :class:`RaggedKDADecodeConfig`.
        mesh: Optional mesh for ``shard_map`` execution over the head axis.
        in_specs: Nine input specs (see :meth:`RaggedKDADecode.create_shard_map_wrapper`).
        out_specs: Output specs ``(out, new_recurrent_state)``.

    Returns:
        ``(out [R, H, V] fp32, new_recurrent_state [S, H, K, V])``.
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _ragged_kda_decode_executor(
        RaggedKDADecode(),
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        state_indices,
        valid,
        has_initial_state,
        use_qk_l2norm=use_qk_l2norm,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )


def ragged_kda_decode_head_sharded(
    query: Float[Array, "num_rows num_heads head_dim"],
    key: Float[Array, "num_rows num_heads head_dim"],
    value: Float[Array, "num_rows num_heads v_dim"],
    beta: Float[Array, "num_rows num_heads"],
    decay: Float[Array, "num_rows num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    state_indices: Int[Array, "num_rows"] | None,
    valid: Bool[Array, "num_rows"],
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    /,
    *,
    mesh: object | None,
    head_axis: object | None,
    use_qk_l2norm: bool = True,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDADecodeConfig | None = None,
) -> tuple[Array, Array]:
    """Run :func:`ragged_kda_decode` with the heads sharded over ``head_axis`` of ``mesh``.

    With a mesh whose ``head_axis`` has more than one device, the executor runs the operation under ``jax.shard_map``:
    every device advances its own heads of every row (rows, row metadata and pool slots replicated, heads split), with
    no collectives. Otherwise the call is unsharded.

    Args:
        query: Post-conv queries ``[R, H, K]``.
        key: Post-conv keys ``[R, H, K]``.
        value: Post-conv values ``[R, H, V]``.
        beta: Write strengths ``[R, H]``.
        decay: Per-channel log-decay ``[R, H, K]``.
        recurrent_state: State pool ``[S, H, K, V]``.
        state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
        valid: ``[R]`` rows that advance.
        has_initial_state: Optional ``[R]`` flags.
        mesh: Mesh carrying ``head_axis`` (or ``None``).
        head_axis: Mesh axis name (or tuple of names) the heads are sharded over (or ``None``).
        use_qk_l2norm: L2-normalise queries and keys.
        platform: Optional platform override.
        cfg: Optional :class:`RaggedKDADecodeConfig`.

    Returns:
        ``(out [R, H, V] fp32, new_recurrent_state [S, H, K, V])``.

    Raises:
        ValueError: If the head count is not divisible by the size of ``head_axis``.
    """
    head_shards = mesh_axis_size(mesh, head_axis)
    kwargs = dict(use_qk_l2norm=use_qk_l2norm, platform=platform, cfg=cfg)
    arrays = (query, key, value, beta, decay, recurrent_state, state_indices, valid, has_initial_state)
    if mesh is None or head_axis is None or head_shards <= 1:
        return ragged_kda_decode(*arrays, **kwargs)
    num_heads = int(query.shape[1])
    if num_heads % head_shards:
        raise ValueError(
            f"ragged_kda_decode: {num_heads} heads are not divisible by the {head_shards}-way axis {head_axis}"
        )
    row_head = PartitionSpec(None, head_axis, None)
    replicated = PartitionSpec()
    state_spec = PartitionSpec(None, head_axis, None, None)
    in_specs = (
        row_head,
        row_head,
        row_head,
        PartitionSpec(None, head_axis),
        row_head,
        state_spec,
        replicated,
        replicated,
        replicated,
    )
    return ragged_kda_decode(*arrays, mesh=mesh, in_specs=in_specs, out_specs=(row_head, state_spec), **kwargs)


__all__ = ("RaggedKDADecode", "ragged_kda_decode", "ragged_kda_decode_head_sharded")

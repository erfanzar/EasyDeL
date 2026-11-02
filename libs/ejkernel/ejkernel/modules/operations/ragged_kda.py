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

"""Ragged (flat-token) KDA with per-channel decay over a recurrent-state pool (``ragged_kda``).

Serving form of the GLM-5-Next / Kimi-Linear KDA recurrence: the tokens of all scheduled requests share one flat
buffer ``[T, H, D]``, ``query_start_loc`` delimits each request and ``state_indices`` names the pool slot that holds
its ``[H, K, V]`` fp32 state. Single-token rows take one vectorised decode step; longer rows run the chunked KDA math
on their own live chunks only, so the cost follows the real token count instead of ``rows * bucket``.

The XLA implementation is the reference and the fallback. The recurrence is independent per head, so on a mesh the
executor runs it under ``jax.shard_map`` over the head axis (no collectives); use :func:`ragged_kda_head_sharded`.
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
from .configs import RaggedKDAConfig
from .ragged_causal_conv1d import mesh_axis_size

_CFG_FIELDS = frozenset(getattr(RaggedKDAConfig, "__dataclass_fields__", ()))
_NUM_ARRAYS = 9  # query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices, has_initial_state


def _normalize_cfg(cfg: RaggedKDAConfig | dict | None) -> RaggedKDAConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return RaggedKDAConfig()
    if isinstance(cfg, dict):
        return RaggedKDAConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


class RaggedKDA(Kernel[RaggedKDAConfig, tuple[Array, Array]]):
    """Ragged per-channel-decay KDA over a recurrent-state pool (XLA reference; accelerator backends optional)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="ragged_kda"``."""
        super().__init__(op_id="ragged_kda")

    def get_impl(self, cfg: RaggedKDAConfig):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``ragged_kda`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "num_tokens num_heads head_dim"],
        key: Float[Array, "num_tokens num_heads head_dim"],
        value: Float[Array, "num_tokens num_heads v_dim"],
        beta: Float[Array, "num_tokens num_heads"],
        decay: Float[Array, "num_tokens num_heads head_dim"],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        query_start_loc: Int[Array, "num_rows_plus_1"],
        state_indices: Int[Array, "num_rows"] | None,
        has_initial_state: Bool[Array, "num_rows"] | None = None,
        *,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        decay_lower_bound: float | None = None,
        precision: str = "highest",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDAConfig,
        **_,
    ) -> tuple[Array, Array]:
        """Execute the selected backend.

        Args:
            query: Post-conv queries ``[T, H, K]``.
            key: Post-conv keys ``[T, H, K]``.
            value: Post-conv values ``[T, H, V]``.
            beta: Write strengths ``[T, H]``.
            decay: Per-channel log-decay ``[T, H, K]``.
            recurrent_state: State pool ``[S, H, K, V]``.
            query_start_loc: Cumulative row offsets ``[R + 1]``.
            state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
            has_initial_state: Optional ``[R]`` flags (``False`` rows start from zero).
            chunk_size: Chunk length of the multi-token path.
            use_qk_l2norm: L2-normalise queries and keys.
            decay_lower_bound: Promised lower bound of every log-decay, or ``None``.
            precision: Precision of the non-inverse chunk dots.
            platform: Optional platform override.
            cfg: Operation config (inverse, pairwise).

        Returns:
            ``(out [T, H, V] fp32, new_recurrent_state [S, H, K, V])``.
        """
        cfg = _normalize_cfg(cfg)
        if platform is not None:
            cfg = RaggedKDAConfig(
                inverse=cfg.inverse,
                pairwise=cfg.pairwise,
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
            query_start_loc,
            state_indices,
            has_initial_state,
            chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm,
            decay_lower_bound=decay_lower_bound,
            precision=precision,
            inverse=cfg.inverse,
            pairwise=cfg.pairwise,
        )

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "num_tokens num_heads head_dim"],
        key: Float[Array, "num_tokens num_heads head_dim"],
        value: Float[Array, "num_tokens num_heads v_dim"],
        beta: Float[Array, "num_tokens num_heads"],
        decay: Float[Array, "num_tokens num_heads head_dim"],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        query_start_loc: Int[Array, "num_rows_plus_1"],
        state_indices: Int[Array, "num_rows"] | None,
        has_initial_state: Bool[Array, "num_rows"] | None = None,
        *,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        decay_lower_bound: float | None = None,
        precision: str = "highest",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDAConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper over the head axis.

        ``in_specs`` has one entry per positional array ``(query, key, value, beta, decay, recurrent_state,
        query_start_loc, state_indices, has_initial_state)``; the entry of a ``None`` array is ignored. Only the head
        axis may be sharded: the token axis carries the recurrence and every row metadata array must be replicated.

        Args:
            query: Post-conv queries ``[T, H, K]``.
            key: Post-conv keys ``[T, H, K]``.
            value: Post-conv values ``[T, H, V]``.
            beta: Write strengths ``[T, H]``.
            decay: Per-channel log-decay ``[T, H, K]``.
            recurrent_state: State pool ``[S, H, K, V]``.
            query_start_loc: Cumulative row offsets ``[R + 1]``.
            state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
            has_initial_state: Optional ``[R]`` flags.
            chunk_size: Chunk length of the multi-token path.
            use_qk_l2norm: L2-normalise queries and keys.
            decay_lower_bound: Promised lower bound of every log-decay, or ``None``.
            precision: Precision of the non-inverse chunk dots.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh (a SpectraX mesh is unwrapped).
            in_specs: Nine input specs (see above).
            out_specs: Output specs ``(out, new_recurrent_state)``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the spec count is wrong or the token / feature axes of ``query`` are sharded.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if len(in_specs) != _NUM_ARRAYS:
            raise ValueError(f"in_specs must have {_NUM_ARRAYS} entries, got {len(in_specs)}")
        q_spec = tuple(in_specs[0]) + (None,) * (3 - len(tuple(in_specs[0])))
        if q_spec[0] is not None or q_spec[2] is not None:
            raise ValueError(f"ragged_kda shard_map requires unsharded token/feature axes, got query spec {q_spec}")
        arrays = (
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            has_initial_state,
        )
        present = tuple(i for i, a in enumerate(arrays) if a is not None)
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(*local):
            full = [None] * _NUM_ARRAYS
            for i, a in zip(present, local, strict=True):
                full[i] = a
            return self.run(
                *full,
                chunk_size=chunk_size,
                use_qk_l2norm=use_qk_l2norm,
                decay_lower_bound=decay_lower_bound,
                precision=precision,
                platform=platform,
                cfg=run_cfg,
            )

        shard_map_fn = jax.shard_map(
            _wrapped,
            mesh=mesh_to_jax_mesh(mesh),
            in_specs=tuple(in_specs[i] for i in present),
            out_specs=tuple(out_specs),
            check_vma=check_vma,
        )
        return shard_map_fn, tuple(arrays[i] for i in present)

    def heuristic_cfg(self, inv: Invocation[RaggedKDAConfig, tuple[Array, Array]]) -> RaggedKDAConfig:
        """Return the default config: automatic chunk inverse and pairwise form, automatic platform."""
        return RaggedKDAConfig(inverse="auto", pairwise="auto", platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedKDAConfig, tuple[Array, Array]]):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_ragged_kda_executor: Executor[RaggedKDAConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback="heuristics",
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("ragged-kda", cfg_type=RaggedKDAConfig),
    )
)


def ragged_kda(
    query: Float[Array, "num_tokens num_heads head_dim"],
    key: Float[Array, "num_tokens num_heads head_dim"],
    value: Float[Array, "num_tokens num_heads v_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    query_start_loc: Int[Array, "num_rows_plus_1"],
    state_indices: Int[Array, "num_rows"] | None,
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    /,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDAConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> tuple[Array, Array]:
    """Ragged per-channel-decay KDA over a recurrent-state pool, through the ejkernel executor.

    Request ``r`` owns tokens ``[query_start_loc[r], query_start_loc[r + 1])`` of the flat stream and the state in
    pool slot ``state_indices[r]``. Per request and head, ``S_t = diag(exp(g_t)) S_{t-1} + k_t (beta_t (v_t - k_t^T
    diag(exp(g_t)) S_{t-1}))^T`` and ``o_t = S_t^T q_t`` (queries scaled by ``K**-0.5``, optionally L2-normalised
    with the keys). Tokens at or past ``query_start_loc[R]`` are padding (zero output, no state change); rows with no
    tokens keep their slot unchanged.

    Args:
        query: Post-conv queries ``[T, H, K]``.
        key: Post-conv keys ``[T, H, K]``.
        value: Post-conv values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``), ideally fp32.
        recurrent_state: State pool ``[S, H, K, V]`` (fp32 recommended; the recurrence always runs in fp32).
        query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
        state_indices: Pool slot of every row ``[R]``; out-of-range slots mark padding rows. ``None`` is the
            identity layout (row ``r`` owns slot ``r``, ``S >= R``): single-token rows then update their slots in
            place instead of gathering and scattering pool-sized buffers.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state instead of their slot.
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys first.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound`` (e.g. a safe-gate bound). With
            ``cfg.pairwise="auto"`` a tight enough bound selects the factored MXU pairwise decay. ``None`` keeps the
            dense form. Values below the bound are a contract violation (possible overflow).
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        platform: Optional platform override (``"xla"`` / ``"pallas"``).
        cfg: Optional :class:`RaggedKDAConfig`.
        mesh: Optional mesh for ``shard_map`` execution over the head axis.
        in_specs: Nine input specs (see :meth:`RaggedKDA.create_shard_map_wrapper`).
        out_specs: Output specs ``(out, new_recurrent_state)``.

    Returns:
        ``(out [T, H, V] fp32, new_recurrent_state [S, H, K, V])`` (the pool keeps its dtype).
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _ragged_kda_executor(
        RaggedKDA(),
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        query_start_loc,
        state_indices,
        has_initial_state,
        chunk_size=chunk_size,
        use_qk_l2norm=use_qk_l2norm,
        decay_lower_bound=decay_lower_bound,
        precision=precision,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )


def ragged_kda_head_sharded(
    query: Float[Array, "num_tokens num_heads head_dim"],
    key: Float[Array, "num_tokens num_heads head_dim"],
    value: Float[Array, "num_tokens num_heads v_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    query_start_loc: Int[Array, "num_rows_plus_1"],
    state_indices: Int[Array, "num_rows"] | None,
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    /,
    *,
    mesh: object | None,
    head_axis: object | None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDAConfig | None = None,
) -> tuple[Array, Array]:
    """Run :func:`ragged_kda` with the heads sharded over ``head_axis`` of ``mesh``.

    With a mesh whose ``head_axis`` has more than one device, the executor runs the operation under ``jax.shard_map``:
    every device advances its own heads of every request (tokens, row metadata and pool slots replicated, heads
    split), with no collectives. Otherwise the call is unsharded.

    Args:
        query: Post-conv queries ``[T, H, K]``.
        key: Post-conv keys ``[T, H, K]``.
        value: Post-conv values ``[T, H, V]``.
        beta: Write strengths ``[T, H]``.
        decay: Per-channel log-decay ``[T, H, K]``.
        recurrent_state: State pool ``[S, H, K, V]``.
        query_start_loc: Cumulative row offsets ``[R + 1]``.
        state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
        has_initial_state: Optional ``[R]`` flags.
        mesh: Mesh carrying ``head_axis`` (or ``None``).
        head_axis: Mesh axis name (or tuple of names) the heads are sharded over (or ``None``).
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promised lower bound of every log-decay, or ``None``.
        precision: Precision of the non-inverse chunk dots.
        platform: Optional platform override.
        cfg: Optional :class:`RaggedKDAConfig`.

    Returns:
        ``(out [T, H, V] fp32, new_recurrent_state [S, H, K, V])``.

    Raises:
        ValueError: If the head count is not divisible by the size of ``head_axis``.
    """
    head_shards = mesh_axis_size(mesh, head_axis)
    kwargs = dict(
        chunk_size=chunk_size,
        use_qk_l2norm=use_qk_l2norm,
        decay_lower_bound=decay_lower_bound,
        precision=precision,
        platform=platform,
        cfg=cfg,
    )
    arrays = (query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices, has_initial_state)
    if mesh is None or head_axis is None or head_shards <= 1:
        return ragged_kda(*arrays, **kwargs)
    num_heads = int(query.shape[1])
    if num_heads % head_shards:
        raise ValueError(f"ragged_kda: {num_heads} heads are not divisible by the {head_shards}-way axis {head_axis}")
    token_head = PartitionSpec(None, head_axis, None)
    replicated = PartitionSpec()
    state_spec = PartitionSpec(None, head_axis, None, None)
    in_specs = (
        token_head,
        token_head,
        token_head,
        PartitionSpec(None, head_axis),
        token_head,
        state_spec,
        replicated,
        replicated,
        replicated,
    )
    return ragged_kda(*arrays, mesh=mesh, in_specs=in_specs, out_specs=(token_head, state_spec), **kwargs)


__all__ = ("RaggedKDA", "ragged_kda", "ragged_kda_head_sharded")

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

"""Ragged (flat-token) KDA: fused causal conv, per-channel decay and per-token state capture (``ragged_kda``).

Serving form of the GLM-5-Next / Kimi-Linear KDA layer. The tokens of all scheduled rows share one flat buffer;
``query_start_loc`` delimits each row and ``state_indices`` names the base slot ``g`` of the row's slot group in the
pools: the recurrent state ``[S, H, K, V]`` and, when the conv runs inside the op, the q / k / v conv windows
``[S, dim, d_conv]``. With the six conv arrays (weights and window pools) the op takes the raw q / k / v projections and
runs the depthwise causal conv (SiLU fused) itself; without them the inputs are already convolved (the round-4 call).

Per row: read the windows and the state from slot ``g + read_offset`` (zeros without an initial state); process the
row's tokens; rows of at most ``short_row_max`` (``W``) tokens write the windows and state as they stand after token
``commit_lens - 1 + j`` into slot ``g + j`` for ``j = 0 .. lens - commit_lens`` (a speculative verify row captures every
draft position; an ordinary row writes its final state into ``g``), longer rows run the chunked math over their own
live chunks and write their final state into ``g``. With ``W = 1`` this is the ordinary serving step.

The XLA implementation is the reference and the fallback. Everything is independent per head, so on a mesh the
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
# The op's positional arrays, in order. The first nine are the post-conv (round-4) call; the rest are optional.
ARRAY_NAMES = (
    "query",
    "key",
    "value",
    "beta",
    "decay",
    "recurrent_state",
    "query_start_loc",
    "state_indices",
    "has_initial_state",
    "read_offset",
    "commit_lens",
    "q_conv_weight",
    "k_conv_weight",
    "v_conv_weight",
    "q_conv_state",
    "k_conv_state",
    "v_conv_state",
)
_NUM_ARRAYS = len(ARRAY_NAMES)
_NUM_BASE_ARRAYS = 9
_CONV_INDEX = ARRAY_NAMES.index("q_conv_weight")


def _normalize_cfg(cfg: RaggedKDAConfig | dict | None) -> RaggedKDAConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return RaggedKDAConfig()
    if isinstance(cfg, dict):
        return RaggedKDAConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


def _trim(arrays: tuple[Array | None, ...]) -> tuple[Array | None, ...]:
    """Drop trailing ``None`` arrays past the nine post-conv ones (so a round-4 call keeps its nine arrays)."""
    n = len(arrays)
    while n > _NUM_BASE_ARRAYS and arrays[n - 1] is None:
        n -= 1
    return tuple(arrays[:n])


def _num_outputs(arrays: tuple[Array | None, ...]) -> int:
    """5 outputs with the conv arrays (out + three window pools + state pool), else 2 (out + state pool)."""
    return 5 if len(arrays) > _CONV_INDEX and arrays[_CONV_INDEX] is not None else 2


class RaggedKDA(Kernel[RaggedKDAConfig, tuple[Array, ...]]):
    """Ragged KDA with the causal conv fused and per-token state capture (XLA reference; accelerator backends)."""

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
        read_offset: Int[Array, "num_rows"] | None = None,
        commit_lens: Int[Array, "num_rows"] | None = None,
        q_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
        k_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
        v_conv_weight: Float[Array, "value_dim d_conv"] | None = None,
        q_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "num_slots value_dim d_conv"] | None = None,
        *,
        short_row_max: int = 1,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        decay_lower_bound: float | None = None,
        precision: str = "highest",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDAConfig,
        **_,
    ) -> tuple[Array, ...]:
        """Execute the selected backend (arguments as in :func:`ragged_kda`).

        Returns:
            With the conv arrays ``(out, q_conv_state, k_conv_state, v_conv_state, recurrent_state)``, else
            ``(out, recurrent_state)``.
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
            read_offset,
            commit_lens,
            q_conv_weight,
            k_conv_weight,
            v_conv_weight,
            q_conv_state,
            k_conv_state,
            v_conv_state,
            short_row_max=int(short_row_max),
            chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm,
            decay_lower_bound=decay_lower_bound,
            precision=precision,
            inverse=cfg.inverse,
            pairwise=cfg.pairwise,
        )

    def create_shard_map_wrapper(
        self,
        *arrays: Array | None,
        short_row_max: int = 1,
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

        ``arrays`` are the op's positional arrays in :data:`ARRAY_NAMES` order (the nine post-conv ones, optionally
        followed by the rest); ``in_specs`` has one entry per passed array (the entry of a ``None`` array is ignored).
        Only head and conv-feature axes may be sharded: the token axis carries the recurrence, and every row metadata
        array must be replicated.

        Args:
            *arrays: The op's positional arrays (see :func:`ragged_kda`).
            short_row_max: ``W``.
            chunk_size: Chunk length of the chunk path.
            use_qk_l2norm: L2-normalise queries and keys.
            decay_lower_bound: Promised lower bound of every log-decay, or ``None``.
            precision: Precision of the non-inverse chunk dots.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh (a SpectraX mesh is unwrapped).
            in_specs: One input spec per passed array.
            out_specs: Output specs: ``(out, recurrent_state)``, or with the conv arrays ``(out, q_conv_state,
                k_conv_state, v_conv_state, recurrent_state)``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the array, spec or output-spec counts are wrong, or the token / head-dim axes of ``query``
                are sharded.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if not _NUM_BASE_ARRAYS <= len(arrays) <= _NUM_ARRAYS:
            raise ValueError(f"ragged_kda takes {_NUM_BASE_ARRAYS} to {_NUM_ARRAYS} arrays, got {len(arrays)}")
        if len(in_specs) != len(arrays):
            raise ValueError(f"in_specs must have {len(arrays)} entries (one per array), got {len(in_specs)}")
        num_out = _num_outputs(arrays)
        if len(out_specs) != num_out:
            raise ValueError(f"out_specs must have {num_out} entries, got {len(out_specs)}")
        q_spec = tuple(in_specs[0]) + (None,) * (3 - len(tuple(in_specs[0])))
        if q_spec[0] is not None or q_spec[2] is not None:
            raise ValueError(f"ragged_kda shard_map requires unsharded token/feature axes, got query spec {q_spec}")
        num_passed = len(arrays)
        present = tuple(i for i, a in enumerate(arrays) if a is not None)
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(*local):
            full = [None] * num_passed
            for i, a in zip(present, local, strict=True):
                full[i] = a
            return self.run(
                *full,
                short_row_max=short_row_max,
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

    def heuristic_cfg(self, inv: Invocation[RaggedKDAConfig, tuple[Array, ...]]) -> RaggedKDAConfig:
        """Return the default config: automatic chunk inverse and pairwise form, automatic platform."""
        return RaggedKDAConfig(inverse="auto", pairwise="auto", platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedKDAConfig, tuple[Array, ...]]):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_ragged_kda_executor: Executor[RaggedKDAConfig, tuple[Array, ...]] = Executor(
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
    read_offset: Int[Array, "num_rows"] | None = None,
    commit_lens: Int[Array, "num_rows"] | None = None,
    q_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
    k_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
    v_conv_weight: Float[Array, "value_dim d_conv"] | None = None,
    q_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
    k_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
    v_conv_state: Float[Array, "num_slots value_dim d_conv"] | None = None,
    /,
    *,
    short_row_max: int = 1,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDAConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> tuple[Array, ...]:
    """Ragged KDA over a flat token stream: fused causal conv, recurrence, per-token state capture.

    Row ``r`` owns tokens ``[query_start_loc[r], query_start_loc[r + 1])`` (``lens`` of them) and the slot group that
    starts at its base slot ``g = state_indices[r]`` (``g = r`` when ``state_indices`` is ``None``). With
    ``a = read_offset[r]`` and ``c = commit_lens[r]``:

    1. the conv windows and the recurrent state are read from slot ``g + a`` (zeros when ``has_initial_state[r]`` is
       ``False``);
    2. every token is processed causally: with the conv arrays the depthwise conv with SiLU on q / k / v (the output in
       the stream dtype, ``bfloat16`` for 1-byte float streams), then per head
       ``S = diag(exp(g_t)) S + k (beta (v - k^T diag(exp(g_t)) S))^T`` and ``o = S^T q`` (queries scaled by
       ``K**-0.5``, queries and keys optionally L2-normalised);
    3. rows of at most ``short_row_max`` tokens write the windows and state as they stand after token ``c - 1 + j``
       into slot ``g + j`` for ``j = 0 .. lens - c`` (every token with the decode-step arithmetic, the state rounded to
       the pool dtype after every token); longer rows run the chunked math and write their final windows and state
       into ``g`` (``commit_lens`` is ignored for them).

    A row reads before it writes. Rows must own disjoint slot groups, and ``read_offset`` stays inside the row's
    group. Rows without tokens and rows whose slots (``g``, ``g + a``, ``g + lens - c``) fall out of range produce zero
    output and write nothing; tokens at or past ``query_start_loc[R]`` are padding (zero output).

    Args:
        query: Queries ``[T, H, K]``: raw (pre-conv) projections with the six conv arrays, post-conv without them.
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``), ideally fp32.
        recurrent_state: State pool ``[S, H, K, V]`` (the recurrence runs in fp32; stored in the pool's dtype).
        query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
        state_indices: Base slot of every row ``[R]``, or ``None`` (``g = r``, needs ``S >= R``).
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from zero windows and a zero state.
        read_offset: Optional ``[R]`` read offsets ``a`` (zero when ``None``).
        commit_lens: Optional ``[R]`` commit lengths ``c`` (``lens`` when ``None``).
        q_conv_weight: Depthwise causal conv taps ``[H * K, d_conv]`` (the last tap multiplies the current token), or
            ``None`` (post-conv inputs). The six conv arrays are given together or not at all.
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: Conv-window pool ``[S, H * K, d_conv]`` (oldest tap first), or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.
        short_row_max: ``W = 1 + K_max``: rows of at most ``W`` tokens take the short-row path (static).
        chunk_size: Chunk length of the chunk path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound`` (e.g. a safe-gate bound). With
            ``cfg.pairwise="auto"`` a tight enough bound selects the factored MXU pairwise decay. ``None`` keeps the
            dense form. Values below the bound are a contract violation (possible overflow).
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        platform: Optional platform override (``"xla"`` / ``"pallas"``).
        cfg: Optional :class:`RaggedKDAConfig`.
        mesh: Optional mesh for ``shard_map`` execution over the head axis.
        in_specs: One input spec per passed array (see :meth:`RaggedKDA.create_shard_map_wrapper`).
        out_specs: Output specs (2 or 5, see the return value).

    Returns:
        With the conv arrays ``(out [T, H, V] fp32, q_conv_state, k_conv_state, v_conv_state, recurrent_state)``,
        without them ``(out, recurrent_state)``. The pools keep their dtypes; every slot no live row writes is
        unchanged.
    """
    arrays = _trim(
        (
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            read_offset,
            commit_lens,
            q_conv_weight,
            k_conv_weight,
            v_conv_weight,
            q_conv_state,
            k_conv_state,
            v_conv_state,
        )
    )
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _ragged_kda_executor(
        RaggedKDA(),
        *arrays,
        short_row_max=int(short_row_max),
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
    read_offset: Int[Array, "num_rows"] | None = None,
    commit_lens: Int[Array, "num_rows"] | None = None,
    q_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
    k_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
    v_conv_weight: Float[Array, "value_dim d_conv"] | None = None,
    q_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
    k_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
    v_conv_state: Float[Array, "num_slots value_dim d_conv"] | None = None,
    /,
    *,
    mesh: object | None,
    head_axis: object | None,
    short_row_max: int = 1,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDAConfig | None = None,
) -> tuple[Array, ...]:
    """Run :func:`ragged_kda` with the heads (and conv features) sharded over ``head_axis`` of ``mesh``.

    With a mesh whose ``head_axis`` has more than one device, the executor runs the operation under ``jax.shard_map``:
    every device advances its own heads of every row (tokens, row metadata and pool slots replicated; heads, the conv
    features ``H * D`` of weights, windows and stream, and the state pool's head axis split), with no collectives.
    The conv features of head ``h`` are ``[h * D, (h + 1) * D)``, so a contiguous feature split is a head split.
    Otherwise the call is unsharded.

    Args:
        query: Queries ``[T, H, K]`` (raw with the conv arrays, post-conv without).
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: Write strengths ``[T, H]``.
        decay: Per-channel log-decay ``[T, H, K]``.
        recurrent_state: ``[S, H, K, V]``.
        query_start_loc: ``[R + 1]``.
        state_indices: ``[R]`` base slots, or ``None``.
        has_initial_state: Optional ``[R]``.
        read_offset: Optional ``[R]``.
        commit_lens: Optional ``[R]``.
        q_conv_weight: ``[H * K, d_conv]`` or ``None``.
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.
        mesh: Mesh carrying ``head_axis`` (or ``None``).
        head_axis: Mesh axis name (or tuple of names) the heads are sharded over (or ``None``).
        short_row_max: ``W``.
        chunk_size: Chunk length of the chunk path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promised lower bound of every log-decay, or ``None``.
        precision: Precision of the non-inverse chunk dots.
        platform: Optional platform override.
        cfg: Optional :class:`RaggedKDAConfig`.

    Returns:
        As :func:`ragged_kda`.

    Raises:
        ValueError: If the head count is not divisible by the size of ``head_axis``.
    """
    head_shards = mesh_axis_size(mesh, head_axis)
    kwargs = dict(
        short_row_max=int(short_row_max),
        chunk_size=chunk_size,
        use_qk_l2norm=use_qk_l2norm,
        decay_lower_bound=decay_lower_bound,
        precision=precision,
        platform=platform,
        cfg=cfg,
    )
    arrays = _trim(
        (
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            read_offset,
            commit_lens,
            q_conv_weight,
            k_conv_weight,
            v_conv_weight,
            q_conv_state,
            k_conv_state,
            v_conv_state,
        )
    )
    if mesh is None or head_axis is None or head_shards <= 1:
        return ragged_kda(*arrays, **kwargs)
    num_heads = int(query.shape[1])
    if num_heads % head_shards:
        raise ValueError(f"ragged_kda: {num_heads} heads are not divisible by the {head_shards}-way axis {head_axis}")
    token_head = PartitionSpec(None, head_axis, None)
    feature = PartitionSpec(head_axis, None)
    window = PartitionSpec(None, head_axis, None)
    state_spec = PartitionSpec(None, head_axis, None, None)
    replicated = PartitionSpec()
    in_specs = (
        token_head,  # query
        token_head,  # key
        token_head,  # value
        PartitionSpec(None, head_axis),  # beta
        token_head,  # decay
        state_spec,  # recurrent_state
        replicated,  # query_start_loc
        replicated,  # state_indices
        replicated,  # has_initial_state
        replicated,  # read_offset
        replicated,  # commit_lens
        feature,  # q_conv_weight
        feature,  # k_conv_weight
        feature,  # v_conv_weight
        window,  # q_conv_state
        window,  # k_conv_state
        window,  # v_conv_state
    )[: len(arrays)]
    if _num_outputs(arrays) == 5:
        out_specs = (token_head, window, window, window, state_spec)
    else:
        out_specs = (token_head, state_spec)
    return ragged_kda(*arrays, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs)


__all__ = ("ARRAY_NAMES", "RaggedKDA", "ragged_kda", "ragged_kda_head_sharded")

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

"""Fused chunked KDA with per-channel decay (``kda_chunk``).

The whole inter-chunk recurrence of the chunked KDA forward (per-channel log-decay, GLM-5-Next / Kimi-Linear
style) runs in one Pallas TPU ``pallas_call`` whose grid walks the chunks with the fp32 state in VMEM, and the
backward is a second reverse-sweep ``pallas_call``. The XLA implementation (a rematerialised ``lax.scan`` over
chunks) is the reference and the fallback for other backends and untileable shapes.

A Mosaic custom call cannot be GSPMD-partitioned: on a multi-device mesh pass ``mesh`` / ``in_specs`` /
``out_specs`` so the executor runs the kernel under ``jax.shard_map`` over the batch and head axes (the recurrence
is independent per row and per head, so no collectives are needed).
"""

from __future__ import annotations

from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import KdaChunkConfig

_CFG_FIELDS = frozenset(getattr(KdaChunkConfig, "__dataclass_fields__", ()))


def _normalize_cfg(cfg: KdaChunkConfig | dict | None) -> KdaChunkConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return KdaChunkConfig()
    if isinstance(cfg, dict):
        return KdaChunkConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


class KdaChunk(Kernel[KdaChunkConfig, tuple[Array, ...]]):
    """Chunked per-channel-decay KDA (Pallas TPU forward + backward; XLA reference/fallback)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="kda_chunk"``."""
        super().__init__(op_id="kda_chunk")

    def get_impl(self, cfg: KdaChunkConfig):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``kda_chunk`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch num_heads seq_len head_dim"],
        value: Float[Array, "batch num_heads seq_len v_dim"],
        beta: Float[Array, "batch num_heads seq_len"],
        decay: Float[Array, "batch num_heads seq_len head_dim"],
        initial_state: Float[Array, "batch num_heads head_dim v_dim"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        *,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        continue_initial_state: bool = False,
        return_chunk_states: bool = False,
        precision: str = "highest",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: KdaChunkConfig,
        **_,
    ) -> tuple[Array, ...]:
        """Execute the selected backend.

        Args:
            query: Queries ``[B, H, L, K]``.
            key: Keys ``[B, H, L, K]``.
            value: Values ``[B, H, L, V]``.
            beta: Delta-rule gates ``[B, H, L]``.
            decay: Per-channel log-decay ``[B, H, L, K]``.
            initial_state: Optional ``[B, H, K, V]``.
            segment_ids: Optional packed-row ids ``[B, L]``.
            chunk_size: Chunk length.
            use_qk_l2norm: L2-normalise queries and keys.
            continue_initial_state: Cached-continuation packed-row semantics.
            return_chunk_states: Also return the state after every chunk.
            precision: Precision of the non-inverse dots.
            platform: Optional platform override.
            cfg: Operation config (block_h, backward, bwd_reduce).

        Returns:
            ``(out, final_state)`` (+ ``chunk_states``).
        """
        cfg = _normalize_cfg(cfg)
        if platform is not None:
            cfg = KdaChunkConfig(
                block_h=cfg.block_h,
                backward=cfg.backward,
                bwd_reduce=cfg.bwd_reduce,
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
            initial_state,
            segment_ids,
            chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm,
            continue_initial_state=continue_initial_state,
            return_chunk_states=return_chunk_states,
            precision=precision,
            block_h=cfg.block_h,
            backward=cfg.backward,
            bwd_reduce=cfg.bwd_reduce,
        )

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch num_heads seq_len head_dim"],
        value: Float[Array, "batch num_heads seq_len v_dim"],
        beta: Float[Array, "batch num_heads seq_len"],
        decay: Float[Array, "batch num_heads seq_len head_dim"],
        initial_state: Float[Array, "batch num_heads head_dim v_dim"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        *,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        continue_initial_state: bool = False,
        return_chunk_states: bool = False,
        precision: str = "highest",
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: KdaChunkConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper over the batch/head axes.

        ``in_specs`` has one entry per positional array ``(query, key, value, beta, decay, initial_state,
        segment_ids)``; entries for ``None`` arrays are ignored. The sequence and feature axes must be unsharded
        (the recurrence is serial along the sequence).

        Args:
            query: Queries ``[B, H, L, K]``.
            key: Keys ``[B, H, L, K]``.
            value: Values ``[B, H, L, V]``.
            beta: Gates ``[B, H, L]``.
            decay: Per-channel log-decay ``[B, H, L, K]``.
            initial_state: Optional ``[B, H, K, V]``.
            segment_ids: Optional ``[B, L]``.
            chunk_size: Chunk length.
            use_qk_l2norm: L2-normalise queries and keys.
            continue_initial_state: Cached-continuation packed-row semantics.
            return_chunk_states: Also return the per-chunk states.
            precision: Precision of the non-inverse dots.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh.
            in_specs: Seven input specs (see above).
            out_specs: Output specs ``(out, final_state[, chunk_states])``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the spec count is wrong or the sequence/feature axes of ``query`` are sharded.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if len(in_specs) != 7:
            raise ValueError(f"in_specs must have 7 entries, got {len(in_specs)}")
        q_spec = tuple(in_specs[0]) + (None,) * (4 - len(tuple(in_specs[0])))
        if q_spec[2] is not None or q_spec[3] is not None:
            raise ValueError(f"kda_chunk shard_map requires unsharded sequence/feature axes, got query spec {q_spec}")
        arrays = (query, key, value, beta, decay, initial_state, segment_ids)
        present = tuple(i for i, a in enumerate(arrays) if a is not None)
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(*local):
            full = [None] * 7
            for i, a in zip(present, local, strict=True):
                full[i] = a
            return self.run(
                *full,
                chunk_size=chunk_size,
                use_qk_l2norm=use_qk_l2norm,
                continue_initial_state=continue_initial_state,
                return_chunk_states=return_chunk_states,
                precision=precision,
                platform=platform,
                cfg=run_cfg,
            )

        shard_map_fn = shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=tuple(in_specs[i] for i in present),
            out_specs=tuple(out_specs),
            check_vma=check_vma,
        )
        return shard_map_fn, tuple(arrays[i] for i in present)

    def heuristic_cfg(self, inv: Invocation[KdaChunkConfig, tuple[Array, ...]]) -> KdaChunkConfig:
        """Return the measured default: one head per grid step, Pallas backward, VPU reductions."""
        return KdaChunkConfig(block_h=1, backward="pallas", bwd_reduce="vpu", platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[KdaChunkConfig, tuple[Array, ...]]):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_kda_chunk_executor: Executor[KdaChunkConfig, tuple[Array, ...]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback="heuristics",
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("kda-chunk", cfg_type=KdaChunkConfig),
    )
)


def kda_chunk(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len v_dim"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len head_dim"],
    initial_state: Float[Array, "batch num_heads head_dim v_dim"] | None = None,
    segment_ids: Int[Array, "batch seq_len"] | None = None,
    /,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    continue_initial_state: bool = False,
    return_chunk_states: bool = False,
    precision: str = "highest",
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: KdaChunkConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> tuple[Array, ...]:
    """Chunked KDA with per-channel decay through the ejkernel executor.

    Per row and head, ``S_t = diag(exp(g_t)) S_{t-1} + k_t (beta_t (v_t - k_t^T diag(exp(g_t)) S_{t-1}))^T`` and
    ``o_t = S_t^T q_t`` (queries scaled by ``K**-0.5``, optionally L2-normalised with the keys), evaluated in chunks.

    Args:
        query: Queries ``[B, H, L, K]``.
        key: Keys ``[B, H, L, K]``.
        value: Values ``[B, H, L, V]``.
        beta: Delta-rule gates ``[B, H, L]``.
        decay: Per-token per-channel log-decay ``[B, H, L, K]`` (``<= 0``).
        initial_state: Optional state ``[B, H, K, V]``.
        segment_ids: Optional packed-row ids ``[B, L]``; the state resets when the id changes, negative ids are
            padding.
        chunk_size: Chunk length (power of two in ``[8, 128]`` for the Pallas kernel).
        use_qk_l2norm: L2-normalise queries and keys first.
        continue_initial_state: With ``segment_ids``: padding does not break a run and the first live run inherits
            ``initial_state``.
        return_chunk_states: Also return the state after every chunk ``[B, C, H, K, V]``.
        precision: ``"highest"`` or ``"default"`` for the non-inverse dots.
        platform: Optional platform override (``"pallas"`` / ``"xla"``).
        cfg: Optional :class:`KdaChunkConfig`.
        mesh: Optional mesh for ``shard_map`` execution (required for the Pallas kernel on a multi-device mesh).
        in_specs: Seven input specs ``(query, key, value, beta, decay, initial_state, segment_ids)``.
        out_specs: Output specs ``(out, final_state[, chunk_states])``.

    Returns:
        ``(out [B, H, L, V] fp32, final_state [B, H, K, V] fp32)`` (+ ``chunk_states``).
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _kda_chunk_executor(
        KdaChunk(),
        query,
        key,
        value,
        beta,
        decay,
        initial_state,
        segment_ids,
        chunk_size=chunk_size,
        use_qk_l2norm=use_qk_l2norm,
        continue_initial_state=continue_initial_state,
        return_chunk_states=return_chunk_states,
        precision=precision,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )


__all__ = ("KdaChunk", "kda_chunk")

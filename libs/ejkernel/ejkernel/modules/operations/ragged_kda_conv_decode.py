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


"""Fused per-token KDA decode step with its causal convs over in-place pools (``ragged_kda_conv_decode``).

The whole KDA decode step of an eSurge pure-decode window in one operation: token ``t`` of the packed stream advances
pool slot ``token_slots[t]`` (out of range: the token outputs zeros and writes nothing), from the slot's conv windows and
recurrent state (``token_init[t]``) or from zeros. Per advancing token, each of q / k / v takes one depthwise causal conv
step (window shift, ``d_conv`` taps in fp32, SiLU), then q / k are L2-normalised, q is scaled by ``K**-0.5`` and the
recurrent state takes one per-channel delta-rule step::

    S = diag(exp(g)) S;   S += k (beta (v - k^T S))^T;   o = S^T q

The new windows and state are written back to the slot. On TPU one Pallas kernel does all of it with the pools updated in
place (bit-identical to the XLA conv glue followed by the ``ragged_kda_decode`` Pallas kernel in a standalone unsharded
program; see ``EXACT_ROUNDING`` for why a larger graph can differ); the XLA implementation is
the reference and the fallback. Every step is independent per head, so on a mesh the executor runs it under
``jax.shard_map`` over the head axis (no collectives): :func:`ragged_kda_conv_decode_head_sharded`. Advancing tokens
must name distinct slots.
"""

from __future__ import annotations

import typing as tp
from typing import Literal

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._pallas.tpu.ragged_kda_conv_decode._pallas_impl import unsupported_reason
from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.modules.base import mesh_to_jax_mesh
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedKDAConvDecodeConfig
from .ragged_causal_conv1d import mesh_axis_size

_CFG_FIELDS = frozenset(getattr(RaggedKDAConvDecodeConfig, "__dataclass_fields__", ()))
_NUM_ARGS = 10  # query, key, value, beta, decay, conv_weights, conv_pools, recurrent_state, token_slots, token_init


def _normalize_cfg(cfg: RaggedKDAConvDecodeConfig | dict | None) -> RaggedKDAConvDecodeConfig:
    """Restore an executor-serialised config (or build the default)."""
    if cfg is None:
        return RaggedKDAConvDecodeConfig()
    if isinstance(cfg, dict):
        return RaggedKDAConvDecodeConfig.from_dict({k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return cfg


def _dtype_name(dtype: tp.Any) -> str:
    return jnp.dtype(dtype).name


class RaggedKDAConvDecode(Kernel[RaggedKDAConvDecodeConfig, tuple[Array, tuple[Array, Array, Array, Array]]]):
    """Fused per-token KDA decode step with its causal convs (Pallas TPU; XLA reference)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="ragged_kda_conv_decode"``."""
        super().__init__(op_id="ragged_kda_conv_decode")

    def get_impl(self, cfg: RaggedKDAConvDecodeConfig):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``ragged_kda_conv_decode`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "num_tokens key_dim"],
        key: Float[Array, "num_tokens key_dim"],
        value: Float[Array, "num_tokens value_dim"],
        beta: Float[Array, "num_tokens num_heads"],
        decay: Float[Array, "num_tokens num_heads head_dim"],
        conv_weights: tuple[Array, Array, Array],
        conv_pools: tuple[Array, Array, Array],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        token_slots: Int[Array, "num_tokens"],
        token_init: Bool[Array, "num_tokens"],
        *,
        conv_dtype: str | None = None,
        use_qk_l2norm: bool = True,
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDAConvDecodeConfig,
        **_,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """Execute the selected backend.

        Args:
            query: Raw (pre-conv) queries ``[T, H*K]``.
            key: Raw keys ``[T, H*K]``.
            value: Raw values ``[T, H*V]``.
            beta: Write strengths ``[T, H]`` (sigmoid applied).
            decay: Per-channel log-decay ``[T, H, K]``.
            conv_weights: Depthwise conv kernels ``(q, k, v)``, each ``[dim, d_conv]``.
            conv_pools: Conv windows ``(q, k, v)``, each ``[S, dim, d_conv]``.
            recurrent_state: State pool ``[S, H, K, V]``.
            token_slots: ``[T]`` pool slot advanced by each token (out of range: none).
            token_init: ``[T]`` whether that slot's windows and state carry over.
            conv_dtype: Dtype name of the conv outputs (``None``: the query dtype).
            use_qk_l2norm: L2-normalise queries and keys.
            platform: Optional platform override.
            cfg: Operation config.

        Returns:
            ``(out [T, H, V] fp32, (q_pool, k_pool, v_pool, recurrent_state))``.
        """
        cfg = _normalize_cfg(cfg)
        if platform is not None:
            cfg = RaggedKDAConvDecodeConfig(
                block_tokens=cfg.block_tokens,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        out, pools = impl(
            query,
            key,
            value,
            beta,
            decay,
            tuple(conv_weights),
            tuple(conv_pools),
            recurrent_state,
            token_slots,
            token_init,
            conv_dtype=conv_dtype,
            use_qk_l2norm=use_qk_l2norm,
            block_tokens=int(cfg.block_tokens),
        )
        return out, tuple(pools)

    def create_shard_map_wrapper(
        self,
        query: Array,
        key: Array,
        value: Array,
        beta: Array,
        decay: Array,
        conv_weights: tuple[Array, Array, Array],
        conv_pools: tuple[Array, Array, Array],
        recurrent_state: Array,
        token_slots: Array,
        token_init: Array,
        *,
        conv_dtype: str | None = None,
        use_qk_l2norm: bool = True,
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: RaggedKDAConvDecodeConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[tp.Any, ...] | None = None,
        out_specs: tuple[tp.Any, ...] | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper over the head axis.

        ``in_specs`` has one entry per positional argument (``conv_weights`` / ``conv_pools`` take a tuple of three
        specs); the token axis and the per-token slot tables must be replicated.

        Args:
            query: Raw queries ``[T, H*K]``.
            key: Raw keys ``[T, H*K]``.
            value: Raw values ``[T, H*V]``.
            beta: Write strengths ``[T, H]``.
            decay: Per-channel log-decay ``[T, H, K]``.
            conv_weights: ``(q, k, v)`` ``[dim, d_conv]``.
            conv_pools: ``(q, k, v)`` ``[S, dim, d_conv]``.
            recurrent_state: ``[S, H, K, V]``.
            token_slots: ``[T]``.
            token_init: ``[T]``.
            conv_dtype: Dtype name of the conv outputs.
            use_qk_l2norm: L2-normalise queries and keys.
            platform: Optional platform override.
            cfg: Operation config.
            mesh: Device mesh (a SpectraX mesh is unwrapped).
            in_specs: Ten input specs (see above).
            out_specs: Output specs ``(out, (q_pool, k_pool, v_pool, recurrent_state))``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)``.

        Raises:
            ValueError: If the spec count is wrong or the token axis of ``query`` is sharded.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if len(in_specs) != _NUM_ARGS:
            raise ValueError(f"in_specs must have {_NUM_ARGS} entries, got {len(in_specs)}")
        q_spec = tuple(in_specs[0]) + (None,) * (2 - len(tuple(in_specs[0])))
        if q_spec[0] is not None:
            raise ValueError(f"ragged_kda_conv_decode shard_map requires an unsharded token axis, got {q_spec}")
        run_cfg = _normalize_cfg(cfg)

        def _wrapped(*local):
            return self.run(*local, conv_dtype=conv_dtype, use_qk_l2norm=use_qk_l2norm, platform=platform, cfg=run_cfg)

        shard_map_fn = jax.shard_map(
            _wrapped,
            mesh=mesh_to_jax_mesh(mesh),
            in_specs=tuple(in_specs),
            out_specs=tuple(out_specs),
            check_vma=check_vma,
        )
        args = (query, key, value, beta, decay, tuple(conv_weights), tuple(conv_pools), recurrent_state)
        return shard_map_fn, (*args, token_slots, token_init)

    def heuristic_cfg(self, inv: Invocation) -> RaggedKDAConvDecodeConfig:
        """Return the default config: blocks from the VMEM capacity, automatic platform."""
        return RaggedKDAConvDecodeConfig(block_tokens=0, platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_ragged_kda_conv_decode_executor: Executor = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback="heuristics",
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("ragged-kda-conv-decode", cfg_type=RaggedKDAConvDecodeConfig),
    )
)


def ragged_kda_conv_decode(
    query: Float[Array, "num_tokens key_dim"],
    key: Float[Array, "num_tokens key_dim"],
    value: Float[Array, "num_tokens value_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    conv_weights: tuple[Array, Array, Array],
    conv_pools: tuple[Array, Array, Array],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    token_slots: Int[Array, "num_tokens"],
    token_init: Bool[Array, "num_tokens"],
    /,
    *,
    conv_dtype: tp.Any = None,
    use_qk_l2norm: bool = True,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDAConvDecodeConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[tp.Any, ...] | None = None,
    out_specs: tuple[tp.Any, ...] | None = None,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Fused per-token KDA decode step with its causal convs, through the ejkernel executor (see the module docstring).

    Args:
        query: Raw (pre-conv) queries ``[T, H*K]``.
        key: Raw keys ``[T, H*K]``.
        value: Raw values ``[T, H*V]``.
        beta: Write strengths ``[T, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``), ideally fp32.
        conv_weights: Depthwise conv kernels ``(q, k, v)``, each ``[dim, d_conv]`` (last tap = current token).
        conv_pools: Conv windows ``(q, k, v)``, each ``[S, dim, d_conv]`` (last column = most recent token).
        recurrent_state: State pool ``[S, H, K, V]`` (the step runs in fp32; the pool keeps its dtype).
        token_slots: ``[T]`` pool slot advanced by each token (``< 0`` or ``>= S``: zero output, no write). Advancing
            tokens must name distinct slots.
        token_init: ``[T]`` whether that slot's windows and state carry over (``False``: the row starts a request).
        conv_dtype: Dtype of the conv outputs fed to the recurrence (``None``: the query dtype).
        use_qk_l2norm: L2-normalise queries and keys.
        platform: Optional platform override (``"xla"`` / ``"pallas"``).
        cfg: Optional :class:`RaggedKDAConvDecodeConfig`.
        mesh: Optional mesh for ``shard_map`` execution over the head axis.
        in_specs: Ten input specs (see :meth:`RaggedKDAConvDecode.create_shard_map_wrapper`).
        out_specs: Output specs ``(out, (q_pool, k_pool, v_pool, recurrent_state))``.

    Returns:
        ``(out [T, H, V] fp32, (q_pool, k_pool, v_pool, recurrent_state))``.
    """
    method = "shard_map" if mesh is not None and in_specs is not None and out_specs is not None else None
    return _ragged_kda_conv_decode_executor(
        RaggedKDAConvDecode(),
        query,
        key,
        value,
        beta,
        decay,
        tuple(conv_weights),
        tuple(conv_pools),
        recurrent_state,
        token_slots,
        token_init,
        conv_dtype=_dtype_name(query.dtype if conv_dtype is None else conv_dtype),
        use_qk_l2norm=use_qk_l2norm,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )


def _head_shards(mesh: object | None, head_axis: object | None) -> int:
    return mesh_axis_size(mesh, head_axis) if mesh is not None and head_axis is not None else 1


def ragged_kda_conv_decode_unsupported_reason(
    query: tp.Any,
    key: tp.Any,
    value: tp.Any,
    beta: tp.Any,
    decay: tp.Any,
    conv_weights: tp.Sequence[tp.Any],
    conv_pools: tp.Sequence[tp.Any],
    recurrent_state: tp.Any,
    *,
    conv_dtype: tp.Any = None,
    mesh: object | None = None,
    head_axis: object | None = None,
    cfg: RaggedKDAConvDecodeConfig | None = None,
) -> str | None:
    """Why the TPU kernel would not take this call (each device's share under the head split), or ``None``.

    Callers that keep another implementation for such calls (the XLA reference otherwise runs in the kernel's place)
    use this to choose. Arrays or ``jax.ShapeDtypeStruct`` with the global shapes.

    Args:
        query: ``[T, H*K]``.
        key: ``[T, H*K]``.
        value: ``[T, H*V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        conv_weights: ``(q, k, v)`` ``[dim, d_conv]``.
        conv_pools: ``(q, k, v)`` ``[S, dim, d_conv]``.
        recurrent_state: ``[S, H, K, V]``.
        conv_dtype: Dtype of the conv outputs (``None``: the query dtype).
        mesh: Mesh of the head split (or ``None``).
        head_axis: Mesh axis name (or tuple of names) of the heads (or ``None``).
        cfg: Optional :class:`RaggedKDAConvDecodeConfig` (its ``block_tokens``).

    Returns:
        A short reason, or ``None``.
    """
    shards = _head_shards(mesh, head_axis)
    num_heads = int(beta.shape[-1])
    if num_heads % shards:
        return f"{num_heads} heads do not split over {shards} devices"

    def local(x: tp.Any, axis: int) -> jax.ShapeDtypeStruct:
        shape = list(x.shape)
        shape[axis] //= shards
        return jax.ShapeDtypeStruct(tuple(shape), x.dtype)

    return unsupported_reason(
        local(query, 1),
        local(key, 1),
        local(value, 1),
        local(beta, 1),
        local(decay, 1),
        tuple(local(w, 0) for w in conv_weights),
        tuple(local(p, 1) for p in conv_pools),
        local(recurrent_state, 1),
        query.dtype if conv_dtype is None else conv_dtype,
        block_tokens=int(_normalize_cfg(cfg).block_tokens),
    )


def ragged_kda_conv_decode_head_sharded(
    query: Float[Array, "num_tokens key_dim"],
    key: Float[Array, "num_tokens key_dim"],
    value: Float[Array, "num_tokens value_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    conv_weights: tuple[Array, Array, Array],
    conv_pools: tuple[Array, Array, Array],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    token_slots: Int[Array, "num_tokens"],
    token_init: Bool[Array, "num_tokens"],
    /,
    *,
    mesh: object | None,
    head_axis: object | None,
    conv_dtype: tp.Any = None,
    use_qk_l2norm: bool = True,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: RaggedKDAConvDecodeConfig | None = None,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Run :func:`ragged_kda_conv_decode` with the heads (and the q / k / v features, head-major) split over a mesh axis.

    With a mesh whose ``head_axis`` has more than one device, every device advances its own heads of every token under
    ``jax.shard_map`` (tokens and slot tables replicated; conv windows, kernels and states split with the features),
    with no collectives. Otherwise the call is unsharded.

    Args:
        query: Raw queries ``[T, H*K]``.
        key: Raw keys ``[T, H*K]``.
        value: Raw values ``[T, H*V]``.
        beta: Write strengths ``[T, H]``.
        decay: Per-channel log-decay ``[T, H, K]``.
        conv_weights: ``(q, k, v)`` ``[dim, d_conv]``.
        conv_pools: ``(q, k, v)`` ``[S, dim, d_conv]``.
        recurrent_state: ``[S, H, K, V]``.
        token_slots: ``[T]`` pool slot advanced by each token.
        token_init: ``[T]`` whether that slot's windows and state carry over.
        mesh: Mesh carrying ``head_axis`` (or ``None``).
        head_axis: Mesh axis name (or tuple of names) the heads are sharded over (or ``None``).
        conv_dtype: Dtype of the conv outputs (``None``: the query dtype).
        use_qk_l2norm: L2-normalise queries and keys.
        platform: Optional platform override.
        cfg: Optional :class:`RaggedKDAConvDecodeConfig`.

    Returns:
        ``(out [T, H, V] fp32, (q_pool, k_pool, v_pool, recurrent_state))``.

    Raises:
        ValueError: If the head count is not divisible by the size of ``head_axis``.
    """
    shards = _head_shards(mesh, head_axis)
    kwargs = dict(conv_dtype=conv_dtype, use_qk_l2norm=use_qk_l2norm, platform=platform, cfg=cfg)
    arrays = (query, key, value, beta, decay, tuple(conv_weights), tuple(conv_pools), recurrent_state)
    if shards <= 1:
        return ragged_kda_conv_decode(*arrays, token_slots, token_init, **kwargs)
    num_heads = int(beta.shape[-1])
    if num_heads % shards:
        raise ValueError(f"ragged_kda_conv_decode: {num_heads} heads are not divisible by the {shards}-way {head_axis}")
    features = PartitionSpec(None, head_axis)
    window = PartitionSpec(None, head_axis, None)
    state = PartitionSpec(None, head_axis, None, None)
    taps = PartitionSpec(head_axis, None)
    replicated = PartitionSpec()
    in_specs = (
        features,
        features,
        features,
        features,
        window,
        (taps, taps, taps),
        (window, window, window),
        state,
        replicated,
        replicated,
    )
    out_specs = (window, (window, window, window, state))
    return ragged_kda_conv_decode(
        *arrays, token_slots, token_init, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs
    )


__all__ = (
    "RaggedKDAConvDecode",
    "ragged_kda_conv_decode",
    "ragged_kda_conv_decode_head_sharded",
    "ragged_kda_conv_decode_unsupported_reason",
)

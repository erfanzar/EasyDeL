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

"""mHC (manifold-constrained hyper-connection) collapse / expand operations.

The residual of DeepSeek-V4-class decoders is ``hc`` parallel streams ``[B, S, hc, W]``. At every sub-layer site
``mhc_collapse`` predicts the mixing gates from an RMS-normalised projection of the streams and collapses them into
the sub-layer input; ``mhc_expand`` writes the sub-layer output back onto the streams through the Sinkhorn-projected
``comb`` mixer. See :mod:`ejkernel.kernels._xla.mhc_fused._xla_impl_fwd` for the math.

Platform support:
    - XLA (``Backend.ANY``): the reference and the DEFAULT. The heuristic configuration selects it on every
      backend, because on TPU v5p the XLA site measured faster than the fused one end to end.
    - Pallas (``Backend.TPU``): one sweep over the streams per pass with an analytic backward. Selected only by an
      explicit ``platform="pallas"`` / ``cfg``. Its collapse is faster than XLA's at long sequences; its expand is
      VPU-bound and slower (numbers in :mod:`ejkernel.kernels._pallas.tpu.mhc_fused._pallas_impl`).

Both operations support ``method="shard_map"`` through the executor; the hidden (last) axis of the streams' spec
names the reduction axis of the two hidden contractions, and the body must run with ``check_vma=False``.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from jax import Array
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Float

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel

from ..base import detect_platform
from .configs import MHCFusedConfig

_PlatformArg = Literal["pallas", "xla", "auto"] | None


def _resolve_cfg(cfg: MHCFusedConfig, platform: _PlatformArg) -> MHCFusedConfig:
    """Apply a per-call ``platform`` override to ``cfg`` (XLA always uses ``backend="any"``)."""
    if platform is None:
        return cfg
    return dataclasses.replace(cfg, platform=platform, backend="any" if platform == "xla" else cfg.backend)


def _hidden_axis(spec: PartitionSpec | None):
    """The mesh axis (or axes) of the hidden (last) entry of a ``[B, S, hc, W]`` stream spec, or ``None``."""
    if spec is None or len(spec) < 4:
        return None
    entry = spec[3]
    if entry is None or (isinstance(entry, tuple) and not entry):
        return None
    return entry


def _check_shard_map_args(mesh, in_specs, out_specs, check_vma: bool) -> None:
    if mesh is None or in_specs is None or out_specs is None:
        raise ValueError("mhc: mesh, in_specs and out_specs are required for shard_map execution")
    if check_vma:
        raise ValueError(
            "mhc: shard_map execution requires check_vma=False (the backward rules return per-device partial "
            "gate / parameter cotangents that the check_vma=False transpose reduces)"
        )


class MHCCollapse(Kernel[MHCFusedConfig, tuple[Array, Array, Array]]):
    """Gate prediction and read-weighted collapse of one mHC site: ``(post, comb, collapsed)``."""

    def __init__(self):
        super().__init__(op_id="mhc_collapse")

    def get_impl(self, cfg: MHCFusedConfig):
        """Resolve the registered backend for ``cfg``.

        Args:
            cfg: Dispatch configuration.

        Returns:
            The registered implementation callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "batch seq hc width"],
        fn: Float[Array, "mix hc width"],
        base: Float[Array, "mix"],
        scale: Float[Array, "three"],
        hidden_size: int | None = None,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        rms_eps: float = 1e-6,
        axis_name: str | tuple[str, ...] | None = None,
        platform: _PlatformArg = None,
        *,
        cfg: MHCFusedConfig,
    ) -> tuple[Float[Array, "batch seq hc"], Float[Array, "batch seq hc hc"], Float[Array, "batch seq width"]]:
        """Dispatch to the selected backend.

        Args:
            x: Residual streams ``[B, S, hc, W]``.
            fn: Gate predictor ``[(2 + hc) hc, hc, W]``.
            base: Gate biases ``[(2 + hc) hc]``.
            scale: Logit scales ``[3]``.
            hidden_size: Full hidden width for the RMS denominator (``None``: from the streams).
            sinkhorn_iters: Sinkhorn-Knopp iterations.
            eps: Sinkhorn / ``pre`` floor.
            rms_eps: RMS epsilon.
            axis_name: Mesh axis the hidden is sharded over (inside ``shard_map``), or ``None``.
            platform: Optional per-call platform override.
            cfg: Dispatch configuration (tile rows and MXU pieces for Pallas).

        Returns:
            ``post [B, S, hc]`` fp32, ``comb [B, S, hc, hc]`` fp32, ``collapsed [B, S, W]`` in ``x.dtype``.
        """
        cfg = _resolve_cfg(cfg, platform)
        return self.get_impl(cfg)(
            x,
            fn,
            base,
            scale,
            hidden_size=hidden_size,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            rms_eps=rms_eps,
            axis_name=axis_name,
            tile_rows=cfg.tile_rows,
            fn_pieces=cfg.fn_pieces,
            grad_pieces=cfg.grad_pieces,
        )

    def heuristic_cfg(self, inv: Invocation[MHCFusedConfig, tuple[Array, Array, Array]]) -> MHCFusedConfig:
        """XLA by default (measured faster end to end); Pallas only on request."""
        return MHCFusedConfig(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[MHCFusedConfig, tuple[Array, Array, Array]]):
        """Single deterministic configuration (no autotuning)."""
        return [self.heuristic_cfg(inv)]

    def create_shard_map_wrapper(
        self,
        x: Float[Array, "batch seq hc width"],
        fn: Float[Array, "mix hc width"],
        base: Float[Array, "mix"],
        scale: Float[Array, "three"],
        hidden_size: int | None = None,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        rms_eps: float = 1e-6,
        axis_name: str | tuple[str, ...] | None = None,
        platform: _PlatformArg = None,
        *,
        cfg: MHCFusedConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` over ``(x, fn, base, scale)``.

        ``axis_name`` defaults to the hidden entry of ``in_specs[0]``; the stats and the logit cotangent are summed
        over it inside the body. ``base`` / ``scale`` should be replicated (``P()``).

        Args:
            x: Residual streams ``[B, S, hc, W]`` (global).
            fn: Gate predictor ``[(2 + hc) hc, hc, W]`` (global).
            base: Gate biases.
            scale: Logit scales.
            hidden_size: Full hidden width (``None``: local width times the hidden-axis size).
            sinkhorn_iters: Sinkhorn-Knopp iterations.
            eps: Sinkhorn / ``pre`` floor.
            rms_eps: RMS epsilon.
            axis_name: Hidden-shard mesh axis override.
            platform: Optional per-call platform override.
            cfg: Dispatch configuration.
            mesh: Device mesh.
            in_specs: Specs for ``(x, fn, base, scale)``.
            out_specs: Specs for ``(post, comb, collapsed)``.
            check_vma: Must be ``False``.

        Returns:
            ``(shard_map_fn, call_args)``.
        """
        import jax

        _check_shard_map_args(mesh, in_specs, out_specs, check_vma)
        axis = axis_name if axis_name is not None else _hidden_axis(in_specs[0])

        def body(x_, fn_, base_, scale_):
            return self.run(
                x_,
                fn_,
                base_,
                scale_,
                hidden_size=hidden_size,
                sinkhorn_iters=sinkhorn_iters,
                eps=eps,
                rms_eps=rms_eps,
                axis_name=axis,
                platform=platform,
                cfg=cfg,
            )

        shard_map_fn = jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        return shard_map_fn, (x, fn, base, scale)


class MHCExpand(Kernel[MHCFusedConfig, Array]):
    """Residual write of one mHC site: ``post * out + comb^T @ streams`` (gates quantised to the stream dtype)."""

    def __init__(self):
        super().__init__(op_id="mhc_expand")

    def get_impl(self, cfg: MHCFusedConfig):
        """Resolve the registered backend for ``cfg``.

        Args:
            cfg: Dispatch configuration.

        Returns:
            The registered implementation callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "batch seq hc width"],
        out: Float[Array, "batch seq width"],
        post: Float[Array, "batch seq hc"],
        comb: Float[Array, "batch seq hc hc"],
        platform: _PlatformArg = None,
        *,
        cfg: MHCFusedConfig,
    ) -> Float[Array, "batch seq hc width"]:
        """Dispatch to the selected backend.

        Args:
            x: Residual streams ``[B, S, hc, W]``.
            out: Sub-layer output ``[B, S, W]``.
            post: Write gates ``[B, S, hc]``.
            comb: Mixer ``[B, S, hc, hc]`` indexed ``[source, target]``.
            platform: Optional per-call platform override.
            cfg: Dispatch configuration.

        Returns:
            New streams ``[B, S, hc, W]`` in ``x.dtype``.
        """
        cfg = _resolve_cfg(cfg, platform)
        return self.get_impl(cfg)(x, out, post, comb, tile_rows=cfg.tile_rows)

    def heuristic_cfg(self, inv: Invocation[MHCFusedConfig, Array]) -> MHCFusedConfig:
        """XLA by default (the Pallas expand is VPU-bound and slower); Pallas only on request."""
        return MHCFusedConfig(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[MHCFusedConfig, Array]):
        """Single deterministic configuration (no autotuning)."""
        return [self.heuristic_cfg(inv)]

    def create_shard_map_wrapper(
        self,
        x: Float[Array, "batch seq hc width"],
        out: Float[Array, "batch seq width"],
        post: Float[Array, "batch seq hc"],
        comb: Float[Array, "batch seq hc hc"],
        platform: _PlatformArg = None,
        *,
        cfg: MHCFusedConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` over ``(x, out, post, comb)``; the body has no collective.

        Args:
            x: Residual streams (global).
            out: Sub-layer output (global).
            post: Write gates (global).
            comb: Mixer (global).
            platform: Optional per-call platform override.
            cfg: Dispatch configuration.
            mesh: Device mesh.
            in_specs: Specs for ``(x, out, post, comb)``.
            out_specs: Spec for the new streams.
            check_vma: Must be ``False``.

        Returns:
            ``(shard_map_fn, call_args)``.
        """
        import jax

        _check_shard_map_args(mesh, in_specs, out_specs, check_vma)

        def body(x_, out_, post_, comb_):
            return self.run(x_, out_, post_, comb_, platform=platform, cfg=cfg)

        shard_map_fn = jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        return shard_map_fn, (x, out, post, comb)


_collapse_executor: Executor[MHCFusedConfig, tuple[Array, Array, Array]] = Executor(
    ConfigSelectorChain(cache=ConfigCache(), policy=AutotunePolicy(allow_autotune=False))
)
_expand_executor: Executor[MHCFusedConfig, Array] = Executor(
    ConfigSelectorChain(cache=ConfigCache(), policy=AutotunePolicy(allow_autotune=False))
)


def mhc_collapse(
    x: Float[Array, "batch seq hc width"],
    fn: Float[Array, "mix hc width"],
    base: Float[Array, "mix"],
    scale: Float[Array, "three"],
    /,
    hidden_size: int | None = None,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    rms_eps: float = 1e-6,
    axis_name: str | tuple[str, ...] | None = None,
    *,
    platform: _PlatformArg = None,
    cfg: MHCFusedConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec, ...] | None = None,
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> tuple[Float[Array, "batch seq hc"], Float[Array, "batch seq hc hc"], Float[Array, "batch seq width"]]:
    """``(post, comb, collapsed)`` of one mHC site.

    Args:
        x: Residual streams ``[B, S, hc, W]`` (``W`` may be a tensor-parallel shard of the hidden).
        fn: Gate predictor ``[(2 + hc) hc, hc, W]``, the same shard.
        base: Gate biases ``[(2 + hc) hc]``.
        scale: Logit scales ``[3]`` (pre, post, comb).
        hidden_size: Full hidden width for the RMS denominator ``hc * hidden_size``; ``None`` takes it from the
            streams (local width times the hidden-axis size).
        sinkhorn_iters: Sinkhorn-Knopp iterations for ``comb``.
        eps: Sinkhorn / ``pre`` floor.
        rms_eps: RMS epsilon.
        axis_name: Mesh axis the hidden is sharded over when already inside ``shard_map``, or ``None``.
        platform: Implementation override; default (``None``) is XLA via the heuristic configuration.
        cfg: Optional dispatch configuration (``tile_rows`` / ``fn_pieces`` / ``grad_pieces`` for Pallas).
        mesh: When given with ``in_specs`` / ``out_specs``, run under the executor's ``shard_map`` path.
        in_specs: Specs for ``(x, fn, base, scale)``.
        out_specs: Specs for ``(post, comb, collapsed)``.

    Returns:
        ``post [B, S, hc]`` fp32, ``comb [B, S, hc, hc]`` fp32, ``collapsed [B, S, W]`` in ``x.dtype``.
    """
    kwargs = dict(
        hidden_size=hidden_size,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        rms_eps=rms_eps,
        axis_name=axis_name,
        platform=platform,
        _cfg=cfg,
    )
    if mesh is not None:
        return _collapse_executor(
            MHCCollapse(),
            x,
            fn,
            base,
            scale,
            method="shard_map",
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
            **kwargs,
        )
    return _collapse_executor(MHCCollapse(), x, fn, base, scale, **kwargs)


def mhc_expand(
    x: Float[Array, "batch seq hc width"],
    out: Float[Array, "batch seq width"],
    post: Float[Array, "batch seq hc"],
    comb: Float[Array, "batch seq hc hc"],
    /,
    *,
    platform: _PlatformArg = None,
    cfg: MHCFusedConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch seq hc width"]:
    """New streams ``post[..., None] * out[..., None, :] + einsum("bsji,bsjd->bsid", comb, x)``.

    ``post`` / ``comb`` are cast to ``x.dtype`` before the multiply (the residual write's quantisation).

    Args:
        x: Residual streams ``[B, S, hc, W]``.
        out: Sub-layer output ``[B, S, W]``.
        post: Write gates ``[B, S, hc]``.
        comb: Mixer ``[B, S, hc, hc]`` indexed ``[source, target]``.
        platform: Implementation override; default (``None``) is XLA via the heuristic configuration.
        cfg: Optional dispatch configuration (``tile_rows`` for Pallas).
        mesh: When given with ``in_specs`` / ``out_specs``, run under the executor's ``shard_map`` path.
        in_specs: Specs for ``(x, out, post, comb)``.
        out_specs: Spec for the new streams.

    Returns:
        New streams ``[B, S, hc, W]`` in ``x.dtype``.
    """
    if mesh is not None:
        return _expand_executor(
            MHCExpand(),
            x,
            out,
            post,
            comb,
            platform=platform,
            _cfg=cfg,
            method="shard_map",
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
    return _expand_executor(MHCExpand(), x, out, post, comb, platform=platform, _cfg=cfg)


__all__ = ("MHCCollapse", "MHCExpand", "MHCFusedConfig", "mhc_collapse", "mhc_expand")

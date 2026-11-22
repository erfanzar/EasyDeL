# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Factory for the ``ParallelLinear.distributed_matmul`` injectable.

Builds explicit-collective matmul engines backed by the unified ejkernel
collective ops (``all_reduce`` today; SparseCore / fused variants slot in
later without touching any model code).  The engine replaces only the einsum
in :meth:`ParallelLinear.forward`; scaling and bias stay in the layer.

Engine semantics per direction on the canonical EasyDeL TP layout:

    - ``"row"``: activations carry a TP-sharded feature axis, the kernel is
      ``PartitionSpec(tp, None)``.  The engine runs a local dot inside
      ``shard_map`` and finishes with the layout-matched differentiable
      collective: a ``reduce_scatter`` when the resolved hidden state keeps
      its feature dim TP-sharded (the canonical EasyDeL layout — half the
      wire bytes of an all-reduce), otherwise an ``all_reduce``.  Leading
      (batch/sequence) dims keep their model-resolved sharding — specs are
      derived from ``config.runtime_sharding_resolver`` at call time, so
      dp/fsdp/sp-sharded training batches stay sharded.
    - ``"column"``: the canonical layout needs no collective in the forward
      pass (activations replicated over tp, kernel column-sharded), so no
      engine is built and the factory returns ``None``.

Decline protocol: the returned callable gives back ``None`` whenever it
cannot take over (no mesh on the operands, no active TP axis, shapes not
divisible by the TP degree, the TP axis appearing on a leading dim, or a
weight whose declared/actual layout is not exactly the canonical
row-parallel ``P(tp, None)`` — e.g. the RowWise training layout with an
(fsdp, sp)-sharded output dim) and :meth:`ParallelLinear.forward` falls
through to the plain einsum, keeping GSPMD behavior.  Under tracing the
operands carry no ``NamedSharding``; the seam then consults the layout
DECLARED at wiring time through ``config.runtime_sharding_resolver``
instead of blindly pinning the canonical layout.

``"auto"`` currently resolves to the ``"xla"`` engine: MEASURED (v5p-8,
2026-08-10) it compiles to a program bit-identical to the GSPMD einsum path
(jitted train-step loss/grad deltas 0.0), while keeping the wiring live so
future engines (``"sparse_core"``, quantized fusions, multi-slice
hierarchies) are a config flip.  The Pallas ring engines measured slower
than XLA on single-slice v5p and stay explicit opt-ins; see
``.claude/projects/checkpoint-quantization.md``.
"""

from __future__ import annotations

import typing as tp

import jax
from ejkernel.modules import AllReduceConfig, ReduceScatterConfig
from ejkernel.modules import all_reduce as ej_all_reduce
from ejkernel.modules import reduce_scatter as ej_reduce_scatter
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec
from spectrax import common_types

from easydel.infra.sharding import partition_spec_for_mesh, pick_array_mesh, resolve_stage_mesh

from ._linear_quantized import _axis_names, _extract_tp_axis_name

if tp.TYPE_CHECKING:
    from jaxtyping import Array

    from easydel.infra.base_config import EasyDeLBaseConfig

VALID_COLLECTIVE_MATMUL_IMPLS: tuple[str, ...] = ("none", "auto", "xla", "pallas_ring")

_IMPL_TO_PLATFORM: dict[str, str] = {"auto": "xla", "xla": "xla", "pallas_ring": "pallas"}
_IMPL_TO_BACKEND: dict[str, str] = {"auto": "any", "xla": "any", "pallas_ring": "tpu"}


def _resolve_hidden_layout(
    config: EasyDeLBaseConfig | None,
    shape: tuple[int, ...],
    mesh: tp.Any,
    tp_axis: str,
) -> tuple[tuple[tp.Any, ...], str | None] | None:
    """Resolve the mode-appropriate hidden-state layout for the engine.

    Uses the config's runtime sharding resolver (the same source model code
    uses for hidden-state constraints) so dp/fsdp-sharded batches and
    sp-sharded sequences stay sharded inside the engine's ``shard_map``, and
    so the collective matches the downstream layout: when the resolved
    hidden state keeps its feature dim TP-sharded (the canonical EasyDeL
    layout), the comm-optimal finisher is a reduce-scatter, not an
    all-reduce.

    Args:
        config: The model config carrying ``runtime_sharding_resolver``;
            ``None`` yields fully replicated leading dims and an all-reduce
            finisher.
        shape: Full input shape (including the trailing feature dim).
        mesh: Active mesh (axis-membership/divisibility checks).
        tp_axis: The TP axis name; must not shard a leading dim for a
            row-parallel reduction.

    Returns:
        ``(leading_spec_entries, out_feature_axis)`` where
        ``out_feature_axis`` is ``tp_axis`` when the output should stay
        feature-sharded (reduce-scatter finisher) or ``None`` for a
        replicated output (all-reduce finisher); or ``None`` when the
        resolved layout is unusable (TP on a leading dim, or a dim not
        divisible by its mesh axes).
    """
    leading: list[tp.Any] = [None] * (len(shape) - 1)
    out_feature_axis: str | None = None
    if config is not None:
        resolved = config.runtime_sharding_resolver.resolve(
            dynamic_axes=common_types.HiddenStateSharding,
            shape=tuple(int(dim) for dim in shape),
        )
        if len(resolved) == len(shape):
            resolved_tuple = tuple(resolved)
            leading = list(resolved_tuple[:-1])
            if tp_axis in _axis_names(resolved_tuple[-1]):
                out_feature_axis = tp_axis

    for dim, entry in zip(shape[:-1], leading, strict=False):
        names = _axis_names(entry)
        if tp_axis in names:
            return None
        group = 1
        for name in names:
            if name not in mesh.axis_names:
                return None
            group *= int(mesh.shape[name])
        if group > 1 and int(dim) % group != 0:
            return None
    return tuple(leading), out_feature_axis


def make_distributed_matmul(
    direction: tp.Literal["row", "column"] | None,
    collective_matmul_impl: str,
    precision: jax.lax.PrecisionLike = None,
    config: EasyDeLBaseConfig | None = None,
    weight_layout: tp.Any = None,
) -> tp.Callable[[Array, Array], Array | None] | None:
    """Build a collective-matmul engine for a :class:`ParallelLinear` layer.

    Args:
        direction: The layer's parallelism direction (``_direction``).  Only
            ``"row"`` layers get an engine; everything else returns ``None``.
        collective_matmul_impl: Engine selector from
            ``EasyDeLBaseConfig.collective_matmul_impl`` — ``"none"`` (no
            engine), ``"auto"`` (best measured engine; currently the XLA
            one), ``"xla"``, or ``"pallas_ring"``.
        precision: The layer's matmul precision, applied to the local dot.
        config: The model config; supplies the runtime sharding resolver so
            leading (batch/sequence) dims keep their sharding and so the
            weight's DECLARED layout can be resolved under tracing.  ``None``
            treats leading dims as replicated and restricts the engine to
            eagerly-sharded operands (tracing declines).
        weight_layout: The layer's declared weight sharding layout
            (``ParallelLinear.sharding_layout``).  ``None`` falls back to the
            constructor default for row-parallel layers
            (:class:`spectrax.common_types.RowWise`).

    Returns:
        A callable ``(inputs, kernel) -> Array | None`` implementing the
        decline protocol, or ``None`` when no engine applies.

    Raises:
        ValueError: If ``collective_matmul_impl`` is not a known engine name.
    """
    if collective_matmul_impl not in VALID_COLLECTIVE_MATMUL_IMPLS:
        raise ValueError(
            f"collective_matmul_impl must be one of {VALID_COLLECTIVE_MATMUL_IMPLS}, got {collective_matmul_impl!r}."
        )
    if collective_matmul_impl == "none" or direction != "row":
        return None

    ar_cfg = AllReduceConfig(
        mode="auto",
        platform=_IMPL_TO_PLATFORM[collective_matmul_impl],
        backend=_IMPL_TO_BACKEND[collective_matmul_impl],
    )
    rs_cfg = ReduceScatterConfig(
        mode="auto",
        platform=_IMPL_TO_PLATFORM[collective_matmul_impl],
        backend=_IMPL_TO_BACKEND[collective_matmul_impl],
    )

    def _active_axis_names(entry: tp.Any, mesh: tp.Any) -> tuple[str, ...]:
        """Mesh axes in *entry* that actually partition (size > 1)."""
        return tuple(name for name in _axis_names(entry) if name in mesh.axis_names and int(mesh.shape[name]) > 1)

    def _declared_kernel_spec(mesh: tp.Any, kernel_shape: tuple[int, ...]) -> PartitionSpec | None:
        """Resolve the wiring-time DECLARED weight layout on ``mesh``.

        Used when the kernel operand carries no ``NamedSharding`` (tracing).
        ``None`` means the layout cannot be resolved and the engine declines.
        """
        if config is None:
            return None
        layout = weight_layout if weight_layout is not None else common_types.RowWise
        try:
            return config.runtime_sharding_resolver.resolve_layout(layout, shape=kernel_shape, mesh=mesh)
        except Exception:
            return None

    def distributed_matmul(inputs: Array, kernel: Array) -> Array | None:
        """Row-parallel dot + explicit all-reduce; ``None`` declines to einsum."""
        if kernel.ndim != 2 or inputs.ndim < 2:
            return None
        mesh = pick_array_mesh(inputs, kernel)
        if mesh is None:
            return None
        mesh = resolve_stage_mesh(mesh, arr=inputs)
        if mesh is None:
            return None

        kernel_spec = partition_spec_for_mesh(kernel, mesh)
        if _extract_tp_axis_name(kernel_spec, "row", mesh) is None:
            # Under tracing the operands carry no NamedSharding. Resolve the
            # layout DECLARED at wiring time through the config's sharding
            # resolver — never a blind canonical pin — so the decline checks
            # below judge the layout the model actually assigns.
            kernel_spec = _declared_kernel_spec(mesh, tuple(int(dim) for dim in kernel.shape))
            if kernel_spec is None:
                return None
        tp_axis = _extract_tp_axis_name(kernel_spec, "row", mesh)
        if tp_axis is None:
            return None
        tp_size = int(mesh.shape[tp_axis])
        if tp_size <= 1:
            return None
        if int(kernel.shape[0]) % tp_size != 0 or int(inputs.shape[-1]) % tp_size != 0:
            return None
        # Decline protocol: the fused op supports exactly the canonical
        # row-parallel weight layout P(tp, None). A weight whose output dim is
        # (fsdp, sp)-sharded (the RowWise training layout at fsdp/sp > 1) or
        # whose input dim is co-sharded with other axes would be force-
        # gathered / re-laid-out by the shard_map in_specs — decline to the
        # einsum path so GSPMD keeps the declared layout.
        if _active_axis_names(kernel_spec[0], mesh) != (tp_axis,):
            return None
        if len(kernel_spec) > 1 and _active_axis_names(kernel_spec[1], mesh):
            return None

        layout = _resolve_hidden_layout(config, tuple(inputs.shape), mesh, tp_axis)
        if layout is None:
            return None
        leading, out_feature_axis = layout
        if out_feature_axis is not None and int(kernel.shape[-1]) % tp_size != 0:
            out_feature_axis = None

        if out_feature_axis is None:

            def _body(x_local: Array, w_local: Array) -> Array:
                y_partial = jnp.einsum("...i,io->...o", x_local, w_local, precision=precision)
                return ej_all_reduce(y_partial, tp_axis, cfg=ar_cfg)

        else:

            def _body(x_local: Array, w_local: Array) -> Array:
                y_partial = jnp.einsum("...i,io->...o", x_local, w_local, precision=precision)
                return ej_reduce_scatter(
                    y_partial,
                    tp_axis,
                    scatter_axis=y_partial.ndim - 1,
                    cfg=rs_cfg,
                )

        return shard_map(
            _body,
            mesh=mesh,
            in_specs=(PartitionSpec(*leading, tp_axis), PartitionSpec(tp_axis, None)),
            out_specs=PartitionSpec(*leading, out_feature_axis),
            check_vma=False,
        )(inputs, kernel)

    return distributed_matmul


__all__ = ("VALID_COLLECTIVE_MATMUL_IMPLS", "make_distributed_matmul")

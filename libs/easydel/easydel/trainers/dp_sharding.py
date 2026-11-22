# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Single-mesh ``spmd_axis_name='dp'`` backward lowering and deferred post-scan ``dp`` reduction."""

from __future__ import annotations

import collections.abc
import functools
import re
import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.loss_utils import LossMetrics
from easydel.infra.sharding import (
    active_dp_spmd_axis,
    dp_spmd_vmap_scope,
    is_mpmd_mesh,
    strip_compound_axis_from_spec,
)

_HOOKS_INSTALLED = False
_ORIG_SHARD_MAP = jax.shard_map

DP_REDUCE_COMPILE_OPTIONS: dict[str, str] = {
    "xla_tpu_enable_async_collective_fusion": "true",
    "xla_tpu_enable_async_collective_fusion_fuse_all_reduce": "true",
    "xla_tpu_enable_sparse_core_collective_offload_all_reduce": "false",
}


def install_dp_spmd_hooks() -> None:
    """Install idempotent trace-time hooks for ``dp_spmd_vmap_scope`` and singleton ``shard_map`` unbatching."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    import eformer.escale.partition.manager as ef_mgr
    import spectrax.sharding.manager as spx_mgr
    import spectrax.sharding.partition as spx_part
    from jax._src import core
    from jax._src import shard_map as jax_smap
    from jax._src.interpreters import batching

    orig_spx_ax = spx_mgr.PartitionAxis.resolve_axis
    orig_ef_ax = ef_mgr.PartitionAxis.resolve_axis
    orig_san = spx_part.explain_partition_spec_sanitization
    orig_smap_batch = jax_smap._shard_map_batch

    def spx_resolve_axis(self, *args, **kwargs):
        out = orig_spx_ax(self, *args, **kwargs)
        dp_ax = active_dp_spmd_axis()
        return strip_compound_axis_from_spec(out, dp_ax) if dp_ax is not None else out

    def ef_resolve_axis(self, *args, **kwargs):
        out = orig_ef_ax(self, *args, **kwargs)
        dp_ax = active_dp_spmd_axis()
        return strip_compound_axis_from_spec(out, dp_ax) if dp_ax is not None else out

    def san(spec, *, mesh=None, shape=None):
        dp_ax = active_dp_spmd_axis()
        if dp_ax is not None and spec is not None:
            ps = spec if isinstance(spec, PartitionSpec) else PartitionSpec(*tuple(spec))
            spec = strip_compound_axis_from_spec(ps, dp_ax)
        return orig_san(spec, mesh=mesh, shape=shape)

    def unbatched_singleton_smap_batch(
        trace, prim, fun, in_tracers, mesh, in_specs, check_vma, newly_manual_axes, debug_info
    ):
        spmd_axis_name = trace.axis_data.spmd_name
        if spmd_axis_name is not None and trace.axis_data.name is core.no_axis_name:
            in_vals, in_dims = jax_smap.unzip2(map(trace.to_batch_info, in_tracers))
            used = {n for spec in in_specs for n in jax_smap.used_axis_names(spec)}
            if not (set(spmd_axis_name) & used) and all(n in mesh.shape for n in spmd_axis_name):
                sz_prod = jax_smap.prod(mesh.shape[n] for n in spmd_axis_name)
                if trace.axis_data.size == sz_prod:
                    new_in_specs = [
                        sp if d is None else jax_smap.pxla.batch_spec(sp, d, spmd_axis_name)
                        for sp, d in zip(in_specs, in_dims, strict=True)
                    ]
                    any_batched = any(d is not None for d in in_dims)

                    def fun_unbatched(*args):
                        u_args = [
                            a if d is None else jax.lax.squeeze(a, (d,))
                            for a, d in zip(args, in_dims, strict=True)
                        ]
                        ans, out_specs = fun(*u_args).unpack_aux()
                        if any_batched:
                            out_dims = (0,) * len(ans)
                            ans = ans.map(lambda o: jax.lax.expand_dims(o, (0,)))
                        else:
                            out_dims = (None,) * len(ans)
                        new_out_specs = jax_smap._batch_out_specs(
                            spmd_axis_name, trace.axis_data.explicit_mesh_axis, out_dims, out_specs
                        )
                        return ans.with_aux(out_dims).with_aux(tuple(new_out_specs))

                    new_params = dict(
                        mesh=mesh,
                        in_specs=new_in_specs,
                        check_vma=False,
                        newly_manual_axes=newly_manual_axes,
                        debug_info=debug_info,
                    )
                    with (
                        core.set_current_trace(trace.parent_trace),
                        core.remove_explicit_mesh_axis_names(trace.axis_data.explicit_mesh_axis),
                    ):
                        out_vals = prim.bind(*in_vals, subfuns=(fun_unbatched,), **new_params)
                    make_tracer = functools.partial(
                        batching.BatchTracer, trace, source_info=jax_smap.source_info_util.current()
                    )
                    out_vals, out_dims = out_vals.unpack_aux()
                    return out_vals.map2(out_dims, make_tracer)
        return orig_smap_batch(
            trace, prim, fun, in_tracers, mesh, in_specs, check_vma, newly_manual_axes, debug_info
        )

    spx_mgr.PartitionAxis.resolve_axis = spx_resolve_axis
    ef_mgr.PartitionAxis.resolve_axis = ef_resolve_axis
    spx_part.explain_partition_spec_sanitization = san
    jax_smap._shard_map_batch = unbatched_singleton_smap_batch
    batching.BatchTrace.process_shard_map = unbatched_singleton_smap_batch
    _HOOKS_INSTALLED = True


def _unescape_regex_literal(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def resolve_graphstate_specs(state: tp.Any, mesh: tp.Any, tree: tp.Any | None = None) -> tp.Any:
    """Resolve a ``PartitionSpec`` pytree for ``tree`` (defaults to ``state.graphstate``)."""
    from easydel.infra.utils import jax_path_to_string

    if tree is None:
        tree = state.graphstate
    model = getattr(state, "model", None)
    if model is None:
        return jtu.tree_map(
            lambda x: x.sharding.spec if isinstance(getattr(x, "sharding", None), NamedSharding) else PartitionSpec(),
            tree,
        )

    prefix, suffix = r"^(?:.*/)?", r"(?:/.*)?$"
    rules = model.resolve_shardings_regex(mesh=mesh)
    literal: dict[str, int] = {}
    regex_rules: list[tuple[int, re.Pattern[str]]] = []
    for idx, (pattern, _sh) in enumerate(rules):
        lit = None
        if pattern.startswith(prefix) and pattern.endswith(suffix):
            cand = _unescape_regex_literal(pattern[len(prefix) : -len(suffix)])
            if prefix + "/".join(re.escape(c) for c in cand.split("/")) + suffix == pattern:
                lit = cand
        if lit is None:
            regex_rules.append((idx, re.compile(pattern)))
        elif lit not in literal:
            literal[lit] = idx

    def _to_spec(sharding, leaf) -> PartitionSpec:
        spec = sharding.spec if isinstance(sharding, NamedSharding) else PartitionSpec(*tuple(sharding))
        if hasattr(leaf, "ndim") and len(spec) > leaf.ndim:
            spec = PartitionSpec(*tuple(spec)[: leaf.ndim])
        return spec

    def _spec_for_path(path_str: str, leaf) -> PartitionSpec:
        if not hasattr(leaf, "shape"):
            return PartitionSpec()
        xsh = getattr(leaf, "sharding", None)
        if isinstance(xsh, NamedSharding) and len(xsh.spec) > 0:
            return xsh.spec
        comps = path_str.split("/")
        best: int | None = None
        for i in range(len(comps)):
            acc = comps[i]
            for j in range(i + 1, len(comps) + 1):
                if j > i + 1:
                    acc = acc + "/" + comps[j - 1]
                k = literal.get(acc)
                if k is not None and (best is None or k < best):
                    best = k
        for idx, rx in regex_rules:
            if best is not None and idx > best:
                break
            if rx.search(path_str):
                best = idx
                break
        return PartitionSpec() if best is None else _to_spec(rules[best][1], leaf)

    return jtu.tree_map_with_path(lambda p, leaf: _spec_for_path(jax_path_to_string(p), leaf), tree)


def _spec_mentions_axis(spec: PartitionSpec, axis_name: str) -> bool:
    for entry in spec:
        if entry == axis_name or (isinstance(entry, tuple) and axis_name in entry):
            return True
    return False


def _is_mrope_position_leaf(key: tp.Any, leaf: tp.Any, batch_size: int) -> bool:
    """Return whether ``leaf`` is a rank-3 mRoPE ``(3, batch_size, seq_len)`` position tensor."""
    if not isinstance(key, str) or not (key == "position_ids" or key.endswith("_position_ids")):
        return False
    shape = getattr(leaf, "shape", None)
    return shape is not None and len(shape) == 3 and int(shape[0]) == 3 and int(shape[1]) == batch_size


def _batch_compatible_with_dp_vmap(batch: collections.abc.Mapping[str, tp.Any], batch_size: int) -> bool:
    """Return False when ``batch`` contains non-scalar leaves unaligned with ``batch_size``.

    Flattened multimodal/packed-embed leaves (e.g. ``pixel_values``, ``image_embeds``,
    ``image_grid_thw``) cannot be sliced uniformly across DP replicas; disqualifying
    them lets the caller fall back to standard SPMD execution instead of replicating
    unaligned vision tensors onto every DP slice.
    """
    from .training_utils import SHARED_GENERATION_MODEL_INPUT_KEYS

    for path, leaf in jtu.tree_flatten_with_path(batch)[0]:
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        key = next((entry.key for entry in reversed(path) if isinstance(entry, jtu.DictKey)), None)
        if key in SHARED_GENERATION_MODEL_INPUT_KEYS:
            continue
        if _is_mrope_position_leaf(key, leaf, batch_size):
            continue
        if int(shape[0]) != batch_size:
            return False
    return True


def can_use_dp_spmd_vmap(
    *,
    state: tp.Any,
    batch_size: int,
    minibatch_size: int,
    partition_spec: PartitionSpec | None,
    batch: collections.abc.Mapping[str, tp.Any] | None = None,
    dp_axis: str = "dp",
) -> bool:
    """Return whether single-mesh ``vmap(..., spmd_axis_name=dp_axis)`` applies to this training step."""
    if active_dp_spmd_axis() is not None:
        return False
    model = getattr(state, "model", None)
    mesh = getattr(model, "mesh", None)
    if mesh is None:
        return False
    cfg_mesh = getattr(getattr(model, "config", None), "mesh", None)
    if is_mpmd_mesh(mesh) or is_mpmd_mesh(cfg_mesh):
        return False
    jax_mesh = mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh
    shape = dict(getattr(jax_mesh, "shape", {}) or {})
    dp_size = int(shape.get(dp_axis, 1))
    if dp_size <= 1 or int(shape.get("pp", 1)) > 1:
        return False
    if batch_size <= 0 or minibatch_size <= 0 or batch_size % dp_size != 0 or minibatch_size % dp_size != 0:
        return False
    if isinstance(partition_spec, PartitionSpec) and len(partition_spec) > 0:
        b_entry = partition_spec[0]
        b_axes = (b_entry,) if isinstance(b_entry, str) else (tuple(b_entry) if isinstance(b_entry, tuple) else ())
        if dp_axis not in b_axes:
            return False
        shard_prod = 1
        for ax in b_axes:
            shard_prod *= int(shape.get(ax, 1))
        if minibatch_size % shard_prod != 0:
            return False
    if batch is not None and not _batch_compatible_with_dp_vmap(batch, batch_size):
        return False
    return True


def _lift_tree_with_specs(tree: tp.Any, specs_tree: tp.Any, jax_mesh: jax.sharding.Mesh, dp_axis: str = "dp") -> tp.Any:
    """Zero-copy local expand_dims onto ``PartitionSpec(dp_axis, *spec)``."""

    def _lift_leaf(x, spec):
        if not isinstance(x, jax.Array):
            return x
        sp = spec if isinstance(spec, PartitionSpec) else PartitionSpec()
        return _ORIG_SHARD_MAP(
            lambda z: z[None],
            mesh=jax_mesh,
            in_specs=sp,
            out_specs=PartitionSpec(dp_axis, *sp),
            check_vma=False,
        )(x)

    if specs_tree is None:
        return jtu.tree_map(lambda x: _lift_leaf(x, PartitionSpec()), tree)
    return jtu.tree_map(_lift_leaf, tree, specs_tree)


def _split_batch_for_dp(
    batch: collections.abc.Mapping[str, tp.Any],
    *,
    batch_size: int,
    partition_spec: PartitionSpec | None,
    jax_mesh: jax.sharding.Mesh,
    dp_axis: str = "dp",
) -> dict[str, tp.Any]:
    shape = dict(jax_mesh.shape)
    dp_size = int(shape[dp_axis])
    fsdp_size = int(shape.get("fsdp", 1))
    first_entry = partition_spec[0] if isinstance(partition_spec, PartitionSpec) and len(partition_spec) > 0 else None
    fsdp_major = first_entry == ("fsdp", dp_axis)

    def _split_leaf(path, b):
        if not isinstance(b, jax.Array):
            return b
        key = next((entry.key for entry in reversed(path) if isinstance(entry, jtu.DictKey)), None)
        if _is_mrope_position_leaf(key, b, batch_size):
            if fsdp_major and fsdp_size > 1:
                return (
                    b.reshape((3, fsdp_size, dp_size, batch_size // (fsdp_size * dp_size), b.shape[2]))
                    .transpose((2, 0, 1, 3, 4))
                    .reshape((dp_size, 3, batch_size // dp_size, b.shape[2]))
                )
            return b.reshape((3, dp_size, batch_size // dp_size, b.shape[2])).swapaxes(0, 1)
        if b.ndim == 0 or b.shape[0] != batch_size:
            return _ORIG_SHARD_MAP(
                lambda z: z[None],
                mesh=jax_mesh,
                in_specs=PartitionSpec(),
                out_specs=PartitionSpec(dp_axis),
                check_vma=False,
            )(b)
        if fsdp_major and fsdp_size > 1:
            return (
                b.reshape((fsdp_size, dp_size, batch_size // (fsdp_size * dp_size), *b.shape[1:]))
                .swapaxes(0, 1)
                .reshape((dp_size, batch_size // dp_size, *b.shape[1:]))
            )
        return b.reshape((dp_size, batch_size // dp_size, *b.shape[1:]))

    return jtu.tree_map_with_path(_split_leaf, dict(batch))


def dp_grad_shardings(param_shardings_tree: tp.Any, mesh: tp.Any, dp_axis: str = "dp") -> tp.Any:
    """Build ``NamedSharding(mesh, PartitionSpec(dp_axis, *spec))`` pytree for unreduced per-``dp`` gradients."""
    jax_mesh = mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh

    def _one(sh):
        if isinstance(sh, NamedSharding):
            return NamedSharding(jax_mesh, PartitionSpec(dp_axis, *sh.spec))
        if isinstance(sh, PartitionSpec):
            return NamedSharding(jax_mesh, PartitionSpec(dp_axis, *sh))
        return NamedSharding(jax_mesh, PartitionSpec(dp_axis))

    return jtu.tree_map(_one, param_shardings_tree, is_leaf=lambda s: isinstance(s, (NamedSharding, PartitionSpec)))


def reduce_dp_weighted(
    grads_dp: tp.Any,
    metrics_dp: tp.Any,
    weight_dp: jax.Array,
    grad_specs_tree: tp.Any,
    jax_mesh: jax.sharding.Mesh,
    *,
    dp_axis: str = "dp",
    normalize_grads: bool = True,
) -> tuple[tp.Any, tp.Any, jax.Array]:
    """Reduce per-``dp``-slice gradients and metrics in ``float32`` via ``shard_map`` ``psum``."""
    dp_size = int(jax_mesh.shape[dp_axis])
    g_leaves, g_treedef = jtu.tree_flatten(grads_dp)
    sp_leaves = g_treedef.flatten_up_to(grad_specs_tree)
    g_out_specs = tuple(
        (sp.spec if isinstance(sp, NamedSharding) else (sp if isinstance(sp, PartitionSpec) else PartitionSpec()))
        for sp in sp_leaves
    )
    g_in_specs = tuple(PartitionSpec(dp_axis, *sp) for sp in g_out_specs)

    has_weight_sum = isinstance(metrics_dp, LossMetrics) and metrics_dp.weight_sum is not None
    raw_ws_dp = metrics_dp.weight_sum if has_weight_sum else jnp.zeros_like(weight_dp)

    m_leaves, m_treedef = jtu.tree_flatten(metrics_dp)
    m_in_specs = tuple(PartitionSpec(dp_axis) for _ in m_leaves)
    m_out_specs = tuple(PartitionSpec() for _ in m_leaves)

    def _body(g_loc_tuple, m_loc_tuple, w_loc, ws_loc):
        w = w_loc.reshape(()).astype(jnp.float32)
        total_w = jax.lax.psum(w, dp_axis)
        denom = jnp.where(total_w > 0, total_w, jnp.float32(1.0))
        coeff = jnp.where(
            total_w > 0,
            jnp.where(w > 0, w / denom, jnp.float32(0.0)),
            jnp.float32(1.0 / dp_size),
        )
        if normalize_grads:
            g_red = tuple(
                jax.lax.psum(
                    jnp.where(w > 0, g[0].astype(jnp.float32) * coeff, jnp.float32(0.0)),
                    dp_axis,
                ).astype(g.dtype)
                for g in g_loc_tuple
            )
        else:
            g_red = tuple(
                jax.lax.psum(
                    jnp.where(w > 0, g[0].astype(jnp.float32) * w, jnp.float32(0.0)),
                    dp_axis,
                )
                for g in g_loc_tuple
            )
        m_red = tuple(
            jax.lax.psum(
                jnp.where(w > 0, m[0].astype(jnp.float32) * coeff, jnp.float32(0.0)),
                dp_axis,
            ).astype(m.dtype)
            for m in m_loc_tuple
        )
        ws_sum = jax.lax.psum(ws_loc.reshape(()).astype(jnp.float32), dp_axis)
        return g_red, m_red, total_w, ws_sum

    g_red_tuple, m_red_tuple, total_weight, total_ws = _ORIG_SHARD_MAP(
        _body,
        mesh=jax_mesh,
        in_specs=(g_in_specs, m_in_specs, PartitionSpec(dp_axis), PartitionSpec(dp_axis)),
        out_specs=(g_out_specs, m_out_specs, PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )(tuple(g_leaves), tuple(m_leaves), weight_dp, raw_ws_dp)

    grads = jtu.tree_unflatten(g_treedef, list(g_red_tuple))
    metrics = jtu.tree_unflatten(m_treedef, list(m_red_tuple))
    if has_weight_sum:
        metrics = metrics.replace(weight_sum=total_ws.astype(raw_ws_dp.dtype))
    return grads, metrics, total_weight


def dp_spmd_step_gradients(
    *,
    student_state: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    batch_size: int,
    minibatch_size: int,
    partition_spec: PartitionSpec | None,
    slice_grad_and_weight_fn: tp.Callable[[tp.Any, tp.Any, dict[str, tp.Any]], tuple[tp.Any, tp.Any, jax.Array]],
    extra_state: tp.Any = None,
    normalize_grads: bool = True,
    defer_dp_reduce: bool = False,
    dp_axis: str = "dp",
) -> tuple[tp.Any, tp.Any, jax.Array] | None:
    """Compute ``(grads, metrics, total_weight)`` via single-mesh ``vmap(spmd_axis_name=dp_axis)``.

    When ``defer_dp_reduce`` is ``True``, returns the unreduced per-``dp``-slice ``(grads_dp, metrics_dp, weight_dp)``
    so a split-step driver can execute :func:`reduce_dp_weighted` in a dedicated XLA executable compiled with
    :data:`DP_REDUCE_COMPILE_OPTIONS` (async collective fusion enabled, SparseCore all-reduce offload disabled).
    """
    if not _batch_compatible_with_dp_vmap(batch, batch_size):
        return None
    install_dp_spmd_hooks()
    mesh = student_state.model.mesh
    jax_mesh = mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh

    student_gs_specs = resolve_graphstate_specs(student_state, mesh)
    for sp in jtu.tree_leaves(student_gs_specs):
        if isinstance(sp, PartitionSpec) and _spec_mentions_axis(sp, dp_axis):
            return None
    student_go_specs = resolve_graphstate_specs(student_state, mesh, tree=student_state.graphother)
    for sp in jtu.tree_leaves(student_go_specs):
        if isinstance(sp, PartitionSpec) and _spec_mentions_axis(sp, dp_axis):
            return None

    if extra_state is not None:
        extra_gs_specs = resolve_graphstate_specs(extra_state, mesh)
        for sp in jtu.tree_leaves(extra_gs_specs):
            if isinstance(sp, PartitionSpec) and _spec_mentions_axis(sp, dp_axis):
                return None
        extra_go_specs = resolve_graphstate_specs(extra_state, mesh, tree=extra_state.graphother)
        for sp in jtu.tree_leaves(extra_go_specs):
            if isinstance(sp, PartitionSpec) and _spec_mentions_axis(sp, dp_axis):
                return None
    else:
        extra_gs_specs = extra_go_specs = None

    s_gs_dp = _lift_tree_with_specs(student_state.graphstate, student_gs_specs, jax_mesh, dp_axis)
    s_go_dp = _lift_tree_with_specs(student_state.graphother, student_go_specs, jax_mesh, dp_axis)
    batch_dp = _split_batch_for_dp(
        batch, batch_size=batch_size, partition_spec=partition_spec, jax_mesh=jax_mesh, dp_axis=dp_axis
    )

    if extra_state is not None:
        e_gs_dp = _lift_tree_with_specs(extra_state.graphstate, extra_gs_specs, jax_mesh, dp_axis)
        e_go_dp = _lift_tree_with_specs(extra_state.graphother, extra_go_specs, jax_mesh, dp_axis)

        def _one_slice_with_extra(s_gs, s_go, e_gs, e_go, b_slice):
            s_st = student_state.replace(graphstate=s_gs, graphother=s_go, opt_state=None)
            e_st = extra_state.replace(graphstate=e_gs, graphother=e_go, opt_state=None)
            return slice_grad_and_weight_fn(s_st, e_st, b_slice)

        with dp_spmd_vmap_scope(dp_axis):
            grads_dp, metrics_dp, weight_dp = jax.vmap(
                _one_slice_with_extra,
                in_axes=(0, 0, 0, 0, 0),
                out_axes=0,
                spmd_axis_name=dp_axis,
            )(s_gs_dp, s_go_dp, e_gs_dp, e_go_dp, batch_dp)
    else:

        def _one_slice(s_gs, s_go, b_slice):
            s_st = student_state.replace(graphstate=s_gs, graphother=s_go, opt_state=None)
            return slice_grad_and_weight_fn(s_st, None, b_slice)

        with dp_spmd_vmap_scope(dp_axis):
            grads_dp, metrics_dp, weight_dp = jax.vmap(
                _one_slice,
                in_axes=(0, 0, 0),
                out_axes=0,
                spmd_axis_name=dp_axis,
            )(s_gs_dp, s_go_dp, batch_dp)

    if defer_dp_reduce:
        return grads_dp, metrics_dp, weight_dp

    return reduce_dp_weighted(
        grads_dp,
        metrics_dp,
        weight_dp,
        student_gs_specs,
        jax_mesh,
        dp_axis=dp_axis,
        normalize_grads=normalize_grads,
    )

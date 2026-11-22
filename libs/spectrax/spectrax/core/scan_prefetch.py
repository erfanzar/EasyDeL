# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer scans with next-layer weight prefetch (``ModuleList.scan(prefetch_axes=...)``).

The plain layer scan slices each layer's stacked, sharded parameters inside the ``lax.scan`` body and lets the SPMD
partitioner all-gather them at first use, at the top of the layer that needs them. This lowering owns the whole
loop, forward and backward, as one :func:`jax.custom_vjp`:

* Forward: the body for slot ``i`` consumes the gathered weights ``G_i`` from the loop carry and issues the gather
  of slot ``i + 1`` (a sharding constraint to the leaf's spec with ``prefetch_axes`` removed), so the transfer can
  overlap a whole layer of compute.
* Backward: hand-written reverse loops recompute each layer from its saved input carry and the carried ``G_i``
  while issuing the gather of slot ``i - 1``. A prefetched weight's cotangent is constrained back to its sharded
  spec, which is the reduce-scatter the partitioner emits for the plain scan.
* Residuals: only the per-layer input carry; parameters reach the backward as the custom_vjp's own inputs.

Values the layer function closes over and that are being differentiated are hoisted with
:func:`jax.closure_convert` and receive their accumulated cotangents. Only 2-D floating leaves whose sharding names
a gather axis, and whose path holds none of the ``exclude`` keys, are prefetched; every other leaf keeps the plain
scan's sharding constraint. The lowering supports reverse-mode differentiation only (``custom_vjp``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from .state import State

logger = logging.getLogger("spectrax.scan_prefetch")


def barrier_array_leaves(tree):
    """Apply ``jax.lax.optimization_barrier`` to the array leaves of ``tree``.

    Non-array leaves pass through unchanged. The barrier is the identity on values; it only removes scheduling
    freedom (for example, sinking one layer's weight-gradient reductions past the next layer's backward).

    Args:
        tree: Any pytree.

    Returns:
        ``tree`` with its array leaves routed through one optimization barrier.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    picks = [isinstance(x, jax.Array) for x in leaves]
    arrays = [x for x, ok in zip(leaves, picks, strict=True) if ok]
    if not arrays:
        return tree
    barred = iter(jax.lax.optimization_barrier(tuple(arrays)))
    return jax.tree_util.tree_unflatten(
        treedef, [next(barred) if ok else x for x, ok in zip(leaves, picks, strict=True)]
    )


def _strip_axes(entry, axes: frozenset[str]):
    """Remove the gather axes from one PartitionSpec entry."""
    if entry is None:
        return None
    if isinstance(entry, str):
        return None if entry in axes else entry
    kept = tuple(a for a in entry if a not in axes)
    if not kept:
        return None
    return kept[0] if len(kept) == 1 else kept


def _spec_names(entry) -> tuple[str, ...]:
    """Mesh axis names used by one PartitionSpec entry."""
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)
    return tuple(entry)


def gathered_sharding(spec: NamedSharding, axes: Sequence[str]) -> NamedSharding:
    """Return ``spec`` with ``axes`` removed from every dimension (replicated over them).

    Args:
        spec: The leaf's sharded spec.
        axes: Mesh axes to gather over.

    Returns:
        The gathered :class:`NamedSharding` on the same mesh.
    """
    axes = frozenset(axes)
    return NamedSharding(spec.mesh, PartitionSpec(*[_strip_axes(e, axes) for e in spec.spec]))


def spec_mentions(spec: NamedSharding | None, axes: Sequence[str]) -> bool:
    """Return whether ``spec`` shards any dimension over one of ``axes``.

    Args:
        spec: A :class:`NamedSharding` or ``None``.
        axes: Mesh axis names.

    Returns:
        ``True`` when at least one dimension names one of ``axes``.
    """
    if spec is None:
        return False
    axes = frozenset(axes)
    return any(name in axes for entry in spec.spec for name in _spec_names(entry))


def _is_float(x) -> bool:
    return bool(jnp.issubdtype(x.dtype, jnp.inexact))


@dataclass(frozen=True)
class PositionPlan:
    """Flat view of one scan position: leaf order, treedef, and which leaves are prefetched.

    Attributes:
        gdef: Graph template bound for every slot of the position.
        treedef: Treedef of the position's raw state dict.
        paths: ``keystr`` of every leaf, in flat order.
        length: Number of layers of the position per block.
        specs: Per leaf, the plain scan's :class:`NamedSharding` constraint or ``None``.
        prefetch: Per leaf, whether it is gathered one slot ahead.
        floating: Per leaf, whether its dtype is inexact.
        gathered: Per leaf, the gathered :class:`NamedSharding` (prefetched leaves) or ``None``.
        stacked: Per leaf, the sharded spec of the ``[blocks, length, ...]`` stack or ``None``.
    """

    gdef: object
    treedef: object
    paths: tuple[str, ...]
    length: int
    specs: tuple[object, ...]
    prefetch: tuple[bool, ...]
    floating: tuple[bool, ...]
    gathered: tuple[object, ...]
    stacked: tuple[object, ...]

    @property
    def pf_idx(self) -> tuple[int, ...]:
        """Flat indices of the prefetched leaves."""
        return tuple(i for i, p in enumerate(self.prefetch) if p)

    @property
    def other_idx(self) -> tuple[int, ...]:
        """Flat indices of the leaves that are not prefetched."""
        return tuple(i for i, p in enumerate(self.prefetch) if not p)


def plan_position(
    gdef,
    stacked: State,
    specs: State | None,
    length: int,
    axes: Sequence[str],
    *,
    exclude: Sequence[str] = (),
    block_axis: bool,
) -> tuple[PositionPlan, list]:
    """Flatten a position's stacked state and decide per leaf whether it is prefetched.

    Args:
        gdef: The position's graph template.
        stacked: Stacked state, ``[blocks, length, ...]`` leaves when ``block_axis`` else ``[length, ...]``.
        specs: State-shaped per-layer sharding constraints (``None`` without a mesh).
        length: Layers per block of the position.
        axes: Mesh axes to gather over.
        exclude: State keys; a leaf whose path contains one of them is never prefetched.
        block_axis: Whether ``stacked`` already carries the leading block axis.

    Returns:
        ``(plan, leaves)`` with the flat leaves normalised to ``[blocks, length, ...]``.
    """
    keyed, treedef = jax.tree_util.tree_flatten_with_path(stacked.raw())
    paths = tuple(jax.tree_util.keystr(k) for k, _ in keyed)
    excluded = frozenset(exclude)
    leaves = [v if block_axis else v[None] for _, v in keyed]
    spec_by_path: dict[str, object] = {}
    if specs is not None:
        for k, v in jax.tree_util.tree_leaves_with_path(specs.raw()):
            spec_by_path[jax.tree_util.keystr(k)] = v
    per_spec, prefetch, floating, gathered = [], [], [], []
    for (keypath, _), path, leaf in zip(keyed, paths, leaves, strict=True):
        spec = spec_by_path.get(path)
        isf = _is_float(leaf)
        pf = (
            isf
            and leaf.ndim == 4
            and isinstance(spec, NamedSharding)
            and spec_mentions(spec, axes)
            and not any(getattr(k, "key", None) in excluded for k in keypath)
        )
        per_spec.append(spec)
        prefetch.append(pf)
        floating.append(isf)
        gathered.append(gathered_sharding(spec, axes) if pf else None)
    plan = PositionPlan(
        gdef=gdef,
        treedef=treedef,
        paths=paths,
        length=length,
        specs=tuple(per_spec),
        prefetch=tuple(prefetch),
        floating=tuple(floating),
        gathered=tuple(gathered),
        stacked=tuple(_stacked_sharding(s) for s in per_spec),
    )
    return plan, leaves


def _stacked_sharding(spec) -> NamedSharding | None:
    """Layer spec with two leading replicated axes for the ``[blocks, length, ...]`` stack."""
    if not isinstance(spec, NamedSharding):
        return None
    return NamedSharding(spec.mesh, PartitionSpec(None, None, *spec.spec))


def _constrain_stacked(plan: PositionPlan, leaves):
    return [
        x if x is None or s is None else jax.lax.with_sharding_constraint(x, s)
        for x, s in zip(leaves, plan.stacked, strict=True)
    ]


def convert_layer(plan: PositionPlan, leaves: list, apply: Callable, carry) -> tuple[Callable, list]:
    """Closure-convert one position's layer function ``(carry, flat_leaves) -> carry``.

    Args:
        plan: The position plan.
        leaves: The position's flat leaves ``[blocks, length, ...]`` (only their slot avals are used).
        apply: ``apply(gdef, state, carry) -> carry``.
        carry: Example carry (array leaves).

    Returns:
        ``(converted, consts)`` from :func:`jax.closure_convert`; ``converted(carry, flat, *consts)``.
    """

    def layer(c, flat):
        state = State._from_raw(jax.tree_util.tree_unflatten(plan.treedef, list(flat)))
        return apply(plan.gdef, state, c)

    converted, consts = jax.closure_convert(layer, carry, [x[0, 0] for x in leaves])
    return converted, list(consts)


def _gather(plan: PositionPlan, pf_leaves):
    return [jax.lax.with_sharding_constraint(x, plan.gathered[i]) for i, x in zip(plan.pf_idx, pf_leaves, strict=True)]


def _constrain_other(plan: PositionPlan, other_leaves):
    out = []
    for i, x in zip(plan.other_idx, other_leaves, strict=True):
        spec = plan.specs[i]
        out.append(jax.lax.with_sharding_constraint(x, spec) if spec is not None else x)
    return out


def _full_leaves(plan: PositionPlan, other_leaves, gathered_leaves):
    leaves = [None] * len(plan.paths)
    for i, x in zip(plan.other_idx, other_leaves, strict=True):
        leaves[i] = x
    for i, x in zip(plan.pf_idx, gathered_leaves, strict=True):
        leaves[i] = x
    return leaves


def _split_float(tree):
    """``(float leaves, other leaves, rebuild)`` for a pytree of arrays."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    isf = [_is_float(x) for x in leaves]
    floats = [x for x, f in zip(leaves, isf, strict=True) if f]
    rest = [x for x, f in zip(leaves, isf, strict=True) if not f]

    def rebuild(floats_, rest_):
        fi, ri = iter(floats_), iter(rest_)
        return jax.tree_util.tree_unflatten(treedef, [next(fi) if f else next(ri) for f in isf])

    return floats, rest, rebuild


def _maybe_barrier(tree, on: bool):
    return barrier_array_leaves(tree) if on else tree


def _slot(leaves, idx, b, j):
    return [leaves[i][b, j] for i in idx]


def _layer_forward(plan, layer, barrier, carry, other, gathered):
    converted, consts = layer
    other = _constrain_other(plan, other)
    carry, other, gathered = _maybe_barrier((carry, other, gathered), barrier)
    return converted(carry, _full_leaves(plan, other, gathered), *consts)


def _layer_backward(plan, layer, barrier, defer_dw, carry_in, other, gathered, ct_floats, acc):
    """Recompute one slot from its saved input carry; return its cotangents and the updated const accumulators."""
    converted, consts = layer
    carry_in, other, gathered, ct_floats = _maybe_barrier((carry_in, other, gathered, ct_floats), barrier)
    c_floats, c_rest, rebuild = _split_float(carry_in)
    of_idx = [k for k, i in enumerate(plan.other_idx) if plan.floating[i]]
    kf_idx = [k for k, c in enumerate(consts) if _is_float(c)]

    def f(cf, of, g, kf):
        other_full = list(other)
        for k, x in zip(of_idx, of, strict=True):
            other_full[k] = x
        consts_full = list(consts)
        for k, x in zip(kf_idx, kf, strict=True):
            consts_full[k] = x
        out = converted(rebuild(cf, c_rest), _full_leaves(plan, _constrain_other(plan, other_full), g), *consts_full)
        return _split_float(out)[0]

    _, vjp = jax.vjp(f, c_floats, [other[k] for k in of_idx], list(gathered), [consts[k] for k in kf_idx])
    ct_c, ct_of, ct_g, ct_k = vjp(list(ct_floats))
    ct_pf = [jax.lax.with_sharding_constraint(g, plan.specs[i]) for i, g in zip(plan.pf_idx, ct_g, strict=True)]
    if barrier:
        if defer_dw:
            ct_c, ct_pf = barrier_array_leaves((ct_c, ct_pf))
        else:
            ct_c, ct_of, ct_pf = barrier_array_leaves((ct_c, ct_of, ct_pf))
    return ct_c, list(ct_of), ct_pf, [a + k for a, k in zip(acc, ct_k, strict=True)]


def _position_forward(plan, layer, barrier, carry, leaves, b, gathered_first, after):
    """Run one position of one block.

    Returns ``(carry, gathered weights of the slot after this position or None, saved input carries [length, ...])``.
    """
    gathered_after = _gather(after[0], after[1]) if after is not None else None
    if plan.length == 1:
        saved = jax.tree_util.tree_map(lambda x: x[None], carry)
        carry = _layer_forward(plan, layer, barrier, carry, _slot(leaves, plan.other_idx, b, 0), gathered_first)
        return carry, gathered_after, saved
    length = plan.length

    def body(state, j):
        c, g = state
        g_next = _gather(plan, _slot(leaves, plan.pf_idx, b, (j + 1) % length))
        out = _layer_forward(plan, layer, barrier, c, _slot(leaves, plan.other_idx, b, j), g)
        return (out, g_next), c

    (carry, _), saved = jax.lax.scan(body, (carry, gathered_first), jnp.arange(length))
    return carry, gathered_after, saved


def _position_backward(plan, layer, barrier, defer_dw, ct, leaves, b, saved, gathered_last, before, acc):
    """Backward of one position of one block, slots in reverse.

    Returns ``(ct_carry, gathered weights of the slot before this position or None, float-other cts, prefetched cts,
    const accumulators)``; the per-slot cotangents carry a leading ``[length]`` axis.
    """
    gathered_before = _gather(before[0], before[1]) if before is not None else None
    if plan.length == 1:
        carry_in = jax.tree_util.tree_map(lambda x: x[0], saved)
        ct, ct_of, ct_pf, acc = _layer_backward(
            plan, layer, barrier, defer_dw, carry_in, _slot(leaves, plan.other_idx, b, 0), gathered_last, ct, acc
        )
        return ct, gathered_before, [x[None] for x in ct_of], [x[None] for x in ct_pf], acc
    length = plan.length

    def body(state, xs):
        c, g, a = state
        j, carry_in = xs
        g_prev = _gather(plan, _slot(leaves, plan.pf_idx, b, (j - 1) % length))
        c, ct_of_j, ct_pf_j, a = _layer_backward(
            plan, layer, barrier, defer_dw, carry_in, _slot(leaves, plan.other_idx, b, j), g, c, a
        )
        return (c, g_prev, a), (ct_of_j, ct_pf_j)

    (ct, _, acc), (ct_of, ct_pf) = jax.lax.scan(
        body,
        (ct, gathered_last, acc),
        (jnp.arange(length), saved),
        reverse=True,
        unroll=length if defer_dw else 1,
    )
    return ct, gathered_before, ct_of, ct_pf, acc


def scan_with_weight_prefetch(
    plans: Sequence[PositionPlan],
    stacked_leaves: Sequence[list],
    layers: Sequence[tuple[Callable, list]],
    blocks: int,
    init_carry,
    *,
    barrier: bool,
    defer_dw: bool = False,
):
    """Run ``blocks`` repeats of the positions in ``plans`` with next-slot weight prefetch.

    Args:
        plans: One :class:`PositionPlan` per position, in execution order within a block.
        stacked_leaves: Per position, the flat leaves ``[blocks, length, ...]``.
        layers: Per position, ``(converted, consts)`` from :func:`convert_layer`.
        blocks: Number of repeats of the position sequence.
        init_carry: Initial carry (array leaves).
        barrier: Per-layer ``optimization_barrier`` (forward: over ``(carry, other leaves, gathered weights)``;
            backward: over the recompute inputs and over the cotangents).
        defer_dw: Keep the non-prefetched leaves' cotangents out of the cotangent barrier and unroll each
            position's reverse loop into the block body, so their per-layer all-reduces can be combined.

    Returns:
        The final carry.
    """
    npos = len(plans)
    last = npos - 1
    convs = [layer[0] for layer in layers]
    of_idx = [[k for k, i in enumerate(plan.other_idx) if plan.floating[i]] for plan in plans]

    def forward(carry0, lv, ks):
        def after_of(p, b):
            if p + 1 < npos:
                return plans[p + 1], _slot(lv[p + 1], plans[p + 1].pf_idx, b, 0)
            if blocks == 1:
                return None
            return plans[0], _slot(lv[0], plans[0].pf_idx, (b + 1) % blocks, 0)

        def block_body(state, b):
            c, g = state
            saved = []
            for p in range(npos):
                c, g, s = _position_forward(plans[p], (convs[p], ks[p]), barrier, c, lv[p], b, g, after_of(p, b))
                saved.append(s)
            return (c, g), tuple(saved)

        g0 = _gather(plans[0], _slot(lv[0], plans[0].pf_idx, 0, 0))
        if blocks == 1:
            (c, _), saved = block_body((carry0, g0), 0)
            return c, jax.tree_util.tree_map(lambda x: x[None], saved)
        (c, _), saved = jax.lax.scan(block_body, (carry0, g0), jnp.arange(blocks))
        return c, saved

    def backward(lv, ks, saved, ct_floats):
        def before_of(p, b):
            if p > 0:
                q, bb = p - 1, b
            elif blocks == 1:
                return None
            else:
                q, bb = last, (b - 1) % blocks
            return plans[q], _slot(lv[q], plans[q].pf_idx, bb, plans[q].length - 1)

        def block_body(state, xs):
            ct, g, acc = state
            b, saved_b = xs
            acc = list(acc)
            cts = [None] * npos
            for p in reversed(range(npos)):
                ct, g, ct_of, ct_pf, acc[p] = _position_backward(
                    plans[p],
                    (convs[p], ks[p]),
                    barrier,
                    defer_dw,
                    ct,
                    lv[p],
                    b,
                    saved_b[p],
                    g,
                    before_of(p, b),
                    acc[p],
                )
                cts[p] = (ct_of, ct_pf)
            return (ct, g, tuple(acc)), tuple(cts)

        g_last = _gather(plans[last], _slot(lv[last], plans[last].pf_idx, blocks - 1, plans[last].length - 1))
        acc0 = tuple([jnp.zeros_like(k) for k in ks[p] if _is_float(k)] for p in range(npos))
        if blocks == 1:
            saved_0 = jax.tree_util.tree_map(lambda x: x[0], saved)
            (ct, _, acc), cts = block_body((ct_floats, g_last, acc0), (0, saved_0))
            return ct, acc, jax.tree_util.tree_map(lambda x: x[None], cts)
        (ct, _, acc), cts = jax.lax.scan(
            block_body, (ct_floats, g_last, acc0), (jnp.arange(blocks), saved), reverse=True
        )
        return ct, acc, cts

    init_carry = jax.tree_util.tree_map(jnp.asarray, init_carry)
    c_leaves, c_tree = jax.tree_util.tree_flatten(init_carry)
    c_float = [_is_float(x) for x in c_leaves]

    def constrain_all(lv):
        # Without this, propagation may replicate the whole stack outside the loop.
        return [_constrain_stacked(plan, list(lp)) for plan, lp in zip(plans, lv, strict=True)]

    @jax.custom_vjp
    def run(carry0, lv, ks):
        return forward(carry0, constrain_all(lv), ks)[0]

    def run_fwd(carry0, lv, ks):
        lv = constrain_all(lv)
        carry, saved = forward(carry0, lv, ks)
        return carry, (lv, ks, saved)

    def run_bwd(res, ct_carry):
        lv, ks, saved = res
        lv = constrain_all(lv)
        ct_c, acc, cts = backward(lv, ks, saved, _split_float(ct_carry)[0])
        ct_iter = iter(ct_c)
        ct_carry0 = jax.tree_util.tree_unflatten(c_tree, [next(ct_iter) if f else None for f in c_float])
        ct_lv = []
        for p, plan in enumerate(plans):
            ct_of, ct_pf = cts[p]
            full = [None] * len(plan.paths)
            for k, x in zip(of_idx[p], ct_of, strict=True):
                full[plan.other_idx[k]] = x
            for i, x in zip(plan.pf_idx, ct_pf, strict=True):
                full[i] = x
            ct_lv.append(_constrain_stacked(plan, full))
        ct_ks = []
        for p in range(npos):
            acc_iter = iter(acc[p])
            ct_ks.append([next(acc_iter) if _is_float(k) else None for k in ks[p]])
        return ct_carry0, ct_lv, ct_ks

    run.defvjp(run_fwd, run_bwd)
    return run(init_carry, [list(lp) for lp in stacked_leaves], [list(layer[1]) for layer in layers])


def log_plan(plans: Sequence[PositionPlan], axes: Sequence[str]) -> None:
    """Log, per position, which leaves are prefetched.

    Args:
        plans: The position plans.
        axes: The gather axes.
    """
    for p, plan in enumerate(plans):
        picked = [plan.paths[i] for i in plan.pf_idx]
        logger.debug(
            "weight prefetch: position %d (length %d): %d of %d leaves gathered over %s: %s",
            p,
            plan.length,
            len(picked),
            len(plan.paths),
            tuple(axes),
            ", ".join(picked),
        )

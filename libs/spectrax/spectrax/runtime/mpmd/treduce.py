# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`treduce` — schedule-driven microbatch reduction over a function.

This is spectrax's port of jaxpp's ``treduce``: a primitive that binds
a user-provided body function, a microbatch count, and a pipeline
:class:`Schedule` into the traced jaxpr. When the containing function
is compiled by :func:`sxjit`, the MPMD compiler detects the
``pscan_p`` equations, unrolls the body across microbatches, and
reorders the per-microbatch tasks according to the schedule to drive
interleaved pipeline-parallel execution (GPipe, 1F1B, ZeroBubble).

User-facing API::

    @sxjit(mesh=mesh)
    def train_step(params, batch):
        micro_grad = jax.value_and_grad(loss_fn)
        losses, grads = treduce(
            lambda mb: micro_grad(params, mb),
            batch,
            schedule=Std1F1B(mpmd_mesh.mpmd_dim),
        )
        return losses, grads

Schedules currently supported by the MPMD compiler path are GPipe,
Std1F1B / Eager1F1B, ZeroBubbleH1, and InterleavedH1.

**Operations** (passed via ``operation=``) define per-microbatch
accumulation:

* :class:`Add` — sum per-microbatch values (for gradients).
* :class:`Concat` — stack per-microbatch values along a leading axis
  (for losses, predictions).
* Custom: any dataclass with ``state(aval)`` and ``update(state, value, idx)``.

Default operation is ``(Concat, Add)``: first return value stacked,
subsequent return values summed. Matches jaxpp's convention so a
``value_and_grad``-style body returns concatenated losses + summed grads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import ClosedJaxpr, Jaxpr, Primitive, Var
from jax.interpreters import mlir

__all__ = [
    "Add",
    "Concat",
    "Max",
    "Op",
    "pscan_p",
    "treduce",
    "treduce_i",
]


class Op:
    """Base class for microbatch accumulation operations.

    Subclasses define how per-microbatch results are combined into a
    single result across all microbatches.
    """

    def state(self, aval: core.AbstractValue) -> jax.Array:
        """Return the identity accumulator for a per-microbatch output described by ``aval``.

        Called once at the beginning of a :func:`treduce` to initialise
        the running state. The returned array's shape and dtype should
        be compatible with whatever :meth:`update` will fold in for
        each microbatch.

        Args:
            aval: Abstract value of one microbatch's contribution
                (matches the body output's per-microbatch aval).

        Returns:
            A concrete JAX array used as the starting accumulator.
        """
        raise NotImplementedError

    def update(self, state: jax.Array, value: jax.Array, idx: jax.Array) -> jax.Array:
        """Fold microbatch ``idx``'s ``value`` into the running ``state``.

        The op is invoked once per microbatch in the order chosen by the
        active pipeline schedule (which need not be sequential). ``idx``
        is the microbatch index supplied as a traced :class:`jax.Array`
        so :class:`Concat`-style ops can write into a stacking buffer.

        Args:
            state: Current accumulator returned by the previous
                :meth:`update` (or :meth:`state` on the first call).
            value: Per-microbatch contribution returned by the user
                body for microbatch ``idx``.
            idx: Microbatch index (rank-0 ``int32`` array).

        Returns:
            The new accumulator value.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Add(Op):
    """Accumulate per-microbatch values by summation.

    Used for gradients: ``Add`` over N microbatch grads gives the total
    gradient for the full batch.
    """

    def state(self, aval: core.AbstractValue) -> jax.Array:
        """Return a zero accumulator with ``aval``'s shape and dtype.

        Args:
            aval: Abstract value describing a single microbatch's grad
                shape — the resulting buffer holds the running sum.

        Returns:
            ``jnp.zeros(aval.shape, dtype=aval.dtype)``.
        """
        return jnp.zeros(aval.shape, dtype=aval.dtype)

    def update(self, state: jax.Array, value: jax.Array, idx: jax.Array) -> jax.Array:
        """Add the microbatch contribution ``value`` to ``state``.

        ``idx`` is intentionally unused — :class:`Add` is associative and
        commutative across microbatches so the accumulation order picked
        by the active schedule does not affect the result.

        Args:
            state: Current sum.
            value: New microbatch contribution.
            idx: Microbatch index (ignored).

        Returns:
            ``state + value``.
        """
        del idx
        return jax.lax.add(state, value)


@dataclass(frozen=True)
class Concat(Op):
    """Stack per-microbatch values along a new leading axis.

    Used for losses, predictions, or any per-microbatch scalar/tensor
    that should be collected (not summed) across microbatches.
    """

    length: int

    def state(self, aval: core.AbstractValue) -> jax.Array:
        """Return a zero-initialised stacking buffer of shape ``(length, *aval.shape)``.

        The leading axis has size ``length`` (the microbatch count) so
        :meth:`update` can write each microbatch's value into its own
        slot without reallocation.

        Args:
            aval: Abstract value of one microbatch's contribution.

        Returns:
            A zero buffer with shape ``(self.length, *aval.shape)`` and
            dtype ``aval.dtype``.
        """
        return jnp.zeros((self.length, *aval.shape), dtype=aval.dtype)

    def update(self, state: jax.Array, value: jax.Array, idx: jax.Array) -> jax.Array:
        """Insert ``value`` at index ``idx`` along the leading axis of ``state``.

        Uses :func:`jax.lax.dynamic_update_index_in_dim` so that ``idx``
        may be a traced :class:`jax.Array` — necessary because the
        microbatch index is not always a Python literal under
        schedule-driven dispatch.

        Args:
            state: The stacking buffer.
            value: The microbatch contribution to write.
            idx: Microbatch index (rank-0 ``int32`` array).

        Returns:
            ``state`` with ``value`` written into slot ``idx``.
        """
        return jax.lax.dynamic_update_index_in_dim(state, value, idx, axis=0)


@dataclass(frozen=True)
class Max(Op):
    """Accumulate per-microbatch values by elementwise maximum.

    Suitable for max-reduction quantities like absolute-value norms or
    per-batch numerical-stability statistics. Like :class:`Add`, ``Max``
    is associative and commutative so the schedule's microbatch order is
    irrelevant.
    """

    def state(self, aval: core.AbstractValue) -> jax.Array:
        """Return a ``-inf`` accumulator with ``aval``'s shape and dtype.

        Starting from negative infinity makes the first :meth:`update`
        return that microbatch's value verbatim — the identity element
        for elementwise ``max`` over real-valued data.

        Args:
            aval: Abstract value of one microbatch's contribution.

        Returns:
            A ``-inf``-filled buffer shaped like ``aval``.
        """
        return jnp.full(aval.shape, -jnp.inf, dtype=aval.dtype)

    def update(self, state: jax.Array, value: jax.Array, idx: jax.Array) -> jax.Array:
        """Return the elementwise maximum of ``state`` and ``value``.

        Args:
            state: Current running max.
            value: New microbatch contribution.
            idx: Microbatch index (ignored — order-independent).

        Returns:
            ``jnp.maximum(state, value)``.
        """
        del idx
        return jax.lax.max(state, value)


pscan_p = Primitive("spectrax_pscan")
pscan_p.multiple_results = True


class _HashableSchedule:
    """Wrap a :class:`Schedule` instance in a hashable ``id(...)``-based container.

    JAX 0.7+ requires primitive equation parameters to implement
    ``__hash__`` and ``__eq__``. Schedule dataclasses are mutable
    (``@dataclass`` without ``frozen=True``), so we wrap them here.
    The wrapper compares by object identity — two distinct Schedule
    instances never equal even if their fields match. This is
    conservative but safe for caching purposes.
    """

    __slots__ = ("schedule",)

    def __init__(self, schedule: object) -> None:
        """Store ``schedule`` unchanged; comparisons and hashes go by ``id``.

        Args:
            schedule: Pipeline schedule object controlling forward/backward execution order.
        """
        self.schedule = schedule

    def __hash__(self) -> int:
        """Hash by object identity of the wrapped schedule.

        Using ``id(...)`` keeps the wrapper hashable even though the
        underlying schedule dataclass may be mutable. Two distinct
        schedule instances with the same field values still collide
        only if they happen to live at the same Python ``id`` (which
        would require GC reuse).

        Returns:
            Result described by this helper.
        """
        return id(self.schedule)

    def __eq__(self, other: object) -> bool:
        """Two wrappers are equal iff they hold the same schedule instance.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return isinstance(other, _HashableSchedule) and self.schedule is other.schedule

    def __repr__(self) -> str:
        """Pass through the underlying schedule's repr for readability.

        Returns:
            Result described by this helper.
        """
        return f"_HashableSchedule({self.schedule!r})"


class _HashableOps:
    """Wrap a tuple of :class:`Op` instances in a hashable container.

    Same rationale as :class:`_HashableSchedule`: ``Op`` subclasses are
    ``@dataclass(frozen=True)`` so they're already hashable
    individually, but wrapping the tuple uniformly simplifies param
    handling.
    """

    __slots__ = ("ops",)

    def __init__(self, ops: tuple) -> None:
        """Store the ops tuple; equality follows the contained operation values.

        Args:
            ops: Ops value consumed by this operation.
        """
        self.ops = ops

    def __hash__(self) -> int:
        """Hash by operation values so equivalent op tuples share cache entries.

        Returns:
            Result described by this helper.
        """
        return hash(self.ops)

    def __eq__(self, other: object) -> bool:
        """Two wrappers are equal iff their operation tuples are equal.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return isinstance(other, _HashableOps) and self.ops == other.ops

    def __repr__(self) -> str:
        """Render the wrapper as ``_HashableOps(<ops repr>)`` for debugging.

        Returns:
            Result described by this helper.
        """
        return f"_HashableOps({self.ops!r})"


@pscan_p.def_impl
def _pscan_impl(
    *args,
    jaxpr,
    fn_jaxpr,
    loss_jaxpr,
    body_mode,
    grad_tree,
    ops,
    n_mubatches,
    n_consts,
    schedule,
    n_outs,
):
    """Eager / trace-less fallback: unroll the body sequentially.

    Schedule is ignored in the eager path — it only matters inside
    ``@sxjit`` where the MPMD compiler intercepts ``pscan_p``. This
    fallback is used for debugging / single-device smoke tests.

    ``args`` layout: ``(*consts, *init_state)``.

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.
        fn_jaxpr: Fn jaxpr value consumed by this operation.
        loss_jaxpr: Loss jaxpr value consumed by this operation.
        body_mode: Body mode value consumed by this operation.
        grad_tree: Grad tree value consumed by this operation.
        ops: Ops value consumed by this operation.
        n_mubatches: N mubatches value consumed by this operation.
        n_consts: N consts value consumed by this operation.
        schedule: Pipeline schedule object controlling forward/backward execution order.
        n_outs: N outs value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del body_mode, grad_tree, loss_jaxpr, schedule, n_outs, fn_jaxpr, ops
    consts = list(args[:n_consts])
    init_state = list(args[n_consts:])
    for mb in range(n_mubatches):
        mb_arr = jnp.asarray(mb, dtype=jnp.int32)
        out = core.eval_jaxpr(jaxpr.jaxpr, consts, mb_arr, *init_state)
        init_state = list(out)
    return init_state


@pscan_p.def_abstract_eval
def _pscan_abstract(
    *args,
    jaxpr,
    fn_jaxpr,
    loss_jaxpr,
    body_mode,
    grad_tree,
    ops,
    n_mubatches,
    n_consts,
    schedule,
    n_outs,
):
    """Abstract eval: output avals are the body's outvars (loop state).

    Args:
        jaxpr: JAXPR being inspected, rewritten, split, or executed.
        fn_jaxpr: Fn jaxpr value consumed by this operation.
        loss_jaxpr: Loss jaxpr value consumed by this operation.
        body_mode: Body mode value consumed by this operation.
        grad_tree: Grad tree value consumed by this operation.
        ops: Ops value consumed by this operation.
        n_mubatches: N mubatches value consumed by this operation.
        n_consts: N consts value consumed by this operation.
        schedule: Pipeline schedule object controlling forward/backward execution order.
        n_outs: N outs value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del args, body_mode, fn_jaxpr, grad_tree, loss_jaxpr, ops, n_mubatches, n_consts, schedule, n_outs
    return [v.aval for v in jaxpr.jaxpr.outvars]


def _unwrap_schedule(schedule: object) -> object:
    """Return the underlying schedule object from a :class:`_HashableSchedule` wrapper.

    Args:
        schedule: Pipeline schedule object controlling forward/backward execution order.

    Returns:
        Return the underlying schedule object from a :class:`_HashableSchedule` wrapper.
    """
    return schedule.schedule if isinstance(schedule, _HashableSchedule) else schedule


def _unwrap_ops(ops: object) -> tuple:
    """Return the underlying ops tuple from a :class:`_HashableOps` wrapper.

    Args:
        ops: Ops value consumed by this operation.

    Returns:
        Return the underlying ops tuple from a :class:`_HashableOps` wrapper.
    """
    return ops.ops if isinstance(ops, _HashableOps) else tuple(ops)


def _pscan_lowering(
    ctx,
    *args,
    jaxpr,
    fn_jaxpr,
    loss_jaxpr,
    body_mode,
    grad_tree,
    ops,
    n_mubatches,
    n_consts,
    schedule,
    n_outs,
):
    """MLIR lowering: emit a straight-line unrolled loop.

    When ``sxjit`` doesn't intercept the ``pscan_p`` equation (e.g.
    the user called ``jax.jit`` on a ``treduce``-containing function
    directly), we fall back to an unrolled loop in HLO. Not optimal
        for pipeline parallelism but numerically correct.

    Args:
        ctx: Ctx value consumed by this operation.
        jaxpr: JAXPR being inspected, rewritten, split, or executed.
        fn_jaxpr: Fn jaxpr value consumed by this operation.
        loss_jaxpr: Loss jaxpr value consumed by this operation.
        body_mode: Body mode value consumed by this operation.
        grad_tree: Grad tree value consumed by this operation.
        ops: Ops value consumed by this operation.
        n_mubatches: N mubatches value consumed by this operation.
        n_consts: N consts value consumed by this operation.
        schedule: Pipeline schedule object controlling forward/backward execution order.
        n_outs: N outs value consumed by this operation.
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
    """
    del body_mode, ctx, args, fn_jaxpr, grad_tree, jaxpr, loss_jaxpr, ops, n_mubatches, n_consts, schedule, n_outs
    raise NotImplementedError(
        "pscan_p MLIR lowering requires @sxjit. Call the function "
        "inside an @sxjit decorator so the schedule-driven compiler "
        "can intercept and unroll the loop per pipeline rank."
    )


mlir.register_lowering(pscan_p, _pscan_lowering)


def _is_scalar_aval(aval: object) -> bool:
    """Return ``True`` iff ``aval`` is a rank-0 array-like abstract value.

    Args:
        aval: Aval value consumed by this operation.

    Returns:
        Return ``True`` iff ``aval`` is a rank-0 array-like abstract value.
    """
    return hasattr(aval, "shape") and tuple(aval.shape) == ()


def _prune_closed_jaxpr_to_outputs(
    closed_jaxpr: ClosedJaxpr,
    keep_outvars: tuple[Var, ...],
) -> ClosedJaxpr:
    """Return ``closed_jaxpr`` pruned to the dataflow needed for ``keep_outvars``.

    Constvars and invars are preserved so the resulting closed jaxpr keeps the
    same call signature as the original body trace. This lets the compiled MPMD
    path reuse the same resolved consts while dropping dead reverse-pass eqns
        from pre-differentiated bodies.

    Args:
        closed_jaxpr: Closed JAXPR being inspected, rewritten, split, or executed.
        keep_outvars: Keep outvars value consumed by this operation.

    Returns:
        Return ``closed_jaxpr`` pruned to the dataflow needed for ``keep_outvars``.
    """
    needed: set[int] = {id(v) for v in keep_outvars}
    kept_rev: list[object] = []
    for eqn in reversed(closed_jaxpr.jaxpr.eqns):
        eqn_outvars = [v for v in eqn.outvars if isinstance(v, Var)]
        if not any(id(v) in needed for v in eqn_outvars):
            continue
        kept_rev.append(eqn)
        for invar in eqn.invars:
            if isinstance(invar, Var):
                needed.add(id(invar))

    pruned = Jaxpr(
        constvars=list(closed_jaxpr.jaxpr.constvars),
        invars=list(closed_jaxpr.jaxpr.invars),
        outvars=list(keep_outvars),
        eqns=list(reversed(kept_rev)),
        effects=closed_jaxpr.jaxpr.effects,
    )
    return ClosedJaxpr(pruned, closed_jaxpr.consts)


def _probe_body(
    fun: Callable[[jax.Array]],
    probe_idx: jax.Array,
) -> tuple[str, ClosedJaxpr, ClosedJaxpr | None, object | None, object]:
    """Trace ``fun`` once and classify its output convention.

    Supported conventions:

        * scalar loss: ``fun(i) -> scalar``
        * pre-differentiated: ``fun(i) -> (scalar, grads_pytree)``

    Args:
        fun: Fun value consumed by this operation.
        probe_idx: Probe idx value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    out_shape = jax.eval_shape(fun, probe_idx)
    fn_jaxpr = jax.make_jaxpr(fun)(probe_idx)
    out_tree = jax.tree.structure(out_shape)

    if isinstance(out_shape, (tuple, list)) and len(out_shape) == 2:
        loss_shape, grad_shape = out_shape
        loss_leaves = jax.tree.leaves(loss_shape)
        if len(loss_leaves) != 1 or not _is_scalar_aval(loss_leaves[0]):
            raise ValueError("treduce: pre-differentiated bodies must return `(scalar_loss, grads_pytree)`.")
        loss_jaxpr = _prune_closed_jaxpr_to_outputs(
            fn_jaxpr,
            (fn_jaxpr.jaxpr.outvars[0],),
        )
        return "prediff", fn_jaxpr, loss_jaxpr, jax.tree.structure(grad_shape), out_tree

    out_leaves = jax.tree.leaves(out_shape)
    if len(out_leaves) == 1 and _is_scalar_aval(out_leaves[0]):
        return "scalar_loss", fn_jaxpr, fn_jaxpr, None, out_tree

    raise ValueError("treduce: body must return either a scalar loss or `(scalar_loss, grads_pytree)`.")


def treduce(
    fun: Callable[[object]],
    xs: object,
    schedule: object,
    axis: int = 0,
    operation: object = None,
) -> object:
    """Reduce ``fun`` over microbatches of ``xs`` under a pipeline schedule.

    ``xs`` is a pytree with a common leading microbatch axis (``axis``).
    ``fun(microbatch)`` runs on each microbatch in schedule-driven order
    (not necessarily sequential) and its outputs are combined via
    ``operation``.

    Args:
        fun: User function ``microbatch -> output`` or
            ``microbatch -> (loss, grads)``.
        xs: Pytree of arrays; each array's ``axis``-th dim is the
            microbatch axis.
        schedule: Pipeline schedule from :mod:`spectrax.runtime.schedules`.
            Determines the order of forward/backward phases across
            microbatches and ranks.
        axis: The microbatch axis in each leaf of ``xs``. Default 0.
        operation: Per-output accumulation ops. Default is
            ``(Concat, Add, Add, ...)`` — first output concatenated,
            rest summed (matches ``value_and_grad`` output shape). If a
            sequence is supplied, it must cover every output leaf.

    Returns:
        Tree with the same structure as ``fun``'s output, accumulated
        across all microbatches.
    """
    flat_leaves = jax.tree.leaves(xs)
    if not flat_leaves:
        raise ValueError("treduce: xs is empty.")
    length = flat_leaves[0].shape[axis]

    def wrap(i: jax.Array) -> object:
        """Select the ``i``-th microbatch from ``xs`` and call ``fun``.

        Args:
            i: I value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        mb = jax.tree.map(lambda arr: jax.lax.dynamic_index_in_dim(arr, i, axis=axis, keepdims=False), xs)
        return fun(mb)

    return treduce_i(wrap, length, schedule, operation=operation)


def treduce_i(
    fun: Callable[[jax.Array]],
    length: int,
    schedule: object,
    operation: object = None,
) -> object:
    """Reduce ``fun(i)`` for ``i`` in ``[0, length)`` under a pipeline schedule.

    Unlike :func:`treduce`, this passes the microbatch index directly
    to ``fun`` — use when your body wants to compute the microbatch
    slice itself (e.g. dynamic_slice of a padded batch).

    Args:
        fun: Function ``i -> output``.
        length: Number of microbatches.
        schedule: Pipeline schedule.
        operation: Per-output accumulation operation(s).

    Supported body conventions:

    * ``fun(i) -> scalar``: eager mode returns concatenated losses; inside
      ``@sxjit`` the compiled MPMD path synthesizes gradients with respect
      to the captured model argument and returns ``(losses, grads)``.
    * ``fun(i) -> (scalar, grads_pytree)``: eager and compiled paths both
      return ``(losses, summed_grads)``.

    Returns:
        Accumulated output tree.
    """
    probe_idx = jnp.zeros((), dtype=jnp.int32)
    body_mode, fn_jaxpr, loss_jaxpr, grad_tree, out_tree = _probe_body(fun, probe_idx)
    out_avals = [v.aval for v in fn_jaxpr.jaxpr.outvars]
    n_outs = len(out_avals)

    if operation is None:
        ops = [Concat(length) if i == 0 else Add() for i in range(n_outs)]
    elif isinstance(operation, Op):
        ops = [operation] * n_outs
    elif isinstance(operation, (list, tuple)):
        if len(operation) < n_outs:
            raise ValueError(
                "treduce: operation sequence is shorter than the body output leaves; "
                f"got {len(operation)} ops for {n_outs} outputs."
            )
        else:
            ops = list(operation)[:n_outs]
    else:
        raise TypeError(
            f"treduce: `operation` must be None, an Op, or a sequence of Op; got {type(operation).__name__}."
        )

    init_state = [op.state(aval) for op, aval in zip(ops, out_avals, strict=True)]

    def body_wrapped(*args):
        """Call ``fun(i)`` and fold each output into the running accumulator via ``ops``.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
        """
        i = args[0]
        state = list(args[1:])
        outs = list(jax.tree.leaves(fun(i)))
        new_state = [op.update(s, v, i) for op, s, v in zip(ops, state, outs, strict=True)]
        return new_state

    body_jaxpr = jax.make_jaxpr(body_wrapped)(probe_idx, *init_state)
    n_consts = len(body_jaxpr.consts)

    results = pscan_p.bind(
        *body_jaxpr.consts,
        *init_state,
        jaxpr=body_jaxpr,
        fn_jaxpr=fn_jaxpr,
        loss_jaxpr=loss_jaxpr,
        body_mode=body_mode,
        grad_tree=grad_tree,
        ops=_HashableOps(tuple(ops)),
        n_mubatches=length,
        n_consts=n_consts,
        schedule=_HashableSchedule(schedule),
        n_outs=n_outs,
    )
    return jax.tree_util.tree_unflatten(out_tree, results)

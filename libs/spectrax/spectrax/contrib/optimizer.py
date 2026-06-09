# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`Optimizer` — a pytree-registered wrapper around :mod:`optax`.

A pytree-registered wrapper around :mod:`optax` for spectrax's module/state
split: the optimizer is a JAX pytree, so it flows through
:func:`spectrax.jit`, :func:`jax.jit`, :func:`spectrax.vmap`, and
other transforms without special handling. Its dynamic children are
the optax state and the step counter; the gradient transformation and
selector are carried as static aux (JAX never traces them).

The public update is **functional** — :meth:`Optimizer.update` takes
the current trainable :class:`~spectrax.State` plus grads and returns
the *new* parameters / optimizer pair, without mutating anything. That's
what makes it safe to compile: jit has to be able to re-run the same
function on new inputs and produce the same outputs, which isn't true
if the optimizer silently mutates its own state on each call.

Eager-mode sugar is provided via :meth:`Optimizer.apply_eager` which
wraps the functional update with an ``spx.update`` write-back so
training loops that live outside a transform can still use the
familiar ``opt.apply_eager(model, grads)`` shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import jax

try:
    import optax as _optax
except ImportError as _e:
    _optax = None
    _optax_import_error = _e
else:
    _optax_import_error = None

from ..core.graph import export, update
from ..core.module import Module
from ..core.selector import Selector, SelectorSugar, as_selector
from ..core.state import State

if TYPE_CHECKING:
    import optax

__all__ = ["MultiOptimizer", "Optimizer"]


class _OptaxModule(Protocol):
    """Subset of :mod:`optax` used by this wrapper."""

    def apply_updates(self, params: State, updates: State) -> State:
        """Apply optax updates to a :class:`State` tree.

        Args:
            params: Parameter mapping or primitive parameter dictionary.
            updates: Updates value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        ...


OptState = object
StepValue = int | jax.Array


def _require_optax() -> _OptaxModule:
    """Return the imported :mod:`optax` module, raising if it isn't installed.

    The optax import is attempted once at module load and cached in
    the module-level ``_optax``; this helper centralises the
    user-facing error so every public entry point can call it on
    demand without duplicating the install hint.

    Returns:
        The imported :mod:`optax` module.

    Raises:
        ImportError: optax is not importable. The message includes a
            ``pip install`` hint and chains the original import error.
    """
    if _optax is None:
        raise ImportError(
            "spectrax.contrib.Optimizer requires optax. Install with "
            "`pip install spectrax-lib[contrib]` or `pip install optax`."
        ) from _optax_import_error
    return cast(_OptaxModule, _optax)


@jax.tree_util.register_pytree_node_class
class Optimizer:
    """A pytree-registered, module-aware :mod:`optax` wrapper.

    Holds:

    * ``tx`` — a :class:`optax.GradientTransformation` (static aux).
    * ``selector`` — a :class:`~spectrax.Selector` picking the
      trainable subset (static aux).
    * ``opt_state`` — the optax state pytree (dynamic children).
    * ``step`` — a scalar JAX int tracking call count (dynamic child).

    Construct with :meth:`Optimizer.create` (reads the initial trainable
    :class:`State` off a live module) or with the bare constructor
    (mostly used by :meth:`tree_unflatten`).

    Uses ``__slots__`` to skip the per-instance ``__dict__`` allocation
    — the Optimizer is flattened/unflattened once per jit call in a
    hot training loop, so the fixed-slot layout meaningfully reduces
    per-step dispatch overhead versus a plain class.

    Example — jittable training step::

        opt = spx.contrib.Optimizer.create(model, optax.adamw(3e-4))

        @spx.jit
        def step(opt, parameters, rest, gdef, x, y):
            def loss_fn(parameters):
                m = spx.bind(gdef, parameters.overlay(rest))
                return ((m(x) - y) ** 2).mean()
            loss, grads = jax.value_and_grad(loss_fn)(parameters)
            new_parameters, new_opt = opt.update(parameters, grads)
            return loss, new_parameters, new_opt

    Example — eager loop::

        opt = spx.contrib.Optimizer.create(model, optax.adamw(3e-4))
        for x, y in batches:
            grads = spx.grad(loss)(model, x, y)
            opt = opt.apply_eager(model, grads)   # mutates model in-place, returns new opt
    """

    __slots__ = ("opt_state", "selector", "step", "tx")

    def __init__(
        self,
        tx: optax.GradientTransformation,
        selector: Selector,
        opt_state: OptState,
        step: StepValue,
    ) -> None:
        """Low-level constructor; prefer :meth:`create`.

        Kept public so :meth:`tree_unflatten` can rebuild instances
        without calling :mod:`optax`'s ``tx.init`` again.

        Args:
            tx: The :class:`optax.GradientTransformation` to apply.
            selector: The :class:`~spectrax.Selector` picking the
                trainable subset.
            opt_state: An optax state pytree, normally produced by
                ``tx.init`` (or carried across a jit boundary).
            step: A scalar JAX int (typically ``int32``) tracking
                how many updates have been applied.
        """
        self.tx = tx
        self.selector = selector
        self.opt_state = opt_state
        self.step = step

    @classmethod
    def create(
        cls,
        module: Module,
        tx: optax.GradientTransformation,
        *,
        wrt: SelectorSugar = "parameters",
    ) -> Optimizer:
        """Build an :class:`Optimizer` whose optax state is sized to ``wrt``.

        Exports the live module, partitions its :class:`State` by
        ``wrt``, and runs ``tx.init`` on the trainable portion *only*
        — so Adam moments, SGD velocities, etc. exist only for the
        parameters that will actually be differentiated.

        Args:
            module: The source of initial parameter shapes/dtypes.
            tx: An :class:`optax.GradientTransformation`.
            wrt: :data:`~spectrax.core.selector.SelectorSugar` picking
                which variables train. Default ``"parameters"``; use a
                Variable subclass (e.g. ``nn.LoraParameter``) or a
                composite selector for fine-tuning.

        Returns:
            A fresh :class:`Optimizer` with ``step = 0``.
        """
        _require_optax()
        sel = as_selector(wrt)
        _gdef, state = export(module)
        parameters, _rest = sel.partition_state(module, state)
        opt_state = tx.init(parameters)
        step = jax.numpy.asarray(0, dtype=jax.numpy.int32)
        return cls(tx=tx, selector=sel, opt_state=opt_state, step=step)

    def update(self, parameters: State, grads: State) -> tuple[State, Optimizer]:
        """Apply one optax step functionally and return ``(new_parameters, new_opt)``.

        Does **not** touch any live :class:`Module`; the caller is
        responsible for writing ``new_parameters`` back with
        :func:`spectrax.update` if they want eager-mode propagation.

        Safe to call inside :func:`jax.jit` / :func:`spectrax.jit`.

        Args:
            parameters: The trainable :class:`State` the gradients were
                taken against. Must have the same pytree shape as the
                ``parameters`` seen by :meth:`create`.
            grads: Gradient :class:`State` with the same shape as
                ``parameters``.

        Returns:
            ``(new_parameters, new_optimizer)``. ``new_optimizer`` has
            ``step`` incremented and its ``opt_state`` advanced by
            one optax step.
        """
        tx = self.tx
        updates, new_opt_state = tx.update(grads, self.opt_state, parameters)
        optax_mod = _require_optax()
        new_parameters = optax_mod.apply_updates(parameters, updates)
        new_opt = Optimizer(
            tx=self.tx,
            selector=self.selector,
            opt_state=new_opt_state,
            step=self.step + 1,
        )
        return new_parameters, new_opt

    def apply_eager(self, module: Module, grads: State) -> Optimizer:
        """Eager-mode sugar: run :meth:`update` and write results back to ``module``.

        Not safe inside a transform — it mutates the live module via
        :func:`spectrax.update`. Use this in hand-rolled Python loops
        where you don't want to thread ``parameters`` through yourself.

        Args:
            module: Live module whose trainable variables should be
                updated in place.
            grads: Gradient :class:`State` for the trainable subset.

        Returns:
            The new :class:`Optimizer` (with advanced ``opt_state`` and
            ``step``). The returned optimizer is **not** the same
            object as ``self``; rebind ``opt = opt.apply_eager(...)``.
        """
        _gdef, state = export(module)
        parameters, _rest = self.selector.partition_state(module, state)
        new_parameters, new_opt = self.update(parameters, grads)
        update(module, new_parameters)
        return new_opt

    def tree_flatten(self) -> tuple[tuple[OptState, StepValue], tuple[optax.GradientTransformation, Selector]]:
        """Pytree flatten: expose ``opt_state`` and ``step`` as dynamic leaves.

        The optax transformation and the selector are carried as
        static ``aux_data`` — they must be hashable (optax
        transformations are namedtuple-shaped so comparing by identity
        is enough; spectrax :class:`Selector` s are frozen dataclasses).

        Returns:
            ``((opt_state, step), (tx, selector))`` — JAX's flatten
            convention.
        """
        return (self.opt_state, self.step), (self.tx, self.selector)

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[optax.GradientTransformation, Selector],
        children: tuple[OptState, StepValue],
    ) -> Optimizer:
        """Rebuild an :class:`Optimizer` from its flattened representation.

        Inverse of :meth:`tree_flatten`. Calls :meth:`__init__` directly
        so no further :mod:`optax` work is performed.

        Args:
            aux: The static aux pair ``(tx, selector)``.
            children: The dynamic leaves ``(opt_state, step)``.

        Returns:
            A new :class:`Optimizer` instance with the given parts.
        """
        opt_state, step = children
        tx, selector = aux
        return cls(tx=tx, selector=selector, opt_state=opt_state, step=step)


@jax.tree_util.register_pytree_node_class
class MultiOptimizer:
    """Compose several :class:`Optimizer` s each over a disjoint param slice.

    Per-slice selectors are resolved against the live module at
    :meth:`create` time and cached as a frozenset of ``(kind, path)``
    tuples per sub, so :meth:`update` can run entirely from
    :class:`State` + frozenset lookups with no further module walks —
    safe inside :func:`jax.jit` / :func:`spectrax.jit`.

    The cached path frozensets are carried as static pytree aux (so
    jit can hash them), and each sub :class:`Optimizer` contributes
    its own dynamic children. Sub-optimizers are assumed to cover
    disjoint slices; overlapping selectors produce undefined
    last-writer-wins behaviour.

    Useful for per-collection learning rates (e.g. ``1e-4`` on base
    weights + ``1e-3`` on LoRA adapters).
    """

    def __init__(
        self,
        subs: list[Optimizer],
        owned_paths: tuple[frozenset[tuple[str, str]], ...],
    ) -> None:
        """Low-level constructor; prefer :meth:`create`.

        Args:
            subs: One :class:`Optimizer` per slice, in stable order.
            owned_paths: Parallel tuple of ``frozenset`` s identifying
                the ``(collection, path)`` pairs each sub owns.
                Resolved once by :meth:`create` so ``update`` doesn't
                re-walk the module tree.
        """
        self.subs = list(subs)
        self.owned_paths = owned_paths

    @classmethod
    def create(
        cls,
        module: Module,
        by_selector: dict[SelectorSugar, optax.GradientTransformation],
    ) -> MultiOptimizer:
        """Build one :class:`Optimizer` per ``(selector, tx)`` pair.

        Each selector is applied to ``module`` once to produce the
        frozenset of owned ``(kind, path)`` pairs — cached as static
        aux so downstream updates don't touch the live module.

        Args:
            module: Live module whose variables the selectors will
                pick from.
            by_selector: Mapping from :data:`SelectorSugar` values
                (``"parameters"``, ``"lora"``, :class:`~spectrax.nn.LoraParameter`,
                composite selectors, …) to :mod:`optax`
                transformations.

        Returns:
            A :class:`MultiOptimizer` ready for functional updates.
        """
        subs: list[Optimizer] = []
        owned: list[frozenset[tuple[str, str]]] = []
        for sel_sugar, tx in by_selector.items():
            sel = as_selector(sel_sugar)
            sub = Optimizer.create(module, tx, wrt=sel)
            paths = frozenset((v.kind, p) for p, v in sel.apply(module))
            subs.append(sub)
            owned.append(paths)
        return cls(subs, tuple(owned))

    def update(self, parameters: State, grads: State) -> tuple[State, MultiOptimizer]:
        """Dispatch each sub-optimizer's update over its owned slice.

        The full ``parameters`` / ``grads`` are sliced per sub using the
        cached path frozensets, each sub runs independently, and the
        results are overlaid back onto ``parameters`` to produce
        ``new_parameters``. Order is stable; subs are evaluated in the
        order passed to :meth:`create`.

        Args:
            parameters: Full trainable :class:`State` — union of every
                sub's owned slice. Leaves outside every sub's
                ownership pass through unchanged.
            grads: Gradient :class:`State` shaped like ``parameters``.

        Returns:
            ``(new_parameters, new_multi_optimizer)``.
        """
        new_subs: list[Optimizer] = []
        new_parameters = parameters
        for sub, paths in zip(self.subs, self.owned_paths, strict=True):
            sub_parameters = _slice_state(new_parameters, paths)
            sub_grads = _slice_state(grads, paths)
            updated_parameters, new_sub = sub.update(sub_parameters, sub_grads)
            new_parameters = new_parameters.overlay(updated_parameters)
            new_subs.append(new_sub)
        return new_parameters, MultiOptimizer(new_subs, self.owned_paths)

    def apply_eager(self, module: Module, grads: State) -> MultiOptimizer:
        """Eager-mode: run :meth:`update` and write results back to ``module``.

        Not safe inside a transform (it mutates the live module via
        :func:`spectrax.update`). The trainable slice is rebuilt from
        ``module``'s current state by unioning every sub's owned
        ``(kind, path)`` set, so leaves outside any sub's ownership
        pass through unmodified.

        Args:
            module: Live module whose trainable variables should be
                updated in place.
            grads: Gradient :class:`State` for the union of trainable
                slices.

        Returns:
            The new :class:`MultiOptimizer` with each sub's state
            advanced one step. Rebind: ``mopt = mopt.apply_eager(...)``.
        """
        _gdef, state = export(module)
        all_paths: set[tuple[str, str]] = set().union(*self.owned_paths) if self.owned_paths else set()
        parameters = _slice_state(state, all_paths)
        new_parameters, new_multi = self.update(parameters, grads)
        update(module, new_parameters)
        return new_multi

    def tree_flatten(self) -> tuple[tuple[Optimizer, ...], tuple[frozenset[tuple[str, str]], ...]]:
        """Pytree flatten: subs are dynamic children, owned-paths is static aux.

        Each sub :class:`Optimizer` is itself pytree-registered, so
        flattening recursively produces a flat list of optax leaves.

        Returns:
            ``(tuple_of_subs, owned_paths)`` — JAX's flatten convention.
        """
        return tuple(self.subs), self.owned_paths

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[frozenset[tuple[str, str]], ...],
        children: tuple[Optimizer, ...],
    ) -> MultiOptimizer:
        """Rebuild a :class:`MultiOptimizer` from its flattened representation.

        Args:
            aux: The static ``owned_paths`` tuple.
            children: The dynamic sub-optimizer tuple.

        Returns:
            A new :class:`MultiOptimizer` with the given parts.
        """
        return cls(list(children), aux)


def _slice_state(state: State, paths: set[tuple[str, str]] | frozenset[tuple[str, str]]) -> State:
    """Project ``state`` down to the leaves at the given ``(collection, path)`` pairs.

    Used by :class:`MultiOptimizer` to materialise the per-sub
    parameter / gradient slices passed into each
    :meth:`Optimizer.update`. Leaves outside ``paths`` are omitted
    entirely so the resulting :class:`State` has the same shape as
    what :mod:`optax` saw at ``init`` time.

    Args:
        state: The full state to project.
        paths: Iterable of ``(collection, path)`` keys to keep.

    Returns:
        A new :class:`State` containing only the listed leaves.
    """
    out = State()
    for c, p in paths:
        value = state.get(c, p, default=None)
        if value is not None:
            out.set(c, p, value)
    return out

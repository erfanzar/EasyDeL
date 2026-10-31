# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.contrib.optimizer`.

Covers both the functional update path (``opt.update(parameters, grads)``)
and the eager-sugar path (``opt.apply_eager(module, grads)``), plus
pytree-compatibility with :func:`jax.jit` / :func:`jax.tree_util` so
the wrapper composes inside traced training steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

optax = pytest.importorskip("optax")

from spectrax.contrib.optimizer import MultiOptimizer, Optimizer  # noqa: E402
from spectrax.core.graph import export  # noqa: E402
from spectrax.core.module import Module  # noqa: E402
from spectrax.core.state import State  # noqa: E402
from spectrax.nn.linear import Linear  # noqa: E402
from spectrax.nn.lora import LoraParameter, wrap_lora  # noqa: E402
from spectrax.rng.rngs import Rngs  # noqa: E402
from spectrax.transforms.grad import value_and_grad  # noqa: E402


class NestedLoRA(Module):
    """Tiny module whose LoRA leaves live under a dotted path."""

    def __init__(self, *, rngs: Rngs):
        """Create a LoRA-wrapped child layer."""
        super().__init__()
        self.fc = wrap_lora(Linear(2, 2, rngs=rngs), rank=1, rngs=rngs)

    def forward(self, x):
        """Forward through the wrapped projection."""
        return self.fc(x)


def test_optimizer_create_constructs_with_adamw():
    """``Optimizer.create`` wraps an optax transform and starts at step 0."""
    m = Linear(4, 4, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.adamw(1e-3))
    assert int(opt.step) == 0
    assert opt.opt_state is not None


def test_optimizer_opt_state_sized_to_selection():
    """``tx.init`` sees only the trainable slice — smaller for narrower selectors.

    Using ``wrt="parameters"`` allocates Adam moments for every weight /
    bias tensor; using ``wrt="lora"`` on a module with no LoRA
    parameters leaves only the optax scalar book-keeping (step
    counter) behind. Comparing totals quantifies the gap.
    """
    m = Linear(4, 4, rngs=Rngs(0))
    opt_params = Optimizer.create(m, optax.adam(1e-3))
    opt_lora = Optimizer.create(m, optax.adam(1e-3), wrt="lora")
    nparams = sum(v.size for v in jax.tree.leaves(opt_params.opt_state) if hasattr(v, "size"))
    nlora = sum(v.size for v in jax.tree.leaves(opt_lora.opt_state) if hasattr(v, "size"))
    assert nparams > 0
    assert nlora < nparams


def test_optimizer_update_is_functional():
    """``update(parameters, grads)`` returns new ``(parameters, optimizer)`` — no mutation."""
    m = Linear(2, 2, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.sgd(0.1))
    _gdef, state = export(m)
    params, _ = opt.selector.partition_state(m, state)
    grads = State({c: {p: jnp.zeros_like(v) for p, v in d.items()} for c, d in params.raw().items()})

    new_params, new_opt = opt.update(params, grads)
    assert int(new_opt.step) == 1
    assert int(opt.step) == 0
    assert new_opt is not opt
    for c, d in params.raw().items():
        for p, v in d.items():
            assert jnp.allclose(new_params.raw()[c][p], v)


def test_optimizer_update_decreases_convex_loss():
    """One SGD step on MSE drops the loss."""
    m = Linear(2, 2, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.sgd(0.1))
    x = jnp.asarray([[1.0, 2.0]])
    y = jnp.asarray([[0.0, 0.0]])

    def loss_fn(mod):
        """Sum-of-squares loss."""
        return ((mod(x) - y) ** 2).sum()

    l0 = loss_fn(m)
    _loss, grads = value_and_grad(loss_fn)(m)
    new_opt = opt.apply_eager(m, grads)
    l1 = loss_fn(m)
    assert float(l1) < float(l0)
    assert int(new_opt.step) == 1


def test_optimizer_apply_eager_mutates_live_module():
    """``apply_eager`` syncs updated parameters back to the module in place."""
    m = Linear(2, 2, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.sgd(0.1))
    x = jnp.asarray([[1.0, 2.0]])
    y = jnp.asarray([[0.0, 0.0]])

    def loss_fn(mod):
        """Sum-of-squares loss."""
        return ((mod(x) - y) ** 2).sum()

    w_before = jnp.asarray(m.weight.value)
    _loss, grads = value_and_grad(loss_fn)(m)
    opt = opt.apply_eager(m, grads)
    assert not jnp.allclose(m.weight.value, w_before)


def test_optimizer_is_pytree():
    """Optimizer flattens/unflattens cleanly via the jax pytree utilities."""
    m = Linear(2, 2, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.adamw(1e-3))
    leaves, treedef = jax.tree_util.tree_flatten(opt)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, Optimizer)
    assert int(rebuilt.step) == int(opt.step)
    assert rebuilt.tx is opt.tx
    assert rebuilt.selector == opt.selector


def test_optimizer_flows_through_jax_jit():
    """The pytree registration lets ``Optimizer`` pass through :func:`jax.jit`.

    A fully jit-compiled training step carrying ``(parameters, opt)``
    through as pytree arguments, returning new ones on the other side.
    """
    m = Linear(2, 2, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.sgd(0.1))
    _gdef, state = export(m)
    params, _ = opt.selector.partition_state(m, state)
    x = jnp.ones((1, 2))
    y = jnp.zeros((1, 2))

    @jax.jit
    def step(opt, params, x, y):
        """Jitted single-step training: forward + grad + optax update."""

        def loss_fn(params):
            """Rebind the live module with ``parameters`` and score MSE."""
            import spectrax as spx

            gdef, full = export(m)
            _p, rest = opt.selector.partition_state(m, full)
            merged = params.merge(rest)
            rebuilt = spx.bind(gdef, merged)
            return ((rebuilt(x) - y) ** 2).sum()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        new_params, new_opt = opt.update(params, grads)
        return loss, new_params, new_opt

    loss, _new_params, new_opt = step(opt, params, x, y)
    jax.block_until_ready(loss)
    assert int(new_opt.step) == 1
    assert isinstance(new_opt, Optimizer)


def test_multi_optimizer_create_and_update():
    """:class:`MultiOptimizer` dispatches per-selector and returns a fresh instance."""
    m = Linear(2, 2, rngs=Rngs(0))
    mopt = MultiOptimizer.create(m, {"parameters": optax.sgd(0.01)})
    assert len(mopt.subs) == 1
    assert len(mopt.owned_paths) == 1

    _gdef, state = export(m)
    params = State({c: dict(d) for c, d in state.raw().items()})
    zero_grads = State({c: {p: jnp.zeros_like(v) for p, v in d.items()} for c, d in state.raw().items()})

    new_params, new_mopt = mopt.update(params, zero_grads)
    for c, d in params.raw().items():
        for p, v in d.items():
            assert jnp.allclose(new_params.raw()[c][p], v)
    assert int(new_mopt.subs[0].step) == 1


def test_multi_optimizer_handles_nested_lora_paths():
    """Nested dotted paths are sliced correctly for per-selector updates."""
    m = NestedLoRA(rngs=Rngs(0))
    mopt = MultiOptimizer.create(m, {"parameters": optax.sgd(0.01), "lora": optax.adam(0.01)})
    _gdef, state = export(m)
    params = state.filter("parameters", "lora", copy=True)
    grads = params.map(lambda leaf: jnp.ones_like(leaf), copy=True)

    lora_before = params.get("lora", "fc.lora_a")
    new_params, new_mopt = mopt.update(params, grads)

    assert int(new_mopt.subs[0].step) == 1
    assert int(new_mopt.subs[1].step) == 1
    assert new_params.get("parameters", "fc.base_module.weight") is not None
    assert new_params.get("lora", "fc.lora_a") is not None
    assert not jnp.allclose(new_params.get("lora", "fc.lora_a"), lora_before)


def test_multi_optimizer_is_pytree():
    """:class:`MultiOptimizer` is a pytree (children = subs, aux = owned paths)."""
    m = Linear(2, 2, rngs=Rngs(0))
    mopt = MultiOptimizer.create(m, {"parameters": optax.sgd(0.01)})
    leaves, treedef = jax.tree_util.tree_flatten(mopt)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, MultiOptimizer)
    assert len(rebuilt.subs) == len(mopt.subs)


def test_optimizer_missing_optax_raises_helpfully():
    """Calling :meth:`Optimizer.create` without optax raises ``ImportError``."""
    import spectrax.contrib.optimizer as mod

    saved = mod._optax
    try:
        mod._optax = None
        mod._optax_import_error = ImportError("no optax")
        with pytest.raises(ImportError, match="optax"):
            Optimizer.create(Linear(2, 2, rngs=Rngs(0)), object())
    finally:
        mod._optax = saved


def test_optimizer_wrt_class_selector_works():
    """``wrt=`` accepts a :class:`~spectrax.Variable` subclass directly.

    A bare :class:`~spectrax.nn.Linear` has no
    :class:`~spectrax.nn.LoraParameter` members, so ``tx.init`` runs
    over an empty :class:`State`. Optax still allocates a scalar step
    counter; assert that the array-sized leaves are absent, not the
    book-keeping.
    """
    m = Linear(4, 4, rngs=Rngs(0))
    opt = Optimizer.create(m, optax.adam(1e-3), wrt=LoraParameter)
    assert int(opt.step) == 0
    total_bytes = sum(
        v.size * v.dtype.itemsize for v in jax.tree.leaves(opt.opt_state) if hasattr(v, "size") and v.size > 1
    )
    assert total_bytes == 0

# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``spx.remat`` — gradient checkpointing.

``spx.remat`` trades activation memory for recomputation: intermediate
activations produced during the forward pass are discarded and
recomputed during the backward pass. For an N-block stack this turns
peak memory from ``O(N * activation)`` into ``O(activation)`` — the
classic memory-vs-compute knob.

Two usage modes are demonstrated:

* **Function-style**: ``spx.remat(fn)`` wraps a per-call function.
  Useful when you want fine-grained control at the call site.
* **Class-style**: ``spx.remat(MyModule)`` returns a subclass whose
  ``forward`` is permanently checkpointed. Build instances normally;
  the backward pass recomputes every block's forward without any
  extra ``spx.remat(...)`` in the model body.

Both modes are numerically equivalent to the un-rematted version —
gradients match within float noise. The saving only shows up under a
memory profiler on a real GPU/TPU run, so this example asserts
numerical equivalence instead.

Run::

    python -m examples.03_transformations.04_remat
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

import spectrax as spx


def loss_fn(model, x):
    """Scalar loss: the mean of ``model(x)``, convenient for autodiff."""
    return model(x).mean()


def main():
    """Compare function-remat and class-remat gradients against plain autodiff."""
    base = spx.nn.MLPBlock(features=8, hidden_features=32, rngs=spx.Rngs(0))
    x = jnp.ones((4, 8))

    baseline_grads = spx.grad(loss_fn)(base, x)

    def rematted_forward(m, inp):
        """Checkpointed forward — activations recomputed in backward."""
        return m(inp)

    def fn_loss(m, inp):
        """Loss through the function-style checkpointed call."""
        return spx.remat(rematted_forward)(m, inp).mean()

    fn_grads = spx.grad(fn_loss)(base, x)

    RematMLP = spx.remat(spx.nn.MLPBlock)
    class_model = RematMLP(features=8, hidden_features=32, rngs=spx.Rngs(0))
    class_grads = spx.grad(loss_fn)(class_model, x)

    def _max_abs_delta(ref_grads, other_grads):
        """Infinity-norm of the element-wise gradient difference."""
        ref_leaves = {(c, p): arr for c, p, arr in ref_grads.items()}
        other_leaves = {(c, p): arr for c, p, arr in other_grads.items()}
        return max(float(jnp.max(jnp.abs(ref_leaves[k] - other_leaves[k]))) for k in ref_leaves)

    fn_delta = _max_abs_delta(baseline_grads, fn_grads)
    class_delta = _max_abs_delta(baseline_grads, class_grads)

    print(f"remat class name: {type(class_model).__name__}")
    print(f"|fn-remat grads - baseline|_inf:    {fn_delta:.2e}")
    print(f"|class-remat grads - baseline|_inf: {class_delta:.2e}")
    print(f"equivalent: {fn_delta < 1e-5 and class_delta < 1e-5}")


if __name__ == "__main__":
    main()

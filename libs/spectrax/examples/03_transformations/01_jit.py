# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``spx.jit`` — compile a ``Module.__call__``.

``spx.jit`` is a drop-in for :func:`jax.jit` that understands
:class:`~spectrax.Module` arguments. Internally it exports each module
to ``(GraphDef, State)``, caches the compiled XLA executable keyed by
the graph-def tuple, and re-applies declared mutations on the way out.

Key concepts demonstrated:

* First call traces + compiles; second call hits the compile cache
  (visible via ``fn._spx_compile_cache``).
* Passing a freshly-built but structurally-identical module also hits
  the structural cache — no recompile.
* ``donate_argnums=`` indexes into the compiled function's
  ``(states, stripped_args, stripped_kwargs)`` tuple, not the user
  signature. We donate the ``stripped_args`` slot (index 1) whose
  buffer is safe to consume in-place.

Run::

    python -m examples.03_transformations.01_jit
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import spectrax as spx


def main():
    """Build a small Linear, call jit'd forward twice, and report cache stats."""
    model = spx.nn.Linear(8, 4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 8))

    @spx.jit
    def forward(m, inp):
        """Jit-compiled forward pass through ``m``."""
        return m(inp)

    y1 = forward(model, x)
    jax.block_until_ready(y1)
    compiles_after_first = len(forward._spx_compile_cache)

    y2 = forward(model, x)
    jax.block_until_ready(y2)
    compiles_after_second = len(forward._spx_compile_cache)

    twin = spx.nn.Linear(8, 4, rngs=spx.Rngs(1))
    y3 = forward(twin, x)
    jax.block_until_ready(y3)
    compiles_after_twin = len(forward._spx_compile_cache)

    @spx.jit(donate_argnums=(1,))
    def forward_donated(m, inp):
        """Jit with input-arg donation — XLA may reuse ``inp``'s buffer."""
        return m(inp) + 1.0

    y4 = forward_donated(model, jnp.ones((2, 8)))
    jax.block_until_ready(y4)

    print(f"output shape: {y1.shape}")
    print(f"compiles after 1st call:  {compiles_after_first}  (trace+compile)")
    print(f"compiles after 2nd call:  {compiles_after_second}  (cache hit, no trace)")
    print(f"compiles after twin model: {compiles_after_twin}  (same GraphDef -> still 1)")
    print(f"donated variant output norm: {float(jnp.linalg.norm(y4)):.4f}")


if __name__ == "__main__":
    main()

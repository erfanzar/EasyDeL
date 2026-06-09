# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3 one-step forward demo.

Builds :class:`examples.models.llama.Llama3` at a deliberately tiny
configuration that fits comfortably on a laptop CPU, runs a single
forward pass on random token ids, and prints the output logit
shape alongside the total parameter count.

The point of this example is not training but **wiring**: show
how the shared :mod:`examples.models.llama` module plugs into the
surrounding :mod:`examples.02_implementation_guide` walkthrough.
The same :class:`~examples.models.llama.Llama3` class appears in
the MPMD training examples under :mod:`examples.07_mpmd`.

Run::

    python -m examples.02_implementation_guide.01_llama3
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx

from ..models.llama import Llama3, Llama3Config


def count_parameters(model) -> int:
    """Sum the element counts of every :class:`spectrax.Parameter` in ``model``."""
    return sum(v.value.size for _, v in spx.iter_variables(model) if isinstance(v, spx.Parameter))


def main():
    """Build a tiny Llama 3, run one forward pass, and print shape + param count."""
    cfg = Llama3Config(
        vocab=256,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        ffn=256,
        n_layers=2,
    )
    model = Llama3(cfg, rngs=spx.Rngs(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (4, 32), 0, cfg.vocab)
    logits = model(ids)
    jax.block_until_ready(logits)
    print(f"llama3 logits shape: {logits.shape}")
    print(f"llama3 parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()

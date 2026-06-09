# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qwen 2.5 one-step forward demo.

Mirror of :mod:`examples.02_implementation_guide.01_llama3` for the
:class:`examples.models.qwen.Qwen` model: build a tiny config, run
a single forward pass on random token ids, and print the output
logit shape and total parameter count.

This example highlights where Qwen differs from Llama — notably
the QKV bias and the larger RoPE base — while sharing the same
single-argument ``forward(x)`` composition used throughout the
:mod:`examples` suite.

Run::

    python -m examples.02_implementation_guide.02_qwen
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx

from ..models.qwen import Qwen, QwenConfig


def count_parameters(model) -> int:
    """Sum the element counts of every :class:`spectrax.Parameter` in ``model``."""
    return sum(v.value.size for _, v in spx.iter_variables(model) if isinstance(v, spx.Parameter))


def main():
    """Build a tiny Qwen 2.5, run one forward pass, and print shape + param count."""
    cfg = QwenConfig(
        vocab=256,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        ffn=256,
        n_layers=2,
    )
    model = Qwen(cfg, rngs=spx.Rngs(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (4, 32), 0, cfg.vocab)
    logits = model(ids)
    jax.block_until_ready(logits)
    print(f"qwen logits shape: {logits.shape}")
    print(f"qwen parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()

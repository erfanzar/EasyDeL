# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inject LoRA adapters into a pretrained-style module.

Builds a tiny two-layer MLP, then swaps each :class:`spx.nn.Linear`
for a :class:`spx.nn.LoRA` wrapper using :func:`spx.nn.wrap_lora`.
Compares trainable parameter counts before and after, and confirms
forward outputs match at step 0 (LoRA ``B`` is zero-initialized).

Run::

    python -m examples.04_surgery.02_lora_injection
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn


class MLP(spx.Module):
    """Pretend pretrained MLP: two linears separated by a GELU."""

    def __init__(self, d: int, rngs: spx.Rngs):
        """Construct ``fc1`` / ``fc2`` linears with a GELU between."""
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2, rngs=rngs)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d * 2, d, rngs=rngs)

    def __call__(self, x):
        """Forward ``fc2(act(fc1(x)))``."""
        return self.fc2(self.act(self.fc1(x)))


def count_parameters(module: spx.Module) -> int:
    """Sum the sizes of every :class:`spx.Parameter` in ``module``."""
    return sum(v.value.size for _, v in spx.iter_variables(module, select=spx.of_type(spx.Parameter)))


def count_lora_only(module: spx.Module) -> int:
    """Sum the sizes of LoRA-collection cells (``lora_a`` / ``lora_b``)."""
    total = 0
    for _, v in spx.iter_variables(module, select="lora"):
        total += v.value.size
    return total


def main():
    """Build MLP, wrap linears with LoRA adapters, compare counts + outputs."""
    base = MLP(d=16, rngs=spx.Rngs(0))
    base_params = count_parameters(base)
    print(f"base trainable parameters: {base_params}")

    wrapped = MLP(d=16, rngs=spx.Rngs(0))
    wrapped.fc1 = nn.wrap_lora(wrapped.fc1, rank=4, rngs=spx.Rngs(1))
    wrapped.fc2 = nn.wrap_lora(wrapped.fc2, rank=4, rngs=spx.Rngs(2))

    total_after = count_parameters(wrapped)
    lora_only = count_lora_only(wrapped)
    print(f"wrapped total parameters: {total_after}")
    print(f"lora-only parameters: {lora_only}")
    print(f"non-adapter parameters unchanged: {total_after == base_params}")

    x = jax.random.normal(jax.random.PRNGKey(7), (2, 16))
    y_base = base(x)
    y_wrapped = wrapped(x)
    max_err = float(jnp.max(jnp.abs(y_base - y_wrapped)))
    print(f"outputs match at step 0 (zero-init B): max_err={max_err:.2e}")

    for path, mod in spx.iter_modules(wrapped, select=nn.LoRA):
        print(f"  LoRA at {path!r}: rank={mod.lora_a.value.shape[-1]}")


if __name__ == "__main__":
    main()

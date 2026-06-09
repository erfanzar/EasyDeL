# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Demonstrate ``fuse_1f1b`` and ``fuse_zb`` flags on ``spx.run``.

Runs Std1F1B with and without steady-state fusion and verifies that
the resulting losses match. Fusion reduces XLA dispatch overhead
without changing the numerical result.

Run::

    python -m examples.07_mpmd.07_fused_tasks
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx
from spectrax.runtime.schedules import Std1F1B
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def cross_entropy(logits, labels):
    """Mean token-level cross-entropy loss for language-model training."""
    return -(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1])).sum(-1).mean()


def main():
    """Compare unfused vs fused Std1F1B training and print both losses."""
    cfg = Llama3Config(
        vocab=1024,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        ffn=512,
        n_layers=4,
    )
    mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis="pp")

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        ids = jax.random.randint(jax.random.PRNGKey(1), (8, 16), 0, cfg.vocab)
        labels = jax.random.randint(jax.random.PRNGKey(2), (8, 16), 0, cfg.vocab)

        loss_plain, _ = spx.run(
            model,
            inputs=ids,
            targets=labels,
            mesh=mesh,
            mode="train",
            loss_fn=cross_entropy,
            schedule=Std1F1B(microbatches=4),
        )
        jax.block_until_ready(loss_plain)

        loss_fused, _ = spx.run(
            model,
            inputs=ids,
            targets=labels,
            mesh=mesh,
            mode="train",
            loss_fn=cross_entropy,
            schedule=Std1F1B(microbatches=4),
            fuse_1f1b=True,
        )
        jax.block_until_ready(loss_fused)

    print(f"unfused: {float(loss_plain):.4f}")
    print(f"fused:   {float(loss_fused):.4f}")


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the same Llama 3 model under all 9 pipeline schedules.

Demonstrates that every schedule produces the same loss for a fixed
model and input. Prints a comparison table of scheduler name vs loss.

Run::

    python -m examples.07_mpmd.06_all_schedulers
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx
from spectrax.runtime.schedules import (
    DualPipeV,
    Eager1F1B,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Std1F1B,
    ZeroBubbleH1,
)
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def cross_entropy(logits, labels):
    """Mean token-level cross-entropy loss for language-model training."""
    return -(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1])).sum(-1).mean()


def main():
    """Train under all 9 schedules and print a loss comparison table."""
    cfg = Llama3Config(
        vocab=1024,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        ffn=512,
        n_layers=4,
    )
    mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis="pp")

    schedules = [
        ("GPipe", GPipe(microbatches=4)),
        ("Std1F1B", Std1F1B(microbatches=4)),
        ("Eager1F1B", Eager1F1B(microbatches=4)),
        ("ZeroBubbleH1", ZeroBubbleH1(microbatches=4)),
        ("InterleavedH1", InterleavedH1(microbatches=4, virtual_stages=2)),
        ("KimiK2", KimiK2(microbatches=4, virtual_stages=2)),
        ("DualPipeV", DualPipeV(microbatches=4)),
        ("InterleavedGPipe", InterleavedGPipe(microbatches=4, virtual_stages=2)),
        ("Interleaved1F1BPlusOne", Interleaved1F1BPlusOne(microbatches=4, virtual_stages=2)),
    ]

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        ids = jax.random.randint(jax.random.PRNGKey(1), (8, 16), 0, cfg.vocab)
        labels = jax.random.randint(jax.random.PRNGKey(2), (8, 16), 0, cfg.vocab)

        print(f"{'Schedule':30s} {'Loss':>10s}")
        print("-" * 42)
        for name, sch in schedules:
            loss, _ = spx.run(
                model,
                inputs=ids,
                targets=labels,
                mesh=mesh,
                mode="train",
                loss_fn=cross_entropy,
                schedule=sch,
            )
            jax.block_until_ready(loss)
            print(f"{name:30s} {float(loss):10.4f}")


if __name__ == "__main__":
    main()

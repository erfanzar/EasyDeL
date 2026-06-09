# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Manual ``pp_stage`` annotation for pipeline stage placement.

Shows how to override automatic stage assignment by setting
``pp_stage`` on individual submodules. Demonstrates both string
aliases (``"first"``) and integer stage indices.

Run::

    python -m examples.07_mpmd.08_pp_stage_assignment
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx
from spectrax.runtime.primitives.split import auto_split
from spectrax.runtime.schedules import GPipe
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def cross_entropy(logits, labels):
    """Mean token-level cross-entropy loss for language-model training."""
    return -(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1])).sum(-1).mean()


def main():
    """Annotate submodules with pp_stage, split, and inspect stage contents."""
    cfg = Llama3Config(
        vocab=1024,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        ffn=512,
        n_layers=4,
    )
    n_pp = 4

    with logical_axis_rules(FSDP_TP_RULES):
        model = Llama3(cfg, rngs=spx.Rngs(0))
        model.embed.pp_stage = "first"
        model.head.pp_stage = 2

        stages = auto_split(model, n_pp)
        print(f"split into {len(stages)} stages")
        for i, stage in enumerate(stages):
            attrs = [n for n in vars(stage) if not n.startswith("_")]
            print(f"  stage {i}: {attrs}")

        model_default = Llama3(cfg, rngs=spx.Rngs(0))
        stages_default = auto_split(model_default, n_pp)
        print("\ndefault split (no annotations):")
        for i, stage in enumerate(stages_default):
            attrs = [n for n in vars(stage) if not n.startswith("_")]
            print(f"  stage {i}: {attrs}")

    mesh = spx.create_mesh(axis_dims=(n_pp, 1, -1, 1, 1, 1), mpmd_axis="pp")
    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model_run = Llama3(cfg, rngs=spx.Rngs(0))
        model_run.embed.pp_stage = "first"
        model_run.head.pp_stage = 2
        ids = jax.random.randint(jax.random.PRNGKey(1), (8, 16), 0, cfg.vocab)
        labels = jax.random.randint(jax.random.PRNGKey(2), (8, 16), 0, cfg.vocab)
        loss, _ = spx.run(
            model_run,
            inputs=ids,
            targets=labels,
            mesh=mesh,
            mode="train",
            loss_fn=cross_entropy,
            schedule=GPipe(microbatches=4),
        )
        jax.block_until_ready(loss)
    print(f"\ntrain loss (head on stage 2): {float(loss):.4f}")


if __name__ == "__main__":
    main()

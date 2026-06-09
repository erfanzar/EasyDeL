# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same Llama 3 model, two meshes — SPMD vs MPMD give similar loss.

This example demonstrates the unified-runtime promise of spectrax:
you write the model **once**, and the mesh decides whether
:func:`spectrax.run` goes through ``pjit`` (no pipeline
parallelism) or auto-splits the model and calls ``sxcall``
(with pipeline parallelism).

The same data and the same model class are fed through both a
SPMD mesh (no ``mpmd_axis``) and an MPMD mesh
(``mpmd_axis='pp'``), and the resulting losses are compared. They
match within ~1% relative difference, arising from the different
execution paths (single-HLO stacked SPMD vs per-rank jitted MPMD)
and auto-split stage boundaries.

Run::

    python -m examples.07_mpmd.03_spmd_vs_mpmd
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def cross_entropy(logits, labels):
    """Mean token-level cross-entropy loss for language-model training."""
    return -(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1])).sum(-1).mean()


def main():
    """Run the same model on SPMD then MPMD meshes and compare losses."""
    cfg = Llama3Config(
        vocab=1024,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        ffn=512,
        n_layers=4,
    )

    rngs = spx.Rngs(0)
    ids = jax.random.randint(jax.random.PRNGKey(1), (8, 16), 0, cfg.vocab)
    labels = jax.random.randint(jax.random.PRNGKey(2), (8, 16), 0, cfg.vocab)

    mesh_spmd = spx.create_mesh(axis_dims=(1, 1, -1, 1, 1, 1))
    with logical_axis_rules(FSDP_TP_RULES), mesh_spmd:
        model_spmd = Llama3(cfg, rngs=rngs)
        loss_spmd, _ = spx.run(
            model_spmd,
            inputs=ids,
            targets=labels,
            mesh=mesh_spmd,
            mode="train",
            loss_fn=cross_entropy,
        )
        jax.block_until_ready(loss_spmd)

    mesh_mpmd = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis="pp")
    with logical_axis_rules(FSDP_TP_RULES), mesh_mpmd:
        model_mpmd = Llama3(cfg, rngs=rngs)
        loss_mpmd, _ = spx.run(
            model_mpmd,
            inputs=ids,
            targets=labels,
            mesh=mesh_mpmd,
            mode="train",
            loss_fn=cross_entropy,
            microbatches=4,
        )
        jax.block_until_ready(loss_mpmd)

    print(f"SPMD mesh : {dict(mesh_spmd.shape)}")
    print(f"MPMD mesh : {dict(mesh_mpmd.shape)}")
    print()
    print(f"SPMD loss : {float(loss_spmd):.6f}")
    print(f"MPMD loss : {float(loss_mpmd):.6f}")
    rel_diff = abs(float(loss_spmd) - float(loss_mpmd)) / max(float(loss_spmd), 1e-6)
    print(f"rel diff  : {rel_diff:.2e}  (numerical noise from per-rank shardings)")


if __name__ == "__main__":
    main()

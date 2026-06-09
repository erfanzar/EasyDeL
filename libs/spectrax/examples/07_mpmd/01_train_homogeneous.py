# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train a Llama 3 with one ``spx.run`` call.

This example is the 50-LOC version of pipeline-parallel training.
The model is the standard :class:`examples.models.llama.Llama3` (embed +
blocks + head), written **once** with a single ``Llama3`` class.
``spx.run`` auto-splits :attr:`Llama3.blocks` across the mesh's MPMD
axis — no pipeline-aware bookkeeping in user code. Because embed and
head differ from the repeated transformer blocks, the resulting stages
are heterogeneous; spectrax routes them through the MPMD runtime
automatically. SPMD-only runs just drop ``mpmd_axis=`` from
:func:`spectrax.create_mesh`.

Key concepts demonstrated:

* ``mode='train'`` returns ``(loss, grads)`` given a user-supplied
  ``loss_fn``.
* Adding ``mpmd_axis="pp"`` to :func:`spectrax.create_mesh` enables
  pipeline parallelism — the model stays the same, ``spx.run`` does
  the split.
* Intra-stage FSDP+TP composes from the model's Linear ``sharding=``
  annotations under an active ``logical_axis_rules`` context.

Run::

    python -m examples.07_mpmd.01_train_homogeneous
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx
from spectrax.runtime.schedules import GPipe, Std1F1B
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def cross_entropy(logits, labels):
    """Mean token-level cross-entropy loss for language-model training."""
    return -(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1])).sum(-1).mean()


def main():
    """Build a small Llama 3, run training with GPipe and Std1F1B, and compare losses."""
    cfg = Llama3Config(
        vocab=1024,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        ffn=512,
        n_layers=4,
    )
    mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis="pp")
    print(f"mesh: {dict(mesh.shape)}  is_mpmd={mesh.is_mpmd}")

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        ids = jax.random.randint(jax.random.PRNGKey(1), (8, 16), 0, cfg.vocab)
        labels = jax.random.randint(jax.random.PRNGKey(2), (8, 16), 0, cfg.vocab)

        loss_gpipe, grads_gpipe = spx.run(
            model,
            inputs=ids,
            targets=labels,
            mesh=mesh,
            mode="train",
            loss_fn=cross_entropy,
            schedule=GPipe(microbatches=4),
        )
        jax.block_until_ready(loss_gpipe)

        loss_1f1b, _grads_1f1b = spx.run(
            model,
            inputs=ids,
            targets=labels,
            mesh=mesh,
            mode="train",
            loss_fn=cross_entropy,
            schedule=Std1F1B(microbatches=4),
        )
        jax.block_until_ready(loss_1f1b)

    print(f"\nGPipe   loss: {float(loss_gpipe):.4f}")
    print(f"Std1F1B loss: {float(loss_1f1b):.4f}")
    print("rank 0 grad shardings (FSDP+TP):")
    for _c, p, arr in list(grads_gpipe[0].items())[:6]:
        if "weight" in p:
            print(f"  {p:30s} {arr.shape!s:20s} {arr.sharding.spec}")


if __name__ == "__main__":
    main()

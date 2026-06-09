# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PP x FSDP x TP composition on a 3-D mesh — same Llama, no model changes.

This example activates all three parallelism strategies at once on
a 3-D mesh ``(pp, fsdp, tp)``:

* :func:`spectrax.run` auto-splits the model along the ``pp`` axis.
* Intra-stage tensor parallelism (TP) falls out of the model's
  logical sharding annotations resolved against ``FSDP_TP_RULES``.
* Intra-stage fully-sharded data parallelism (FSDP) is similarly
  activated by those same annotations when ``fsdp > 1``.

The important part: the model class itself does not change at all
between SPMD-only, PP-only, or PP x FSDP x TP runs — the mesh and
the logical-axis rules are what drive the difference.

Default mesh here is ``pp=2, fsdp=1, tp=2`` on 4 chips; bump
``fsdp`` on bigger meshes.

Run on 4+ chips::

    python -m examples.07_mpmd.05_pp_fsdp_tp
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
    """Train one step under PP x FSDP x TP and print per-parameter grad shardings."""
    if len(jax.devices()) < 4:
        print("This example needs >=4 chips; skipping.")
        return

    cfg = Llama3Config(
        vocab=1024,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        ffn=512,
        n_layers=2,
    )
    mesh = spx.create_mesh(axis_dims=(2, 1, 1, 1, 2, 1), mpmd_axis="pp")
    print(f"mesh: {dict(mesh.shape)}  is_mpmd={mesh.is_mpmd}")

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        ids = jax.random.randint(jax.random.PRNGKey(1), (8, 16), 0, cfg.vocab)
        labels = jax.random.randint(jax.random.PRNGKey(2), (8, 16), 0, cfg.vocab)
        loss, grads = spx.run(
            model,
            inputs=ids,
            targets=labels,
            mesh=mesh,
            mode="train",
            loss_fn=cross_entropy,
            microbatches=4,
        )
        jax.block_until_ready(loss)

    print(f"\nPP x FSDP x TP train loss: {float(loss):.4f}\n")
    print("rank 0 grad shardings (showing FSDP+TP layout):")
    for _c, p, arr in grads[0].items():
        if "weight" in p:
            print(f"  {p:30s} {arr.shape!s:20s} {arr.sharding.spec}")


if __name__ == "__main__":
    main()

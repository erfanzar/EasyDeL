# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3 prefill inference via ``spx.run(..., mode='forward')``.

This example reuses the exact same :class:`examples.models.llama.Llama3`
model class from the training example and runs it in inference
("prefill") mode. The only differences from training are that
``mode='forward'`` is requested and no ``loss_fn``/``targets`` are
passed — :func:`spectrax.run` then returns logits directly rather
than ``(loss, grads)``.

Pipeline parallelism is still enabled (``mpmd_axis='pp'``) to show
that MPMD prefill is a one-line switch from MPMD training.

Run::

    python -m examples.07_mpmd.02_inference_forward
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

import spectrax as spx
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def main():
    """Construct a small Llama 3, run one prefill forward pass, and print logit stats."""
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
        ids = jax.random.randint(jax.random.PRNGKey(1), (2, 64), 0, cfg.vocab)
        logits = spx.run(model, inputs=ids, mesh=mesh, mode="forward")
        jax.block_until_ready(logits)

    print(f"\nlogits shape   : {logits.shape}  (batch, seq, vocab)")
    print(f"logits sharding: {logits.sharding.spec}")
    print(f"logits stats   : mean={float(logits.mean()):+.4f}  std={float(logits.std()):.4f}")


if __name__ == "__main__":
    main()

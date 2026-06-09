# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Greedy autoregressive decode via ``spx.run(..., mode='forward')``.

Decode **is** forward. There is no separate ``generate`` mode in
spectrax — if you don't compute a backward, you're decoding. Each
call in this example re-runs the full prompt + already-generated
suffix through the pipeline, which is O(seq^2) but also the
simplest possible demonstration.

For O(seq) decode with a KV cache, the idea is to put the cache in
stage ``state`` and thread the returned state back in on the next
call; the underlying pipeline primitive supports that directly and
this will ship through :func:`spectrax.run` once the ``state=``
plumbing lands.

Run::

    python -m examples.07_mpmd.04_decode_loop
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.sharding import logical_axis_rules

from ..models.llama import FSDP_TP_RULES, Llama3, Llama3Config


def main():
    """Run a tiny greedy decode loop from a random prompt."""
    cfg = Llama3Config(
        vocab=512,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        ffn=256,
        n_layers=2,
    )
    mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis="pp")
    print(f"mesh: {dict(mesh.shape)}  is_mpmd={mesh.is_mpmd}\n")

    BATCH = 1
    PROMPT_LEN = 4
    GEN_STEPS = 6

    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = Llama3(cfg, rngs=spx.Rngs(0))
        ids = jax.random.randint(
            jax.random.PRNGKey(42),
            (BATCH, PROMPT_LEN),
            0,
            cfg.vocab,
        )
        print(f"prompt tokens : {ids[0].tolist()}")

        for step in range(GEN_STEPS):
            logits = spx.run(model, inputs=ids, mesh=mesh, mode="forward")
            jax.block_until_ready(logits)
            next_id = jnp.argmax(logits[:, -1:, :], axis=-1)
            ids = jnp.concatenate([ids, next_id], axis=1)
            print(f"step {step}: generated {int(next_id[0, 0]):4d}  context now {ids.shape[1]} tokens")

        print(f"\nfinal sequence: {ids[0].tolist()}")


if __name__ == "__main__":
    main()

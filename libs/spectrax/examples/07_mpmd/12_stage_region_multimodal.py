# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Marker-based multimodal MPMD with ``sxstage_region``.

Multimodal models often have two independent paths before the final
loss: for example, a vision tower and a text tower. If both paths use
plain ``sxstage_iter`` markers, the parent scheduler sees one long
sequence and tries to map it as a single pipeline. ``sxstage_region``
marks each tower as its own logical stage sequence:

* region ``vision``: V0 -> V1
* region ``text``  : T0 -> T1

The parent function stays a normal JAX function in eager/JIT mode; the
markers only matter when ``sxjit`` builds the true MPMD schedule.

Run::

    python -m examples.07_mpmd.12_stage_region_multimodal
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import spectrax as spx
from spectrax.runtime.mpmd.markers import stage_region_specs
from spectrax.runtime.types import MpMdMesh

vision_region = spx.sxstage_region("vision", schedule=spx.GPipe(microbatches=2))
text_region = spx.sxstage_region("text", schedule=spx.GPipe(microbatches=2))


def multimodal_score(image_features: jax.Array, token_features: jax.Array) -> jax.Array:
    """Toy vision+text score with independent region-local stage markers."""

    def vision_path(x: jax.Array) -> jax.Array:
        x = jnp.tanh(1.5 * x + 0.25)
        x = spx.sxstage_iter(x, stage=0)
        return jnp.sin(x)

    def text_path(x: jax.Array) -> jax.Array:
        x = jnp.cos(x - 0.5)
        x = spx.sxstage_iter(x, stage=0)
        return jnp.tanh(x)

    vision = vision_region(vision_path)(image_features)
    text = text_region(text_path)(token_features)
    return (vision * text).mean()


def make_two_rank_mesh() -> MpMdMesh | None:
    """Return a two-rank mesh when at least two devices are visible."""
    devices = jax.devices()[:2]
    if len(devices) < 2:
        return None
    return MpMdMesh(Mesh(np.asarray(devices, dtype=object).reshape(2), ("pp",)), "pp")


def main() -> None:
    """Compare eager execution with scheduled MPMD when devices are available."""
    image = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4)
    text = jnp.linspace(0.5, 2.0, 16, dtype=jnp.float32).reshape(4, 4)
    eager = multimodal_score(image, text)
    print(f"eager score: {float(eager):+.6f}")

    jaxpr = jax.make_jaxpr(multimodal_score)(image, text).jaxpr
    region_names = [spec.name for spec in stage_region_specs(jaxpr)]
    print("stage regions in jaxpr:", region_names)
    print("logical layout: V0 -> V1, then T0 -> T1")

    mesh = make_two_rank_mesh()
    if mesh is None:
        print("scheduled sxjit demo skipped: need at least two visible JAX devices")
        return

    @spx.sxjit(mesh=mesh, schedule=spx.GPipe(microbatches=2), batch_argnums=(0, 1))
    def scheduled(image_features: jax.Array, token_features: jax.Array) -> jax.Array:
        """Scheduled true-MPMD version of :func:`multimodal_score`."""
        return multimodal_score(image_features, token_features)

    out = scheduled(image, text)
    jax.block_until_ready(out)
    plan = scheduled._mpmd_state["schedule_plan"]
    print(f"scheduled score: {float(out):+.6f}")
    print("serial region plan:", plan["serial_region_plan"])
    print("logical stage -> physical rank:", tuple(plan["loc_for_logical"]))


if __name__ == "__main__":
    main()

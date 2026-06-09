# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Graph seam microbenchmarks: export/bind/clone/pop/update/tree_state."""

from __future__ import annotations

from collections.abc import Callable

from flax import nnx

import spectrax as spx

from .. import models


def _cases_for(model_name: str, model_factory) -> dict[str, tuple[Callable, Callable, Callable]]:
    """Return ``{case_name: (setup_fn, spx_fn, nnx_fn)}`` for a given model."""

    def setup_spx():
        """Build the spectrax model for this case."""
        mdl, _ = getattr(models, f"spx_{model_factory}")()
        return mdl

    def setup_nnx():
        """Build the nnx model for this case."""
        mdl, _ = getattr(models, f"nnx_{model_factory}")()
        return mdl

    spx_mdl = setup_spx()
    nnx_mdl = setup_nnx()

    spx_gdef, spx_state = spx.export(spx_mdl)
    nnx_gdef, nnx_state = nnx.split(nnx_mdl)

    out: dict[str, tuple[Callable, Callable]] = {}

    out[f"{model_name}/export"] = (
        lambda: spx.export(spx_mdl),
        lambda: nnx.split(nnx_mdl),
    )
    out[f"{model_name}/bind"] = (
        lambda: spx.bind(spx_gdef, spx_state),
        lambda: nnx.merge(nnx_gdef, nnx_state),
    )
    out[f"{model_name}/update"] = (
        lambda: spx.update(spx_mdl, spx_state),
        lambda: nnx.update(nnx_mdl, nnx_state),
    )
    out[f"{model_name}/clone"] = (
        lambda: spx.clone(spx_mdl),
        lambda: nnx.clone(nnx_mdl),
    )
    out[f"{model_name}/tree_state"] = (
        lambda: spx.tree_state(spx_mdl),
        lambda: nnx.state(nnx_mdl),
    )
    return out


def build():
    """Build the full graph-seam case set."""
    cases: dict[str, tuple[Callable, Callable]] = {}
    cases.update(_cases_for("mlp12x1024", "mlp"))
    cases.update(_cases_for("xfmr_d512", "transformer"))
    return cases

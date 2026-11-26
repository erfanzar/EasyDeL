# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The README must name the backends that exist, and only those.

It described Eyvan as a control plane "across GCP Compute Engine and Cloud TPU
fleets" long after Kubernetes was proven on real GKE and an on-premises provider
and a plugin ABI had shipped. Nobody reading it would have known they could run
it anywhere else, which is a worse failure than a missing feature: the feature
was there and the document turned people away from it.
"""

from __future__ import annotations

import pathlib

from eyvan.capacity.drivers.registry import DEFAULT_CAPACITY_BACKENDS

_README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_every_registered_backend_is_named_in_the_readme() -> None:
    """A backend nobody documents is a backend nobody adopts."""

    text = _README.read_text(encoding="utf-8")
    missing = [name for name in DEFAULT_CAPACITY_BACKENDS.names() if name not in text]

    assert not missing, f"the README names no backend for: {', '.join(sorted(missing))}"


def test_the_readme_does_not_claim_eyvan_is_gcp_only() -> None:
    """The claim that made a multi-substrate control plane look single-cloud."""

    text = _README.read_text(encoding="utf-8").lower()

    for claim in ("gcp only", "gcp-only", "only on gcp", "requires gcp"):
        assert claim not in text, f"the README still claims {claim!r}"


def test_the_readme_separates_running_somewhere_from_proving_it_there() -> None:
    """Breadth and evidence are different axes and must not be blurred.

    Kubernetes runs on any conformant cluster; the evidence is one GKE run. An
    on-premises host is code with no real SSH behind it yet. A reader deciding
    where to deploy needs both facts, not the flattering one.
    """

    text = _README.read_text(encoding="utf-8")

    assert "qualification" in text.lower()
    assert "no run against a real SSH host" in text

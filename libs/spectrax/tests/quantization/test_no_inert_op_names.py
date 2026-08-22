# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guard: every op name a rule can claim must actually be consulted somewhere.

A rule naming an op that nothing looks up is worse than a rejected one.
It is accepted, stamped onto matching modules, counted in the "N modules
matched" log line, and changes nothing — the user reads a confirmation
that their model is quantized while it trains at full precision for that
op.

The vocabulary is split by who implements the layer. spectrax ships the
dense layers, so it must consult ``dot_general`` and ``einsum`` itself.
It ships no grouped-matmul layer, so ``ragged_dot`` is declared here for
consumers to consult — easydel's mixture-of-experts path does, and its
own test suite asserts that. Pinning the split means adding a name
forces a deliberate choice about which side owns it, instead of the name
quietly doing nothing.
"""

from __future__ import annotations

import pathlib

import spectrax
from spectrax.quantization import DEFAULT_OP_NAMES

_SPECTRAX_CONSULTED = {"dot_general", "einsum"}
"""Ops spectrax's own layers must look up."""

_CONSUMER_CONSULTED = {"ragged_dot"}
"""Ops declared for consumers, because spectrax ships no layer performing them."""


def _consulted_in_spectrax() -> set[str]:
    """Scan the shipped spectrax source for op names passed to the rule lookup.

    Returns:
        The set of op-name literals appearing in a ``rule_for(...)`` call.
    """
    root = pathlib.Path(spectrax.__file__).parent
    found: set[str] = set()
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for op in DEFAULT_OP_NAMES:
            if f'rule_for(module, "{op}")' in text or f'rule_for(self, "{op}")' in text:
                found.add(op)
    return found


def test_the_vocabulary_is_exactly_the_two_documented_groups():
    """Adding an op name must be a deliberate choice about which side consults it."""
    assert set(DEFAULT_OP_NAMES) == _SPECTRAX_CONSULTED | _CONSUMER_CONSULTED, (
        "DEFAULT_OP_NAMES changed. Decide whether the new name is consulted by a spectrax layer or "
        "by a consumer, add it to the matching set here, and make sure something actually looks it up."
    )


def test_spectrax_consults_every_op_it_owns():
    """The names spectrax claims for its own layers must really be looked up."""
    inert = _SPECTRAX_CONSULTED - _consulted_in_spectrax()
    assert not inert, (
        f"{sorted(inert)} can be named in a rule but no spectrax layer consults it, so such a rule "
        f"would be silently inert. Wire a layer to call rule_for(..., <op>) or move the name to the "
        f"consumer-consulted set."
    )

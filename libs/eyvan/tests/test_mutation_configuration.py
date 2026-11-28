# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mutation comparisons must not silently change the property-test corpus."""

import tomllib
from pathlib import Path


def test_optional_scheduler_mutation_campaign_pins_its_hypothesis_seed() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text())
    arguments = configuration["tool"]["mutmut"]["pytest_add_cli_args"]

    assert [argument for argument in arguments if argument.startswith("--hypothesis-seed=")] == [
        "--hypothesis-seed=20260910"
    ]

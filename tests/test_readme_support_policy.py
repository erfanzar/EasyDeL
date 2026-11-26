# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The installation entry point must match the supported interpreter range."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_python_support_matches_package_metadata():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    readme = (ROOT / "README.md").read_text()
    match = re.search(r"Eyvan supports Python `([^`]+)`", readme)
    assert match is not None
    assert match.group(1) == metadata["project"]["requires-python"]
    assert "py3.10" not in readme

# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyDeL library code reaches ejkernel only through its public modules (no ``_``-prefixed submodules)."""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.cpu_ok

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "easydel"


def _private_ejkernel_imports(path: pathlib.Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if parts[0] == "ejkernel" and any(part.startswith("_") for part in parts[1:]):
                found.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno} {module}")
    return found


def test_library_imports_no_private_ejkernel_modules():
    offenders = [hit for path in sorted(PACKAGE.rglob("*.py")) for hit in _private_ejkernel_imports(path)]
    assert offenders == [], "private ejkernel imports:\n" + "\n".join(offenders)

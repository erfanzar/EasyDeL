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

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_ROOT = REPO_ROOT / "easydel" / "inference"


def _spx_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name for alias in node.names if alias.name == "SpecTrax" or alias.name.startswith("SpecTrax.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "spectrax" or module.startswith("spectrax."):
                imports.append(module)
    return imports


def test_inference_graph_rebinding_uses_spectrax() -> None:
    runtime_files = [
        INFERENCE_ROOT / "esurge" / "mixins" / "lifecycle.py",
        INFERENCE_ROOT / "esurge" / "runners" / "model_runner.py",
        INFERENCE_ROOT / "esurge" / "runners" / "executors" / "model_executor.py",
        INFERENCE_ROOT / "vwhisper" / "core.py",
        INFERENCE_ROOT / "vwhisper" / "generation.py",
    ]

    for path in runtime_files:
        source = path.read_text()
        assert "import spectrax as spx" in source
        assert "spx.bind(" in source or "spx.export(" in source
        assert "nn.merge(" not in source
        assert "nn.split(" not in source
        assert "SpecTrax.SpecTrax" not in source

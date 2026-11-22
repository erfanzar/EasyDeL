# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Check optional TileLang ring signatures without requiring a GPU installation."""

import ast
from pathlib import Path

import pytest


@pytest.mark.cpu_ok
def test_tilelang_registered_ring_signature_matches_xla_contract():
    """Direct registry callers share parameter names/defaults and broadcast axes."""
    root = Path(__file__).resolve().parents[1] / "ejkernel" / "kernels"

    def signature(backend):
        tree = ast.parse((root / backend / "ring_attention" / "_interface.py").read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ring_attention"
        )
        return (
            [arg.arg for arg in function.args.args],
            [ast.unparse(node) for node in function.args.defaults],
            {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args},
        )

    names, defaults, annotations = signature("_xla")
    tile_names, tile_defaults, tile_annotations = signature("_tilelang")
    assert tile_names == names
    assert tile_defaults == defaults
    for name in ("bias", "softmax_aux", "precision"):
        assert tile_annotations[name] == annotations[name]

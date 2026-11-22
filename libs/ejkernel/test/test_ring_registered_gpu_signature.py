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

"""Inspect optional GPU public contracts without importing GPU-only dependencies."""

import ast
from pathlib import Path

import pytest


@pytest.mark.cpu_ok
def test_triton_registered_ring_signature_matches_xla_precision_and_broadcast_axes():
    """Public parameter names, defaults, bias/sink shapes must agree across backends."""
    root = Path(__file__).resolve().parents[1] / "ejkernel" / "kernels"

    def signature(backend):
        tree = ast.parse((root / backend / "ring_attention" / "_interface.py").read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ring_attention"
        )
        names = [arg.arg for arg in function.args.args]
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
        defaults = [ast.unparse(node) for node in function.args.defaults]
        return names, annotations, defaults

    xla_names, xla_annotations, xla_defaults = signature("_xla")
    gpu_names, gpu_annotations, gpu_defaults = signature("_triton")
    assert gpu_names == xla_names
    assert gpu_defaults == xla_defaults
    for name in ("bias", "softmax_aux", "precision"):
        assert gpu_annotations[name] == xla_annotations[name]

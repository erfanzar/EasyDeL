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

"""TileLang registry and standalone test coverage checks."""

from __future__ import annotations

import ast
from pathlib import Path

import ejkernel.kernels._tilelang  # noqa: F401
from ejkernel.kernels._registry import Platform, kernel_registry

_TEST_ROOT = Path(__file__).resolve().parent
_META_TESTS = frozenset({"test_honesty.py", "test_registry_coverage.py"})


def _algorithms_for(platform):
    algorithms = set()
    for algorithm in kernel_registry.list_algorithms():
        if any(spec.platform == platform for spec in kernel_registry.list_implementations(algorithm)):
            algorithms.add(algorithm)
    return algorithms


def _module_string_constants(tree):
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _algorithm_from_arg(node, constants):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _algorithms_referenced_by_test(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = _module_string_constants(tree)
    algorithms = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"_tl", "_xla"} or not node.args:
            continue
        algorithm = _algorithm_from_arg(node.args[0], constants)
        if algorithm is not None:
            algorithms.add(algorithm)
    return algorithms


def _standalone_test_coverage():
    coverage = {}
    untagged_files = []
    for path in sorted(_TEST_ROOT.glob("test_*.py")):
        if path.name in _META_TESTS:
            continue
        algorithms = _algorithms_referenced_by_test(path)
        if not algorithms:
            untagged_files.append(path.name)
            continue
        for algorithm in algorithms:
            coverage.setdefault(algorithm, set()).add(path.name)
    return coverage, untagged_files


def test_every_xla_algorithm_has_tilelang_registration():
    missing = _algorithms_for(Platform.XLA) - _algorithms_for(Platform.TILELANG)
    assert not missing, f"tile-lang missing registration for: {sorted(missing)}"


def test_tilelang_registered_algorithms_have_standalone_tests():
    registered = _algorithms_for(Platform.TILELANG)
    coverage, untagged_files = _standalone_test_coverage()
    covered = set(coverage)
    missing = registered - covered
    unknown = covered - registered
    assert not untagged_files, 'TileLang standalone tests must call `_tl("algorithm")`:\n' + "\n".join(untagged_files)
    assert not missing, "TileLang registered algorithms without standalone tests:\n" + "\n".join(sorted(missing))
    assert not unknown, "TileLang tests reference unregistered algorithms:\n" + "\n".join(sorted(unknown))

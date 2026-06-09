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

"""Static checks for operation-level autotune candidate ownership."""

from __future__ import annotations

import ast
from pathlib import Path

OPERATIONS_DIR = Path(__file__).resolve().parents[3] / "ejkernel" / "modules" / "operations"


def _operation_modules() -> list[Path]:
    """Return operation implementation modules that contain Kernel subclasses."""
    return [
        path
        for path in sorted(OPERATIONS_DIR.glob("*.py"))
        if path.name not in {"__init__.py", "configs.py"} and "Kernel[" in path.read_text()
    ]


def _is_kernel_subclass(node: ast.ClassDef) -> bool:
    """Return whether a class directly subclasses ``Kernel[...]`` or ``Kernel``."""
    for base in node.bases:
        if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) and base.value.id == "Kernel":
            return True
        if isinstance(base, ast.Name) and base.id == "Kernel":
            return True
    return False


def test_every_operation_kernel_declares_gpu_and_tpu_candidates():
    """Every operation-level Kernel must own both GPU and TPU candidate methods."""
    missing: list[str] = []
    for path in _operation_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not _is_kernel_subclass(node):
                continue
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            for method in ("candidate_cfgs_gpu", "candidate_cfgs_tpu"):
                if method not in methods:
                    missing.append(f"{path.relative_to(OPERATIONS_DIR)}::{node.name}.{method}")
    assert not missing


def test_platform_specific_candidates_do_not_emit_auto_configs():
    """GPU/TPU candidate methods must benchmark concrete platforms, not ``auto``."""
    offenders: list[str] = []
    for path in _operation_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in {"candidate_cfgs_gpu", "candidate_cfgs_tpu"}:
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                for kw in call.keywords:
                    if kw.arg == "platform" and isinstance(kw.value, ast.Constant) and kw.value.value == "auto":
                        offenders.append(f"{path.relative_to(OPERATIONS_DIR)}:{kw.value.lineno}:{node.name}")
    assert not offenders


def test_operation_modules_do_not_use_del_shims():
    """Operation wrappers should honor arguments or route explicitly instead of deleting them."""
    offenders: list[str] = []
    for path in _operation_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Delete):
                offenders.append(f"{path.relative_to(OPERATIONS_DIR)}:{node.lineno}")
    assert not offenders

# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-light helpers shared by tool/reasoning parsers and serving utilities."""

from __future__ import annotations

import importlib.util
import os
import sys
import typing as tp
from types import ModuleType


def import_from_path(module_name: str, file_path: str | os.PathLike) -> ModuleType:
    """Load a plugin file, registering it before executing its module body.

    Args:
        module_name: Name under which the plugin is registered in ``sys.modules``.
        file_path: Absolute or relative path to the Python plugin file.

    Returns:
        The executed module object, also available in ``sys.modules``.

    Raises:
        ModuleNotFoundError: If importlib cannot create a spec for the path.
        FileNotFoundError: If the source file does not exist.
        AssertionError: If the module spec has no loader.
        Exception: Exceptions from plugin execution propagate unchanged.

    Notes:
        Matches the existing parser loaders: registration precedes execution,
        enabling self-imports and decorators, and remains in place if execution
        raises. A pre-existing module with the same name is replaced.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def is_list_of(
    value: object,
    typ: tp.Any,
    *,
    check: tp.Literal["first", "all"] = "first",
    raise_on_invalid_check: bool = True,
) -> bool:
    """Check list elements with the parser and serving compatibility policies.

    Args:
        value: Value to inspect; non-lists always return ``False``.
        typ: Type or tuple of types accepted by ``isinstance``.
        check: ``"first"`` checks only the first element; ``"all"`` checks all
            elements. Both modes accept empty lists.
        raise_on_invalid_check: Raise for an unknown mode (the parser-helper
            behavior). Set to ``False`` to return ``False`` instead, as the
            eSurge compatibility wrapper does.

    Returns:
        Whether the requested list elements match ``typ``. The first-element
        heuristic does not guarantee that every element has the expected type.

    Raises:
        AssertionError: If a list is passed with an unknown mode and
            ``raise_on_invalid_check`` is true.
        TypeError: If an inspected element is checked against an invalid type.
    """
    if not isinstance(value, list):
        return False
    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    if check == "all":
        return all(isinstance(v, typ) for v in value)
    if raise_on_invalid_check:
        tp.assert_never(check)
    return False

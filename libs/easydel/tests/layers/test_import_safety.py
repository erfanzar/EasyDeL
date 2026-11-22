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

"""Array-free regressions for cache/indexer import ordering."""

import builtins
import os
import runpy
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_ok


def test_indexer_primitives_load_without_importing_operations(monkeypatch):
    from easydel.layers.indexer import _primitives

    original_import = builtins.__import__

    def import_without_operations(name, *args, **kwargs):
        if name == "easydel.operations" or name.startswith("easydel.operations."):
            raise ImportError("operations cannot load while caching is partially initialized")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_operations)
    exports = runpy.run_path(_primitives.__file__)
    assert callable(exports["topk_selection_mask"])
    assert "topk_selection_mask" in exports["__all__"]


@pytest.mark.parametrize("first", ["easydel.caching", "easydel.layers.indexer", "easydel.operations"])
def test_cold_imports_expose_cache_and_selection_apis(first):
    script = f"""
import importlib
importlib.import_module({first!r})
from easydel.caching import UnifiedAttentionCacheView
from easydel.layers.indexer import topk_selection_mask
assert UnifiedAttentionCacheView.__name__ == 'UnifiedAttentionCacheView'
assert callable(topk_selection_mask)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "ENABLE_DISTRIBUTED_INIT": "0", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

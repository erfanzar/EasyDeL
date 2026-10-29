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

from __future__ import annotations

import sys

# When executed as a script (python test/test_*.py), Python puts the test
# directory at sys.path[0], which can shadow stdlib modules (e.g. test/types).
# Drop that entry and prepend repo root to keep stdlib resolution correct.
if sys.path:
    p0 = sys.path[0].replace("\\", "/")
    if p0.endswith("/test"):
        sys.path.pop(0)

file_path = __file__.replace("\\", "/")
if "/test/" in file_path:
    repo_root = file_path.rsplit("/test/", 1)[0]
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import importlib.util  # noqa
import os  # noqa
import subprocess  # noqa
import textwrap  # noqa

import pytest  # noqa


def test_importing_ejkernel_tree_does_not_trigger_jax_backend_init():
    script = textwrap.dedent("""
        import importlib
        import jax._src.xla_bridge as xb

        def _assert_not_initialized(where: str):
            if xb.backends_are_initialized():
                raise RuntimeError(f"jax backend initialized during import: {where}")

        _assert_not_initialized("startup")

        import ejkernel
        _assert_not_initialized("import ejkernel")

        module_names = [
            "ejkernel.errors",
            "ejkernel.kernels",
            "ejkernel.modules",
            "ejkernel.quantization",
            "ejkernel.ops",
            "ejkernel.types",
            "ejkernel.utils",
            "ejkernel.xla_utils",
            "ejkernel.benchmarks",
            "ejkernel.callib",
            "ejkernel.loggings",
        ]
        for name in module_names:
            importlib.import_module(name)
            _assert_not_initialized(name)

        from ejkernel import benchmarks, callib, errors, kernels, loggings, modules  # noqa: F401
        _assert_not_initialized("from ejkernel import benchmarks/callib/errors/kernels/loggings/modules")
        """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    assert proc.returncode == 0, f"stdout:\\n{proc.stdout}\\nstderr:\\n{proc.stderr}"


if __name__ == "__main__":
    pytest.main([__file__])

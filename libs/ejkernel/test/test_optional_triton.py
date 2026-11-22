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

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest


def test_import_works_without_triton_installed():
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("Triton is installed; this test only validates the optional-dependency path.")

    import ejkernel

    assert ejkernel.kernels.triton is None
    assert ejkernel.utils.triton is None


def test_triton_call_errors_cleanly_when_triton_missing():
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("Triton is installed; this test only validates the optional-dependency path.")

    import ejkernel

    with pytest.raises(ValueError, match=r"triton.*installed"):
        ejkernel.callib.triton_call((), kernel=None, out_shape=(), grid=1)


def test_import_works_when_triton_imports_are_blocked():
    script = textwrap.dedent("""
        import builtins

        _orig_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError("No module named 'triton'", name="triton")
            return _orig_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _fake_import

        import ejkernel
        assert ejkernel.kernels.triton is None

        from ejkernel.callib import ejit
        assert callable(ejit)
        """)
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, f"stdout:\\n{proc.stdout}\\nstderr:\\n{proc.stderr}"


def test_import_works_when_cute_cuda_bindings_are_missing():
    script = textwrap.dedent("""
        import builtins
        import importlib.util

        _orig_import = builtins.__import__
        _orig_find_spec = importlib.util.find_spec
        _fake_cutlass_spec = object()

        def _fake_find_spec(name, *args, **kwargs):
            if name == "cutlass":
                return _fake_cutlass_spec
            return _orig_find_spec(name, *args, **kwargs)

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cuda" or name.startswith("cuda."):
                raise ModuleNotFoundError("No module named 'cuda'", name="cuda")
            return _orig_import(name, globals, locals, fromlist, level)

        importlib.util.find_spec = _fake_find_spec
        builtins.__import__ = _fake_import

        import ejkernel
        assert ejkernel.kernels.cute is None
        """)
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, f"stdout:\\n{proc.stdout}\\nstderr:\\n{proc.stderr}"

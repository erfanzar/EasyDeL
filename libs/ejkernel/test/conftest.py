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

import importlib.util
import os
import shutil
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("EJKERNEL_AUTOTUNE_POLICY", "heuristics")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_gpu_enable_command_buffer=".strip()

_TRITON_ONLY_TEST_BASENAMES = {
    # Cross-backend comparisons that require Triton.
    "test_flash_attention_xla_triton.py",
    "test_native_sparse_attention_xla_triton.py",
    "test_ragged_page_attention_v3_reference.py",
}


def _has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


@cache
def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@lru_cache(maxsize=1)
def _has_gpu_backend() -> bool:
    if not _has_module("jax"):
        return False
    try:
        import jax
    except Exception:
        return False
    try:
        devices = jax.devices()
    except Exception:
        return False
    return bool(devices) and devices[0].platform == "gpu"


@lru_cache(maxsize=1)
def _has_nvcc() -> bool:
    return shutil.which("nvcc") is not None


def _triton_available() -> bool:
    return _has_triton() and _has_gpu_backend()


def _cute_available() -> bool:
    return _has_module("cutlass") and _has_gpu_backend()


def _cuda_available() -> bool:
    return _has_gpu_backend() and _has_nvcc()


def _tilelang_available() -> bool:
    return _has_module("tilelang") and _has_module("jax_tvm_ffi") and _has_gpu_backend()


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:
    parts = set(collection_path.parts)
    is_kernel_test = "test" in parts and "kernels" in parts
    if is_kernel_test and "_triton" in parts and not _triton_available():
        return True
    if is_kernel_test and "_cute" in parts and not _cute_available():
        return True
    if is_kernel_test and "_cuda" in parts and not _cuda_available():
        return True
    if is_kernel_test and "_tilelang" in parts and not _tilelang_available():
        return True

    # Ignore standalone Triton-dependent tests outside test/kernels/_triton.
    return collection_path.name in _TRITON_ONLY_TEST_BASENAMES and not _triton_available()

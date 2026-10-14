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

"""Build utilities for the ragged paged attention v3 CUDA kernel.

This module provides functions to compile the CUDA source code for the
ragged paged attention v3 kernel into a shared library (``.so``) using
CMake. It handles GPU architecture detection, incremental rebuilds based
on source file modification timestamps, and multi-architecture builds.

The build process can be configured through environment variables:

    ``EJKERNEL_CUDA_ARCH``
        Override the detected GPU compute capability (e.g., ``"90"`` for
        SM 9.0). When unset, the architecture is auto-detected via
        ``nvidia-smi``.

    ``EJKERNEL_CUDA_ARCHS``
        Semicolon- or comma-separated list of architectures to build for
        (e.g., ``"80;90"``). Enables multi-architecture builds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jax.ffi as ffi


def _normalize_arch(value: str) -> str:
    """Normalize a CUDA architecture string to a compact numeric form.

    Strips whitespace, converts to lowercase, removes ``sm_`` and
    ``compute_`` prefixes, and collapses dotted versions (e.g., ``"8.0"``)
    into their concatenated form (``"80"``).

    Args:
        value: Raw architecture string, such as ``"sm_80"``,
            ``"compute_8.0"``, or ``"90"``.

    Returns:
        Normalized architecture string containing only digits
        (e.g., ``"80"``, ``"90"``).
    """
    val = value.strip().lower().replace("sm_", "").replace("compute_", "")
    if "." in val:
        major, minor = val.split(".", 1)
        return f"{major}{minor[:1]}"
    return val


def _parse_arch_list(value: str) -> list[str]:
    """Parse a delimited string of CUDA architectures into a list.

    Accepts semicolon- or comma-separated architecture values and
    normalizes each entry using :func:`_normalize_arch`.

    Args:
        value: Delimited architecture string, e.g., ``"80;90"`` or
            ``"sm_80, compute_90"``.

    Returns:
        List of normalized architecture strings (e.g., ``["80", "90"]``).
    """
    parts = [p for p in value.replace(",", ";").split(";") if p.strip()]
    return [_normalize_arch(p) for p in parts]


def _detect_cuda_arch() -> str:
    """Detect the CUDA compute capability of the current GPU.

    Resolution order:
        1. The ``EJKERNEL_CUDA_ARCH`` environment variable, if set.
        2. The compute capability reported by ``nvidia-smi``.
        3. Falls back to ``"80"`` (Ampere / SM 8.0) if detection fails.

    Returns:
        Normalized architecture string (e.g., ``"80"``, ``"90"``).
    """
    override = os.getenv("EJKERNEL_CUDA_ARCH")
    if override:
        return _normalize_arch(override)
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = output.strip().splitlines()[0].strip()
        if line:
            return _normalize_arch(line)
    except Exception:
        pass
    return "80"


def _latest_mtime(paths: list[Path]) -> float:
    """Return the most recent modification time across directory trees.

    Recursively walks each directory in *paths* and returns the latest
    ``st_mtime`` found among all regular files. Used to determine
    whether a rebuild of the CUDA library is necessary.

    Args:
        paths: List of directory root paths to scan.

    Returns:
        The latest modification time as a float (seconds since epoch),
        or ``0.0`` if no files are found or no paths exist.
    """
    latest = 0.0
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                latest = max(latest, p.stat().st_mtime)
    return latest


def build_cuda_lib() -> Path:
    """Build the ragged paged attention v3 CUDA shared library.

    Compiles the C/CUDA sources located in ``csrc/ragged_page_attention_v3``
    into a shared library using CMake. The library is placed in the
    ``_build`` subdirectory of this package and named according to the
    target GPU architecture (e.g.,
    ``libejkernel_ragged_page_attention_v3_cuda_sm80.so``).

    Incremental builds are supported: the library is only recompiled
    when the source files are newer than the existing artifact.

    When the ``EJKERNEL_CUDA_ARCHS`` environment variable is set, the
    function builds for all specified architectures in a single CMake
    invocation.

    Returns:
        Path to the compiled shared library file for the detected (or
        overridden) GPU architecture.

    Raises:
        RuntimeError: If CMake is not installed, the build fails, or the
            expected shared library is missing after a successful build.
    """
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent.parent.parent
    csrc_root = repo_root / "csrc" / "ragged_page_attention_v3"
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _detect_cuda_arch()
    lib_path = build_dir / f"libejkernel_ragged_page_attention_v3_cuda_sm{arch}.so"
    archs_env = os.getenv("EJKERNEL_CUDA_ARCHS")
    archs = _parse_arch_list(archs_env) if archs_env else []
    if archs and arch not in archs:
        archs.insert(0, arch)

    vendor_mtime = _latest_mtime([csrc_root])
    if archs:
        libs = [build_dir / f"libejkernel_ragged_page_attention_v3_cuda_sm{a}.so" for a in archs]
        if all(p.exists() and p.stat().st_mtime >= vendor_mtime for p in libs):
            return lib_path
    else:
        if lib_path.exists() and lib_path.stat().st_mtime >= vendor_mtime:
            return lib_path

    include_dir = Path(ffi.include_dir())
    cuda_compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

    cmake_cmd = [
        "cmake",
        "-S",
        str(csrc_root),
        "-B",
        str(build_dir),
        f"-DEJKERNEL_CUDA_ARCHS={';'.join(archs)}" if archs else f"-DEJKERNEL_CUDA_ARCH={arch}",
        f"-DEJKERNEL_JAX_FFI_INCLUDE={include_dir}",
    ]
    if Path(cuda_compiler).exists():
        cmake_cmd.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
    build_jobs = os.getenv("EJKERNEL_CUDA_BUILD_JOBS") or str(max(1, min(os.cpu_count() or 1, 16)))
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel", build_jobs]
    if not archs:
        build_cmd.extend(["--target", f"ejkernel_ragged_page_attention_v3_cuda_sm{arch}"])

    try:
        subprocess.run(cmake_cmd, check=True, capture_output=True, text=True)
        subprocess.run(build_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("cmake not found; please install CMake and CUDA toolkit.") from exc
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or "CUDA build failed."
        raise RuntimeError(msg) from exc

    if not lib_path.exists():
        raise RuntimeError(f"CUDA build succeeded but library not found at {lib_path}")
    return lib_path

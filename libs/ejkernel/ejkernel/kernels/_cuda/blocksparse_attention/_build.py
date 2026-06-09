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

"""Build utilities for the CUDA block-sparse attention shared library.

This module handles the detection of the GPU compute capability, CMake-based
compilation of the CUDA C++ source tree located under ``csrc/blocksparse_attention``,
and caching of the resulting shared library (``.so``) so that rebuilds only
occur when the source files have been modified.

Environment Variables:
    EJKERNEL_CUDA_ARCH: Override the detected GPU compute capability with a
        specific SM architecture (e.g., ``"90"`` or ``"sm_90"``).
    EJKERNEL_CUDA_ARCHS: Semicolon- or comma-separated list of SM architectures
        to build for (e.g., ``"80;90"``). When set, the library is built for
        all listed architectures in addition to the detected one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jax.ffi as ffi


def _normalize_arch(value: str) -> str:
    """Normalize a CUDA architecture string to a compact numeric form.

    Strips whitespace, lowercases the input, removes ``sm_`` and ``compute_``
    prefixes, and converts dotted versions (e.g., ``"8.0"``) to concatenated
    digits (e.g., ``"80"``).

    Args:
        value: Raw architecture string such as ``"sm_80"``, ``"8.0"``, or
            ``"compute_90"``.

    Returns:
        A normalized architecture string containing only digits, e.g. ``"80"``.
    """
    val = value.strip().lower().replace("sm_", "").replace("compute_", "")
    if "." in val:
        major, minor = val.split(".", 1)
        return f"{major}{minor[:1]}"
    return val


def _parse_arch_list(value: str) -> list[str]:
    """Parse a delimited string of CUDA architectures into a list of normalized values.

    Accepts semicolon- or comma-separated architecture specifications and
    normalizes each entry using :func:`_normalize_arch`.

    Args:
        value: Delimited architecture list, e.g. ``"80;90"`` or ``"sm_80,sm_90"``.

    Returns:
        A list of normalized architecture strings, e.g. ``["80", "90"]``.
    """
    parts = [p for p in value.replace(",", ";").split(";") if p.strip()]
    return [_normalize_arch(p) for p in parts]


def _detect_cuda_arch() -> str:
    """Detect the CUDA compute capability of the current GPU.

    Resolution order:
        1. The ``EJKERNEL_CUDA_ARCH`` environment variable, if set.
        2. The compute capability reported by ``nvidia-smi``.
        3. Falls back to ``"80"`` (Ampere) when detection fails.

    Returns:
        A normalized architecture string (e.g. ``"80"``).
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
    """Return the most recent modification time across all files under the given paths.

    Recursively walks each directory in *paths* and tracks the latest
    ``st_mtime`` value. Non-existent paths are silently skipped.

    Args:
        paths: Directory paths to scan recursively.

    Returns:
        The latest modification timestamp as a float (seconds since epoch),
        or ``0.0`` if no files are found.
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
    """Build (or locate a cached) CUDA shared library for block-sparse attention.

    The function performs the following steps:

    1. Determines the target SM architecture via :func:`_detect_cuda_arch`.
    2. Checks whether the compiled ``.so`` already exists and is up-to-date
       relative to the C++/CUDA source files under ``csrc/blocksparse_attention``.
    3. If a rebuild is needed, invokes CMake to configure and compile the
       library, placing build artifacts in a ``_build`` subdirectory next to
       this file.

    The resulting shared library is named
    ``libejkernel_blocksparse_attention_cuda_sm{arch}.so``.

    Returns:
        The :class:`~pathlib.Path` to the compiled shared library.

    Raises:
        RuntimeError: If CMake is not found, the build process fails, or the
            library is not found at the expected path after a successful build.
    """
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent.parent.parent
    csrc_root = repo_root / "csrc" / "blocksparse_attention"
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _detect_cuda_arch()
    lib_path = build_dir / f"libejkernel_blocksparse_attention_cuda_sm{arch}.so"
    archs_env = os.getenv("EJKERNEL_CUDA_ARCHS")
    archs = _parse_arch_list(archs_env) if archs_env else []
    if archs and arch not in archs:
        archs.insert(0, arch)

    vendor_mtime = _latest_mtime([csrc_root])
    if archs:
        libs = [build_dir / f"libejkernel_blocksparse_attention_cuda_sm{a}.so" for a in archs]
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
        build_cmd.extend(["--target", f"ejkernel_blocksparse_attention_cuda_sm{arch}"])

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

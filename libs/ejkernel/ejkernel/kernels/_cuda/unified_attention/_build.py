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

"""CMake-based build system for the unified attention CUDA shared library.

Detects the local GPU architecture, invokes CMake to compile the CUDA
source tree located in ``csrc/unified_attention/``, and returns the path
to the resulting shared library. The build is incremental: if the library
already exists and its modification time is newer than any source file,
the build step is skipped.

Environment variables:
    EJKERNEL_CUDA_ARCH: Override the detected SM architecture
        (e.g. ``"80"`` or ``"sm_80"``). When set, ``nvidia-smi`` is not
        invoked.
    EJKERNEL_CUDA_ARCHS: Semicolon- or comma-separated list of
        architectures to compile for (e.g. ``"80;90"``). The detected
        architecture is always prepended if not already present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jax.ffi as ffi


def _normalize_arch(value: str) -> str:
    """Normalize a CUDA architecture string to a bare numeric form.

    Strips whitespace, converts to lowercase, removes ``sm_`` and
    ``compute_`` prefixes, and collapses dotted versions (e.g. ``"8.0"``)
    into their compact equivalents (e.g. ``"80"``).

    Args:
        value: Raw architecture string such as ``"sm_80"``, ``"8.0"``,
            or ``"compute_90"``.

    Returns:
        A compact numeric string (e.g. ``"80"`` or ``"90"``).
    """
    val = value.strip().lower().replace("sm_", "").replace("compute_", "")
    if "." in val:
        major, minor = val.split(".", 1)
        return f"{major}{minor[:1]}"
    return val


def _parse_arch_list(value: str) -> list[str]:
    """Parse a delimited string of CUDA architectures into a list.

    Accepts semicolon- or comma-separated architecture tokens and
    normalizes each one via :func:`_normalize_arch`.

    Args:
        value: Delimited architecture string, e.g. ``"80;90"`` or
            ``"sm_80, sm_90"``.

    Returns:
        List of normalized architecture strings (e.g. ``["80", "90"]``).
    """
    parts = [p for p in value.replace(",", ";").split(";") if p.strip()]
    return [_normalize_arch(p) for p in parts]


def _detect_cuda_arch() -> str:
    """Detect the CUDA SM architecture of the first visible GPU.

    Resolution order:
        1. The ``EJKERNEL_CUDA_ARCH`` environment variable, if set.
        2. The compute capability reported by ``nvidia-smi``.
        3. Fallback to ``"80"`` (Ampere / A100) if detection fails.

    Returns:
        Normalized SM architecture string (e.g. ``"80"``).
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
    """Return the most recent modification time across all files under *paths*.

    Recursively walks each directory in *paths* and returns the maximum
    ``st_mtime`` found. Non-existent paths are silently skipped.

    Args:
        paths: Root directories (or files) to scan.

    Returns:
        The latest modification time as a POSIX timestamp float, or
        ``0.0`` if no files were found.
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
    """Build (or locate) the unified attention CUDA shared library.

    Determines the target SM architecture, checks whether the shared
    library is already up-to-date with respect to the C/CUDA sources in
    ``csrc/unified_attention/``, and triggers a CMake configure + build
    when necessary.

    When the environment variable ``EJKERNEL_CUDA_ARCHS`` is set, multiple
    architecture-specific libraries are compiled in a single build. The
    returned path always corresponds to the *detected* (or overridden)
    architecture so that the caller can load it immediately.

    Returns:
        Absolute path to the compiled ``.so`` shared library.

    Raises:
        RuntimeError: If CMake is not found on ``PATH``, if the CMake
            configure or build step fails, or if the expected library
            file does not exist after a successful build.
    """
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent.parent.parent
    csrc_root = repo_root / "csrc" / "unified_attention"
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _detect_cuda_arch()
    lib_path = build_dir / f"libejkernel_unified_attention_cuda_sm{arch}.so"
    archs_env = os.getenv("EJKERNEL_CUDA_ARCHS")
    archs = _parse_arch_list(archs_env) if archs_env else []
    if archs and arch not in archs:
        archs.insert(0, arch)

    vendor_mtime = _latest_mtime([csrc_root])
    if archs:
        libs = [build_dir / f"libejkernel_unified_attention_cuda_sm{a}.so" for a in archs]
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
        build_cmd.extend(["--target", f"ejkernel_unified_attention_cuda_sm{arch}"])

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

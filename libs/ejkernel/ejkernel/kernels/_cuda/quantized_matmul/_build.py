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

"""Build utilities for the quantized matrix multiplication CUDA kernel.

This module handles the compilation and caching of the CUDA shared library
that implements the quantized matrix multiplication kernel. It detects the
target GPU architecture (via ``nvidia-smi`` or the ``EJKERNEL_CUDA_ARCH``
environment variable) and invokes CMake to build the kernel from C++/CUDA
sources located in ``csrc/quantized_matmul/``.

Build artifacts are placed in a ``_build/`` subdirectory alongside this
module and are reused across runs. Rebuilds are triggered automatically
when source files are newer than the cached library.

Environment Variables:
    EJKERNEL_CUDA_ARCH: Override the detected GPU compute capability
        (e.g., ``"89"`` for SM 8.9). When set, ``nvidia-smi`` is not queried.
    EJKERNEL_CUDA_ARCHS: Semicolon- or comma-separated list of additional
        architectures to compile for (e.g., ``"80;89;90"``). The detected
        architecture is always included.
    EJKERNEL_CUTLASS_INCLUDE: Override the CUTLASS C++ header include path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jax.ffi as ffi


def _normalize_arch(value: str) -> str:
    """Normalize a CUDA architecture string to a compact numeric form.

    Strips whitespace, converts to lowercase, removes ``sm_`` or
    ``compute_`` prefixes, and collapses dotted version strings
    (e.g., ``"8.9"``) into a two-digit code (``"89"``).

    Args:
        value: Raw architecture string, e.g. ``"sm_89"``, ``"8.9"``,
            or ``"compute_90"``.

    Returns:
        Normalized architecture code such as ``"89"`` or ``"90"``.
    """
    val = value.strip().lower().replace("sm_", "").replace("compute_", "")
    if "." in val:
        major, minor = val.split(".", 1)
        return f"{major}{minor[:1]}"
    return val


def _parse_arch_list(value: str) -> list[str]:
    """Parse a comma- or semicolon-separated list of CUDA architectures.

    Each entry is normalized via :func:`_normalize_arch`. Empty entries
    resulting from consecutive delimiters are silently ignored.

    Args:
        value: Delimited architecture string, e.g. ``"80;89;90"`` or
            ``"sm_80, sm_89"``.

    Returns:
        List of normalized architecture codes, e.g. ``["80", "89", "90"]``.
    """
    parts = [p for p in value.replace(",", ";").split(";") if p.strip()]
    return [_normalize_arch(p) for p in parts]


def _detect_cuda_arch() -> str:
    """Detect the CUDA compute capability of the current GPU.

    Resolution order:
        1. The ``EJKERNEL_CUDA_ARCH`` environment variable, if set.
        2. The first GPU reported by ``nvidia-smi --query-gpu=compute_cap``.
        3. Falls back to ``"80"`` (Ampere A100) when detection fails.

    Returns:
        Normalized architecture code (e.g. ``"89"``).
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

    Recursively walks each directory in *paths* and returns the maximum
    ``st_mtime`` found. Non-existent paths are silently skipped.

    Args:
        paths: Root directories (or files) to scan recursively.

    Returns:
        The latest modification timestamp as a POSIX float, or ``0.0``
        if no files are found.
    """
    latest = 0.0
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                latest = max(latest, p.stat().st_mtime)
    return latest


def _find_cutlass_include(repo_root: Path) -> Path:
    """Find a CUTLASS C++ header include directory."""
    candidates: list[Path] = []
    override = os.getenv("EJKERNEL_CUTLASS_INCLUDE")
    if override:
        candidates.append(Path(override))
    candidates.append(repo_root / "csrc" / "cutlass" / "include")

    for candidate in candidates:
        if (candidate / "cutlass" / "gemm" / "device" / "gemm.h").exists():
            return candidate

    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        "CUTLASS C++ headers were not found for the CUDA quantized_matmul backend. "
        "Run `git submodule update --init csrc/cutlass`, or set "
        "EJKERNEL_CUTLASS_INCLUDE to a CUDA CUTLASS include directory containing "
        "cutlass/gemm/device/gemm.h.\n"
        f"Checked:\n{formatted}"
    )


def _cmake_cache_value(cache_path: Path, key: str) -> str | None:
    """Read one string/filepath value from an existing CMake cache."""
    if not cache_path.exists():
        return None
    prefix = f"{key}:"
    try:
        for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(prefix):
                _, value = line.split("=", 1)
                return value
    except OSError:
        return None
    return None


def _invalidate_stale_cmake_cache(build_dir: Path, *, cuda_compiler: str, arch: str, archs: list[str]) -> None:
    """Remove stale CMake configure state when static build inputs change."""
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return
    expected_archs = ";".join(archs) if archs else ""
    expected_arch = "" if archs else arch
    cached_compiler = _cmake_cache_value(cache_path, "CMAKE_CUDA_COMPILER")
    cached_arch = _cmake_cache_value(cache_path, "EJKERNEL_CUDA_ARCH")
    cached_archs = _cmake_cache_value(cache_path, "EJKERNEL_CUDA_ARCHS")
    if (
        (cached_compiler is not None and Path(cached_compiler) != Path(cuda_compiler))
        or (cached_arch is not None and cached_arch != expected_arch)
        or (cached_archs is not None and cached_archs != expected_archs)
    ):
        shutil.rmtree(build_dir / "CMakeFiles", ignore_errors=True)
        for name in ("CMakeCache.txt", "Makefile", "cmake_install.cmake"):
            try:
                (build_dir / name).unlink()
            except FileNotFoundError:
                pass


def build_cuda_lib() -> Path:
    """Build (or locate a cached) CUDA shared library for quantized matmul.

    The function performs the following steps:

    1. Detects the target compute capability via :func:`_detect_cuda_arch`.
    2. Checks whether a cached ``.so`` already exists and is up-to-date
       relative to the C++/CUDA source tree in ``csrc/quantized_matmul/``.
    3. If a rebuild is needed, invokes CMake to configure and compile the
       library, placing build artifacts in ``_build/`` next to this module.
    4. Returns the path to the compiled shared library.

    When the ``EJKERNEL_CUDA_ARCHS`` environment variable is set, the
    function additionally builds for each listed architecture and still
    returns the library targeting the detected (or overridden) architecture.

    Returns:
        Absolute path to the compiled shared library, e.g.
        ``<package>/_build/libejkernel_qmm_cuda_sm89.so``.

    Raises:
        RuntimeError: If CMake is not installed, the build fails, or the
            expected library file is missing after a successful build.
    """
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent.parent.parent
    csrc_root = repo_root / "csrc" / "quantized_matmul"
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _detect_cuda_arch()
    lib_path = build_dir / f"libejkernel_qmm_cuda_sm{arch}.so"
    archs_env = os.getenv("EJKERNEL_CUDA_ARCHS")
    archs = _parse_arch_list(archs_env) if archs_env else []
    if archs and arch not in archs:
        archs.insert(0, arch)

    vendor_mtime = _latest_mtime([csrc_root])
    if archs:
        libs = [build_dir / f"libejkernel_qmm_cuda_sm{a}.so" for a in archs]
        if all(p.exists() and p.stat().st_mtime >= vendor_mtime for p in libs):
            return lib_path
    else:
        if lib_path.exists() and lib_path.stat().st_mtime >= vendor_mtime:
            return lib_path

    include_dir = Path(ffi.include_dir())
    cutlass_include = _find_cutlass_include(repo_root)
    cuda_compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    _invalidate_stale_cmake_cache(build_dir, cuda_compiler=cuda_compiler, arch=arch, archs=archs)

    cmake_cmd = [
        "cmake",
        "-S",
        str(csrc_root),
        "-B",
        str(build_dir),
        f"-DEJKERNEL_CUDA_ARCHS={';'.join(archs)}" if archs else f"-DEJKERNEL_CUDA_ARCH={arch}",
        f"-DEJKERNEL_JAX_FFI_INCLUDE={include_dir}",
        f"-DEJKERNEL_CUTLASS_INCLUDE={cutlass_include}",
    ]
    if Path(cuda_compiler).exists():
        cmake_cmd.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
    build_jobs = os.getenv("EJKERNEL_CUDA_BUILD_JOBS") or str(max(1, min(os.cpu_count() or 1, 16)))
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel", build_jobs]
    if not archs:
        build_cmd.extend(["--target", f"ejkernel_qmm_cuda_sm{arch}"])

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

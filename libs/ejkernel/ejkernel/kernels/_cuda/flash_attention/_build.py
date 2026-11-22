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

"""Build utilities for the CUDA Flash Attention shared library.

This module handles the CMake-based compilation of the CUDA C++ Flash
Attention kernels into a shared library (``.so``) that is later loaded via
:mod:`ctypes` and registered with the JAX FFI runtime. It provides:

* GPU compute-capability detection (via ``nvidia-smi`` or environment
  variable overrides).
* Incremental rebuild logic that skips compilation when the shared library
  is already up-to-date with respect to the source tree timestamps.
* Multi-architecture builds when the ``EJKERNEL_CUDA_ARCHS`` environment
  variable is set.

Environment Variables:
    EJKERNEL_CUDA_ARCH: Override the detected GPU compute capability with a
        single SM version (e.g., ``"90"`` for SM 9.0).
    EJKERNEL_CUDA_ARCHS: Semicolon- or comma-separated list of SM
        versions to build for (e.g., ``"80;86;90"``).
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

    Strips whitespace, the ``sm_`` or ``compute_`` prefix, and converts
    dotted versions (e.g., ``"8.6"``) to concatenated digits (``"86"``).

    Args:
        value: Raw architecture string such as ``"sm_80"``, ``"8.6"``, or
            ``"compute_90"``.

    Returns:
        A short numeric string like ``"80"`` or ``"86"``.
    """
    val = value.strip().lower().replace("sm_", "").replace("compute_", "")
    if "." in val:
        major, minor = val.split(".", 1)
        return f"{major}{minor[:1]}"
    return val


def _parse_arch_list(value: str) -> list[str]:
    """Parse a semicolon- or comma-separated architecture list.

    Each entry is normalized via :func:`_normalize_arch`.

    Args:
        value: Delimited string of architecture identifiers, e.g.
            ``"80;86"`` or ``"sm_80,sm_90"``.

    Returns:
        List of normalized architecture strings.
    """
    parts = [p for p in value.replace(",", ";").split(";") if p.strip()]
    return [_normalize_arch(p) for p in parts]


def _detect_cuda_arch() -> str:
    """Detect the CUDA compute capability of the current GPU.

    Resolution order:

    1. The ``EJKERNEL_CUDA_ARCH`` environment variable (if set).
    2. The first GPU reported by ``nvidia-smi --query-gpu=compute_cap``.
    3. Falls back to ``"80"`` (Ampere A100) when detection fails.

    Returns:
        Normalized architecture string (e.g., ``"80"`` for SM 8.0).
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

    Recursively walks every directory in *paths* and returns the maximum
    ``st_mtime`` found among all regular files. Non-existent roots are
    silently skipped.

    Args:
        paths: Root directories (or files) to scan.

    Returns:
        The latest modification time as a POSIX timestamp (float).
        Returns ``0.0`` if no files are found.
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
        if (candidate / "cutlass" / "cutlass.h").exists():
            return candidate

    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        "CUTLASS C++ headers were not found for the CUDA flash_attention backend. "
        "Run `git submodule update --init csrc/cutlass`, or set "
        "EJKERNEL_CUTLASS_INCLUDE to a CUDA CUTLASS include directory containing "
        "cutlass/cutlass.h.\n"
        f"Checked:\n{formatted}"
    )


def build_cuda_lib() -> Path:
    """Build (or locate an up-to-date) CUDA Flash Attention shared library.

    The function performs the following steps:

    1. Detect the target SM architecture.
    2. Check whether a shared library already exists and is newer than the
       CUDA and CUTLASS source trees; if so, return it immediately.
    3. Otherwise, invoke CMake to configure and build the library.

    The build artefacts are placed in a ``_build/`` directory next to this
    source file.

    Returns:
        Absolute :class:`~pathlib.Path` to the compiled ``.so`` file for
        the detected (or overridden) compute capability.

    Raises:
        RuntimeError: If CMake is not installed, the build fails, or the
            expected library file is missing after a successful build.
    """
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent.parent.parent
    csrc_root = repo_root / "csrc" / "flash_attention"
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _detect_cuda_arch()
    lib_path = build_dir / f"libejkernel_flash_attention_cuda_sm{arch}.so"
    archs_env = os.getenv("EJKERNEL_CUDA_ARCHS")
    archs = _parse_arch_list(archs_env) if archs_env else []
    if archs and arch not in archs:
        archs.insert(0, arch)

    vendor_mtime = _latest_mtime(
        [
            repo_root / "csrc" / "cutlass",
            csrc_root,
        ]
    )
    if archs:
        libs = [build_dir / f"libejkernel_flash_attention_cuda_sm{a}.so" for a in archs]
        if all(p.exists() and p.stat().st_mtime >= vendor_mtime for p in libs):
            return lib_path
    else:
        if lib_path.exists() and lib_path.stat().st_mtime >= vendor_mtime:
            return lib_path

    include_dir = Path(ffi.include_dir())
    cutlass_include = _find_cutlass_include(repo_root)
    cuda_compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

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
        build_cmd.extend(["--target", f"ejkernel_flash_attention_cuda_sm{arch}"])

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


def build_cuda_libs(archs: list[str] | None = None) -> list[Path]:
    """Pre-build CUDA shared libraries for one or more SM architectures.

    Sets the ``EJKERNEL_CUDA_ARCHS`` environment variable and delegates to
    :func:`build_cuda_lib` so that a single CMake invocation produces
    libraries for every requested architecture.

    Args:
        archs: List of normalized SM architecture strings (e.g.,
            ``["80", "90"]``). When ``None``, the currently detected GPU
            architecture is used.

    Returns:
        List of :class:`~pathlib.Path` objects pointing to the built
        shared libraries, one per architecture.
    """
    if archs is None:
        archs = [_detect_cuda_arch()]
    os.environ["EJKERNEL_CUDA_ARCHS"] = ";".join(archs)
    build_cuda_lib()
    root = Path(__file__).resolve().parent
    build_dir = root / "_build"
    return [build_dir / f"libejkernel_flash_attention_cuda_sm{a}.so" for a in archs]

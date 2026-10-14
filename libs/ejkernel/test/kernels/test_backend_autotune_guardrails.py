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

"""Static guardrails that keep timed autotuning out of backend kernels."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KERNELS_ROOT = _REPO_ROOT / "ejkernel" / "kernels"

_BANNED_PATTERNS = {
    "@triton.autotune": re.compile(r"@triton\.autotune\b"),
    "triton.autotune": re.compile(r"\btriton\.autotune\b"),
    "@autotune": re.compile(r"@autotune\s*\("),
    "_AUTOTUNE_CACHE": re.compile(r"\b_AUTOTUNE_CACHE\b"),
    "autotune_tilelang_ffi": re.compile(r"\bautotune_tilelang_ffi\b"),
    "backend autotune env": re.compile(r"\bEJKERNEL_[A-Z0-9_]*AUTOTUNE[A-Z0-9_]*\b"),
}


def _kernel_python_files() -> list[Path]:
    return sorted(path for path in _KERNELS_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def test_backend_kernels_do_not_own_runtime_autotune() -> None:
    """Backend kernels must consume concrete configs chosen by operation executors."""
    offenders: list[str] = []
    for path in _kernel_python_files():
        text = path.read_text()
        rel = path.relative_to(_REPO_ROOT)
        for label, pattern in _BANNED_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{rel}:{line}: {label}: {match.group(0)}")

    assert not offenders, "Backend-local autotune is banned under ejkernel/kernels:\n" + "\n".join(offenders)

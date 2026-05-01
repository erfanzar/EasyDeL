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
"""Typed policy values for GDN inference kernels."""

from __future__ import annotations

import typing as tp

KernelTilePolicy: tp.TypeAlias = tp.Literal["auto", "b16", "b8", "b4"]
KERNEL_TILE_POLICIES: frozenset[str] = frozenset(("auto", "b16", "b8", "b4"))


def normalize_kernel_tile_policy(policy: str | None) -> KernelTilePolicy:
    normalized = "auto" if policy is None else str(policy).lower()
    if normalized not in KERNEL_TILE_POLICIES:
        raise ValueError(f"kernel_tile_policy must be one of 'auto', 'b16', 'b8', or 'b4'; got {policy!r}.")
    return tp.cast(KernelTilePolicy, normalized)

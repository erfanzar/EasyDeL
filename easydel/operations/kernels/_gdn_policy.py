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
"""Typed policy values for Gated Delta Rule (GDN) inference kernels.

Defines the allowed string identifiers for the per-row tile size used when
launching the GDN Pallas kernels and provides a small validation helper.

Exports:
    KernelTilePolicy: literal type alias of the supported tile-size names.
    KERNEL_TILE_POLICIES: frozen set of valid tile-policy strings.
    normalize_kernel_tile_policy: validate and normalize a user-supplied policy.
"""

from __future__ import annotations

import typing as tp

KernelTilePolicy: tp.TypeAlias = tp.Literal["auto", "b16", "b8", "b4"]
KERNEL_TILE_POLICIES: frozenset[str] = frozenset(("auto", "b16", "b8", "b4"))


def normalize_kernel_tile_policy(policy: str | None) -> KernelTilePolicy:
    """Validate and normalize a kernel-tile-policy string.

    Accepts a user-provided string (or ``None``) and returns it lower-cased
    after checking that it is one of the supported variants.

    Args:
        policy: A tile-policy identifier. ``None`` is treated as ``"auto"``.
            Valid values (case-insensitive) are ``"auto"``, ``"b16"``, ``"b8"``
            and ``"b4"``, corresponding to autoselect, block-size 16, 8 and 4
            respectively.

    Returns:
        KernelTilePolicy: The lower-cased policy string, narrowed to the
        ``KernelTilePolicy`` literal type.

    Raises:
        ValueError: If ``policy`` is not in ``KERNEL_TILE_POLICIES``.
    """
    normalized = "auto" if policy is None else str(policy).lower()
    if normalized not in KERNEL_TILE_POLICIES:
        raise ValueError(f"kernel_tile_policy must be one of 'auto', 'b16', 'b8', or 'b4'; got {policy!r}.")
    return tp.cast(KernelTilePolicy, normalized)

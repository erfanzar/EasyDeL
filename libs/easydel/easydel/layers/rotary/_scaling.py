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

"""Backward-compatible HF-to-EasyDeL RoPE configuration normalization."""

import typing as tp

from ._utils import rope_scaling_from_rope_parameters as _normalize_rope_parameters


def rope_scaling_from_rope_parameters(
    rope_parameters: dict[str, tp.Any] | None,
    rope_scaling: dict[str, tp.Any] | None,
    default: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any] | None:
    """Normalize HF ``rope_parameters`` while preserving legacy config semantics.

    Explicit ``rope_scaling`` wins and is returned unchanged unless its legacy
    ``type`` key needs a ``rope_type`` alias. Otherwise supported fields are
    copied from ``rope_parameters``. When both inputs are absent, ``default``
    is copied, including an independent copy of each list-valued field.

    Args:
        rope_parameters: HF-style RoPE parameters mapping or ``None``.
        rope_scaling: EasyDeL RoPE scaling mapping or ``None``.
        default: Fallback mapping when both inputs are absent. ``None`` returns
            ``None``. Family-specific presets belong to the calling config.

    Returns:
        A normalized scaling mapping, or ``None`` when no inputs or fallback
        are supplied. Values copied from ``rope_parameters`` retain their
        identity, matching the existing configuration contract.
    """
    if rope_scaling is not None:
        # HF sometimes uses "type" instead of "rope_type"
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]
        return rope_scaling

    if rope_parameters is None:
        if default is None:
            return None
        return {key: (list(value) if isinstance(value, list) else value) for key, value in default.items()}

    # The legacy config bridge recognizes "type" only in explicit scaling,
    # unlike the general rotary bridge. Supply the old default explicitly so
    # normalization and its supported-key list can still share one source.
    parameters = dict(rope_parameters)
    parameters["rope_type"] = rope_parameters.get("rope_type", "default")
    return _normalize_rope_parameters(parameters, None)

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

"""Config-driven lowering options for repeated-layer ``ModuleList.scan`` calls.

``EasyDeLBaseConfig.layer_scan_options`` selects how every lowered layer scan in a model is compiled.
Top-level option keys apply to every scan; a scope key (``"text"``, ``"vision"``, ``"audio"``) holds
overrides for the scans of that tower only::

    config.layer_scan_options = {
        "block_scan": True,
        "vision": {"block_scan": False, "unroll": 2},
    }

``None`` (the default) lets :func:`auto_layer_scan_kwargs` pick options: training text scans on a mesh
whose ``fsdp`` axis is larger than one prefetch the next layer's fsdp-sharded weights. An empty mapping
passes nothing to ``scan`` (plain lowering).
"""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping

LAYER_SCAN_OPTION_KEYS: tuple[str, ...] = (
    "block_scan",
    "barrier",
    "prefetch_axes",
    "prefetch_exclude",
    "prefetch_defer_dw",
    "unroll",
)
LAYER_SCAN_SCOPES: tuple[str, ...] = ("text", "vision", "audio")

_BOOL_KEYS = ("block_scan", "barrier", "prefetch_defer_dw")


def _as_name_tuple(key: str, value: tp.Any, *, allow_none: bool) -> tuple[str, ...] | None:
    if value is None:
        if allow_none:
            return None
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise TypeError(f"layer_scan_options[{key!r}] must be a string or a list of strings, got {value!r}.")


def _validate_section(section: Mapping[str, tp.Any], where: str) -> dict[str, tp.Any]:
    out: dict[str, tp.Any] = {}
    for key, value in section.items():
        if key not in LAYER_SCAN_OPTION_KEYS:
            raise KeyError(
                f"Unknown layer scan option {key!r} in {where}; expected one of {LAYER_SCAN_OPTION_KEYS} "
                f"or a scope in {LAYER_SCAN_SCOPES}."
            )
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                raise TypeError(f"layer_scan_options {where}[{key!r}] must be a bool, got {value!r}.")
            out[key] = value
        elif key == "unroll":
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"layer_scan_options {where}['unroll'] must be a positive int or None, got {value!r}.")
            out[key] = value
        elif key == "prefetch_axes":
            axes = _as_name_tuple(key, value, allow_none=True)
            out[key] = axes if axes else None
        else:
            out[key] = _as_name_tuple(key, value, allow_none=False)
    return out


def normalize_layer_scan_options(options: Mapping[str, tp.Any] | None) -> dict[str, dict[str, tp.Any]]:
    """Validate ``layer_scan_options`` and split it into a default section and per-scope sections.

    Args:
        options: The raw config value: ``None``, or a mapping whose option keys apply to every scan and
            whose scope keys map to per-scope overrides.

    Returns:
        ``{"default": {...}, "<scope>": {...}}`` with only the sections that were given.

    Raises:
        KeyError: For an unknown option or scope name.
        TypeError: For a value of the wrong type.
        ValueError: For an out-of-range ``unroll``.
    """
    if not options:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"layer_scan_options must be a mapping or None, got {type(options).__name__}.")
    default: dict[str, tp.Any] = {}
    scoped: dict[str, dict[str, tp.Any]] = {}
    for key, value in options.items():
        if key in LAYER_SCAN_SCOPES:
            if not isinstance(value, Mapping):
                raise TypeError(f"layer_scan_options[{key!r}] must be a mapping of scan options, got {value!r}.")
            scoped[key] = _validate_section(value, f"[{key!r}]")
        else:
            default.update(_validate_section({key: value}, "(top level)"))
    sections = dict(scoped)
    if default:
        sections["default"] = default
    return sections


def resolve_layer_scan_kwargs(options: Mapping[str, tp.Any] | None, scope: str) -> dict[str, tp.Any]:
    """Return the ``ModuleList.scan`` keyword arguments for one scan scope.

    Args:
        options: The raw ``layer_scan_options`` config value.
        scope: The tower the scan belongs to, one of :data:`LAYER_SCAN_SCOPES`.

    Returns:
        Keyword arguments for ``ModuleList.scan``; empty when nothing is configured for ``scope``.

    Raises:
        KeyError: If ``scope`` is not a known scope.
    """
    if scope not in LAYER_SCAN_SCOPES:
        raise KeyError(f"Unknown layer scan scope {scope!r}; expected one of {LAYER_SCAN_SCOPES}.")
    sections = normalize_layer_scan_options(options)
    return {**sections.get("default", {}), **sections.get(scope, {})}


def auto_layer_scan_kwargs(mesh: tp.Any, scope: str, *, inference: bool) -> dict[str, tp.Any]:
    """Default ``ModuleList.scan`` options when ``layer_scan_options`` is ``None``.

    Args:
        mesh: The model mesh (anything with a ``shape`` mapping), or ``None``.
        scope: The scan scope.
        inference: Whether the scan is traced under inference mode.

    Returns:
        ``{"prefetch_axes": ("fsdp",)}`` for training text scans when the mesh's ``fsdp`` axis is larger
        than one, else ``{}``.
    """
    if inference or scope != "text":
        return {}
    try:
        fsdp = int(mesh.shape["fsdp"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}
    return {"prefetch_axes": ("fsdp",)} if fsdp > 1 else {}


def infer_layer_scan_scope(config: tp.Any) -> str:
    """Guess the scan scope of a tower from its config class name (``*Vision*`` / ``*Audio*`` / text)."""
    name = type(config).__name__.lower()
    if "vision" in name or "visual" in name:
        return "vision"
    if "audio" in name or "speech" in name:
        return "audio"
    return "text"

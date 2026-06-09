# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module/variable location paths and their string codec.

A ``Path`` is a tuple whose components are either:

* ``str`` — an attribute name on a module or a key of a
  :class:`~spectrax.nn.ModuleDict`;
* ``int`` — a positional index in a :class:`~spectrax.nn.Sequential`,
  :class:`~spectrax.nn.ModuleList`, or :class:`~spectrax.nn.ParameterList`.

Paths have a dotted-string form (``"encoder.layers.0.fc.weight"``) used as
the key into :class:`~spectrax.State`. Integer components render as bare
digits. Keys that look numeric or contain ``'.'`` are quoted with a
leading ``'#'`` to keep the codec invertible.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._typing import Path, PathComponent

__all__ = ["Path", "PathComponent", "is_prefix", "join", "path_to_str", "str_to_path"]


def path_to_str(path: Path) -> str:
    """Render a tuple path into its canonical dotted string form.

    Integer components become bare digits (``0`` -> ``"0"``). String
    components that collide with that convention (all-digit or containing
    ``'.'``) are prefixed with ``'#'`` and have their dots escaped.

    Args:
        path: The tuple path to render.

    Returns:
        The canonical dotted string. The empty tuple renders to ``""``.

    Raises:
        TypeError: If a component is neither ``str`` nor ``int``.
    """
    if not path:
        return ""
    parts: list[str] = []
    for c in path:
        if isinstance(c, int):
            parts.append(str(c))
        elif isinstance(c, str):
            if c.isdigit() or "." in c or c.startswith("#"):
                parts.append("#" + c.replace(".", "\\."))
            else:
                parts.append(c)
        else:
            raise TypeError(f"Invalid path component: {c!r}")
    return ".".join(parts)


def str_to_path(s: str) -> Path:
    """Parse a dotted string back into a tuple path.

    The inverse of :func:`path_to_str`. Handles the quote/escape conventions
    that ``path_to_str`` introduces.

    Args:
        s: The dotted string. An empty string yields the empty tuple.

    Returns:
        The tuple path.
    """
    if s == "":
        return ()
    out: list[PathComponent] = []
    buf: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            buf.append(s[i + 1])
            i += 2
            continue
        if c == ".":
            out.append(_decode_component("".join(buf)))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    out.append(_decode_component("".join(buf)))
    return tuple(out)


def _decode_component(c: str) -> PathComponent:
    """Decode a single component string into ``str`` or ``int``.

    Components prefixed with ``'#'`` are treated as quoted strings. Any
    all-digit component is decoded as ``int``. Everything else is a
    plain string.

    Args:
        c: C value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if c.startswith("#"):
        return c[1:]
    if c.isdigit():
        return int(c)
    return c


def is_prefix(prefix: Path, full: Path) -> bool:
    """Return ``True`` iff ``prefix`` is a prefix of ``full``.

    Args:
        prefix: The candidate prefix path.
        full: The path to test against.

    Returns:
        ``True`` when every component of ``prefix`` matches the
        corresponding leading components of ``full``.
    """
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def join(*paths: Iterable[PathComponent]) -> Path:
    """Concatenate any number of component iterables into a single path.

    Args:
        *paths: Arbitrary iterables of :data:`PathComponent` values.

    Returns:
        A single :data:`Path` tuple containing all components in order.
    """
    out: list[PathComponent] = []
    for p in paths:
        out.extend(p)
    return tuple(out)

# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Class resolution by Python import path.

The graph layer stores classes by their fully-qualified ``module.Qualname``
string rather than by object identity. :func:`bind` uses
:func:`resolve_class` to rehydrate those strings into real classes via
``importlib``, without running the user's ``__init__``. There is no
registry dict and no decorator: any importable class is resolvable.
"""

from __future__ import annotations

import importlib

_QNAME_CACHE: dict[type, str] = {}
"""Memoization table keyed by class object.

A class's ``__module__`` + ``__qualname__`` pair is immutable after
definition, so the concatenation can be computed exactly once per type
and reused across every subsequent export/bind cycle. The cache is
process-wide and grows bounded by the number of spectrax classes the
user constructs (typically a few dozen).
"""


def qualified_name(cls: type) -> str:
    """Return the fully-qualified ``module.Qualname`` string for ``cls``.

    This is the canonical string form used to persist class identity
    across the :func:`~spectrax.export` / :func:`~spectrax.bind` seam —
    :func:`resolve_class` maps the same string back to the class object
    via :func:`importlib.import_module`.

    The first call for a given class computes the string and stores it
    in :data:`_QNAME_CACHE`; subsequent calls are dict lookups. Caching
    matters because :func:`qualified_name` runs on every module node and
    variable node during :func:`~spectrax.export`, which is on the hot
    path of every module-aware jit / grad / vmap dispatch.

    Args:
        cls: Any Python class.

    Returns:
        A dotted ``"pkg.module.Class"`` string; nested classes include
        their full qualname chain (``"pkg.module.Outer.Inner"``).
    """
    cached = _QNAME_CACHE.get(cls)
    if cached is not None:
        return cached
    name = f"{cls.__module__}.{cls.__qualname__}"
    _QNAME_CACHE[cls] = name
    return name


def resolve_class(qualified: str) -> type:
    """Import and return the class identified by ``qualified``.

    ``qualified`` must be the form produced by :func:`qualified_name`
    (typically ``"package.module.ClassName"``). Nested classes are
    supported: the longest dotted prefix that successfully imports as a
    module is treated as the module, and the remaining suffix is walked
    via attribute access.

    Args:
        qualified: Fully-qualified class name string.

    Returns:
        The resolved class object.

    Raises:
        ImportError: If no dotted prefix imports as a module or the
            attribute chain does not resolve.
        TypeError: If the resolved object is not a class.
    """
    if "." not in qualified:
        raise ImportError(f"Cannot resolve class: {qualified!r}")
    parts = qualified.split(".")
    mod = None
    split_idx = len(parts) - 1
    while split_idx >= 1:
        module_name = ".".join(parts[:split_idx])
        try:
            mod = importlib.import_module(module_name)
            break
        except ImportError:
            split_idx -= 1
    if mod is None:
        raise ImportError(f"Cannot resolve class: {qualified!r}")
    missing = object()
    obj: object = mod
    for p in parts[split_idx:]:
        obj = getattr(obj, p, missing)
        if obj is missing:
            raise ImportError(f"Class {qualified!r} not found in {qualified.rsplit('.', 1)[0]!r}")
    if not isinstance(obj, type):
        raise TypeError(f"Resolved {qualified!r} is not a class: {obj!r}")
    return obj

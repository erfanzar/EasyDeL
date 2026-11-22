# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Private helper: invalidate ``id()``-keyed cache entries on GC.

Several hot-path caches across spectrax key on ``id(obj)`` so that two
unrelated models / meshes / functions never collide on a plain-value
hash. The risk with raw ``id()`` keys is address reuse: once ``obj``
is garbage-collected, CPython is free to reuse the freed address for
the next object, at which point the cache silently returns a stale
compiled artifact.

:func:`weak_invalidate` wires a :func:`weakref.finalize` callback to the
anchor object so the entry is popped from the cache the moment the
anchor dies — before any new object can claim the freed id.
"""

from __future__ import annotations

import weakref

__all__ = ["weak_invalidate"]


def weak_invalidate(anchor: object, cache: dict[object, object], key: object) -> None:
    """Register a GC callback that pops ``cache[key]`` when ``anchor`` is collected.

    No-op for anchors that cannot be weak-referenced (e.g. tuples of
    plain ints). Cache hygiene is best-effort — if we cannot install a
    finalizer, the entry simply lives until process exit.

    Calling this more than once with the same ``(anchor, cache, key)``
    installs multiple finalizers; each will ``cache.pop(key, None)``,
    which is idempotent.

    Args:
        anchor: The object whose identity backs the cache key. When it
            is finalized, ``cache[key]`` is removed.
        cache: The cache dict to prune.
        key: The key to drop.

    Returns:
        ``None``.
    """
    try:
        weakref.finalize(anchor, cache.pop, key, None)
    except TypeError:
        pass

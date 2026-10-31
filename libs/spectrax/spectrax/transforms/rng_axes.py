# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`split_rngs` and :class:`StateAxes` for transform-aware RNG handling.

Module-level RNG state lives in :class:`~spectrax.Rngs`, which holds one
:class:`~spectrax.RngStream` per logical use-site (``"params"``,
``"dropout"``, …). When a function is run under :func:`~spectrax.vmap`
or :func:`~spectrax.scan` without special handling, those streams are
broadcast unchanged across the mapped axis, so every batch element ends
up drawing *the same* dropout mask or initialization. The two utilities
in this module fix that:

* :func:`split_rngs` is a context manager that produces a list of
  ``axis_size`` independent :class:`~spectrax.Rngs` instances, each
  seeded from a distinct sub-key of the parent.
* :func:`split_stream_keys` is the lower-level primitive that derives
  the raw key array used by :func:`split_rngs`.

:class:`StateAxes` is declarative metadata describing how each variable
collection should be treated by a :func:`~spectrax.vmap` /
:func:`~spectrax.scan` call: ``None`` broadcasts, an ``int`` maps along
that leaf axis, and the string ``"split"`` selects automatic rng
splitting.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core._typing import Array
from ..core.module import Module as _Module
from ..core.variable import Variable
from ..rng.rngs import Rngs, RngStream

__all__ = ["StateAxes", "split_rngs", "split_stream_keys"]


@dataclass(frozen=True)
class StateAxes:
    """Per-collection axis spec for :func:`spectrax.vmap` / :func:`spectrax.scan`.

    Wraps a ``{collection_name: axis_spec}`` mapping that the transform
    consults to decide how each module variable collection should be
    handled along the mapped axis. The valid axis specs are:

    * ``None`` — the collection is broadcast (replicated) along the axis,
      identical to how :func:`jax.vmap` treats an ``in_axes`` entry of
      ``None``.
    * ``int`` — the collection's leaves are mapped along that array axis.
    * ``"split"`` — auto-split rng: each index along the mapped axis
      receives an independent PRNG-derived state, equivalent to threading
      :func:`split_rngs` for that collection at every call site.

    Attributes:
        axes: The underlying mapping; treated as immutable.
    """

    axes: Mapping[str, int | str | None]

    def get(self, collection: str, default: int | str | None = None) -> int | str | None:
        """Return the axis spec configured for ``collection``.

        Args:
            collection: Variable collection name (e.g. ``"parameters"``,
                ``"batch_stats"``, ``"rng"``).
            default: Value to return when ``collection`` has no entry.

        Returns:
            The axis spec for ``collection`` or ``default`` if absent.
        """
        return self.axes.get(collection, default)

    def __iter__(self) -> Iterator[tuple[str, int | str | None]]:
        """Iterate ``(collection, axis_spec)`` pairs in insertion order.

        Returns:
            Iterator over the contained values.
        """
        return iter(self.axes.items())


def split_stream_keys(stream: RngStream, axis_size: int) -> Array:
    """Derive ``axis_size`` independent raw keys from ``stream``.

    Unpacks the stream's ``(raw_key, hi_counter, lo_counter)`` triple,
    folds the two counters into the typed key, splits the result via
    :func:`jax.random.split`, and then rebuilds the stream value with an
    incremented ``lo`` counter (rolling the ``hi`` counter on overflow).
    The returned array carries the *raw* uint32 representation of each
    sub-key so callers can rebind them straight back into a
    :class:`~spectrax.RngStream` without re-typing.

    Args:
        stream: Source :class:`~spectrax.RngStream`. Mutated in place to
            advance its lo counter by one.
        axis_size: Number of independent sub-keys to produce. Must be
            strictly positive.

    Returns:
        A ``(axis_size, key_size)`` ``uint32`` array. Each row is the
        raw key data for one independent sub-stream.

    Raises:
        ValueError: If ``axis_size`` is non-positive.
    """
    if axis_size <= 0:
        raise ValueError(f"axis_size must be > 0, got {axis_size}.")
    raw, hi, lo = stream._unpack()
    typed = jax.random.fold_in(
        jax.random.fold_in(jax.random.wrap_key_data(raw), hi.astype(jnp.int32)), lo.astype(jnp.int32)
    )
    sub = jax.random.split(typed, axis_size)
    new_lo = lo + jnp.uint32(1)
    carry = jnp.where(new_lo == jnp.uint32(0), jnp.uint32(1), jnp.uint32(0))
    stream.value = jnp.concatenate([raw, jnp.array([hi + carry, new_lo], dtype=jnp.uint32)])
    return jax.vmap(jax.random.key_data)(sub)


def _clone_stream(stream: RngStream) -> RngStream:
    """Clone an :class:`~spectrax.RngStream` without sharing mutable counter state.

    Used by :func:`split_rngs` for streams that are *not* in the requested
    ``only`` set: every fork gets its own :class:`~spectrax.RngStream`
    object so that a counter increment on one fork does not propagate to
    its siblings, but the underlying value (raw key + counters) is copied
    by reference so all forks start from the same point in the stream.

    Args:
        stream: Source stream to clone.

    Returns:
        A fresh :class:`~spectrax.RngStream` whose initial value is
        identity-equal to ``stream`` 's value but whose subsequent
        mutations are isolated from ``stream``.
    """
    clone = object.__new__(RngStream)
    Variable.__init__(
        clone,
        stream._raw_get(),
        kind="rng",
        metadata=dict(stream.metadata),
    )
    return clone


@contextlib.contextmanager
def split_rngs(rngs: Rngs, *, axis_size: int, only: tuple[str, ...] | None = None) -> Iterator[list[Rngs]]:
    """Context manager yielding ``axis_size`` independent :class:`~spectrax.Rngs`.

    For every stream named in ``only`` (or every stream when ``only`` is
    ``None``), :func:`split_stream_keys` derives ``axis_size`` independent
    sub-keys; each yielded fork receives one of those sub-keys. Streams
    that were *not* requested in ``only`` are cloned via
    :func:`_clone_stream` so each fork has its own counter state but
    starts from the same point in the original stream.

    The context does not write per-fork state back to the parent ``rngs``
    on exit. Callers who want to retain mutations performed inside the
    block must copy them out explicitly. The parent ``rngs`` *is*
    mutated however during entry, since :func:`split_stream_keys`
    advances each requested stream's counter by one to derive the splits.

    Args:
        rngs: Parent :class:`~spectrax.Rngs` whose streams supply the
            entropy for the splits.
        axis_size: Number of independent forks to produce. Must be
            strictly positive — typically equals the leading axis of a
            batched :func:`~spectrax.vmap` input.
        only: Optional tuple of stream names to split. When ``None``
            (default), every stream registered on ``rngs`` is split.

    Yields:
        A list of ``axis_size`` :class:`~spectrax.Rngs` instances, each
        with the same set of stream names as ``rngs`` but with
        independent key state for the requested streams.

    Raises:
        ValueError: If ``axis_size`` is non-positive.
    """
    if axis_size <= 0:
        raise ValueError(f"axis_size must be > 0, got {axis_size}.")
    names = tuple(rngs._spx_items.keys()) if only is None else tuple(only)
    split_map: dict[str, Array] = {}
    for nm in names:
        stream = rngs.stream(nm)
        split_map[nm] = split_stream_keys(stream, axis_size)
    forks: list[Rngs] = []
    for i in range(axis_size):
        fork = Rngs.__new__(Rngs)
        _Module.__init__(fork)
        object.__setattr__(fork, "_spx_items", {})
        for nm in names:
            key_i = split_map[nm][i]
            fork._spx_items[nm] = RngStream(key_i)
        for nm, s in rngs._spx_items.items():
            if nm not in names:
                fork._spx_items[nm] = _clone_stream(s)
        forks.append(fork)
    try:
        yield forks
    finally:
        pass

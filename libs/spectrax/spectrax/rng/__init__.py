# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit named-stream RNG.

This subpackage owns SpectraX's randomness contract:

* :class:`~spectrax.Rngs` is a :class:`~spectrax.Module` carrying one
  :class:`~spectrax.RngStream` per *named* stream (``"default"``,
  ``"parameters"``, ``"dropout"``, …). Each stream stores a typed PRNG
  key together with a 64-bit counter packed as ``uint32`` words, all
  inside a single JAX array leaf so it travels with the model's state
  through ``jit`` / ``grad`` / ``vmap`` / ``scan`` / ``remat``.
* :func:`spectrax.seed` is a thread-local context manager that pushes a
  default :class:`~spectrax.Rngs` so layer constructors can be called
  without an explicit ``rngs=`` argument.
* :func:`resolve_rngs` is the helper layers use to coerce
  ``Rngs | int | None`` into a concrete :class:`~spectrax.Rngs`.

Streams that have not been declared (e.g. ``rngs.brand_new_stream``
inside a freshly-traced ``jit``) get a deterministic in-transform
fallback: a key derived from the ``"default"`` stream by folding in a
hash of the stream name. Outside transforms the new stream is created
and cached so subsequent accesses advance a single counter.
"""

from .rngs import Rngs, RngStream, resolve_rngs
from .seed import default_rngs, seed

__all__ = ["RngStream", "Rngs", "default_rngs", "resolve_rngs", "seed"]

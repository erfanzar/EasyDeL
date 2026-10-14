# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed exception hierarchy raised by spectrax.

Every exception spectrax raises inherits from :class:`SpecTraxError` so that
user code can catch all spectrax-originated failures with a single
``except spectrax.SpecTraxError`` clause. Individual subclasses describe
the class of failure:

* :class:`CyclicGraphError` — a module graph contains a cycle (a module
  reaches itself by following child references).
* :class:`IllegalMutationError` — a collection that was not declared
  ``mutable=`` was written to during a spectrax transform.
* :class:`LazyInitUnderTransformError` — a lazy layer attempted to
  materialize while a spectrax transform was active.
* :class:`SelectorError` — a :class:`~spectrax.Selector` predicate raised,
  or a selector sugar value could not be coerced.
* :class:`PolicyError` — an invalid dtype policy was applied to a module.
* :class:`GraphStructureError` — the module graph or graph-def violates a
  structural invariant (non-module child, unknown container kind, etc.).
"""

from __future__ import annotations


class SpecTraxError(Exception):
    """Base class for every exception raised by spectrax.

    Catching this class catches all spectrax-originated failures with a
    single ``except spectrax.SpecTraxError`` clause.
    """


class CyclicGraphError(SpecTraxError):
    """Raised when a module's graph contains a cycle.

    A cycle is detected when a traversal of child modules/variables reaches
    a module that is already on the active traversal stack.
    """


class IllegalMutationError(SpecTraxError):
    """Raised when a non-mutable collection changed under a transform.

    SpecTrax transforms capture all :class:`~spectrax.Variable` writes that
    occur while the transform body executes. Writes to collections not
    listed in the ``mutable=`` selector of the transform are forbidden;
    encountering one raises this error.
    """


class LazyInitUnderTransformError(SpecTraxError):
    """Raised when a lazy layer materializes inside a spectrax transform.

    Deferred parameters (e.g. :class:`~spectrax.DeferredParameter`) infer
    their shape from the first input. Running materialization under ``jit`` /
    ``grad`` / ``vmap`` would silently bake the inferred shape into a traced
    program so it is refused eagerly.
    """


class SelectorError(SpecTraxError):
    """Raised for invalid :class:`~spectrax.Selector` usage.

    Raised either when a predicate passed to a selector raises (wrapped and
    re-raised as ``SelectorError``) or when :func:`spectrax.as_selector`
    cannot coerce a value to a :class:`~spectrax.Selector`.
    """


class PolicyError(SpecTraxError):
    """Raised for invalid dtype policy configuration on a module.

    Typically triggered when a :class:`~spectrax.Policy` is assigned to
    a module attribute and the value is not a :class:`~spectrax.Policy`
    instance or ``None``.
    """


class GraphStructureError(SpecTraxError):
    """Raised when a graph or graph-def violates a structural invariant.

    Covers unexpected child types during traversal, missing or malformed
    nodes in a :class:`~spectrax.GraphDef`, non-``str`` keys in a
    :class:`~spectrax.nn.ModuleDict`, and similar low-level structural
    violations.
    """

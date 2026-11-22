# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware :func:`jax.checkpoint` (gradient checkpointing) wrapper.

Two usage modes are supported by the public :func:`remat` entry point:
applied to a function it returns a wrapper that runs the call under
:func:`jax.checkpoint`; applied to a :class:`~spectrax.Module` subclass
it returns a cached subclass whose ``forward`` method is permanently
wrapped, which is the recommended pattern for transformer blocks where
every layer should be rematted identically.

Internally the wrapper uses the standard split/merge shim from
:mod:`spectrax.transforms.split_merge` to convert module arguments to
``(GraphDef, State)``; some keyword arguments (strings, booleans, and
nested containers thereof) are filtered out of the
:func:`jax.checkpoint`-traced payload and closed over as Python
constants because :func:`jax.checkpoint` cannot trace them.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from typing import TypeVar, cast

import jax

from ..core.module import Module
from ..core.selector import SelectorSugar
from .split_merge import (
    _run_pure_body,
    _run_readonly_body,
    apply_mutations,
    locate_and_strip,
    make_pure,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    resolve_mutable,
)

__all__ = ["remat"]

F = TypeVar("F", bound=Callable[..., object])


_REMAT_CLASS_CACHE: dict[tuple, type] = {}
"""Cache of remat'd module subclasses keyed by ``(parent_class, config)``.

Ensures ``spx.remat(Llama3Block)`` returns the *same* class on every
call: avoids re-registering the pytree and lets
:func:`resolve_class` find the remat'd subclass during bind.
"""


def _hashable_cache_key(value: object) -> object:
    """Convert selector-like containers into deterministic, hashable cache-key values.

    The :data:`_REMAT_CLASS_CACHE` is keyed on a tuple that includes
    arbitrary user input (the ``mutable`` selector sugar can be a list,
    tuple, dict, set, …). Python's built-in :func:`hash` rejects mutable
    containers, so this helper canonicalizes dicts to sorted item
    tuples, list/tuple/set/frozenset to recursively-canonicalized
    tuples, and falls back to ``repr`` for any leaf value that is not
    natively hashable.

    Args:
        value: Arbitrary user-facing argument.

    Returns:
        A value with the same equality semantics that is safe to use
        inside a :class:`dict` key.
    """
    if isinstance(value, dict):
        return tuple(sorted((_hashable_cache_key(k), _hashable_cache_key(v)) for k, v in value.items()))
    if isinstance(value, list | tuple | set | frozenset):
        return tuple(_hashable_cache_key(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def remat(
    fn: F | None = None,
    *,
    mutable: SelectorSugar = (),
    prevent_cse: bool = True,
    policy: Callable[..., bool] | None = None,
    static_argnums: int | tuple[int, ...] = (),
) -> F:
    """Module-aware gradient checkpointing.

    Two usage modes:

    * **Function**: ``spx.remat(fn)`` returns a wrapped function that
      runs ``fn`` under :func:`jax.checkpoint`. Per-call wrapping.

    * **Module class**: ``spx.remat(MyBlock)`` where ``MyBlock`` is a
      subclass of :class:`spectrax.Module` returns a new subclass whose
      ``forward`` is checkpointed once. Build instances normally and
      use them in a loop — every block call recomputes during backward
      without per-iteration ``spx.remat(...)`` calls in the model body::

          RematBlock = spx.remat(MyBlock)
          blocks = [RematBlock(cfg, rngs=rngs) for _ in range(N)]
          # inside model.forward:
          for blk in blocks:
              x = blk(x)   # already remat-wrapped

      The returned class is registered in the parent class's module
      namespace so ``spectrax.export`` / ``bind`` can round-trip it.

    Args:
        fn: Function or :class:`Module` subclass to checkpoint; when
            omitted, returns a decorator factory.
        mutable: Selector controlling writable collections.
        prevent_cse, policy, static_argnums: Forwarded verbatim to
            :func:`jax.checkpoint`.

    Returns:
        A wrapped function or a new :class:`~spectrax.Module` subclass
        whose ``forward`` is checkpointed.

    Raises:
        TypeError: If ``prevent_cse`` is not a boolean, or if ``fn`` is
            a class that is not a subclass of :class:`~spectrax.Module`.
    """
    if not isinstance(prevent_cse, bool):
        raise TypeError(f"prevent_cse must be a bool, got {type(prevent_cse).__name__}.")
    if fn is None:
        return cast(
            F,
            lambda f: remat(
                f,
                mutable=mutable,
                prevent_cse=prevent_cse,
                policy=policy,
                static_argnums=static_argnums,
            ),
        )

    if isinstance(fn, type):
        if issubclass(fn, Module):
            return cast(
                F,
                _remat_module_class(
                    fn,
                    mutable=mutable,
                    prevent_cse=prevent_cse,
                    policy=policy,
                    static_argnums=static_argnums,
                ),
            )

    mutable_sel = resolve_mutable(mutable)

    def _should_be_static_kwarg(x: object) -> bool:
        """Return whether ``x`` must be closed over rather than traced by :func:`jax.checkpoint`.

        :func:`jax.checkpoint` traces every input through a fresh JAX
        sub-trace. Strings raise :class:`TypeError` and Python ``bool``
        values raise :class:`jax.errors.TracerBoolConversionError` the
        moment they reach Python-level control flow inside the
        checkpointed body. The wrapper sidesteps both by partitioning
        such kwargs out of the dynamic dict before calling
        :func:`jax.checkpoint` and re-injecting them through a closure
        when the pure body runs.

        The check recurses into tuples, lists, and dicts so a
        ``dict(deterministic=True)`` kwarg is also recognized as
        non-traceable.

        Args:
            x: A keyword-argument value to inspect.

        Returns:
            ``True`` if ``x`` (or any element nested inside it) cannot
            survive a :func:`jax.checkpoint` trace.
        """
        if isinstance(x, (str, bool)):
            return True
        if isinstance(x, (tuple, list)):
            return any(_should_be_static_kwarg(elem) for elem in x)
        if isinstance(x, dict):
            return any(_should_be_static_kwarg(v) for v in x.values())
        return False

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        """Dispatch the call through :func:`jax.checkpoint` wrapped around a pure body.

        Locates module arguments via
        :func:`~spectrax.transforms.split_merge.locate_and_strip`,
        partitions kwargs into traced/static halves via
        :func:`_should_be_static_kwarg`, picks the appropriate pure-body
        factory based on whether the call has the single-positional
        Module fast-path shape, and wraps the result in
        :func:`jax.checkpoint`. After the checkpointed call returns,
        captured mutations are written back to the live module via
        :func:`~spectrax.transforms.split_merge.apply_mutations`.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        refs, stripped_args, stripped_kwargs = locate_and_strip(args, kwargs)

        static_kwargs = {k: v for k, v in stripped_kwargs.items() if _should_be_static_kwarg(v)}
        dynamic_kwargs = {k: v for k, v in stripped_kwargs.items() if k not in static_kwargs}

        readonly = mutable_sel is None

        if readonly and not kwargs and len(refs) == 1 and refs[0].kind == "arg":
            ref = refs[0]
            pure = make_pure_readonly_single_positional(fn, ref)
            checkpointed = jax.checkpoint(
                pure,
                prevent_cse=prevent_cse,
                policy=policy,
                static_argnums=static_argnums,
            )
            locator = int(ref.locator)
            other_args = stripped_args[:locator] + stripped_args[locator + 1 :]
            return checkpointed(ref.state, *other_args)

        if static_kwargs:
            _fn = fn
            _refs = refs
            _static = static_kwargs

            def pure_with_static(
                states: tuple[object, ...],
                stripped_args_: tuple[object, ...],
                stripped_kwargs_: dict[str, object],
            ) -> object:
                """Pure ``(states, stripped_args, stripped_kwargs) -> output`` adapter.

                Re-injects the captured ``static_kwargs`` (those filtered
                out before ``jax.checkpoint`` saw them) at call time, then
                routes through either the readonly or read/write body
                helper depending on the wrapper's mode.

                Args:
                    states: States value consumed by this operation.
                    stripped_args_: Stripped args  value consumed by this operation.
                    stripped_kwargs_: Stripped kwargs  value consumed by this operation.

                Returns:
                    Result described by this helper.
                """
                merged_kwargs = {**stripped_kwargs_, **_static}
                if readonly:
                    return _run_readonly_body(
                        _fn,
                        tuple(r.gdef for r in _refs),
                        tuple(r.module for r in _refs),
                        _refs,
                        states,
                        stripped_args_,
                        merged_kwargs,
                        None,
                    )
                return _run_pure_body(
                    _fn,
                    tuple(r.gdef for r in _refs),
                    tuple(r.module for r in _refs),
                    _refs,
                    states,
                    stripped_args_,
                    merged_kwargs,
                    None,
                )

            pure = pure_with_static
        else:
            pure = make_pure_readonly(fn, refs) if readonly else make_pure(fn, refs)

        checkpointed = jax.checkpoint(
            pure,
            prevent_cse=prevent_cse,
            policy=policy,
            static_argnums=static_argnums,
        )
        states_in = tuple(r.state for r in refs)
        if readonly:
            return checkpointed(states_in, stripped_args, dynamic_kwargs)
        out, new_states = checkpointed(states_in, stripped_args, dynamic_kwargs)
        apply_mutations(refs, list(new_states), mutable_sel)
        return out

    return cast(F, wrapped)


def _remat_module_class(
    cls: type,
    *,
    mutable: SelectorSugar,
    prevent_cse: bool,
    policy: Callable[..., bool] | None,
    static_argnums: int | tuple[int, ...],
) -> type:
    """Build (and cache) a checkpointed subclass of ``cls``.

    Creates a subclass whose ``forward`` is permanently wrapped in
    :func:`remat`, names it ``Remat<Qualname>`` (with dotted qualnames
    flattened to underscores), and injects it into the parent module's
    namespace under the same name so :func:`importlib.import_module` /
    ``resolve_class`` can round-trip it through
    :func:`~spectrax.export` and :func:`~spectrax.bind`.

    The cache key is the tuple ``(cls, hashable(mutable), prevent_cse,
    policy, static_argnums)`` — calling :func:`remat` twice with the
    same settings on the same class returns the *same* subclass, which
    matters for pytree-registration identity and class-equality checks
    inside the graph layer.

    On :class:`ImportError` during parent-module registration (e.g. the
    class was defined in a script that is no longer importable as a
    module), the helper silently swallows the error and the new class
    is only usable in-process.

    Args:
        cls: The :class:`~spectrax.Module` subclass to checkpoint.
        mutable: Selector controlling writable collections.
        prevent_cse: Forwarded to :func:`jax.checkpoint`.
        policy: Forwarded to :func:`jax.checkpoint`.
        static_argnums: Forwarded to :func:`jax.checkpoint`.

    Returns:
        A subclass of ``cls`` whose ``forward`` is wrapped in
        :func:`remat`. Cached so repeated calls return the same class.
    """
    cache_key = (cls, _hashable_cache_key(mutable), prevent_cse, policy, static_argnums)
    cached = _REMAT_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parent_forward = cls.forward
    new_qualname = f"Remat{cls.__qualname__.replace('.', '_')}"

    cached_remat = remat(
        parent_forward,
        mutable=mutable,
        prevent_cse=prevent_cse,
        policy=policy,
        static_argnums=static_argnums,
    )

    class RematSubclass(cls):
        """Subclass of ``cls`` whose ``forward`` is permanently wrapped in :func:`jax.checkpoint`.

        Constructed once per ``(cls, settings)`` combination by
        :func:`_remat_module_class` and reused on every subsequent call.
        Behaves exactly like ``cls`` except that ``forward`` runs under
        gradient checkpointing.
        """

        def forward(self, *args: object, **kwargs: object) -> object:
            """Invoke the parent ``forward`` under the cached :func:`remat` wrapper.

            ``self`` is passed as the first positional argument so the
            wrapper's
            :func:`~spectrax.transforms.split_merge.locate_and_strip`
            can discover the module instance and thread its
            declared-mutable collections through :func:`jax.checkpoint`.

            Args:
                *args: Additional positional arguments forwarded to the wrapped callable or backend.
                **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            return cached_remat(self, *args, **kwargs)

    RematSubclass.__name__ = new_qualname
    RematSubclass.__qualname__ = new_qualname
    RematSubclass.__module__ = cls.__module__

    try:
        parent_module = importlib.import_module(cls.__module__)
        if not hasattr(parent_module, new_qualname):
            setattr(parent_module, new_qualname, RematSubclass)
    except ImportError:
        pass

    _REMAT_CLASS_CACHE[cache_key] = RematSubclass
    return RematSubclass

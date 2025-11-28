"""Utilities for propagating static context into JAX JIT traces.

The functions in this module make it easy to set a lightweight
``NamedTuple`` with compile-time only information and retrieve it inside
JAX-traced code. Because the value is read while tracing, the data is
treated as a constant by JAX which allows the XLA compiler to eliminate
dead code paths that depend on it.

Typical usage::

    from easydel.utils.jit_context import (
        CompilationContext,
        get_jit_context,
        jit_context,
    )

    def forward(x):
        ctx = get_jit_context()
        if ctx.phase == "prefill":
            ...  # logic compiled only for the prefill branch
        else:
            ...  # decode-only code
        return x

    compiled = jax.jit(forward)
    with jit_context(CompilationContext(name="prefill", phase="prefill")):
        compiled(jnp.ones(...))

    # Changing the context will trigger a re-trace with new constants.

Only immutable data (``NamedTuple`` instances or similar tuples) is
accepted to avoid accidental mutations that would break JIT semantics.
"""

from __future__ import annotations

import contextlib
import contextvars
import typing as tp

import jax
from jax._src.lib import xla_client

config_ext = xla_client._xla.config


class CompilationContext(tp.NamedTuple):
    """Basic context payload for compilation-time switches.

    Attributes:
        name: Human readable identifier for debugging/logging.
        phase: Optional phase label such as ``"prefill"`` or ``"decode"``.
        metadata: Arbitrary static metadata. Anything stored here must be
            immutable because the object will be captured during JIT tracing.
        flags: Frozen set of feature flags. Using a ``frozenset`` ensures
            hashability which helps JAX cache compiled programs.
    """

    name: str
    phase: str | None = None
    metadata: tp.Mapping[str, tp.Any] | None = None
    flags: frozenset[str] = frozenset()


ContextPayload = tp.TypeVar("ContextPayload", bound=tuple)

_context_var: contextvars.ContextVar[ContextPayload | None] = contextvars.ContextVar("easydeL_jit_context", default=None)
_context_generation = 0

try:
    _JAX_USER_CONTEXT = jax.make_user_context(default_value=None)
except AttributeError:  # pragma: no cover - older JAX fallback
    _JAX_USER_CONTEXT = None

_JIT_CONTEXT_STATE = getattr(_JAX_USER_CONTEXT, "_obj", None) if _JAX_USER_CONTEXT is not None else None
if _JIT_CONTEXT_STATE is None:
    _JIT_CONTEXT_STATE = config_ext.Config("easydeL_jit_context_state", None, include_in_jit_key=True)


def _get_state_value() -> ContextPayload | None:
    value = _context_var.get()
    return None if value is None else tp.cast(ContextPayload, value)


@contextlib.contextmanager
def jit_context(context: ContextPayload | None) -> tp.Iterator[ContextPayload | None]:
    """Context manager that sets the active JIT context.

    Args:
        context: ``NamedTuple`` payload that should be visible while tracing
            JAX functions. Passing ``None`` temporarily clears the context.

    Yields:
        The context that was set for convenience.
    """

    if context is not None and not isinstance(context, tuple):
        raise TypeError("jit_context expects a NamedTuple (or tuple) payload.")

    global _context_generation
    token = _context_var.set(context)
    prev_generation = _context_generation
    prev_state = _JIT_CONTEXT_STATE.swap_local(context) if _JIT_CONTEXT_STATE else None
    try:
        yield context
    finally:
        if prev_generation == _context_generation:
            _context_var.reset(token)
            if _JIT_CONTEXT_STATE:
                _JIT_CONTEXT_STATE.set_local(prev_state)
        else:  # context was cleared while active
            _context_var.set(None)
            if _JIT_CONTEXT_STATE:
                _JIT_CONTEXT_STATE.set_local(config_ext.unset)


def get_jit_context(expected_type: type[ContextPayload] | None = None) -> ContextPayload:
    """Return the current context or raise if unavailable.

    Args:
        expected_type: Optional ``NamedTuple`` class used to assert the type of
            the stored context. When ``None`` no additional checking is
            performed.

    Returns:
        The active context payload.

    Raises:
        RuntimeError: If no context is currently set.
        TypeError: If ``expected_type`` is provided but the context is of a
            different type.
    """

    context = _get_state_value()
    if context is None:
        raise RuntimeError("JIT context is not available. Use `jit_context(...)` before tracing.")
    if expected_type is not None and not isinstance(context, expected_type):
        raise TypeError(f"Active JIT context is {type(context)!r}, expected {expected_type!r}")
    return context


def peek_jit_context(default: ContextPayload | None = None) -> ContextPayload | None:
    """Return the context if present, otherwise ``default``.

    This helper is useful inside utilities that can operate without forcing an
    active context.
    """

    context = _get_state_value()
    return context if context is not None else default


def clear_jit_context() -> None:
    """Clear any active context without leaving a ``with`` block.

    This is mostly intended for testing helpers.
    """

    global _context_generation
    _context_var.set(None)
    _context_generation += 1
    if _JIT_CONTEXT_STATE:
        _JIT_CONTEXT_STATE.set_local(config_ext.unset)


def is_jit_context_available() -> bool:
    """Check whether a context payload is currently set."""

    return _get_state_value() is not None


__all__ = (
    "CompilationContext",
    "clear_jit_context",
    "get_jit_context",
    "is_jit_context_available",
    "jit_context",
    "peek_jit_context",
)

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from easydel.utils.jit_context import (
    CompilationContext,
    clear_jit_context,
    get_jit_context,
    is_jit_context_available,
    jit_context,
    peek_jit_context,
)


class ExampleContext(NamedTuple):
    name: str
    phase: str


class OtherContext(NamedTuple):
    tag: str


def test_jit_context_roundtrip():
    with jit_context(ExampleContext(name="prefill", phase="prefill")) as ctx:
        assert ctx.name == "prefill"
        active = get_jit_context(ExampleContext)
        assert active.phase == "prefill"
        assert is_jit_context_available()

    assert not is_jit_context_available()
    with pytest.raises(RuntimeError):
        get_jit_context(ExampleContext)


def test_jit_context_type_mismatch():
    with jit_context(ExampleContext(name="prefill", phase="prefill")):
        with pytest.raises(TypeError):
            get_jit_context(OtherContext)


def test_peek_and_clear():
    default = ExampleContext(name="default", phase="decode")
    assert peek_jit_context(default) is default

    with jit_context(ExampleContext(name="custom", phase="prefill")):
        assert peek_jit_context() == ExampleContext(name="custom", phase="prefill")
        clear_jit_context()
        assert not is_jit_context_available()
        assert peek_jit_context(default) is default


@jax.jit
def _contextual_add(x):
    ctx = get_jit_context(CompilationContext)
    delta = 1 if ctx.phase == "prefill" else 2
    return x + delta


def test_jax_jit_reads_context():
    with jit_context(CompilationContext(name="prefill", phase="prefill")):
        out = _contextual_add(jnp.array(0))
        assert int(out) == 1

    with jit_context(CompilationContext(name="decode", phase="decode")):
        out = _contextual_add(jnp.array(0))
        assert int(out) == 2


def test_array_payload_rejected():
    array_payload = jnp.array([1, 2, 3])
    with pytest.raises(TypeError):
        with jit_context(array_payload):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

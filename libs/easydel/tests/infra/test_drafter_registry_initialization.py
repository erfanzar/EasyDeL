"""Concurrent first lookup and retry semantics without importing numerical code."""

import builtins
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor

import pytest
from easydel.infra.spec_decode import DrafterRegistry


class _IsolatedRegistry(DrafterRegistry):
    _builders: tp.ClassVar[dict] = {}
    _aliases: tp.ClassVar[dict] = {}
    _builtins_loaded = False
    _builtins_loading_owner = None
    _builtins_condition = threading.Condition(threading.RLock())


@pytest.fixture
def registry():
    class Registry(_IsolatedRegistry):
        _builders: tp.ClassVar[dict] = {}
        _aliases: tp.ClassVar[dict] = {}
        _builtins_loaded = False
        _builtins_loading_owner = None

    return Registry


@pytest.mark.cpu_ok
def test_concurrent_first_queries_wait_for_registration(monkeypatch, registry):
    real_import = builtins.__import__
    started = threading.Event()
    release = threading.Event()
    querying = threading.Event()
    completed = threading.Event()

    def builder(*args):
        return None

    def delayed_import(name, *args, **kwargs):
        if name == "easydel.inference.speculative":
            started.set()
            assert not registry.has("test_builtin")  # Same-thread reentrancy must not wait on itself.
            assert release.wait(5)
            registry.register("test_builtin", "builtin_alias")(builder)
            return object()
        return real_import(name, *args, **kwargs)

    def concurrent_query():
        querying.set()
        result = registry.resolve("builtin_alias")
        completed.set()
        return result

    monkeypatch.setattr(builtins, "__import__", delayed_import)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.has, "test_builtin")
        try:
            assert started.wait(5)
            second = pool.submit(concurrent_query)
            assert querying.wait(5)
            assert not completed.wait(0.1)
        finally:
            release.set()
        assert first.result(timeout=5)
        assert second.result(timeout=5) is builder
    assert registry.names() == ["test_builtin"]


@pytest.mark.cpu_ok
def test_failed_builtin_import_remains_retryable(monkeypatch, registry):
    real_import = builtins.__import__
    attempts = []

    def builder(*args):
        return None

    def failing_import(name, *args, **kwargs):
        if name == "easydel.inference.speculative":
            attempts.append(name)
            if len(attempts) == 1:
                raise ImportError("first registration import failed")
            registry.register("test_builtin")(builder)
            return object()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ImportError, match="first registration import failed"):
        registry.resolve("test_builtin")
    assert registry.resolve("test_builtin") is builder
    assert len(attempts) == 2

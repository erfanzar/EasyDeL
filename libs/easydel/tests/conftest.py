"""Test-suite compatibility hooks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


def _patch_removed_jax_config_flags() -> None:
    try:
        import jax
    except Exception:
        return
    update = getattr(jax.config, "update", None)
    if update is None or getattr(update, "_easydel_tests_removed_flag_patch", False):
        return

    removed_flags = {"jax_pmap_shmap_merge"}

    def _patched_update(name, value):
        if name in removed_flags:
            return None
        return update(name, value)

    _patched_update._easydel_tests_removed_flag_patch = True  # type: ignore[attr-defined]
    jax.config.update = _patched_update


_patch_removed_jax_config_flags()


# --- Computation tests run on an accelerator (XERXES.md golden rule 5) ---------------------------
# A test file that imports an array / accelerator library is a computation test and must run on a
# TPU or GPU JAX backend; a CPU run of it validates nothing. Pure-Python tests (parsers, configs,
# orchestration) either import none of these libraries or carry ``@pytest.mark.cpu_ok``.
_ARRAY_LIBRARY_IMPORT = re.compile(r"^\s*(?:import|from)\s+(?:jax|torch|spectrax|ejkernel|triton|tilelang|flax)\b", re.M)
_ACCELERATOR_BACKENDS = frozenset({"tpu", "gpu", "cuda", "rocm"})
_ARRAY_LIBRARY_FILES: dict[str, bool] = {}


def _imports_array_library(path) -> bool:
    key = str(path)
    if key not in _ARRAY_LIBRARY_FILES:
        try:
            _ARRAY_LIBRARY_FILES[key] = bool(_ARRAY_LIBRARY_IMPORT.search(Path(key).read_text(encoding="utf-8")))
        except OSError:
            _ARRAY_LIBRARY_FILES[key] = True
    return _ARRAY_LIBRARY_FILES[key]


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "cpu_ok: the test does no array computation, so it may run on a CPU backend")


def pytest_collection_modifyitems(config, items) -> None:
    if config.option.collectonly:
        return
    computation = [
        item for item in items if item.get_closest_marker("cpu_ok") is None and _imports_array_library(item.path)
    ]
    if not computation:
        return
    import jax

    try:
        backend = jax.default_backend()
    except Exception as err:  # e.g. the TPU is held by another process (single libtpu lock)
        backend = f"unavailable: {type(err).__name__}: {str(err).splitlines()[0][:160]}"
    if backend in _ACCELERATOR_BACKENDS:
        return
    raise pytest.UsageError(
        f"{len(computation)} computation test(s) collected on the '{backend}' JAX backend (first: "
        f"{computation[0].nodeid}). Computation tests must run on TPU/GPU, e.g.\n"
        "  env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu uv run pytest ...\n"
        "Mark a test that does no array computation with @pytest.mark.cpu_ok."
    )


# Runtime complement to the collection guard above: a test that reaches JAX only through another
# package (no direct array-library import line) escapes the import scan. Listen during setup/call
# without initializing the backend first (pure-Python tests must not acquire the TPU lock).
# A CPU compilation fails unless marked ``cpu_ok``. Programs already compiled earlier in the process
# (in-memory jit cache) emit no event, so this catches the first computing test rather than every one.
_COMPILE_EVENTS = frozenset(
    {"/jax/core/compile/backend_compile_duration", "/jax/compilation_cache/cache_retrieval_time_sec"}
)


def _cpu_compile_guard(item):
    if item.config.option.collectonly or item.get_closest_marker("cpu_ok") is not None:
        return (yield)
    from jax import monitoring

    compiled: list[str] = []

    def _on_duration(event, duration_secs, **kwargs):
        del duration_secs, kwargs
        if event in _COMPILE_EVENTS:
            compiled.append(event)

    monitoring.register_event_duration_secs_listener(_on_duration)
    try:
        result = yield
    finally:
        monitoring.unregister_event_duration_listener(_on_duration)
    if compiled:
        backend = sys.modules["jax"].default_backend()
        if backend not in _ACCELERATOR_BACKENDS:
            pytest.fail(
                f"{item.nodeid} compiled an XLA program on the '{backend}' JAX backend. Computation tests must run "
                "on TPU/GPU; mark a test that does no array computation with @pytest.mark.cpu_ok.",
                pytrace=False,
            )
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    return (yield from _cpu_compile_guard(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    return (yield from _cpu_compile_guard(item))

"""Relocating the compile caches off the default filesystem.

Serialized XLA executables are the cache that actually grows -- multiple GB on
a host that compiles many shapes. On machines where ``$HOME``, ``/`` and
``/tmp`` share one filesystem, that fills the root volume and takes the whole
machine down. Operators must therefore be able to move this directory, and the
variable they set must be the one that wins:

* ``EJKERNEL_CACHE_ROOT`` relocates the base cache root.
* ``EJKERNEL_COMPILE_CACHE_DIR`` (legacy: ``COMPILE_FUNC_DIR``) relocates the
  serialized-executable directory directly.
* An explicit ``JAX_COMPILATION_CACHE_DIR`` must not be silently overwritten,
  which previously sent JAX's cache back under the default directory with no
  warning and no way to observe it from the environment.

Note that ``EJKERNEL_PERSISTENT_CACHE_DIR`` governs autotuned kernel *configs*,
a different and much smaller cache; it deliberately does not move these.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib
from pathlib import Path

import pytest
from ejkernel.callib._utils import get_cache_dir


def test_cache_root_env_relocates_base_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EJKERNEL_CACHE_ROOT", str(tmp_path))
    resolved = get_cache_dir()
    assert resolved == tmp_path / "ejkernel-cache"
    assert resolved.is_dir()


def test_cache_root_keeps_app_name_leaf_so_apps_stay_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("EJKERNEL_CACHE_ROOT", str(tmp_path))
    assert get_cache_dir("easydel") == tmp_path / "easydel"
    assert get_cache_dir("ejkernel-cache") == tmp_path / "ejkernel-cache"


def test_cache_root_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("EJKERNEL_CACHE_ROOT", "~/ejk-cache-root-test")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert get_cache_dir() == tmp_path / "ejk-cache-root-test" / "ejkernel-cache"


def test_unset_cache_root_uses_platform_default(tmp_path, monkeypatch):
    monkeypatch.delenv("EJKERNEL_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert get_cache_dir() == tmp_path / ".cache" / "ejkernel-cache"


@pytest.mark.parametrize("env_name", ["EJKERNEL_COMPILE_CACHE_DIR", "COMPILE_FUNC_DIR"])
def test_compile_dir_env_relocates_serialized_executables(tmp_path, monkeypatch, env_name):
    """Both the documented name and the legacy one must relocate the big cache."""
    monkeypatch.delenv("EJKERNEL_COMPILE_CACHE_DIR", raising=False)
    monkeypatch.delenv("COMPILE_FUNC_DIR", raising=False)
    monkeypatch.setenv(env_name, str(tmp_path / "compiled"))

    module = importlib.reload(importlib.import_module("ejkernel.callib._ejit"))
    try:
        assert module.COMPILE_FUNC_DIR == tmp_path / "compiled"
        # The env-fingerprint namespace still applies underneath the new root.
        assert module.get_effective_compile_dir().parent == tmp_path / "compiled"
    finally:
        importlib.reload(module)


def test_documented_name_wins_over_legacy_name(tmp_path, monkeypatch):
    monkeypatch.setenv("EJKERNEL_COMPILE_CACHE_DIR", str(tmp_path / "documented"))
    monkeypatch.setenv("COMPILE_FUNC_DIR", str(tmp_path / "legacy"))

    module = importlib.reload(importlib.import_module("ejkernel.callib._ejit"))
    try:
        assert module.COMPILE_FUNC_DIR == tmp_path / "documented"
    finally:
        monkeypatch.delenv("EJKERNEL_COMPILE_CACHE_DIR", raising=False)
        monkeypatch.delenv("COMPILE_FUNC_DIR", raising=False)
        importlib.reload(module)


def test_explicit_jax_cache_dir_is_not_overwritten(tmp_path, monkeypatch):
    """The regression: an operator-set JAX cache dir must survive configuration."""
    import ejkernel.callib._ejit as ej
    import jax

    requested = str(tmp_path / "jax-cache")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", requested)
    monkeypatch.setattr(ej, "_JAX_PERSISTENT_CACHE_CONFIGURED", False)

    ej._configure_jax_persistent_cache()

    assert jax.config.jax_compilation_cache_dir == requested
    assert Path(jax.config.jax_compilation_cache_dir) != ej.get_effective_compile_dir()


def test_without_explicit_jax_cache_dir_the_namespaced_dir_is_used(monkeypatch):
    import ejkernel.callib._ejit as ej
    import jax

    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(ej, "_JAX_PERSISTENT_CACHE_CONFIGURED", False)

    ej._configure_jax_persistent_cache()

    assert jax.config.jax_compilation_cache_dir == str(ej.get_effective_compile_dir())

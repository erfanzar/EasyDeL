"""Guards on the ejit compile save/load path.

Regression tests for the compile-cache poisoning failure mode: serialized XLA
executables are only valid for the exact toolchain/machine that produced them,
and entries at/over protobuf's 2 GiB limit fail metadata parsing on read and
can abort the reader natively (SIGILL) — long after any Python ``except``.

The hardened path must therefore:
* namespace the on-disk cache by an environment fingerprint (foreign entries
  become invisible instead of fatal),
* refuse to write or read entries above the safe size ceiling, and
* write atomically so readers never see partial blobs.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import pytest

import ejkernel.callib._ejit as ej


def _tiny_compiled():
    def f(x):
        return x * 2.0 + 1.0

    x = jnp.arange(4, dtype=jnp.float32)
    return jax.jit(f).lower(x).compile(), x


def test_effective_dir_is_env_namespaced():
    eff = ej.get_effective_compile_dir()
    assert eff.name.startswith("env-")
    assert eff.parent == ej.COMPILE_FUNC_DIR
    assert ej._env_fingerprint() == ej._env_fingerprint(), "fingerprint must be deterministic"
    assert eff.name == f"env-{ej._env_fingerprint()}"


def test_save_is_atomic_and_round_trips(tmp_path):
    compiled, x = _tiny_compiled()
    ej.save_compiled_fn(tmp_path, compiled, prefix="t")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["t-compiled.executable"], f"expected one final file, got {files}"
    assert not any(".tmp." in name for name in files), "no temp-file leftovers"

    loaded = ej.load_compiled_fn(tmp_path, prefix="t")
    assert jnp.allclose(loaded(x), compiled(x))


def test_save_refuses_oversized_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(ej, "MAX_CACHE_ENTRY_BYTES", 8)
    compiled, _ = _tiny_compiled()
    with pytest.warns(UserWarning, match="exceeds the"):
        ej.save_compiled_fn(tmp_path, compiled, prefix="big")
    assert list(tmp_path.iterdir()) == [], "oversized entry must not be written"


def test_load_refuses_oversized_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(ej, "MAX_CACHE_ENTRY_BYTES", 1024)
    bomb = tmp_path / "big-compiled.executable"
    with open(bomb, "wb") as f:
        f.truncate(4096)
    with pytest.warns(UserWarning, match="exceeds the"), pytest.raises(ValueError, match="Refusing to deserialize"):
        ej.load_compiled_fn(tmp_path, prefix="big")


def test_preload_skips_oversized_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(ej, "MAX_CACHE_ENTRY_BYTES", 1024)
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    entry = tmp_path / "somekey"
    entry.mkdir()
    with open(entry / ej.COMPILED_FILE_NAME, "wb") as f:
        f.truncate(4096)
    before = dict(ej.COMPILED_CACHE)
    with pytest.warns(UserWarning, match="exceeds the"):
        ej.load_cached_functions(verbose=True)
    assert ej.COMPILED_CACHE == before, "oversized entry must be skipped, not deserialized"


def test_configure_purges_oversized_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(ej, "MAX_CACHE_ENTRY_BYTES", 1024)
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "_JAX_PERSISTENT_CACHE_CONFIGURED", False)
    bomb = tmp_path / "jit__whatever-deadbeef-cache"
    with open(bomb, "wb") as f:
        f.truncate(4096)
    small = tmp_path / "jit__ok-cafe-cache"
    small.write_bytes(b"ok")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ej._configure_jax_persistent_cache()
    assert not bomb.exists(), "oversized cache entry must be swept on configuration"
    assert small.exists(), "normal entries stay"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

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

import pickle
import warnings

import ejkernel.callib._ejit as ej
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec


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
    assert jnp.array_equal(loaded(x), jnp.array([1.0, 3.0, 5.0, 7.0]))


@pytest.mark.parametrize("no_arguments", [False, True])
def test_round_trip_preserves_nondefault_device(tmp_path, no_arguments):
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local devices")
    device = jax.local_devices()[-1]
    with jax.default_device(device):
        x = jnp.arange(4, dtype=jnp.float32)
        if no_arguments:
            compiled = jax.jit(lambda: jnp.arange(4, dtype=jnp.float32) * 2 + 1).lower().compile()
            args = ()
        else:
            compiled = jax.jit(lambda x: x * 2 + 1).lower(x).compile()
            args = (x,)
    ej.save_compiled_fn(tmp_path, compiled)
    # Loading outside the default-device context must preserve the assignment.
    loaded = ej.load_compiled_fn(tmp_path)
    result = loaded(*args)
    assert result.devices() == {device}
    assert jnp.array_equal(result, jnp.array([1.0, 3.0, 5.0, 7.0]))


def test_round_trip_preserves_noncanonical_multidevice_assignment(tmp_path):
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local devices")
    # Reverse backend order and use only two devices even on a larger host.
    devices = tuple(reversed(jax.local_devices()[:2]))
    mesh = Mesh(np.array(devices, dtype=object), ("shard",))
    sharding = NamedSharding(mesh, PartitionSpec("shard"))
    host_x = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32)
    x = jax.device_put(host_x, sharding)

    def f(x):
        # Unlike uniform elementwise math, global positions expose shard swaps.
        position = jnp.arange(x.shape[0], dtype=x.dtype)
        return x * (position + 1) + position * position

    compiled = jax.jit(f, in_shardings=sharding, out_shardings=sharding).lower(x).compile()
    ej.save_compiled_fn(tmp_path, compiled)
    loaded = ej.load_compiled_fn(tmp_path)
    result = loaded(x)

    position = np.arange(host_x.size, dtype=np.float32)
    expected = host_x * (position + 1) + position * position
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert result.devices() == set(devices)
    expected_indices = {devices[0]: (slice(0, 4),), devices[1]: (slice(4, 8),)}
    assert result.sharding.devices_indices_map(result.shape) == expected_indices
    for shard in result.addressable_shards:
        assert shard.index == expected_indices[shard.device]
        np.testing.assert_array_equal(np.asarray(shard.data), expected[expected_indices[shard.device]])


def test_preload_round_trip(tmp_path, monkeypatch):
    compiled, x = _tiny_compiled()
    ej.save_compiled_fn(tmp_path / "key", compiled)
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    ej.load_cached_functions(verbose=False)
    assert jnp.array_equal(ej.COMPILED_CACHE["key"](x), jnp.array([1.0, 3.0, 5.0, 7.0]))


def test_ejit_disk_hit_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    traces = []

    def f(x):
        traces.append(None)
        return x * 2 + 1

    x = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(ej.ejit(f)(x), jnp.array([1.0, 3.0, 5.0, 7.0]))
    assert len(list(tmp_path.glob("*/compiled.executable"))) == 1
    ej.COMPILED_CACHE.clear()
    jax.clear_caches()
    assert jnp.array_equal(ej.ejit(f)(x), jnp.array([1.0, 3.0, 5.0, 7.0]))
    assert len(traces) == 1, "disk hit must execute the saved artifact without retracing"


def test_load_rejects_legacy_assignmentless_entry(tmp_path):
    # No native deserialization is safe without knowing the device assignment.
    with open(tmp_path / ej.COMPILED_FILE_NAME, "wb") as f:
        pickle.dump((b"not a native executable", None, None), f)
    with pytest.raises(ValueError, match="Legacy ejit cache lacks execution-device metadata"):
        ej.load_compiled_fn(tmp_path)


@pytest.mark.parametrize("mismatch", ["environment", "devices", "version"])
def test_load_rejects_incompatible_metadata(tmp_path, mismatch):
    compiled, _ = _tiny_compiled()
    ej.save_compiled_fn(tmp_path, compiled)
    filename = tmp_path / ej.COMPILED_FILE_NAME
    with open(filename, "rb") as f:
        _, in_tree, out_tree, metadata = pickle.load(f)
    if mismatch == "devices":
        metadata["devices"] = ((-1, -1, "nonexistent device"),)
        message = "execution devices are unavailable"
    elif mismatch == "environment":
        metadata["environment"] = "different toolchain"
        message = "Incompatible ejit compilation environment"
    else:
        metadata["version"] = -1
        message = "Unsupported ejit compiled-cache metadata version"
    with open(filename, "wb") as f:
        # An invalid native payload proves metadata is checked before loading.
        pickle.dump((b"not a native executable", in_tree, out_tree, metadata), f)
    with pytest.raises(ValueError, match=message):
        ej.load_compiled_fn(tmp_path)


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

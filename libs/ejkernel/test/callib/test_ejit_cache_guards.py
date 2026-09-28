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
import subprocess
import sys
import warnings
from collections import namedtuple
from dataclasses import dataclass

import ejkernel.callib._ejit as ej
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
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


_DISK_HIT_TRACES: list[None] = []


def test_ejit_disk_hit_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    _DISK_HIT_TRACES.clear()

    def f(x):
        # Module-level counter: a captured list would be an unhashable closure cell and
        # (correctly) opt this function out of the persistent cache.
        _DISK_HIT_TRACES.append(None)
        return x * 2 + 1

    x = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(ej.ejit(f)(x), jnp.array([1.0, 3.0, 5.0, 7.0]))
    assert len(list(tmp_path.glob("*/compiled.executable"))) == 1
    ej.COMPILED_CACHE.clear()
    jax.clear_caches()
    assert jnp.array_equal(ej.ejit(f)(x), jnp.array([1.0, 3.0, 5.0, 7.0]))
    assert len(_DISK_HIT_TRACES) == 1, "disk hit must execute the saved artifact without retracing"


def test_ejit_static_arg_values_are_keyed(tmp_path, monkeypatch):
    """Different static-arg values must not share a cached executable.

    Regression test: the cache key used to include only the *type* of
    non-array leaves, so ``f(x, 2.0)`` and ``f(x, 3.0)`` collided and the
    second call replayed the executable compiled for the first.
    """
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)

    def f(x, scale):
        return x * scale

    f_jit = ej.ejit(f, static_argnums=(1,))
    x = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(f_jit(x, 2.0), x * 2.0)
    assert jnp.array_equal(f_jit(x, 3.0), x * 3.0)
    # First call again: must still be the scale=2.0 executable.
    assert jnp.array_equal(f_jit(x, 2.0), x * 2.0)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("first", "second"),
    [
        (lax.Precision.DEFAULT, lax.Precision.HIGHEST),
        (jnp.float32, jnp.bfloat16),
        (np.dtype("float32"), np.dtype("float16")),
        (np.int32(1), np.int32(2)),
        ((16, 32), (16, 64)),
        (namedtuple("Tiles", "q kv")(16, 32), namedtuple("Tiles", "q kv")(16, 64)),
        (frozenset({"long", "short"}), frozenset({"long", "full"})),
    ],
    ids=("enum", "dtype-class", "numpy-dtype", "numpy-scalar", "tuple", "namedtuple", "frozenset"),
)
def test_static_signature_distinguishes_values_without_array_computation(first, second):
    assert ej._get_args_signature((), {"config": first}, static_argnames=("config",)) != ej._get_args_signature(
        (), {"config": second}, static_argnames=("config",)
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("first, second", [(2, 3), (2.0, 3.0), (False, True), (2j, 3j)])
def test_dynamic_signature_scalar_values_share_cache_key(first, second):
    assert ej._get_args_signature((1, first), {}) == ej._get_args_signature((1, second), {})
    assert ej._get_args_signature((), {"n": first}) == ej._get_args_signature((), {"n": second})


@pytest.mark.cpu_ok
def test_static_signature_scalar_values_have_distinct_cache_keys():
    assert ej._get_args_signature((2,), {}, static_argnums=(0,)) != ej._get_args_signature((3,), {}, static_argnums=(0,))
    assert ej._get_args_signature((), {"n": 2}, static_argnames=("n",)) != ej._get_args_signature(
        (), {"n": 3}, static_argnames=("n",)
    )


@pytest.mark.cpu_ok
def test_dynamic_signature_int_and_float_have_distinct_cache_keys():
    assert ej._get_args_signature((2,), {}) != ej._get_args_signature((2.0,), {})
    assert ej._get_args_signature((2,), {}) != ej._get_args_signature((np.int32(2),), {})


@pytest.mark.cpu_ok
def test_static_signature_dataclass_and_unhashable_values():
    from ejkernel.ops.utils.datacarrier import BwdParams, FwdParams

    @dataclass
    class Params:
        q_blocksize: int
        window: tuple[int, int]

    first = Params(q_blocksize=64, window=(8, 16))
    second = Params(q_blocksize=128, window=(8, 16))
    assert first.__hash__ is None
    static = ("params",)
    assert ej._get_args_signature((), {"params": first}, static_argnames=static) != ej._get_args_signature(
        (), {"params": second}, static_argnames=static
    )
    assert ej._get_args_signature((), {"params": first}, static_argnames=static) == ej._get_args_signature(
        (), {"params": first}, static_argnames=static
    )
    for parameter_type in (FwdParams, BwdParams):
        assert ej._get_args_signature(
            (), {"params": parameter_type(q_blocksize=64)}, static_argnames=static
        ) != ej._get_args_signature((), {"params": parameter_type(q_blocksize=128)}, static_argnames=static)
    with pytest.raises(ValueError, match="No process-stable ejit cache signature"):
        ej._get_args_signature((), {"unknown": object()}, static_argnames=("unknown",))


class _StaticKindA:
    pass


class _StaticKindB:
    pass


@pytest.mark.cpu_ok
def test_static_signature_distinguishes_non_dtype_classes():
    # ``np.dtype`` maps every non-scalar class to ``object``; such classes must not share a key.
    static = ("kind",)
    assert ej._get_args_signature((), {"kind": _StaticKindA}, static_argnames=static) != ej._get_args_signature(
        (), {"kind": _StaticKindB}, static_argnames=static
    )
    assert ej._get_args_signature((), {"kind": dict}, static_argnames=static) != ej._get_args_signature(
        (), {"kind": list}, static_argnames=static
    )
    assert ej._get_args_signature((), {"kind": jnp.float32}, static_argnames=static) != ej._get_args_signature(
        (), {"kind": jnp.bfloat16}, static_argnames=static
    )


@pytest.mark.cpu_ok
def test_static_signature_stable_between_processes():
    source = """
import jax.numpy as jnp
from jax import lax
from ejkernel.callib._ejit import _get_args_signature
from ejkernel.ops.utils.datacarrier import FwdParams
print(_get_args_signature((), {
    'params': FwdParams(q_blocksize=64, kv_blocksize=128),
    'dtype': jnp.bfloat16,
    'precision': lax.Precision.HIGHEST,
}, static_argnames=('params', 'dtype', 'precision')))
"""
    first = subprocess.check_output([sys.executable, "-c", source], text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", source], text=True).strip()
    assert first == second
    assert "0x" not in first


def _make_scaler(scale):
    def scaled(x):
        return x * scale

    return scaled


@pytest.mark.cpu_ok
def test_closure_signature_distinguishes_captured_values():
    assert ej._get_closure_signature(_make_scaler(2.0)) != ej._get_closure_signature(_make_scaler(3.0))
    assert ej._get_closure_signature(_make_scaler(2.0)) == ej._get_closure_signature(_make_scaler(2.0))
    assert ej._get_closure_signature(lambda x: x) == ()


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "captured",
    [[1, 2], {"a": 1}, np.arange(3), object()],
    ids=("list", "dict", "numpy-array", "address-repr-object"),
)
def test_closure_signature_rejects_unstable_captures(captured):
    def uses_capture(x):
        return x + len(str(captured))

    with pytest.raises(ValueError):
        ej._get_closure_signature(uses_capture)


@pytest.mark.cpu_ok
def test_closure_signature_rejects_unbound_cell():
    def outer():
        def inner():
            return late

        cell_fn = inner
        late = 1  # bound after ``inner`` is created but before the call below
        del late  # leaves the cell empty
        return cell_fn

    with pytest.raises(ValueError, match="unbound"):
        ej._get_closure_signature(outer())


def test_ejit_closure_values_are_keyed(tmp_path, monkeypatch):
    """Identical-source functions with different captured values must not share an executable."""
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    x = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(ej.ejit(_make_scaler(2.0))(x), x * 2.0)
    assert jnp.array_equal(ej.ejit(_make_scaler(3.0))(x), x * 3.0)
    assert len(list(tmp_path.glob("*/compiled.executable"))) == 2


def _make_default_scaler(scale):
    def scaled(x, factor=scale):
        return x * factor

    return scaled


def _make_kwdefault_scaler(scale):
    def scaled(x, *, factor=scale):
        return x * factor

    return scaled


@pytest.mark.cpu_ok
def test_defaults_signature_distinguishes_default_values():
    """Values captured through defaults are keyed like closure cells (r12 S06-1)."""
    for make in (_make_default_scaler, _make_kwdefault_scaler):
        assert ej._get_defaults_signature(make(2.0)) != ej._get_defaults_signature(make(3.0))
        assert ej._get_defaults_signature(make(2.0)) == ej._get_defaults_signature(make(2.0))
    assert ej._get_defaults_signature(lambda x: x) == ()


@pytest.mark.cpu_ok
def test_defaults_signature_keys_module_functions_and_rejects_local_ones():
    """A module-level function default (checkpoint policy) keeps the cache; a local function disables it."""

    def with_policy(x, policy=jax.checkpoint_policies.nothing_saveable):
        return x

    sig = ej._get_defaults_signature(with_policy)
    assert sig and "0x" not in str(sig)

    def local_helper(x):
        return x

    def with_local(x, fn=local_helper):
        return fn(x)

    with pytest.raises(ValueError):
        ej._get_defaults_signature(with_local)


def test_ejit_default_values_are_keyed(tmp_path, monkeypatch):
    """Identical-source factory functions differing only in defaults must not share an executable."""
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    x = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(ej.ejit(_make_default_scaler(2.0))(x), x * 2.0)
    assert jnp.array_equal(ej.ejit(_make_default_scaler(3.0))(x), x * 3.0)
    assert jnp.array_equal(ej.ejit(_make_kwdefault_scaler(2.0))(x), x * 2.0)
    assert jnp.array_equal(ej.ejit(_make_kwdefault_scaler(3.0))(x), x * 3.0)
    assert len(list(tmp_path.glob("*/compiled.executable"))) == 4


class _Scaler:
    def __init__(self, scale):
        self.scale = scale

    def apply(self, x):
        return x * self.scale


def test_ejit_bound_methods_do_not_share_executables(tmp_path, monkeypatch):
    """Two instances' bound methods differ only in receiver state; neither may replay the other (r13 S06-1)."""
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    x = jnp.arange(4, dtype=jnp.float32)
    two, three = _Scaler(2.0), _Scaler(3.0)
    assert jnp.array_equal(ej.ejit(two.apply)(x), x * 2.0)
    assert jnp.array_equal(ej.ejit(three.apply)(x), x * 3.0)
    # Receiver state cannot be fingerprinted, so bound methods never write a persistent entry.
    assert list(tmp_path.glob("*/compiled.executable")) == []


class _CallableScaler:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


def test_ejit_partials_and_callable_instances_fall_back_instead_of_crashing(tmp_path, monkeypatch):
    """Callables without ``__name__`` must fall back to plain jit, not raise AttributeError (r14 S06-1)."""
    import functools

    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    x = jnp.arange(4, dtype=jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fallback warning is expected; an exception is not
        assert jnp.array_equal(ej.ejit(functools.partial(_Scaler(2.0).apply))(x), x * 2.0)
        assert jnp.array_equal(ej.ejit(_CallableScaler(3.0))(x), x * 3.0)
        assert jnp.array_equal(ej.ejit(_CallableScaler(5.0))(x), x * 5.0)
    assert list(tmp_path.glob("*/compiled.executable")) == []


def test_ejit_unstable_closure_skips_persistent_cache(tmp_path, monkeypatch):
    """A function capturing an unhashable value still runs correctly but writes no cache entry."""
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)
    table = [1.0, 2.0, 3.0, 4.0]

    def lookup(x):
        return x * len(table)

    x = jnp.arange(4, dtype=jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a debug log is expected, not a user-visible warning
        assert jnp.array_equal(ej.ejit(lookup)(x), x * 4.0)
    assert list(tmp_path.glob("*/compiled.executable")) == []


def test_ejit_jit_options_are_keyed(tmp_path, monkeypatch):
    """Options that change the executable (here ``inline``/``keep_unused``) must change the key."""
    monkeypatch.setattr(ej, "EFFECTIVE_COMPILE_FUNC_DIR", tmp_path)
    monkeypatch.setattr(ej, "COMPILED_CACHE", {})
    monkeypatch.setattr(ej, "ECACHE_COMPILES", True)
    monkeypatch.setattr(ej, "RECOMPILE_FORCE", False)

    def f(x, y):
        return x * 2.0

    x = jnp.arange(4, dtype=jnp.float32)
    ej.ejit(f, keep_unused=False)(x, x)
    ej.ejit(f, keep_unused=True)(x, x)
    assert len(list(tmp_path.glob("*/compiled.executable"))) == 2


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

"""Host-only ownership regressions: no JAX array allocation or numerical kernels."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import spectrax as spx
from easydel.layers.engram import GatedEngram, _layer

pytestmark = pytest.mark.cpu_ok


def _host_module(monkeypatch):
    # Only host table transport is under test. The neural projections and device
    # placeholder buffers are replaced with NumPy fixtures, not JAX computation.
    monkeypatch.setattr(_layer.jnp, "zeros", np.zeros)
    monkeypatch.setattr(_layer.jnp, "ones", np.ones)
    monkeypatch.setattr(
        GatedEngram,
        "table_sharding",
        lambda self, mesh=None: SimpleNamespace(mesh=SimpleNamespace(devices=np.array([0]))),
    )
    module = GatedEngram.__new__(GatedEngram)
    module.table_dtype = "fp8"
    module.table_memory_kind = "pinned_host"
    module.table_rows = 2
    module.embed = SimpleNamespace(features=32, weight=SimpleNamespace(), scale=SimpleNamespace())
    weight = np.arange(64, dtype=np.float32).reshape(2, 32).astype(_layer.jnp.float8_e4m3fn)
    scale = np.full((2, 1), 127, np.uint8)
    module.set_table(weight, scale)
    return module, weakref.ref(weight), weight.view(np.uint8).copy()


def test_table_is_freed_after_last_model_owner_is_collected(monkeypatch):
    module, allocation, expected = _host_module(monkeypatch)
    np.testing.assert_array_equal(module.gather_host_rows(np.array([1, 0]))[0], expected[[1, 0]])
    owner = weakref.ref(module)
    del module
    gc.collect()
    assert owner() is None
    assert allocation() is None


def test_graphdef_and_rebound_module_keep_table_until_their_last_owner(monkeypatch):
    module, allocation, expected = _host_module(monkeypatch)
    graphdef, state = spx.export(module)
    rebound = spx.bind(graphdef, state)
    module.release_host_table()
    # Explicitly releasing one owner must not invalidate another live model.
    np.testing.assert_array_equal(rebound.gather_host_rows(np.array([1]))[0], expected[[1]])
    del module, graphdef, state
    gc.collect()
    assert allocation() is not None
    np.testing.assert_array_equal(rebound.gather_host_rows(np.array([0]))[0], expected[[0]])
    del rebound
    gc.collect()
    assert allocation() is None


def test_callback_keeps_table_alive_after_source_model_is_collected(monkeypatch):
    module, allocation, expected = _host_module(monkeypatch)
    callbacks = []

    def callback(fn, shapes, ids, **kwargs):
        callbacks.append(fn)
        return fn(ids)

    monkeypatch.setattr(_layer.jax, "pure_callback", callback)
    monkeypatch.setattr(_layer, "_unpack_u8", lambda words, width: words.view(np.uint8)[..., :width])
    np.testing.assert_array_equal(module._host_gather_rows(np.array([1]))[0], expected[[1]])
    del module
    gc.collect()
    assert allocation() is not None
    # The retained callback stands in for the executable's Python callback owner.
    np.testing.assert_array_equal(callbacks[0](np.array([0]))[0].view(np.uint8), expected[[0]])
    callbacks.clear()
    gc.collect()
    assert allocation() is None

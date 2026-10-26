"""Array-free regressions for historical operation import and policy APIs."""

import importlib
import subprocess
import sys
from types import SimpleNamespace

import jax
import pytest

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "module_name,names",
    [
        ("inference_conv1d", ("ragged_causal_conv1d_op",)),
        (
            "inference_gdn",
            (
                "ragged_gated_delta_rule_decode_only",
                "ragged_gated_delta_rule_mixed_prefill",
                "ragged_gated_delta_rule_v2_op",
                "set_gdn_kernel_tile_policy",
            ),
        ),
    ],
)
def test_legacy_function_exports_keep_original_implementations(module_name, names):
    adapter = importlib.import_module(f"easydel.operations.kernels.{module_name}")
    public_kernels = importlib.import_module("ejkernel.modules.operations")
    exported = {}
    exec(f"from easydel.operations.kernels.{module_name} import *", exported)
    for name in names:
        assert name in adapter.__all__
        assert exported[name] is getattr(public_kernels, name)


@pytest.mark.parametrize("forced", [True, False])
@pytest.mark.parametrize("unsupported", [None, "selection", "cache", "mask", "weights", "window", "sinks"])
def test_legacy_sparse_policy_preserves_overrides_and_feature_guards(forced, unsupported):
    from easydel.operations.kernels.dsa_sparse_attention import use_sparse_training_attention

    # Only the public accessor exists: reaching for _softmax_aux would fail.
    layer = SimpleNamespace(
        config=SimpleNamespace(dsa_sparse_training=forced),
        get_softmax_aux=lambda: object() if unsupported == "sinks" else None,
    )
    selected = None if unsupported == "selection" else object()
    cache = object() if unsupported == "cache" else None
    mask = None if unsupported == "mask" else object()
    window = 128 if unsupported == "window" else None
    result = use_sparse_training_attention(layer, selected, cache, mask, unsupported == "weights", window)
    assert result is (forced and unsupported is None)


@pytest.mark.parametrize("forced", [True, False])
@pytest.mark.parametrize("sinks", [False, True])
def test_legacy_sparse_policy_accepts_original_private_hook(forced, sinks):
    from easydel.operations.kernels.dsa_sparse_attention import use_sparse_training_attention

    layer = SimpleNamespace(
        config=SimpleNamespace(dsa_sparse_training=forced),
        _softmax_aux=lambda: object() if sinks else None,
    )
    assert use_sparse_training_attention(layer, object(), None, object(), False, None) is (forced and not sinks)


@pytest.mark.parametrize("accessor", ["get_softmax_aux", "_softmax_aux"])
@pytest.mark.parametrize("unsupported", ["selection", "cache", "mask", "weights", "window"])
def test_legacy_sparse_policy_rejects_unsupported_inputs_before_layer_access(accessor, unsupported):
    from easydel.operations.kernels.dsa_sparse_attention import use_sparse_training_attention

    def unavailable_aux():
        raise AssertionError("Unsupported inputs must not access the layer")

    selected = None if unsupported == "selection" else object()
    cache = object() if unsupported == "cache" else None
    mask = None if unsupported == "mask" else object()
    window = 128 if unsupported == "window" else None
    for layer in (object(), SimpleNamespace(**{accessor: unavailable_aux})):
        assert use_sparse_training_attention(layer, selected, cache, mask, unsupported == "weights", window) is False


@pytest.mark.parametrize("accessor", ["get_softmax_aux", "_softmax_aux"])
def test_legacy_sparse_policy_rejects_sinks_before_config_access(accessor):
    from easydel.operations.kernels.dsa_sparse_attention import use_sparse_training_attention

    layer = SimpleNamespace(**{accessor: lambda: object()})
    assert use_sparse_training_attention(layer, object(), None, object(), False, None) is False


@pytest.mark.parametrize("public_sinks", [False, True])
def test_legacy_sparse_policy_prefers_public_accessor(public_sinks):
    from easydel.operations.kernels.dsa_sparse_attention import use_sparse_training_attention

    layer = SimpleNamespace(
        config=SimpleNamespace(dsa_sparse_training=True),
        get_softmax_aux=lambda: object() if public_sinks else None,
        _softmax_aux=lambda: None if public_sinks else object(),
    )
    assert use_sparse_training_attention(layer, object(), None, object(), False, None) is not public_sinks


@pytest.mark.parametrize(
    "backend,length_offset,expected",
    [("tpu", -1, False), ("tpu", 0, True), ("tpu", 1, True), ("cpu", 1, False), ("gpu", 1, False)],
)
@pytest.mark.parametrize("accessor", ["get_softmax_aux", "_softmax_aux"])
def test_legacy_sparse_policy_preserves_shared_auto_threshold(monkeypatch, backend, length_offset, expected, accessor):
    from easydel.layers.indexer import AUTO_SPARSE_MIN_SEQ as layer_threshold
    from easydel.operations.kernels.dsa_sparse_attention import (
        AUTO_SPARSE_MIN_SEQ,
        use_sparse_training_attention,
    )

    assert AUTO_SPARSE_MIN_SEQ == layer_threshold
    # This replaces backend discovery only; no device computation is performed.
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    layer = SimpleNamespace(config=SimpleNamespace(), **{accessor: lambda: None})
    assert (
        use_sparse_training_attention(
            layer, object(), None, object(), False, None, seq_len=layer_threshold + length_offset
        )
        is expected
    )
    assert use_sparse_training_attention(layer, object(), None, object(), False, None) is False


@pytest.mark.parametrize("first", ["easydel.operations", "easydel.layers.indexer"])
def test_legacy_threshold_import_works_with_both_cold_import_orders(first):
    code = f"""
import importlib
importlib.import_module({first!r})
from easydel.layers.indexer import AUTO_SPARSE_MIN_SEQ as shared
from easydel.operations.kernels.dsa_sparse_attention import AUTO_SPARSE_MIN_SEQ as legacy
assert shared == legacy
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("first", ["easydel.operations", "easydel.layers.indexer"])
def test_legacy_threshold_wildcard_export_preserves_public_names_with_both_cold_import_orders(first):
    code = f"""
import importlib
importlib.import_module({first!r})
adapter = importlib.import_module('easydel.operations.kernels.dsa_sparse_attention')
exported = {{}}
exec('from easydel.operations.kernels.dsa_sparse_attention import *', exported)
from easydel.layers.indexer import AUTO_SPARSE_MIN_SEQ as shared
assert 'AUTO_SPARSE_MIN_SEQ' in exported
assert exported['AUTO_SPARSE_MIN_SEQ'] == shared
# Adding __all__ must not narrow the previous implicit wildcard API.
for name, value in vars(adapter).items():
    if not name.startswith('_'):
        assert name in exported, name
        assert exported[name] is value, name
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

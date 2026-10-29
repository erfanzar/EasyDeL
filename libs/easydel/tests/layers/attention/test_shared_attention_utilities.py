"""Array-free compatibility contracts for shared attention utilities."""

import inspect
import pickle
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers import compressed_window
from easydel.layers.attention import AttentionModule, SoftCappedAttentionModule, _flexible
from easydel.layers.compressed_window import _compressor
from easydel.layers.embeddings._embeddings import Embed, _promote_dtype
from easydel.layers.norms import promote_dtype
from easydel.operations import AttentionOutput
from easydel.operations.kernels import gated_pool_compress
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("first_module", ["easydel.caching", "easydel.operations", "easydel.layers.compressed_window"])
def test_shared_operation_aliases_preserve_fresh_import_orders_and_root_indexer_exports(first_module):
    program = f"""
import importlib
importlib.import_module({first_module!r})
from easydel import layers
from easydel.layers import indexer
for name in (
    'BaseIndexer', 'BlockMaxIndexer', 'BlockMaxIndexerConfig', 'BlockTopKIndexer',
    'CompressedIndexer', 'CompressedIndexerConfig', 'IndexerConfig', 'IndexerKind',
    'IndexerOutput', 'IndexerSelection', 'SelectionSpec', 'SparseIndexer',
    'TokenIndexer', 'TokenIndexerConfig',
):
    assert name in layers.__all__
    assert name in dir(layers)
    assert getattr(layers, name) is getattr(indexer, name)
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("inexact", [None, False, True])
def test_embedding_promotion_preserves_identity_and_missing_values(inexact):
    values = [object(), None]
    assert _promote_dtype is promote_dtype
    assert inspect.signature(Embed).parameters["promote_dtype"].default is promote_dtype
    assert _promote_dtype(values, dtype=None, inexact=inexact) is values
    # No arrays are allocated: missing placeholders remain missing even with an explicit dtype.
    assert _promote_dtype([None, None], dtype="float32", inexact=inexact) == (None, None)


@pytest.mark.parametrize(
    "name",
    [
        "BlockGroupLayout",
        "block_group_layout",
        "gather_group_members",
        "overlap_group_slots",
        "last_closed_overlap",
        "refresh_group_state",
    ],
)
def test_compressor_public_exports_are_operation_owned_compatibility_aliases(name):
    canonical = getattr(gated_pool_compress, name)
    assert getattr(compressed_window, name) is canonical
    assert getattr(_compressor, name) is canonical


def test_group_layout_retains_named_fields_and_pickle_round_trip():
    layout = compressed_window.BlockGroupLayout(
        from_state="old", state_index=0, block_index=1, closed=True, entry=2, refresh=False, refresh_source=3
    )
    assert layout._asdict() == {
        "from_state": "old",
        "state_index": 0,
        "block_index": 1,
        "closed": True,
        "entry": 2,
        "refresh": False,
        "refresh_source": 3,
    }
    assert pickle.loads(pickle.dumps(layout)) == layout


@pytest.mark.parametrize("attribute", ["sinks", "softmax_aux"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_public_softmax_aux_accessor_unwraps_both_supported_attributes(attribute, wrapped):
    layer = object.__new__(AttentionModule)
    value = object()
    object.__setattr__(layer, attribute, SimpleNamespace(value=value) if wrapped else value)
    assert layer.get_softmax_aux() is value


def test_public_softmax_aux_accessor_preserves_precedence_and_legacy_overrides():
    layer = object.__new__(AttentionModule)
    assert layer.get_softmax_aux() is None
    object.__setattr__(layer, "softmax_aux", object())
    object.__setattr__(layer, "sinks", None)
    assert layer.get_softmax_aux() is None

    class CustomAttention(AttentionModule):
        def _softmax_aux(self):
            return "custom-sink"

    assert object.__new__(CustomAttention).get_softmax_aux() == "custom-sink"


class _CapOutput:
    """Host-only stand-in for a backend's output leaf."""

    def __init__(self, cap):
        self.cap = cap

    def astype(self, dtype):
        return self


class _CapOperation:
    metadata = SimpleNamespace(runtime_dtype=None)

    def get_impl_name(self):
        return "soft_cap_contract_backend"

    def __call__(self, *, logits_soft_cap, cache_view, **kwargs):
        return AttentionOutput(attention_outputs=_CapOutput(logits_soft_cap), cache_view=cache_view)


@pytest.mark.parametrize("default_cap", [None, 50.0])
@pytest.mark.parametrize("call_cap", [None, 30.0])
@pytest.mark.parametrize("method", ["forward", "__call__"])
@pytest.mark.parametrize("decode", [False, True])
def test_shared_soft_cap_reaches_actual_dispatch_in_prefill_and_decode(
    monkeypatch, default_cap, call_cap, method, decode
):
    monkeypatch.setattr(_flexible, "_attention_mesh_context", lambda config: nullcontext())
    config = EasyDeLBaseConfig(attn_mechanism="vanilla", decode_attn_mechanism=None)
    module = SoftCappedAttentionModule(config, softmax_scale=1.0, logits_soft_cap=default_cap, requires_cache=False)
    object.__setattr__(module, "impl", _CapOperation())
    object.__setattr__(module, "impl_decode", _CapOperation())
    cache = object.__new__(_flexible.TransformerCacheView) if decode else None
    output = getattr(module, method)(
        None,
        None,
        None,
        mode=common_types.MODE_DECODE if decode else None,
        cache_view=cache,
        logits_soft_cap=call_cap,
    )
    assert output.attention_outputs.cap == (default_cap if call_cap is None else call_cap)
    assert output.cache_view is cache
    assert module.logits_soft_cap == default_cap

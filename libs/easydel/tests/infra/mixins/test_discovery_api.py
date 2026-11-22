"""Public discovery and protocol helpers remain available."""

import typing as tp
from dataclasses import replace

import pytest
from easydel.caching import HybridCache, RecurrentCache, RecurrentCacheView, TransformerCache, TransformerCacheView
from easydel.infra.mixins import protocol
from easydel.infra.mixins.operation_cache import LayerOperationInfo, OperationCacheMixin, OperationsCacheInfo
from easydel.operations.requirements import CacheType, OperationRequirements

pytestmark = pytest.mark.cpu_ok


def layer(index, view, cache_type, slot=None):
    return LayerOperationInfo(
        layer_index=index,
        slot=slot,
        layer_type="full_attention" if cache_type == CacheType.TRANSFORMER else "linear_attention",
        operation_name="test",
        requirements=OperationRequirements.create("test", supported_cache=cache_type, cache_view_class=view),
        supported_cache_types=cache_type,
    )


class Model(OperationCacheMixin):
    def __init__(self, info):
        self.info = info

    def get_operations_cache_info(self):
        return self.info


def test_slot_discovery_and_compatibility_aliases():
    attention = layer(0, TransformerCacheView, CacheType.TRANSFORMER, "attention")
    recurrent = layer(0, RecurrentCacheView, CacheType.RECURRENT, "recurrent")
    info = OperationsCacheInfo(
        layers=[attention, recurrent],
        supports_ragged_pages=True,
        supports_transformer_cache=True,
    )
    model = Model(info)
    assert info.requires_ragged_pages is True
    assert info.requires_transformer_cache is True
    assert info.attention_ratio == 0.5
    assert info.get_layers_by_index(0) == [attention, recurrent]
    assert info.get_layers_by_index(1) == []
    assert info.get_layer_by_index(0) is attention
    assert model.get_layer_cache_requirements(0) is attention
    assert model.get_layer_cache_requirements(1) is None
    assert model.get_unique_cache_view_classes() == {TransformerCacheView, RecurrentCacheView}
    assert model.get_required_cache_class() is HybridCache
    assert Model(replace(info, layers=[attention])).get_required_cache_class() is TransformerCache
    assert Model(replace(info, layers=[recurrent])).get_required_cache_class() is RecurrentCache
    assert OperationsCacheInfo().attention_ratio == 1.0


def test_protocol_mode_annotations_resolve_to_runtime_mode_types():
    from spectrax.common_types import RUNTIME_MODE_TYPES

    # The protocol has overload-only __call__ declarations (typing replaces the
    # runtime attribute with its shared overload dummy). Resolve the same public
    # annotation in the module namespace rather than introspecting that dummy.
    class ModeAnnotation:
        __annotations__ = {"mode": "common_types.RUNTIME_MODE_TYPES | None"}

    resolved = tp.get_type_hints(ModeAnnotation, globalns=vars(protocol))
    assert resolved["mode"] == RUNTIME_MODE_TYPES | None


def test_return_type_adjuster_preserves_runtime_return_and_arguments():
    @protocol.return_type_adjuster(dict)
    def make_dict(key, *, value):
        return {key: value}

    assert make_dict("answer", value=42) == {"answer": 42}

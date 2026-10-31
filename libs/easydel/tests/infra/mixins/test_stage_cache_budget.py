"""Configuration and placement-control regressions; no JAX array computation."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from easydel.caching import (
    RaggedPagesCacheConfig,
    RaggedPagesCacheView,
    TransformerCacheView,
    UnifiedAttentionCacheConfig,
)
from easydel.caching.turboquant_ragged_page import cache as turboquant_cache
from easydel.infra.mixins import sharding
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.infra.mixins.operation_cache import cap_paged_cache_pages, compute_paged_cache_budget
from easydel.infra.mixins.sharding import EasyShardingMixin
from easydel.infra.sharding import AxisPolicy, RuntimeShardingResolver
from easydel.layers.quantization import TurboQuantConfig

pytestmark = pytest.mark.cpu_ok


class Config(SimpleNamespace):
    def get_text_config(self):
        return self

    def get_mask_details(self):
        return {0: "first-mask", 1: "second-mask"}


class Model(EasyGenerationMixin):
    mesh = nullcontext()

    def __init__(self, **kwargs):
        fields = dict(
            num_hidden_layers=2,
            num_key_value_heads=2,
            head_dim=64,
            attn_mechanism="ragged_page_attention_v2",
            kvdtype="bfloat16",
            kv_cache_quantization_config=None,
            mesh=SimpleNamespace(shape={}),
            runtime_sharding_resolver=RuntimeShardingResolver(AxisPolicy.from_partition_axis(None)),
        )
        fields.update(kwargs)
        self.config = Config(**fields)

    def get_operations_cache_view(self):
        return {0: RaggedPagesCacheView, 1: RaggedPagesCacheView}

    def get_operations_cache_view_by_slot(self):
        return {}

    def _pipeline_physical_stage_count(self):
        return 1

    def _standard_ragged_layer_indices_for_cache_budget(self):
        return [0, 1]


@pytest.mark.parametrize("override", [None, 100])
def test_public_budget_charges_busiest_stage_and_caps_overrides(override):
    budget = compute_paged_cache_budget(
        {0: 10, 1: 20, 2: 50},
        available_bytes=510,
        page_size=8,
        data_parallel_size=2,
        stage_layer_groups=(iter(group) for group in ([0, 1], [2])),
        num_pages_override=override,
        max_cache_tokens=75,
    )
    assert (budget.num_pages, budget.page_bytes, budget.num_hidden_layers) == (8, 50, 2)


def test_token_ceiling_cannot_be_exceeded_by_dp_minimum():
    with pytest.raises(ValueError, match="No DP-aligned"):
        cap_paged_cache_pages(100, page_size=16, data_parallel_size=4, max_cache_tokens=63)


@pytest.mark.parametrize("mechanism", ["ragged_page_attention_v2", "unified_attention"])
def test_mixed_builders_share_per_stage_budget_and_token_caps(monkeypatch, mechanism):
    for cls in (RaggedPagesCacheConfig, UnifiedAttentionCacheConfig):
        monkeypatch.setattr(cls, "_compute_free_hbm", staticmethod(lambda **_: 65536))
    model = Model(num_key_value_heads=[2, 4], attn_mechanism=mechanism)
    view = RaggedPagesCacheView
    if mechanism == "unified_attention":
        from easydel.caching import UnifiedAttentionCacheView

        view = UnifiedAttentionCacheView
    model.get_operations_cache_view = lambda: {0: view, 1: view}
    plan = SimpleNamespace(is_enabled=True, layer_to_stage={0: 0, 1: 1}, max_cache_tokens=80)
    configs = model.init_operations_cache_config(batch_size=1, max_length=256, page_size=16, pipeline_plan=plan)
    assert [config.num_pages for config in configs] == [4, 4]
    assert [config.num_hidden_layers for config in configs] == [1, 1]
    assert [config.num_kv_heads for config in configs] == [2, 4]


@pytest.mark.parametrize("max_tokens,expected", [(270, 16), (200, 12), (None, None)])
def test_homogeneous_turboquant_honors_plan_ceiling(monkeypatch, max_tokens, expected):
    monkeypatch.setattr(turboquant_cache, "per_device_hbm_budget_bytes", lambda *_, **__: 1 << 24)
    model = Model(kv_cache_quantization_config=TurboQuantConfig(), mesh=SimpleNamespace(shape={"dp": 4}))
    plan = SimpleNamespace(is_enabled=True, layer_to_stage={0: 0, 1: 1}, max_cache_tokens=max_tokens)
    configs = model.init_operations_cache_config(batch_size=1, max_length=256, page_size=16, pipeline_plan=plan)
    assert all(isinstance(config, turboquant_cache.TurboQuantRaggedPagesCacheConfig) for config in configs)
    assert all(config.num_hidden_layers == 1 for config in configs)
    assert all(config.num_pages % config.data_parallel_size == 0 for config in configs)
    if max_tokens is not None:
        assert [config.num_pages for config in configs] == [expected, expected]
        assert all(config.num_pages * config.page_size <= max_tokens for config in configs)
    else:
        assert all(config.num_pages > 16 for config in configs)


def test_homogeneous_transformer_generation_cache_keeps_stage_ownership(monkeypatch):
    active_stage = []

    class PipelineModel(Model):
        def get_inference_cache_type(self):
            return "transformer"

        @contextmanager
        def _layer_stage_context(self, layer_idx, total_layers):
            assert total_layers == 2
            active_stage.append(layer_idx)
            try:
                yield
            finally:
                active_stage.pop()

    def allocate_view(**kwargs):
        # The stand-in returns observable placement metadata, not arrays.
        return SimpleNamespace(
            owner=tuple(active_stage),
            layer_index=kwargs["layer_index"],
            starts=kwargs["starts"],
            pad_token_id=kwargs["config"].pad_token_id,
            masking_details=kwargs["masking_details"],
        )

    monkeypatch.setattr(TransformerCacheView, "init", staticmethod(allocate_view))
    model = PipelineModel(mesh=nullcontext())
    model._quant_class = lambda **_: None
    cache = model.init_cache(batch_size=1, max_length=128, starts=3, pad_token_id=9)
    assert [view.owner for view in cache.views] == [(0,), (1,)]
    assert [view.layer_index for view in cache.views] == [0, 1]
    assert [view.masking_details for view in cache.views] == ["first-mask", "second-mask"]
    assert all(view.starts == 3 and view.pad_token_id == 9 for view in cache.views)


@pytest.mark.parametrize("addressable", [True, False])
def test_host_gather_never_replicates_whole_state(monkeypatch, addressable):
    live_transfers = []
    used_meshes = []
    mesh = nullcontext()

    class DeviceLeaf:
        def __init__(self, value, replica=False):
            self.value = value
            self.is_fully_addressable = addressable
            self.replica = replica

    def compile_transfer(fn, **kwargs):
        used_meshes.append(kwargs["mesh"])

        def transfer(value):
            assert isinstance(value, DeviceLeaf)
            assert not live_transfers, "Previous replica must be copied to host before starting another leaf"
            replica = DeviceLeaf(value.value, replica=True)
            live_transfers.append(replica)
            return replica

        return transfer

    def host_transfer(value):
        assert isinstance(value, DeviceLeaf)
        if value.replica:
            assert live_transfers.pop() is value
        return value.value

    monkeypatch.setattr(sharding.jax, "Array", DeviceLeaf)
    monkeypatch.setattr(sharding.jax, "device_get", host_transfer)
    monkeypatch.setattr(sharding.spx, "jit", compile_transfer)
    monkeypatch.setattr(sharding, "replicated_named_sharding", lambda _: "replicated")

    class StateModel(EasyShardingMixin):
        def _get_mesh(self, supplied):
            return supplied

        def split_module(self):
            return "graph", {"a": DeviceLeaf(1), "b": DeviceLeaf(2)}, {"c": DeviceLeaf(3)}

        def merge_module(self, graph, state, other):
            return (graph, state, other)

    result = StateModel().gather_model(mesh=mesh)
    assert result == ("graph", {"a": 1, "b": 2}, {"c": 3})
    assert not live_transfers
    assert used_meshes == ([] if addressable else [mesh, mesh, mesh])

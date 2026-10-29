"""Pure-Python regression coverage for generation/cache discovery contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.caching import (
    ParallelHybridCacheView,
    RaggedPagesCacheConfig,
    RaggedPagesCacheView,
    RecurrentCacheView,
)
from easydel.infra.mixins import generation
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.infra.sharding import AxisPolicy, RuntimeShardingResolver

pytestmark = pytest.mark.cpu_ok


class Config(SimpleNamespace):
    def get_text_config(self):
        return self


def make_config(**kwargs):
    return Config(
        num_hidden_layers=2,
        hidden_size=512,
        head_dim=64,
        attn_mechanism="ragged_page_attention_v2",
        kvdtype="bfloat16",
        mesh=SimpleNamespace(shape={}),
        runtime_sharding_resolver=RuntimeShardingResolver(AxisPolicy.from_partition_axis(None)),
        **kwargs,
    )


class CacheModel(EasyGenerationMixin):
    mesh = nullcontext()

    def __init__(self, config):
        self.config = config

    def _standard_ragged_layer_indices_for_cache_budget(self):
        return [0, 1]

    def _pipeline_physical_stage_count(self):
        return 1

    def get_operations_cache_view(self):
        return {0: RaggedPagesCacheView, 1: RaggedPagesCacheView}

    def get_operations_cache_view_by_slot(self):
        return {}


@pytest.fixture
def fixed_hbm(monkeypatch):
    monkeypatch.setattr(RaggedPagesCacheConfig, "_compute_free_hbm", staticmethod(lambda **_: 1 << 24))


@pytest.mark.parametrize("head_fields", [{"num_kv_heads": 4}, {"num_attention_heads": 8}])
def test_public_ragged_config_resolves_head_aliases(fixed_hbm, head_fields):
    model = CacheModel(make_config(**head_fields))
    config = model.create_ragged_page_cache_config(max_length=256, page_size=16)
    assert config.num_kv_heads == next(iter(head_fields.values()))
    assert config.k_headdim == 64
    assert config.num_hidden_layers == 2


def test_public_ragged_config_uses_global_attention_geometry(fixed_hbm):
    model = CacheModel(
        make_config(
            num_key_value_heads=4,
            layer_types=["full_attention", "full_attention"],
            global_head_dim=128,
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
        )
    )
    config = model.create_ragged_page_cache_config(max_length=256, page_size=16)
    assert config.num_kv_heads == 2
    assert config.k_headdim == 128


def test_parallel_hybrid_subclass_keeps_ragged_config(fixed_hbm):
    class CustomRaggedView(RaggedPagesCacheView):
        pass

    class HybridModel(CacheModel):
        def get_operations_cache_view(self):
            return {0: ParallelHybridCacheView, 1: ParallelHybridCacheView}

        def get_operations_cache_view_by_slot(self):
            return {idx: {0: CustomRaggedView, 1: RecurrentCacheView} for idx in range(2)}

        def create_recurrent_cache_config(self, batch_size):
            return SimpleNamespace(batch_size=batch_size)

    model = HybridModel(make_config(num_key_value_heads=4))
    configs = model.init_operations_cache_config(batch_size=1, max_length=256, page_size=16)
    assert all(isinstance(config[0], RaggedPagesCacheConfig) for config in configs)
    assert all(config[0].num_kv_heads == 4 for config in configs)


def test_pipeline_plan_sizes_pool_for_busiest_stage(fixed_hbm):
    model = CacheModel(make_config(num_key_value_heads=4))
    plan = SimpleNamespace(is_enabled=True, layer_to_stage={0: 0, 1: 1}, max_cache_tokens=256)
    configs = model.init_operations_cache_config(batch_size=1, max_length=256, page_size=16, pipeline_plan=plan)
    assert [config.num_hidden_layers for config in configs] == [1, 1]
    assert [config.num_pages for config in configs] == [16, 16]


class ServingModel(EasyGenerationMixin):
    graphdef = "original"
    graphstate = "parameters"
    graphother = "buffers"

    def __init__(self, mechanism="vanilla", **kwargs):
        self.config = Config(attn_mechanism=mechanism, num_attention_heads=8, **kwargs)

    def new_graphdef(self, **updates):
        return updates

    def merge_module(self, graphdef, graphstate, graphother):
        return SimpleNamespace(graphdef=graphdef, graphstate=graphstate, graphother=graphother)


@pytest.mark.parametrize("backend,target", [("cpu", "ragged_page_attention_v3"), ("gpu", "unified_attention")])
@pytest.mark.parametrize("mla", [False, True])
def test_serving_graph_and_model_use_same_attention(monkeypatch, backend, target, mla):
    monkeypatch.setattr(generation.jax, "default_backend", lambda: backend)
    model = ServingModel(**({"kv_lora_rank": 64, "qk_rope_head_dim": 32} if mla else {}))
    adapted = model.esurge_compatible_model
    assert adapted.graphdef == model.esurge_graphdef
    assert adapted.graphdef["attn_mechanism"] == ("multi_latent_ragged_page_attention_v2" if mla else target)
    assert adapted.graphstate == "parameters"
    assert adapted.graphother == "buffers"


def test_compatible_and_compressed_window_models_preserve_identity(monkeypatch):
    monkeypatch.setattr(generation.jax, "default_backend", lambda: "cpu")
    model = ServingModel("ragged_page_attention_v2")
    assert model.esurge_compatible_model is model
    assert model.esurge_graphdef == "original"
    model.esurge_cache_family = "compressed_window"
    model.config.attn_mechanism = "vanilla"
    assert model.esurge_compatible_model is model
    assert model.esurge_graphdef == "original"


def test_scope_is_unique_without_serializable_config_and_propagates_lineage():
    class BrokenConfig:
        def to_dict(self):
            raise ValueError("cannot serialize")

    first, second = ServingModel(), ServingModel()
    first.config = second.config = BrokenConfig()
    first_scope = first._esurge_cache_scope()
    assert first_scope == first._esurge_cache_scope()
    assert first_scope != second._esurge_cache_scope()
    reconstruction = ServingModel()
    reconstruction._esurge_cache_scope_key = first_scope
    assert reconstruction._esurge_cache_scope() == first_scope


class _CompatModel:
    """Stand-in for the generation-compatible module swap."""

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "compat-result"


class SwapModel(EasyGenerationMixin):
    needs_generation_compatible_model = True
    graphstate = "params"

    def __init__(self, gradient_checkpointing="nothing_saveable"):
        self.config = Config(model_type="openelm", gradient_checkpointing=gradient_checkpointing)
        self.generation_config = None
        self.compatible = _CompatModel()

    @property
    def _generate_compatible_graphdef(self):
        return "compat-graph"

    @property
    def _generate_compatible_graphother(self):
        return "compat-other"

    def merge_module(self, graphdef, graphstate, graphother):
        assert (graphdef, graphstate, graphother) == ("compat-graph", "params", "compat-other")
        return self.compatible


def test_generate_routes_to_compatible_model_when_family_opts_in():
    model = SwapModel()
    assert model.generate(None, trace=True) == "compat-result"
    assert model.compatible.calls[0]["_use_generation_compatible_model"] is False


@pytest.mark.parametrize(
    "model",
    [
        # Checkpointing inactive: the swap must stay off even for an opted-in family.
        SwapModel(gradient_checkpointing="none"),
        # Checkpointing active but the family never opted in: no swap either.
        type("NoSwapModel", (SwapModel,), {"needs_generation_compatible_model": False})(),
    ],
)
def test_generate_does_not_swap_without_opt_in_and_checkpointing(model):
    # Without the swap, generate proceeds with generation_config=None and fails
    # on it; an AttributeError here proves the compatible-model branch was skipped.
    with pytest.raises(AttributeError):
        model.generate(None, trace=True)

"""Non-computation regressions for generation/cache geometry integration."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.caching import RaggedPagesCacheConfig, UnifiedAttentionCacheConfig
from easydel.caching.ragged_page import cache as ragged_cache
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.infra.sharding import AxisPolicy, RuntimeShardingResolver
from easydel.modules.openelm.modeling_openelm import OpenELMForCausalLM

pytestmark = pytest.mark.cpu_ok


class Config(SimpleNamespace):
    def get_text_config(self):
        return self


class CacheModel(EasyGenerationMixin):
    mesh = nullcontext()

    def __init__(self, **kwargs):
        fields = dict(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=512,
            head_dim=64,
            attn_mechanism="ragged_page_attention_v2",
            kvdtype="bfloat16",
            pad_token_id=7,
            mesh=SimpleNamespace(shape={}),
            runtime_sharding_resolver=RuntimeShardingResolver(AxisPolicy.from_partition_axis(None)),
        )
        fields.update(kwargs)
        self.config = Config(**fields)

    def _standard_ragged_layer_indices_for_cache_budget(self):
        return [0, 1]

    def _pipeline_physical_stage_count(self):
        return 1


@pytest.fixture
def fixed_hbm(monkeypatch):
    for cls in (RaggedPagesCacheConfig, UnifiedAttentionCacheConfig):
        monkeypatch.setattr(cls, "_compute_free_hbm", staticmethod(lambda **_: 1 << 28))


@pytest.mark.parametrize("mechanism", ["ragged_page_attention_v2", "ragged_page_attention_v3"])
@pytest.mark.parametrize("key_dim,value_dim", [(192, 128), (64, 64)])
def test_expanded_mla_cache_geometry_uses_full_attention_heads(fixed_hbm, mechanism, key_dim, value_dim):
    model = CacheModel(
        attn_mechanism=mechanism,
        kv_lora_rank=32,
        qk_nope_head_dim=key_dim - 32,
        qk_rope_head_dim=32,
        v_head_dim=value_dim,
    )
    transformer = model.create_transformer_cache_config(batch_size=3, max_length=256)
    ragged = model.create_ragged_page_cache_config(max_length=256, page_size=16, max_cache_tokens=1024)
    assert (transformer.key_heads, transformer.value_heads) == (8, 8)
    assert (transformer.key_dim, transformer.value_dim) == (key_dim, value_dim)
    assert (transformer.batch_size, transformer.sequence_length, transformer.pad_token_id) == (3, 256, 7)
    assert (ragged.num_kv_heads, ragged.k_headdim, ragged.v_headdim) == (8, key_dim, value_dim)
    assert ragged.version == mechanism.rsplit("_", 1)[-1]
    assert ragged.num_pages * ragged.page_size <= 1024


def test_standard_gqa_cache_geometry_is_unchanged(fixed_hbm):
    model = CacheModel()
    transformer = model.create_transformer_cache_config(batch_size=1, max_length=256, pad_token_id=9)
    ragged = model.create_ragged_page_cache_config(max_length=256, page_size=16)
    assert (transformer.key_heads, transformer.value_heads, transformer.pad_token_id) == (2, 2, 9)
    assert (transformer.key_dim, transformer.value_dim) == (64, 64)
    assert (ragged.num_kv_heads, ragged.k_headdim, ragged.v_headdim) == (2, 64, 64)


def test_mixed_unified_configs_follow_canonical_slice_policy(fixed_hbm, monkeypatch):
    # A policy change must affect mixed and homogeneous pools identically.
    # No cache arrays are allocated; only integer config fields are inspected.
    monkeypatch.setattr(ragged_cache, "KV_UPDATE_WINDOW_BYTES", 4096)
    monkeypatch.setattr(ragged_cache, "MAX_SLICES_PER_UPDATE_PAGE", 4)
    model = CacheModel(num_key_value_heads=[2, 4], attn_mechanism="unified_attention")
    config = model.create_unified_attention_cache_config(max_length=256, page_size=1)
    assert config.num_slices_per_kv_cache_update_page == 4


def test_mixed_unified_configs_keep_oversized_page_support(fixed_hbm):
    model = CacheModel(num_key_value_heads=[2, 4], attn_mechanism="unified_attention")
    config = model.create_unified_attention_cache_config(max_length=256, page_size=65536)
    assert config.num_slices_per_kv_cache_update_page == 1


class GenerationReached(RuntimeError):
    """Stop before numerical generation if the graph swap was not selected."""


class StopBeforeNumerics:
    _from_model_config = False

    def __deepcopy__(self, memo):
        return self

    def update(self, **kwargs):
        raise GenerationReached


class CompatibleModel:
    def generate(self, **kwargs):
        return kwargs


class SwapModel(EasyGenerationMixin):
    graphstate = "params"

    def __init__(self, capability, checkpointing="nothing_saveable"):
        setattr(self, capability, True)
        self.config = Config(model_type="unrelated_family", gradient_checkpointing=checkpointing)
        self.generation_config = StopBeforeNumerics()

    @property
    def _generate_compatible_graphdef(self):
        return "compatible-graph"

    @property
    def _generate_compatible_graphother(self):
        return "compatible-buffers"

    def merge_module(self, graphdef, graphstate, graphother):
        assert (graphdef, graphstate, graphother) == ("compatible-graph", "params", "compatible-buffers")
        return CompatibleModel()


@pytest.mark.parametrize("capability", ["requires_generation_compatible_graph", "needs_generation_compatible_model"])
def test_generate_accepts_new_capability_and_legacy_alias(capability):
    model = SwapModel(capability)
    result = model.generate(None, trace=True, custom_option="preserved")
    assert result["input_ids"] is None
    assert result["trace"] is True
    assert result["custom_option"] == "preserved"
    assert result["_use_generation_compatible_model"] is False


@pytest.mark.parametrize(
    "checkpointing,kwargs",
    [
        ("none", {}),
        ("nothing_saveable", {"trace": False}),
        ("nothing_saveable", {"_use_generation_compatible_model": False}),
    ],
)
def test_new_capability_preserves_trace_checkpointing_and_recursion_guards(checkpointing, kwargs):
    model = SwapModel("requires_generation_compatible_graph", checkpointing=checkpointing)
    with pytest.raises(GenerationReached):
        model.generate(None, **kwargs)


def test_openelm_declares_consumed_generation_capability():
    # Read the real model's public capability without initializing parameters.
    assert OpenELMForCausalLM.requires_generation_compatible_graph is True
    model = SwapModel("requires_generation_compatible_graph")
    model.requires_generation_compatible_graph = OpenELMForCausalLM.requires_generation_compatible_graph
    assert model.generate(None)["_use_generation_compatible_model"] is False

"""CPU-only cache configuration and generation-dispatch regressions (no array computation)."""

from types import SimpleNamespace

import pytest
from easydel import caching
from easydel.caching import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView, UnifiedAttentionCacheView
from easydel.infra.mixins import generation, operation_cache
from easydel.infra.mixins.generation import EasyGenerationMixin
from transformers import GenerationConfig

from .test_stage_cache_budget import Model

pytestmark = pytest.mark.cpu_ok


def test_original_infra_budget_exports_are_cache_api_aliases():
    assert operation_cache.PagedCacheBudget is caching.PagedCacheBudget
    assert operation_cache.cap_paged_cache_pages is caching.cap_paged_cache_pages
    assert operation_cache.compute_paged_cache_budget is caching.compute_paged_cache_budget
    assert generation.compute_paged_cache_budget is caching.compute_paged_cache_budget


@pytest.mark.parametrize("kind", ["ragged", "unified", "mla", "turboquant"])
@pytest.mark.parametrize("ceiling", [63, 80, None])
def test_supplied_homogeneous_pool_obeys_pipeline_ceiling_without_mutation(monkeypatch, kind, ceiling):
    from easydel.caching.turboquant_ragged_page import cache as turboquant_cache
    from easydel.layers.quantization import TurboQuantConfig

    model = Model(mesh=SimpleNamespace(shape={"dp": 4}))
    cls = caching.RaggedPagesCacheConfig
    view = caching.RaggedPagesCacheView
    dimensions = {"kv_head_dim_size": 64, "kvdtype": caching.canonicalize_dtype("bfloat16")}
    if kind == "unified":
        cls, view = caching.UnifiedAttentionCacheConfig, UnifiedAttentionCacheView
        dimensions = {"head_dim": 64, "kvdtype": caching.canonicalize_dtype("bfloat16")}
        model.config.attn_mechanism = "unified_attention"
    elif kind == "mla":
        cls, view = MLARaggedPagesCacheConfig, MLARaggedPagesCacheView
        dimensions = {"kv_lora_rank": 64, "qk_rope_head_dim": 32, "kvdtype": caching.canonicalize_dtype("bfloat16")}
        model.config.attn_mechanism = "multi_latent_ragged_page_attention_v2"
    elif kind == "turboquant":
        cls = turboquant_cache.TurboQuantRaggedPagesCacheConfig
        model.config.kv_cache_quantization_config = TurboQuantConfig()
        dimensions.pop("kvdtype")
        dimensions["turboquant_config"] = model.config.kv_cache_quantization_config
        monkeypatch.setattr(turboquant_cache, "per_device_hbm_budget_bytes", lambda *_, **__: 1 << 24)
    if kind != "turboquant":
        monkeypatch.setattr(cls, "_compute_free_hbm", staticmethod(lambda **_: 1 << 24))
    original = cls.create(
        mesh=model.config.mesh,
        runtime_sharding_resolver=model.config.runtime_sharding_resolver,
        num_hidden_layers=2,
        num_kv_heads=2,
        max_model_length=256,
        page_size=16,
        **dimensions,
    )
    original_pages = original.num_pages
    model.get_operations_cache_view = lambda: {0: view, 1: view}
    plan = SimpleNamespace(is_enabled=True, layer_to_stage={0: 0, 1: 1}, max_cache_tokens=ceiling)
    supplied = {"unified_config" if kind == "unified" else "ragged_config": original}
    if ceiling == 63:
        with pytest.raises(ValueError, match="No DP-aligned"):
            model.init_operations_cache_config(batch_size=1, max_length=256, pipeline_plan=plan, **supplied)
    else:
        configs = model.init_operations_cache_config(batch_size=1, max_length=256, pipeline_plan=plan, **supplied)
        assert all(isinstance(config, cls) for config in configs)
        assert [config.num_pages for config in configs] == ([4, 4] if ceiling else [original_pages] * 2)
        assert all(config.page_size == 16 and config.data_parallel_size == 4 for config in configs)
        if ceiling is None:
            assert all(config is original for config in configs)
        else:
            assert all(config.num_pages * config.page_size <= ceiling for config in configs)
    assert original.num_pages == original_pages


@pytest.mark.parametrize("cls", [caching.RaggedPagesCacheConfig, caching.UnifiedAttentionCacheConfig])
def test_homogeneous_factories_reject_impossible_dp_ceiling(monkeypatch, cls):
    monkeypatch.setattr(cls, "_compute_free_hbm", staticmethod(lambda **_: 1 << 24))
    model = Model(mesh=SimpleNamespace(shape={"dp": 4}))
    dimensions = {"head_dim": 64} if cls is caching.UnifiedAttentionCacheConfig else {"kv_head_dim_size": 64}
    with pytest.raises(ValueError, match="No DP-aligned"):
        cls.create(
            mesh=model.config.mesh,
            runtime_sharding_resolver=model.config.runtime_sharding_resolver,
            kvdtype=caching.canonicalize_dtype("bfloat16"),
            num_hidden_layers=2,
            num_kv_heads=2,
            max_model_length=256,
            page_size=16,
            max_cache_tokens=63,
            **dimensions,
        )


class Dispatcher(EasyGenerationMixin):
    def _prepare_mask_info_for_generation_step(self, running_token, kwargs):
        return kwargs


@pytest.mark.parametrize("embedded", [False, True])
def test_generation_step_passes_caller_kwargs_without_output_switches(embedded):
    prompt = object()
    cache = object()
    received = {}

    class Model:
        # Simulate SpectraX's generic __call__ with a kwargs-accepting forward.
        # EasyDeL forwards always return their named output type.
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            received.update(kwargs)
            value = inputs_embeds if inputs_embeds is not None else input_ids
            return SimpleNamespace(logits=value, past_key_values=cache)

    kwargs = {"inputs_embeds": prompt} if embedded else {}
    output = Dispatcher()._call_generation_model_step(Model(), prompt, kwargs)
    assert output.logits is prompt
    assert output.past_key_values is cache
    assert "return_dict" not in received
    assert "use_cache" not in received
    assert kwargs == ({"inputs_embeds": prompt} if embedded else {})


@pytest.mark.parametrize("embedded", [False, True])
def test_generation_keeps_legacy_forward_without_return_dict(embedded):
    prompt = object()

    class Model:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids=None, inputs_embeds=None):
            return SimpleNamespace(logits=inputs_embeds if inputs_embeds is not None else input_ids)

    kwargs = {"inputs_embeds": prompt} if embedded else {}
    assert Dispatcher()._call_generation_model_step(Model(), prompt, kwargs).logits is prompt


@pytest.mark.parametrize("embedded", [False, True])
def test_qwen2vl_generation_step_returns_named_outputs(monkeypatch, embedded):
    from easydel.infra.modeling_outputs import BaseModelOutput
    from easydel.modules.qwen2_vl import modeling_qwen2_vl

    hidden, logits, cache = object(), object(), object()
    monkeypatch.setattr(modeling_qwen2_vl, "apply_logical_sharding", lambda value, **kwargs: value)

    class QwenModel:
        forward = modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward
        config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=False,
            runtime_sharding_resolver=None,
        )

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def base_model(self, **kwargs):
            return BaseModelOutput(last_hidden_state=hidden, past_key_values=cache)

        def compute_lm_logits(self, value):
            return logits

        def apply_logit_cap(self, value):
            return value

    model = QwenModel()
    direct = model()
    assert direct.logits is logits
    assert direct.past_key_values is cache
    kwargs = {"inputs_embeds": hidden} if embedded else {}
    output = Dispatcher()._call_generation_model_step(model, object(), kwargs)
    assert output.logits is logits
    assert output.last_hidden_state is hidden
    assert output.past_key_values is cache


@pytest.mark.parametrize("prompt_length", [1, 4])
@pytest.mark.parametrize("pairs", [None, [], [(1, 7), (2, None), (3, 11)]])
def test_forced_decoder_pairs_keep_positions_and_none_placeholders(monkeypatch, prompt_length, pairs):
    class ForcedTokens:
        """Array-free boundary for the real processor's dictionary-only input."""

        def __init__(self, mapping):
            assert isinstance(mapping, dict) and mapping
            self.mapping = mapping

    monkeypatch.setattr(generation, "ForceTokensLogitsProcessor", ForcedTokens)
    config = GenerationConfig(forced_decoder_ids=pairs)
    processors = Dispatcher()._get_logits_processor(config, prompt_length, None)
    if pairs:
        assert len(processors) == 1
        assert processors[0].mapping == {prompt_length: 7, prompt_length + 1: None, prompt_length + 2: 11}
    else:
        assert processors == []
    assert config.forced_decoder_ids == pairs

"""Array-free regressions for persistent serving selection and cached dispatch."""

from types import SimpleNamespace

import pytest
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.mixins import generation
from easydel.modules.qwen2_vl.qwen2_vl_configuration import Qwen2VLConfig

from .test_backend_policy_selection import ServingModel
from .test_cache_budget_dispatch import Dispatcher

pytestmark = pytest.mark.cpu_ok


class CompositeServingModel(ServingModel):
    """Use real composite configs and config reconstruction, without model arrays."""

    new_graphdef = EasyDeLBaseModule.new_graphdef
    _apply_recursive_config_updates = staticmethod(EasyDeLBaseModule._apply_recursive_config_updates)

    def __init__(self, mechanism="vanilla", text_mechanism="vanilla", **kwargs):
        super().__init__()
        self.config = Qwen2VLConfig(
            attn_mechanism=mechanism,
            text_config={"attn_mechanism": text_mechanism},
            **kwargs,
        )
        # Isolate main-mechanism/marker compatibility from the independently
        # tested decode override policy, rather than inheriting config defaults.
        for config in (self.config, self.config.text_config):
            config.decode_attn_mechanism = config.mla_attn_mechanism = None
        self.graphdef = self.config

    def _lazy_init_rebuilt_module(self, config):
        return SimpleNamespace(graphdef=config)


@pytest.mark.parametrize(
    "engine,target,backend", [("ring", "ring_paged", "cpu"), ("triton", "unified_attention", "gpu")]
)
@pytest.mark.parametrize("marker", [None, "opposite"])
@pytest.mark.parametrize("composite", [False, True])
def test_explicit_engine_replaces_missing_or_stale_marker(monkeypatch, engine, target, backend, marker, composite):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: backend)
    marker = ("triton" if engine == "ring" else "ring") if marker == "opposite" else marker
    model = CompositeServingModel(target, target) if composite else ServingModel(target)
    model.config.esurge_decoding_engine = marker
    graph = model.get_esurge_graphdef(engine)
    selected = model.get_esurge_compatible_model(engine)
    for config in (graph, selected.config):
        assert config.esurge_decoding_engine == engine
        assert config.get_text_config().attn_mechanism == target
    assert selected is not model
    assert selected.esurge_compatible_model is selected
    assert selected.esurge_graphdef is selected.graphdef
    assert selected.graphstate is model.graphstate
    assert selected.graphother is model.graphother
    assert model.config.esurge_decoding_engine == marker


@pytest.mark.parametrize("mechanism", ["ring", "ring_paged"])
def test_composite_configured_text_ring_is_not_substituted(monkeypatch, mechanism):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    model = CompositeServingModel("vanilla", mechanism)
    selected = model.esurge_compatible_model
    graph = model.esurge_graphdef
    for config in (graph, selected.config):
        assert config.attn_mechanism == "ring_paged"
        assert config.text_config.attn_mechanism == "ring_paged"
        assert config.esurge_decoding_engine == "ring"
    assert selected.esurge_compatible_model is selected
    assert selected.graphstate is model.graphstate
    assert selected.graphother is model.graphother
    assert model.config.attn_mechanism == "vanilla"
    assert model.config.text_config.attn_mechanism == mechanism


@pytest.mark.parametrize(
    "engine,target,backend", [("ring", "ring_paged", "cpu"), ("triton", "unified_attention", "gpu")]
)
@pytest.mark.parametrize("text_mechanism", ["vanilla", "ring", "unified_attention"])
def test_explicit_selection_checks_composite_text_main_attention(monkeypatch, engine, target, backend, text_mechanism):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: backend)
    model = CompositeServingModel(target, text_mechanism, esurge_decoding_engine=engine)
    compatible, _ = model.resolve_esurge_attention(engine)
    assert compatible is (text_mechanism == target)
    selected = model.get_esurge_compatible_model(engine)
    assert selected.config.text_config.attn_mechanism == target
    assert selected.config.esurge_decoding_engine == engine
    assert selected.esurge_compatible_model is selected
    assert model.config.text_config.attn_mechanism == text_mechanism


def test_composite_none_text_marker_inherits_persisted_root_selection(monkeypatch):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "gpu")
    model = CompositeServingModel("unified_attention", "unified_attention", esurge_decoding_engine="triton")
    model.config.text_config.esurge_decoding_engine = None
    selected = model.esurge_compatible_model
    assert selected.config.esurge_decoding_engine == "triton"
    assert selected.config.text_config.attn_mechanism == "unified_attention"
    # The marker remains authoritative when a later graph refresh needs repair.
    selected.config.text_config.attn_mechanism = "vanilla"
    repaired = selected.esurge_compatible_model
    assert repaired.config.text_config.attn_mechanism == "unified_attention"
    assert repaired.config.esurge_decoding_engine == "triton"


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("preference", [None, False, True])
@pytest.mark.parametrize("signature", ["explicit", "kwargs"])
def test_generation_forwards_caller_cache_preference_unchanged(embedded, preference, signature):
    token, embedding, cache = object(), object(), object()
    seen = {}

    class Model:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids=None, *, inputs_embeds=None, use_cache=None):
            # EasyDeL forwards always return the named output and updated cache;
            # generation must not override what the caller requested.
            seen["use_cache"] = use_cache
            value = inputs_embeds if inputs_embeds is not None else input_ids
            assert (input_ids is None) is embedded
            return SimpleNamespace(logits=value, past_key_values=cache)

    if signature == "kwargs":
        explicit_forward = Model.forward

        def kwargs_forward(self, input_ids=None, **kwargs):
            return explicit_forward(self, input_ids, **kwargs)

        Model.forward = kwargs_forward
    kwargs = {"inputs_embeds": embedding} if embedded else {}
    if preference is not None:
        kwargs["use_cache"] = preference
    output = Dispatcher()._call_generation_model_step(Model(), token, kwargs)
    assert output.logits is (embedding if embedded else token)
    assert output.past_key_values is cache
    assert seen["use_cache"] is preference
    assert kwargs.get("use_cache") is preference


@pytest.mark.parametrize("embedded", [False, True])
def test_generation_does_not_pass_cache_flag_to_legacy_forward(embedded):
    token, embedding = object(), object()

    class Model:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids=None, *, inputs_embeds=None):
            return SimpleNamespace(logits=inputs_embeds if inputs_embeds is not None else input_ids)

    kwargs = {"inputs_embeds": embedding} if embedded else {}
    output = Dispatcher()._call_generation_model_step(Model(), token, kwargs)
    assert output.logits is (embedding if embedded else token)
    assert "use_cache" not in kwargs


@pytest.mark.parametrize("embedded", [False, True])
def test_real_qwen_task_forward_retains_cache_with_disabled_checkpoint_default(monkeypatch, embedded):
    from easydel.infra.modeling_outputs import BaseModelOutput
    from easydel.modules.qwen2_vl import modeling_qwen2_vl

    hidden, logits, cache, token = object(), object(), object(), object()
    monkeypatch.setattr(modeling_qwen2_vl, "apply_logical_sharding", lambda value, **kwargs: value)

    class QwenModel:
        forward = modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward
        config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=False,
            use_cache=False,
            runtime_sharding_resolver=None,
        )

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def base_model(self, **kwargs):
            # The task head must not ask the backbone to drop its cache.
            assert "use_cache" not in kwargs
            return BaseModelOutput(last_hidden_state=hidden, past_key_values=cache)

        def compute_lm_logits(self, value):
            return logits

        def apply_logit_cap(self, value):
            return value

    model = QwenModel()
    assert model().past_key_values is cache
    kwargs = {"inputs_embeds": hidden} if embedded else {}
    output = Dispatcher()._call_generation_model_step(model, token, kwargs)
    assert output.logits is logits
    assert output.past_key_values is cache
    assert model.config.use_cache is False
    assert "use_cache" not in kwargs

"""Host-only explicit backend wiring and graph/state preservation (no JAX arrays)."""

import inspect
from copy import copy
from types import SimpleNamespace

import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.engine import bootstrap
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.infra.elarge.builders import to_esurge_kwargs
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.operations import OperationRegistry
from easydel.operations.requirements import CacheType

pytestmark = pytest.mark.cpu_ok


class HostConfig(SimpleNamespace):
    def get_text_config(self):
        return self


class HostModel(EasyGenerationMixin):
    """Array-free graph/state interface using the real model compatibility policy."""

    def __init__(self, mechanism="vanilla"):
        self.config = HostConfig(attn_mechanism=mechanism, num_attention_heads=4)
        self.graphstate = object()
        self.graphother = object()
        self.graphdef = self.config

    def new_graphdef(self, *, recursive_update=False, **kwargs):
        assert recursive_update
        config = copy(self.config)
        config.__dict__.update(kwargs)
        return config

    def merge_module(self, graphdef, graphstate, graphother):
        model = copy(self)
        model.config = graphdef
        model.graphdef = graphdef
        model.graphstate = graphstate
        model.graphother = graphother
        return model


class RunnerReady(Exception):
    def __init__(self, model):
        self.model = model


@pytest.mark.parametrize("engine", ["ring", "triton"])
def test_elarge_backend_kwargs_bind_to_public_engine_signature(engine):
    kwargs = to_esurge_kwargs({"model": {"name_or_path": "test"}, "esurge": {"decoding_engine": engine}})
    bound = inspect.signature(eSurge).bind(model="test", **kwargs)
    assert bound.arguments["decoding_engine"] == engine


@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("engine,mechanism", [("ring", "ring_paged"), ("triton", "unified_attention")])
def test_actual_engine_constructor_selects_registered_paged_graph_without_mutating_weights(
    monkeypatch, preloaded, engine, mechanism
):
    from easydel.modules.auto import AutoEasyDeLModelForCausalLM

    original = HostModel()
    tokenizer = SimpleNamespace(get_vocab=lambda: {}, eos_token_id=0)
    loads = []

    def load(**kwargs):
        loads.append(kwargs)
        return original

    # Keep the actual engine/bootstrap/loader/compatibility path. Stub only
    # external I/O, parser discovery, worker processes and device allocation.
    monkeypatch.setattr(AutoEasyDeLModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(bootstrap, "_resolve_processor_and_tokenizer", lambda *args: (None, tokenizer, "test"))
    monkeypatch.setattr(bootstrap, "_resolve_parsers", lambda **kwargs: (None, None, None, None))
    monkeypatch.setattr(bootstrap, "get_model_name", lambda model: "test")
    # Reported GPU label is metadata only. No GPU/TPU runtime is initialized.
    monkeypatch.setattr(bootstrap.jax, "default_backend", lambda: "gpu" if engine == "triton" else "cpu")
    monkeypatch.setattr(eSurge, "_apply_data_parallel_axis_to_model", lambda *args: None)
    monkeypatch.setattr(eSurge, "_warn_if_data_parallel_axis_shards_the_model", lambda *args: None)
    monkeypatch.setattr(
        esurge_engine,
        "WorkerManager",
        lambda *args, **kwargs: SimpleNamespace(
            start=lambda **kwargs: (None, None), tokenizer_endpoint=None, detokenizer_endpoint=None
        ),
    )

    def runner_factory(**kwargs):
        raise RunnerReady(kwargs["model"])

    monkeypatch.setattr(esurge_engine, "eSurgeRunner", runner_factory)
    with pytest.raises(RunnerReady) as captured:
        eSurge(model=original if preloaded else "test", decoding_engine=engine, runtime={"max_num_batched_tokens": 32})
    selected = captured.value.model
    assert len(loads) == (0 if preloaded else 1)
    if loads:
        assert loads[0]["config_kwargs"]["attn_mechanism"] == mechanism
    assert selected.config.attn_mechanism == mechanism
    assert selected.config.decode_attn_mechanism == mechanism
    assert selected.config.mla_attn_mechanism == mechanism
    assert selected.config.esurge_decoding_engine == engine
    assert selected.graphstate is original.graphstate
    assert selected.graphother is original.graphother
    assert original.config.attn_mechanism == "vanilla"
    operation = OperationRegistry.get(selected.config.attn_mechanism)
    assert operation.get_impl_name() == mechanism
    assert operation.get_requirements().cache.supported & CacheType.RAGGED_PAGES


@pytest.mark.parametrize("engine", [None, "auto"])
def test_automatic_selection_preserves_existing_public_model_contract(engine):
    original = HostModel("ragged_page_attention_v3")
    assert bootstrap.select_decoding_engine(original, engine) is original


@pytest.mark.parametrize("engine", ["unknown", "RING", "", False, 1])
def test_invalid_backend_fails_before_model_loading(engine, monkeypatch):
    monkeypatch.setattr(esurge_engine, "build_engine_assets", lambda **kwargs: pytest.fail("loaded invalid backend"))
    with pytest.raises(ValueError, match="decoding_engine must"):
        eSurge(model="test", decoding_engine=engine)


def test_explicit_triton_does_not_silently_fall_back_on_cpu(monkeypatch):
    monkeypatch.setattr(bootstrap.jax, "default_backend", lambda: "cpu")
    with pytest.raises(ValueError, match="requires a GPU"):
        bootstrap.select_decoding_engine(HostModel(), "triton")


def test_explicit_selection_rejects_compatibility_substitution():
    class RetargetingModel(HostModel):
        @property
        def esurge_compatible_model(self):
            return HostModel("ragged_page_attention_v3")

    with pytest.raises(ValueError, match="refusing to silently substitute"):
        bootstrap.select_decoding_engine(RetargetingModel(), "ring")


def test_automatic_backend_preserves_real_model_compatibility_policy(monkeypatch):
    class PolicyModel(HostModel):
        esurge_compatible_model = EasyGenerationMixin.esurge_compatible_model
        _esurge_target_attention = EasyGenerationMixin._esurge_target_attention

    monkeypatch.setattr(bootstrap.jax, "default_backend", lambda: "cpu")
    original = PolicyModel()
    selected = bootstrap.select_decoding_engine(original, None)
    assert selected.config.attn_mechanism == "ragged_page_attention_v3"
    assert selected.graphstate is original.graphstate
    assert selected.graphother is original.graphother
    assert original.config.attn_mechanism == "vanilla"


@pytest.mark.parametrize("engine,mechanism", [("ring", "ring_paged"), ("triton", "unified_attention")])
def test_real_string_loader_receives_explicit_backend_before_model_construction(monkeypatch, engine, mechanism):
    from easydel.inference.esurge.config import eSurgeRuntimeConfig
    from easydel.modules.auto import AutoEasyDeLModelForCausalLM
    from easydel.modules.auto.auto_modeling import PreTrainedLoading

    calls = []
    model = HostModel()

    def load(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(AutoEasyDeLModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(bootstrap.jax, "default_backend", lambda: "gpu" if engine == "triton" else "cpu")
    result = bootstrap._load_model(
        "test",
        loading_kwargs=PreTrainedLoading.coerce_config({"pretrained_model_name_or_path": "test"}),
        runtime_config=eSurgeRuntimeConfig.coerce_config({}),
        data_parallelism_axis="dp",
        dtype=None,
        decoding_engine=engine,
    )
    assert result is model
    assert len(calls) == 1
    config = calls[0]["config_kwargs"]
    assert config["attn_mechanism"] == mechanism
    assert config["decode_attn_mechanism"] == mechanism
    assert config["mla_attn_mechanism"] == mechanism
    assert config["esurge_decoding_engine"] == engine


@pytest.mark.parametrize("components_only", [False, True])
@pytest.mark.parametrize("engine,mechanism", [("ring", "ring_paged"), ("triton", "unified_attention")])
def test_graph_component_weight_refresh_preserves_explicit_backend(monkeypatch, components_only, engine, mechanism):
    from tests.inference.esurge.test_engine_stub_behaviors import _SchedulerStub

    original = HostModel()
    updates = []

    class HostState:
        def overlay(self, other):
            assert other is original.graphother
            return self

    state = HostState()

    def bind(graphdef, combined_state):
        assert combined_state is state
        model = HostModel()
        model.config = graphdef
        model.graphdef = graphdef
        model.graphstate = state
        model.graphother = original.graphother
        return model

    monkeypatch.setattr(esurge_engine.spx, "bind", bind)
    monkeypatch.setattr(bootstrap.jax, "default_backend", lambda: "gpu" if engine == "triton" else "cpu")
    instance = _SchedulerStub()
    instance.decoding_engine = engine
    instance.runner.update_model_weights = lambda **kwargs: updates.append(kwargs)
    monkeypatch.setattr(instance, "_build_scheduler", lambda: object())
    instance.update_model_weights(
        model=None if components_only else original,
        graphdef=original.graphdef,
        graphstate=state,
        graphother=original.graphother,
    )
    assert len(updates) == 1
    assert updates[0]["graphdef"].attn_mechanism == mechanism
    assert updates[0]["graphdef"].esurge_decoding_engine == engine
    assert updates[0]["graphstate"] is state
    assert updates[0]["graphother"] is original.graphother
    assert original.config.attn_mechanism == "vanilla"

"""Array-free backend policy and beam named-output boundary regressions."""

from copy import copy
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.engine import bootstrap
from easydel.infra.mixins import generation
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.operations import OperationRegistry
from easydel.operations.requirements import CacheType
from transformers import GenerationConfig

pytestmark = pytest.mark.cpu_ok


class Config(SimpleNamespace):
    def get_text_config(self):
        return self


class ServingModel(EasyGenerationMixin):
    def __init__(self, mechanism="vanilla", **kwargs):
        self.config = Config(attn_mechanism=mechanism, num_attention_heads=8, **kwargs)
        self.graphdef = self.config
        self.graphstate = object()
        self.graphother = object()

    def new_graphdef(self, *, recursive_update=False, **updates):
        assert recursive_update
        config = copy(self.config)
        config.__dict__.update(updates)
        return config

    def merge_module(self, graphdef, graphstate, graphother):
        model = copy(self)
        model.config = model.graphdef = graphdef
        model.graphstate, model.graphother = graphstate, graphother
        return model


@pytest.mark.parametrize("mechanism", ["ring", "ring_paged"])
def test_configured_ring_retains_algorithm_and_uses_registered_paged_allocation(monkeypatch, mechanism):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    original = ServingModel(mechanism)
    graph = original.esurge_graphdef
    selected = original.esurge_compatible_model
    assert graph.attn_mechanism == selected.config.attn_mechanism == "ring_paged"
    assert selected.esurge_compatible_model is selected
    assert selected.esurge_graphdef is selected.graphdef
    assert selected.graphstate is original.graphstate
    assert selected.graphother is original.graphother
    assert original.config.attn_mechanism == mechanism
    operation = OperationRegistry.get(selected.config.attn_mechanism)
    requirements = operation.get_requirements()
    assert requirements.cache.supported & CacheType.RAGGED_PAGES
    assert requirements.cache.cache_view_class.__name__ == "RaggedPagesCacheView"


@pytest.mark.parametrize(
    "engine,target,backend", [("ring", "ring_paged", "cpu"), ("triton", "unified_attention", "gpu")]
)
def test_public_explicit_selection_preserves_graph_state_and_engine_marker(monkeypatch, engine, target, backend):
    # Backend labels only: no device or array computation occurs in this probe.
    monkeypatch.setattr(generation.jax, "default_backend", lambda: backend)
    original = ServingModel()
    compatible, updates = original.resolve_esurge_attention(engine)
    assert compatible is False
    assert updates["attn_mechanism"] == target
    graph = original.get_esurge_graphdef(engine)
    selected = original.get_esurge_compatible_model(engine)
    from_engine = bootstrap.select_decoding_engine(original, engine)
    for config in (graph, selected.config, from_engine.config, selected.esurge_graphdef):
        assert config.attn_mechanism == target
        assert config.decode_attn_mechanism == target
        assert config.mla_attn_mechanism == target
        assert config.esurge_decoding_engine == engine
    assert selected.esurge_compatible_model is selected
    assert from_engine.esurge_compatible_model is from_engine
    assert selected.graphstate is from_engine.graphstate is original.graphstate
    assert selected.graphother is from_engine.graphother is original.graphother
    assert original.config.attn_mechanism == "vanilla"


@pytest.mark.parametrize("engine", [None, "auto"])
def test_auto_policy_remains_unchanged(monkeypatch, engine):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    model = ServingModel()
    assert model.get_esurge_graphdef(engine).attn_mechanism == "ragged_page_attention_v3"
    compatible = ServingModel("ragged_page_attention_v2")
    assert compatible.get_esurge_compatible_model(engine) is compatible


@pytest.mark.parametrize("engine", ["unknown", "RING", "", False, 1])
def test_public_selection_rejects_invalid_engines(engine):
    with pytest.raises(ValueError, match="decoding_engine must"):
        ServingModel().resolve_esurge_attention(engine)


@pytest.mark.parametrize("api", ["get_esurge_graphdef", "get_esurge_compatible_model", "resolve_esurge_attention"])
def test_public_triton_selection_rejects_cpu_without_substitution(monkeypatch, api):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    with pytest.raises(ValueError, match="requires a GPU"):
        getattr(ServingModel(), api)("triton")


def test_ring_rejects_incompatible_model_caches(monkeypatch):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    mla = ServingModel("ring_paged", kv_lora_rank=64, qk_rope_head_dim=32)
    with pytest.raises(ValueError, match="latent MLA"):
        _ = mla.esurge_graphdef
    compressed = ServingModel()
    compressed.esurge_cache_family = "compressed_window"
    assert compressed.esurge_compatible_model is compressed
    with pytest.raises(ValueError, match="compressed-window"):
        compressed.get_esurge_compatible_model("ring")


def test_selection_validates_current_registry(monkeypatch):
    original_get = OperationRegistry.get
    monkeypatch.setattr(OperationRegistry, "get", lambda name: None if name == "ring_paged" else original_get(name))
    with pytest.raises(ValueError, match="requires registered paged attention"):
        ServingModel().get_esurge_compatible_model("ring")


def test_configured_selection_repairs_conflicting_decode_overrides(monkeypatch):
    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    model = ServingModel("ring_paged", esurge_decoding_engine="ring", decode_attn_mechanism="vanilla")
    selected = model.esurge_compatible_model
    assert selected.config.decode_attn_mechanism == "ring_paged"
    assert model.config.decode_attn_mechanism == "vanilla"


@pytest.mark.parametrize("embedded", [False, True])
def test_beam_search_consumes_named_outputs_with_tuple_config(monkeypatch, embedded):
    class ShapeOnly:
        """No array storage or numerical operations: only shape plumbing."""

        def __init__(self, shape=()):
            self.shape = shape
            self.ndim = len(shape)

        def reshape(self, shape):
            return ShapeOnly(shape)

        def __mul__(self, other):
            return self

        def __sub__(self, other):
            return self

    def slice_shape(value, start, size):
        return ShapeOnly(size)

    fake_numpy = SimpleNamespace(
        int32="int32",
        bool_="bool",
        array=lambda *args: ShapeOnly(),
        full=lambda shape, *args, **kwargs: ShapeOnly(shape),
        zeros=lambda shape, **kwargs: ShapeOnly(shape),
        ones=lambda shape: ShapeOnly(shape),
        tile=lambda value, reps: ShapeOnly((1, 2)),
    )
    monkeypatch.setattr(generation, "jnp", fake_numpy)
    monkeypatch.setattr(
        generation,
        "lax",
        SimpleNamespace(dynamic_update_slice=lambda value, *args: value, dynamic_slice=slice_shape),
    )

    class NamedOutputReached(Exception):
        pass

    class Logits:
        def __getitem__(self, index):
            raise NamedOutputReached

    prompt_embedding = object()
    cache = object()

    class BeamModel(EasyGenerationMixin):
        config = Config(is_encoder_decoder=False, use_return_dict=False)
        generation_config = GenerationConfig(max_length=4, pad_token_id=0, eos_token_id=3)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {"inputs_embeds": prompt_embedding} if embedded else {}

        def _prepare_mask_info_for_generation_step(self, token, kwargs):
            return kwargs

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids=None, inputs_embeds=None):
            if embedded:
                assert input_ids is None and inputs_embeds is prompt_embedding
            else:
                assert input_ids.shape == (2, 1)
            # EasyDeL forwards always return their named output type.
            return SimpleNamespace(logits=Logits(), past_key_values=cache)

    model = BeamModel()
    # The real beam body must reach named logits; a tuple produces AttributeError.
    with pytest.raises(NamedOutputReached):
        model._beam_search(ShapeOnly((1, 2, 1)), trace=False, model_kwargs={})
    assert model.config.use_return_dict is False

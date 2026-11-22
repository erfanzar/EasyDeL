"""CPU-only eLarge configuration and reader-boundary regressions."""

import inspect
from types import SimpleNamespace

import datasets
import pytest
from easydel.infra.elarge import builders
from easydel.infra.elarge.model import eLargeModel
from easydel.infra.elarge.processing import materialize_base_config, normalize
from easydel.layers.quantization import QuantizationConfig

pytestmark = pytest.mark.cpu_ok


def config(**sections):
    return {"model": {"name_or_path": "unused-local-model", "task": "causal_lm"}, **sections}


@pytest.mark.parametrize("loader", [builders.to_from_pretrained_kwargs, builders.to_load_state_kwargs])
def test_apply_quantization_alone_preserves_checkpoint_quantizer_selection(loader):
    cfg = config(quantization={"apply_quantization": True})
    normalized = normalize(cfg)
    assert normalized["quantization"].get("model") is None
    kwargs = loader(normalized)
    assert kwargs["quantization_config"] is None
    assert kwargs["apply_quantization"] is True
    assert "quantization_config" not in kwargs["config_kwargs"]
    assert eLargeModel(cfg).get_from_pretrained_kwargs()["quantization_config"] is None


@pytest.mark.parametrize("loader", [builders.to_from_pretrained_kwargs, builders.to_load_state_kwargs])
@pytest.mark.parametrize("typed", [False, True])
def test_apply_quantization_alone_preserves_base_quantizer(loader, typed):
    quantizer = QuantizationConfig(dtype="nf4", group_size=32)
    requested = quantizer if typed else quantizer.to_dict()
    cfg = config(
        quantization={"apply_quantization": True},
        base_config={"values": {"quantization_config": requested}},
    )
    kwargs = loader(cfg)
    assert kwargs["quantization_config"] is None
    assert kwargs["apply_quantization"] is True
    inherited = kwargs["config_kwargs"]["quantization_config"]
    assert (inherited.to_dict() if typed else inherited) == quantizer.to_dict()
    base = materialize_base_config(cfg, prefer="sections")["quantization_config"]
    assert (base.to_dict() if typed else base) == quantizer.to_dict()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("preloaded", [False, True])
def test_public_elarge_runtime_options_reach_engine_boundary(monkeypatch, nested, preloaded):
    import easydel.inference
    from transformers import AutoTokenizer

    values = {"spec_debug_max_traces": "7", "spec_adaptive_conf_threshold": "0.625"}
    raw = {"runtime": values} if nested else values
    elm = eLargeModel(config(esurge=raw))
    assert builders.normalize_esurge_sections(raw)["runtime"] == {
        "spec_debug_max_traces": 7,
        "spec_adaptive_conf_threshold": 0.625,
    }
    # Exercise the setter as well as config-file normalization.
    elm.set_esurge(spec_debug_max_traces="9")
    runtime = builders.to_esurge_kwargs(elm.to_dict())["runtime"]
    assert runtime.spec_debug_max_traces == 9
    assert runtime.spec_adaptive_conf_threshold == 0.625
    model = SimpleNamespace(config=SimpleNamespace())
    monkeypatch.setattr(builders, "build_model", lambda cfg: model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(easydel.inference, "eSurge", lambda **kwargs: SimpleNamespace(**kwargs))
    engine = builders.build_esurge(elm.to_dict(), model=model if preloaded else None)
    assert engine.model is model
    assert engine.runtime.spec_debug_max_traces == 9
    assert engine.runtime.spec_adaptive_conf_threshold == 0.625


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_runtime_trace_budget_validation_is_not_bypassed_by_coercion(value):
    with pytest.raises(ValueError, match="spec_debug_max_traces"):
        builders.to_esurge_kwargs(config(esurge={"runtime": {"spec_debug_max_traces": value}}))


@pytest.mark.parametrize("value", [-0.1, 1.1, "nan", "inf"])
def test_runtime_confidence_validation_reaches_typed_schema(value):
    with pytest.raises(ValueError, match="spec_adaptive_conf_threshold"):
        builders.to_esurge_kwargs(config(esurge={"spec_adaptive_conf_threshold": value}))


@pytest.mark.parametrize("backend", ["ring", "triton"])
@pytest.mark.parametrize("preloaded", [False, True])
def test_decoding_engine_is_forwarded_for_loaded_and_preloaded_models(monkeypatch, backend, preloaded):
    import easydel.inference
    from transformers import AutoTokenizer

    model = SimpleNamespace(config=SimpleNamespace())
    elm = eLargeModel(config()).set_esurge(decoding_engine=backend)
    assert builders.to_esurge_kwargs(elm.to_dict())["decoding_engine"] == backend
    real_constructor = inspect.signature(easydel.inference.eSurge)
    monkeypatch.setattr(builders, "build_model", lambda cfg: model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def engine_boundary(*, model, decoding_engine, **kwargs):
        # Do not let this non-computing substitute accept kwargs that the real
        # public engine rejects before it can select a decoding backend.
        real_constructor.bind(model=model, decoding_engine=decoding_engine, **kwargs)
        return SimpleNamespace(model=model, decoding_engine=decoding_engine)

    monkeypatch.setattr(easydel.inference, "eSurge", engine_boundary)
    engine = builders.build_esurge(elm.to_dict(), model=model if preloaded else None)
    assert engine.model is model
    assert engine.decoding_engine == backend


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_reader_choice_is_observable_through_both_elarge_data_routes(tmp_path, monkeypatch, use_fast_loader, sharded):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"content": "fast-reader"}\n')
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *args, **kwargs: datasets.Dataset.from_list([{"content": "hf-reader"}])
    )
    cfg = config(
        mixture={
            "informs": [{"type": "json", "data_files": str(path)}],
            "streaming": False,
            "shuffle": False,
            "use_fast_loader": use_fast_loader,
        }
    )
    if sharded:
        source = builders.build_sharded_source(cfg)
        rows = [row for shard in source.shard_names for row in source.open_shard(shard)]
    else:
        rows = list(builders.build_dataset(cfg))
    assert [row["text"] for row in rows] == ["fast-reader" if use_fast_loader else "hf-reader"]


@pytest.mark.parametrize("visual", [False, True])
def test_legacy_informs_preserve_filters_and_global_post_filter_caps(visual):
    predicate = lambda row: row["keep"]  # noqa: E731
    inform = {"type": "json", "data_files": "unused.jsonl", "filter_fn": predicate, "num_rows": 8}
    if visual:
        inform["pixel_field"] = "pixels"
    kwargs = builders.to_data_mixture_kwargs(config(mixture={"informs": [inform], "max_samples": 2}))
    actual = kwargs["informs"][0]
    assert actual.filter_fn is predicate
    assert actual.max_samples == 2
    assert actual.num_rows == 8


def generation_boundary(fields):
    from easydel.infra.mixins.generation import EasyGenerationMixin

    class ModelBoundary(EasyGenerationMixin):
        """Real compatibility selection with array-free graph/state storage."""

        def __init__(self, fields, graphstate, graphother):
            self.fields = fields
            self.graphdef = dict(fields)
            self.config = SimpleNamespace(**fields)
            self.config.get_text_config = lambda: self.config
            self.graphstate = graphstate
            self.graphother = graphother

        def new_graphdef(self, *, recursive_update, **fields):
            return {**self.fields, **fields}

        def merge_module(self, graphdef, graphstate, graphother):
            return type(self)(graphdef, graphstate, graphother)

    return ModelBoundary(fields, object(), object())


def test_explicit_ring_selection_uses_registered_backend_and_preserves_state():
    from easydel.inference.esurge.engine.bootstrap import select_decoding_engine
    from easydel.operations import OperationRegistry
    from easydel.operations.requirements import CacheType

    fields = dict.fromkeys(("attn_mechanism", "decode_attn_mechanism", "mla_attn_mechanism"), "vanilla")
    original = generation_boundary(fields)
    selected = select_decoding_engine(original, "ring")
    selected_config = selected.config.get_text_config()
    assert {getattr(selected_config, field) for field in fields} == {"ring_paged"}
    assert original.fields == dict.fromkeys(fields, "vanilla")
    assert selected.graphstate is original.graphstate
    assert selected.graphother is original.graphother
    implementation = OperationRegistry.get("ring_paged")
    assert implementation.get_requirements().cache.supported & CacheType.RAGGED_PAGES


def test_explicit_triton_selection_rejects_cpu_instead_of_silently_substituting():
    from easydel.inference.esurge.engine.bootstrap import select_decoding_engine

    model = generation_boundary({"attn_mechanism": "vanilla"})
    with pytest.raises(ValueError, match="requires a GPU backend"):
        select_decoding_engine(model, "triton")


def test_default_esurge_kwargs_bind_real_constructor_without_initialization():
    from easydel.inference import eSurge

    kwargs = builders.to_esurge_kwargs(config())
    inspect.signature(eSurge).bind(model=object(), processor=object(), **kwargs)


@pytest.mark.parametrize("weight_source", ["inform", "mixture", "unweighted"])
def test_non_streaming_public_build_honors_sampling_weights(tmp_path, weight_source):
    informs = []
    for label, weight in (("a", 1.0), ("b", 0.0)):
        path = tmp_path / f"{label}.jsonl"
        path.write_text(f'{{"content": "{label}"}}\n')
        inform = {"name": label, "type": "json", "data_files": str(path)}
        if weight_source == "inform":
            inform["weight"] = weight
        informs.append(inform)
    mixture = {
        "informs": informs,
        "streaming": False,
        "shuffle": False,
        "stop_strategy": "first_exhausted",
        "cache_dir": str(tmp_path / "cache"),
    }
    if weight_source == "mixture":
        mixture["mixture_weights"] = {"a": 1.0, "b": 0.0}
    elm = eLargeModel(config(mixture=mixture))
    result = elm.build_dataset()
    assert isinstance(result, datasets.Dataset)
    assert list(result["text"]) == (["a", "b"] if weight_source == "unweighted" else ["a"])

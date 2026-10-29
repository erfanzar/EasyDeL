"""Pure configuration regressions for the eLarge facade."""

import importlib
from types import SimpleNamespace

import jax
import pytest
from easydel.infra.elarge.builders import normalize_esurge_sections, to_from_pretrained_kwargs, to_load_state_kwargs
from easydel.infra.elarge.defaults import DEFAULTS
from easydel.infra.elarge.model import eLargeModel
from easydel.infra.elarge.processing import (
    coerce_dtype,
    coerce_precision,
    materialize_base_config,
    normalize,
    normalize_task,
)
from easydel.infra.elarge.types import normalize_trainer_config
from easydel.infra.factory import TaskType
from easydel.layers.quantization import QuantizationConfig
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


def config(**sections):
    return {"model": {"name_or_path": "unused-local-model", "task": "causal-language-model"}, **sections}


def test_normalization_and_dictionary_exports_are_isolated():
    original_dtype = DEFAULTS["loader"]["dtype"]
    raw = config(base_config={"values": {"mask_max_position_embeddings": 1234}})
    first = eLargeModel(raw).set_dtype("fp32").add_dataset("a.json", dataset_type="json")
    second = eLargeModel(config(base_config={"values": {"mask_max_position_embeddings": 4567}}))
    assert DEFAULTS["loader"]["dtype"] == original_dtype
    assert second.to_dict()["loader"]["dtype"] == original_dtype
    assert second.to_dict()["esurge"]["runtime"]["max_model_len"] == 4567
    assert not second.to_dict()["mixture"].get("informs")
    exported = first.to_dict()
    exported["loader"]["dtype"] = "fp16"
    exported["mixture"]["informs"].append({"data_files": "extra"})
    assert first.to_dict()["loader"]["dtype"] == "fp32"
    assert len(first.to_dict()["mixture"]["informs"]) == 1
    assert raw == config(base_config={"values": {"mask_max_position_embeddings": 1234}})


@pytest.mark.parametrize("task", list(TaskType))
def test_canonical_task_values_and_underscore_variants(task):
    assert normalize_task(task.value) == task
    assert normalize_task(task.value.replace("-", "_")) == task
    assert normalize_task(task) == task


@pytest.mark.parametrize("alias, expected", [("lm", TaskType.CAUSAL_LM), ("seq2seq", TaskType.SEQUENCE_TO_SEQUENCE)])
def test_task_aliases_remain_supported(alias, expected):
    assert normalize_task(alias) == expected


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("float32", jax.lax.Precision.HIGHEST),
        ("tensorfloat32", jax.lax.Precision.HIGH),
        ("bfloat16_3x", jax.lax.Precision.HIGH),
        ("fastest", jax.lax.Precision.DEFAULT),
        ("HIGHEST", jax.lax.Precision.HIGHEST),
    ],
)
def test_precision_aliases(alias, expected):
    assert coerce_precision(alias) == expected


def test_unknown_numeric_configuration_uses_default_fallbacks():
    assert coerce_precision("higest") == jax.lax.Precision.DEFAULT
    assert coerce_dtype("bfloat") == jnp.float32
    assert coerce_dtype("bf16") == jnp.bfloat16
    assert coerce_dtype("int8") == jnp.dtype("int8")


def test_ready_kv_quantization_configuration_is_accepted():
    kv = QuantizationConfig(dtype="int8", group_size=64)
    actual = materialize_base_config(normalize(config(quantization={"kv_cache": kv})))["kv_cache_quantization_config"]
    assert isinstance(actual, QuantizationConfig)
    assert actual.to_dict() == kv.to_dict()


@pytest.mark.parametrize("method, dtype", [("nf4", "nf4"), ("a8bit", "int8")])
def test_quantization_setter_reaches_both_loader_contracts(method, dtype):
    elm = eLargeModel(config()).set_quantization(method, group_size=64)
    for kwargs in (elm.get_from_pretrained_kwargs(), to_load_state_kwargs(elm.to_dict())):
        assert kwargs["quantization_config"].dtype == dtype
        assert kwargs["quantization_config"].group_size == 64
        assert kwargs["apply_quantization"] is True
    elm.set_quantization(group_size=32)
    assert elm.get_from_pretrained_kwargs()["quantization_config"].group_size == 32


def test_yaml_quantization_shorthand_and_linear_options_are_consumed():
    kwargs = to_from_pretrained_kwargs(
        config(quantization={"method": "nf4", "linear_pattern": "q_proj", "linear_group_size": 32})
    )
    assert kwargs["quantization_config"].pattern == "q_proj"
    assert kwargs["quantization_config"].group_size == 32
    assert kwargs["apply_quantization"] is True


def test_explicit_disable_quantization_is_preserved():
    kwargs = to_from_pretrained_kwargs(config(quantization={"method": "nf4", "apply_quantization": False}))
    assert kwargs["apply_quantization"] is False


def test_partial_setters_preserve_previous_values():
    elm = eLargeModel(config())
    elm.set_esurge(max_num_seqs=55, hbm_utilization=0.71).set_esurge(tool_parser="hermes")
    elm.set_quantization("nf4", group_size=32, use_qmm_best_config=True).set_quantization(qmm_platform_override="xla")
    elm.set_mixture(batch_size=17, streaming=False, use_fast_loader=False).set_mixture(seed=9)
    saved = elm.to_dict()
    assert saved["esurge"]["runtime"]["max_num_seqs"] == 55
    assert saved["esurge"]["cache"]["hbm_utilization"] == 0.71
    assert saved["quantization"]["group_size"] == 32
    assert saved["quantization"]["use_qmm_best_config"] is True
    assert saved["mixture"]["batch_size"] == 17
    assert saved["mixture"]["streaming"] is False
    assert saved["mixture"]["use_fast_loader"] is False


def test_loading_builders_share_defaults_and_preserve_api_specific_options():
    cfg = config(loader={"from_torch": True, "trust_remote_code": True}, base_config={"values": {"scan_layers": True}})
    pretrained = to_from_pretrained_kwargs(cfg)
    state = to_load_state_kwargs(cfg)
    for key in pretrained.keys() & state.keys() - {"device"}:
        assert pretrained[key] == state[key]
    assert pretrained["config_kwargs"]["scan_layers"] is True
    assert pretrained["from_torch"] is True
    assert "from_torch" not in state
    assert state["device"] == "cpu"
    assert to_load_state_kwargs(config())["config_kwargs"]["scan_layers"] is False


@pytest.mark.parametrize("bad", [{"max_num_seq": 5}, {"runtime": {"max_num_seq": 5}}, {"typo": None}])
def test_unknown_esurge_options_raise(bad):
    with pytest.raises(ValueError, match=r"[Uu]nknown"):
        normalize_esurge_sections(bad)


def test_kv_dtype_survives_esurge_setter_and_reaches_model_config():
    elm = eLargeModel(config()).set_esurge(kv_cache_dtype="fp16", decoding_engine="triton")
    assert elm.to_dict()["esurge"]["model"]["decoding_engine"] == "triton"
    assert elm.get_from_pretrained_kwargs()["config_kwargs"]["kvdtype"] == jnp.float16
    before = elm.to_dict()
    with pytest.raises(ValueError):
        elm.set_esurge(typo=1)
    assert elm.to_dict() == before


def test_kv_dtype_is_applied_to_preloaded_engine_models(monkeypatch):
    import easydel.inference
    from easydel.infra.elarge.builders import build_esurge
    from transformers import AutoTokenizer

    model = SimpleNamespace(config=SimpleNamespace(kvdtype=jnp.bfloat16))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(easydel.inference, "eSurge", lambda **kwargs: SimpleNamespace(**kwargs))
    engine = build_esurge(config(esurge={"kv_cache_dtype": "fp16"}), model=model)
    assert engine.model.config.kvdtype == jnp.float16


def test_summary_includes_sectioned_esurge_values():
    elm = eLargeModel(config()).set_esurge(max_model_len=1234, max_num_seqs=19, hbm_utilization=0.75)
    summary = str(elm)
    assert "1,234 context x 19 sequences" in summary
    assert "75% HBM" in summary


def test_set_model_rebuilds_model_tokenizer_and_state(monkeypatch):
    model_module = importlib.import_module("easydel.infra.elarge.model")
    elm = eLargeModel(config())
    elm._model = object()
    elm._state = object()
    elm._tokenizer = object()
    monkeypatch.setattr(model_module, "build_model", lambda cfg: SimpleNamespace(name=cfg["model"]["name_or_path"]))
    monkeypatch.setattr(model_module.AutoTokenizer, "from_pretrained", lambda name, **kwargs: SimpleNamespace(name=name))
    elm.set_model("new-model")
    assert elm.build_model().name == "new-model"
    assert elm.build_tokenizer().name == "new-model"
    assert elm._state is None


def test_training_arguments_filter_routing_and_trainer_agnostic_options():
    elm = eLargeModel(config(trainer={"trainer_type": "sft", "bucket_configs": None, "max_prompt_length": 16}))

    class CustomArguments:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

    assert elm.build_training_arguments(args_class=CustomArguments, learning_rate=0.25).learning_rate == 0.25
    arguments = elm.build_training_arguments(args_class=CustomArguments, learning_rate=0.5, lerning_rate=0.75)
    assert arguments.learning_rate == 0.5
    assert not hasattr(arguments, "max_prompt_length")
    assert not hasattr(arguments, "lerning_rate")


def test_training_argument_constructor_error_is_not_retried():
    elm = eLargeModel(config())
    attempts = []

    class BrokenArguments:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            raise TypeError("constructor validation failed")

    with pytest.raises(TypeError, match="constructor validation failed"):
        elm.build_training_arguments(args_class=BrokenArguments)
    assert len(attempts) == 1
    assert not set(attempts[0]) & {"bucket_configs", "bucket_rule", "bucket_datasets"}


def test_custom_trainer_accepts_callback_without_name_special_case():
    elm = eLargeModel(config(trainer={"trainer_type": "sft"}))
    elm._model = object()
    elm._tokenizer = object()
    callback = object()

    class Arguments:
        def __init__(self, **kwargs):
            self.values = kwargs

    class RenamedTrainer:
        def __init__(self, *, model, arguments, processing_class, train_dataset, bema_callback):
            self.callback = bema_callback
            self.arguments = arguments

    trainer = elm.build_trainer(
        train_dataset=[{"text": "a"}], trainer_class=RenamedTrainer, args_class=Arguments, bema_callback=callback
    )
    assert trainer.callback is callback
    assert "bema_callback" not in trainer.arguments.values


@pytest.mark.parametrize("alias, canonical", [("nash_md", "nash-md"), ("agentic_moshpit", "agentic-moshpit")])
def test_trainer_aliases_are_canonicalized_before_dispatch(alias, canonical):
    assert normalize_trainer_config({"trainer_type": alias})["trainer_type"] == canonical

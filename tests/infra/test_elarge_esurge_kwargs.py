from easydel.infra.elarge_model.builders import to_esurge_kwargs
from easydel.infra.elarge_model.elarge_model import eLargeModel


def test_to_esurge_kwargs_forwards_string_extra_stops():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"extra_stops": "<|user|>"},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["extra_stops"] == "<|user|>"


def test_to_esurge_kwargs_normalizes_iterable_extra_stops():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"extra_stops": ("<|user|>", "</assistant>")},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["extra_stops"] == ["<|user|>", "</assistant>"]


def test_to_esurge_kwargs_keeps_extra_stops_none_by_default():
    cfg = {"model": {"name_or_path": "dummy-model"}}

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["extra_stops"] is None
    assert kwargs["bind_graphstate_for_aot"] is False


def test_to_esurge_kwargs_defaults_data_parallelism_axis_to_dp():
    cfg = {"model": {"name_or_path": "dummy-model"}}

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["data_parallelism_axis"] == "dp"


def test_to_esurge_kwargs_forwards_data_parallelism_axis():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"data_parallelism_axis": "ep"},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["data_parallelism_axis"] == "ep"


def test_to_esurge_kwargs_forwards_bind_graphstate_for_aot():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"bind_graphstate_for_aot": True},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["bind_graphstate_for_aot"] is True


def test_set_esurge_preserves_parsers_when_omitted_and_clears_when_none():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "esurge": {}}

    elm.set_esurge(tool_parser="openai", reasoning_parser="deepseek_r1")
    assert elm.config["esurge"]["tool_parser"] == "openai"
    assert elm.config["esurge"]["reasoning_parser"] == "deepseek_r1"

    elm.set_esurge(max_num_seqs=8)
    assert elm.config["esurge"]["tool_parser"] == "openai"
    assert elm.config["esurge"]["reasoning_parser"] == "deepseek_r1"

    elm.set_esurge(tool_parser=None, reasoning_parser=None)
    assert elm.config["esurge"]["tool_parser"] is None
    assert elm.config["esurge"]["reasoning_parser"] is None


def test_set_esurge_bind_graphstate_for_aot_override_is_optional():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "esurge": {}}

    elm.set_esurge(bind_graphstate_for_aot=True)
    assert elm.config["esurge"]["bind_graphstate_for_aot"] is True

    elm.set_esurge(max_num_seqs=8)
    assert elm.config["esurge"]["bind_graphstate_for_aot"] is True

    elm.set_esurge(bind_graphstate_for_aot=False)
    assert elm.config["esurge"]["bind_graphstate_for_aot"] is False

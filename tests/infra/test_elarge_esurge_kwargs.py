from easydel.infra.elarge_model.elarge_model import eLargeModel
from easydel.infra.elarge_model.builders import to_esurge_kwargs


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

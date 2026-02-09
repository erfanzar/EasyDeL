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

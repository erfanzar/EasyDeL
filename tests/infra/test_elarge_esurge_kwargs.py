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


def test_to_esurge_kwargs_defaults_distributed_controls():
    cfg = {"model": {"name_or_path": "dummy-model"}}

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["distributed_mode"] is False
    assert kwargs["distributed_role"] == "auto"
    assert kwargs["distributed_world_size"] is None
    assert kwargs["distributed_rank"] is None
    assert kwargs["distributed_control_port"] == 19666
    assert kwargs["distributed_control_bind_host"] == "0.0.0.0"
    assert kwargs["distributed_step_timeout_s"] == 30.0
    assert kwargs["distributed_connect_timeout_s"] == 15.0
    assert kwargs["distributed_verify_sampling_digest"] is True


def test_to_esurge_kwargs_forwards_distributed_controls():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {
            "distributed_mode": True,
            "distributed_role": "worker",
            "distributed_service_name": "esurge-workers.internal",
            "distributed_world_size": 4,
            "distributed_rank": 2,
            "distributed_control_port": 21001,
            "distributed_control_bind_host": "127.0.0.1",
            "distributed_advertise_addr": "10.0.0.12",
            "distributed_auth_token": "secret",
            "distributed_step_timeout_s": 45.0,
            "distributed_connect_timeout_s": 20.0,
            "distributed_verify_sampling_digest": False,
        },
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["distributed_mode"] is True
    assert kwargs["distributed_role"] == "worker"
    assert kwargs["distributed_service_name"] == "esurge-workers.internal"
    assert kwargs["distributed_world_size"] == 4
    assert kwargs["distributed_rank"] == 2
    assert kwargs["distributed_control_port"] == 21001
    assert kwargs["distributed_control_bind_host"] == "127.0.0.1"
    assert kwargs["distributed_advertise_addr"] == "10.0.0.12"
    assert kwargs["distributed_auth_token"] == "secret"
    assert kwargs["distributed_step_timeout_s"] == 45.0
    assert kwargs["distributed_connect_timeout_s"] == 20.0
    assert kwargs["distributed_verify_sampling_digest"] is False


def test_set_esurge_preserves_parsers_when_omitted_and_clears_when_none():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "esurge": {}}

    elm.set_esurge(tool_parser="openai", reasoning_parser="deepseek_r1")
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["tool_parser"] == "openai"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["reasoning_parser"] == "deepseek_r1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(max_num_seqs=8)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["tool_parser"] == "openai"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["reasoning_parser"] == "deepseek_r1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(tool_parser=None, reasoning_parser=None)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["tool_parser"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["reasoning_parser"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_set_esurge_bind_graphstate_for_aot_override_is_optional():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "esurge": {}}

    elm.set_esurge(bind_graphstate_for_aot=True)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["bind_graphstate_for_aot"] is True  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(max_num_seqs=8)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["bind_graphstate_for_aot"] is True  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(bind_graphstate_for_aot=False)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["bind_graphstate_for_aot"] is False  # pyright: ignore[reportTypedDictNotRequiredAccess]

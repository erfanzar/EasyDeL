# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import easydel.inference as inference_module
from easydel.infra.elarge.builders import to_esurge_kwargs
from easydel.infra.elarge.model import eLargeModel
from easydel.scripts.elarge import _run_action


def test_to_esurge_kwargs_forwards_string_extra_stops():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"extra_stops": "<|user|>"},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["parsing"].extra_stops == "<|user|>"


def test_to_esurge_kwargs_normalizes_iterable_extra_stops():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"extra_stops": ("<|user|>", "</assistant>")},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["parsing"].extra_stops == ["<|user|>", "</assistant>"]


def test_to_esurge_kwargs_keeps_extra_stops_none_by_default():
    cfg = {"model": {"name_or_path": "dummy-model"}}

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["parsing"].extra_stops is None
    assert kwargs["runtime"].enable_window_aware_runtime_cap is False


def test_to_esurge_kwargs_forwards_enable_window_aware_runtime_cap():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"enable_window_aware_runtime_cap": False},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["runtime"].enable_window_aware_runtime_cap is False


def test_to_esurge_kwargs_treats_null_window_aware_runtime_cap_as_default_false():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"enable_window_aware_runtime_cap": None},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["runtime"].enable_window_aware_runtime_cap is False


def test_to_esurge_kwargs_defaults_data_parallelism_axis_to_dp():
    cfg = {"model": {"name_or_path": "dummy-model"}}

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["cache"].data_parallelism_axis == "dp"


def test_to_esurge_kwargs_forwards_data_parallelism_axis():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"data_parallelism_axis": "ep"},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["cache"].data_parallelism_axis == "ep"


def test_to_esurge_kwargs_forwards_worker_startup_timeout():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"worker_startup_timeout": "75.5"},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["workers"].worker_startup_timeout == 75.5


def test_to_esurge_kwargs_forwards_async_scheduling():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"async_scheduling": False},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["runtime"].async_scheduling is False


def test_to_esurge_kwargs_forwards_pp_microbatch_policy():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"pp_microbatch_size": "4"},
    }

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["runtime"].pp_microbatch_size == 4
    assert kwargs["runtime"].pp_microbatch_count == "auto"


def test_to_esurge_kwargs_defaults_distributed_controls():
    cfg = {"model": {"name_or_path": "dummy-model"}}

    kwargs = to_esurge_kwargs(cfg)

    assert kwargs["distributed"].distributed_mode is False
    assert kwargs["distributed"].distributed_role == "auto"
    assert kwargs["distributed"].distributed_world_size is None
    assert kwargs["distributed"].distributed_rank is None
    assert kwargs["distributed"].distributed_control_port == 19666
    assert kwargs["distributed"].distributed_control_bind_host == "0.0.0.0"
    assert kwargs["distributed"].distributed_step_timeout_s == 30.0
    assert kwargs["distributed"].distributed_connect_timeout_s == 15.0
    assert kwargs["distributed"].distributed_verify_sampling_digest is True


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

    assert kwargs["distributed"].distributed_mode is True
    assert kwargs["distributed"].distributed_role == "worker"
    assert kwargs["distributed"].distributed_service_name == "esurge-workers.internal"
    assert kwargs["distributed"].distributed_world_size == 4
    assert kwargs["distributed"].distributed_rank == 2
    assert kwargs["distributed"].distributed_control_port == 21001
    assert kwargs["distributed"].distributed_control_bind_host == "127.0.0.1"
    assert kwargs["distributed"].distributed_advertise_addr == "10.0.0.12"
    assert kwargs["distributed"].distributed_auth_token == "secret"
    assert kwargs["distributed"].distributed_step_timeout_s == 45.0
    assert kwargs["distributed"].distributed_connect_timeout_s == 20.0
    assert kwargs["distributed"].distributed_verify_sampling_digest is False


def test_set_esurge_preserves_parsers_when_omitted_and_clears_when_none():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "esurge": {}}

    elm.set_esurge(tool_parser="openai", reasoning_parser="deepseek_r1")
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["parsing"]["tool_parser"] == "openai"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["parsing"]["reasoning_parser"] == "deepseek_r1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(max_num_seqs=8)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["parsing"]["tool_parser"] == "openai"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["parsing"]["reasoning_parser"] == "deepseek_r1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(tool_parser=None, reasoning_parser=None)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["parsing"]["tool_parser"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["parsing"]["reasoning_parser"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_set_esurge_window_aware_runtime_cap_override_is_optional():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "esurge": {}}

    elm.set_esurge(enable_window_aware_runtime_cap=False)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["runtime"]["enable_window_aware_runtime_cap"] is False  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(max_num_seqs=8)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["runtime"]["enable_window_aware_runtime_cap"] is False  # pyright: ignore[reportTypedDictNotRequiredAccess]

    elm.set_esurge(enable_window_aware_runtime_cap=True)
    esurge = elm.config["esurge"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert esurge["runtime"]["enable_window_aware_runtime_cap"] is True  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_serve_action_migrates_deprecated_tool_parser_name_to_esurge(monkeypatch):
    captured = {}

    class FakeServer:
        def __init__(self, surge, **kwargs):
            captured["surge"] = surge
            captured["server_kwargs"] = kwargs

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs

    class FakeElm:
        def __init__(self):
            self.config = {"esurge": {}}
            self.set_esurge_calls = []
            self.validated = False

        def set_esurge(self, **kwargs):
            self.set_esurge_calls.append(kwargs)
            self.config.setdefault("esurge", {}).update(kwargs)

        def validate(self):
            self.validated = True

        @staticmethod
        def build_esurge():
            return "engine"

    monkeypatch.setattr(inference_module, "eSurgeApiServer", FakeServer)

    elm = FakeElm()
    _run_action(
        elm,
        "serve",
        {
            "tool_parser_name": "hermes",
            "host": "127.0.0.1",
            "port": 9000,
        },
    )

    assert elm.validated is True
    assert elm.set_esurge_calls == [{"tool_parser": "hermes"}]
    assert captured["surge"] == "engine"
    assert "tool_parser_name" not in captured["server_kwargs"]
    assert captured["run_kwargs"]["host"] == "127.0.0.1"
    assert captured["run_kwargs"]["port"] == 9000


def test_serve_action_rejects_deprecated_tool_parser_name_when_engine_disagrees():
    class FakeElm:
        def __init__(self):
            self.config = {"esurge": {"tool_parser": "qwen3_xml"}}

        def set_esurge(self, **kwargs):
            raise AssertionError(f"Unexpected set_esurge call: {kwargs}")

        @staticmethod
        def validate():
            raise AssertionError("validate should not run when config is invalid")

        @staticmethod
        def build_esurge():
            raise AssertionError("build_esurge should not run when config is invalid")

    with pytest.raises(SystemExit, match="disagrees with `esurge.tool_parser`"):  # noqa
        _run_action(FakeElm(), "serve", {"tool_parser_name": "hermes"})

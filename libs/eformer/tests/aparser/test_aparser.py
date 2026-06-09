# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for argument parser utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pytest

from eformer.aparser._aparser import Argu, DataClassArgumentParser, string_to_bool


class Mode(Enum):
    fast = "fast"
    slow = "slow"


@dataclass
class Config:
    flag: bool = False
    enabled: bool = True
    count: int = 3
    opt: int | None = None
    mode: Literal["fast", "slow"] = "fast"
    speed: Mode = Mode.fast
    run_id: int = 1
    alias_value: int = Argu(aliases="--alias", default=1)


def test_string_to_bool():
    assert string_to_bool("yes") is True
    assert string_to_bool("no") is False
    assert string_to_bool(True) is True
    assert string_to_bool(False) is False
    with pytest.raises(Exception):  # noqa: B017
        string_to_bool("maybe")


def test_parse_args_into_dataclasses():
    parser = DataClassArgumentParser(Config)
    (cfg,) = parser.parse_args_into_dataclasses(
        args=[
            "--flag",
            "true",
            "--no-enabled",
            "--count",
            "5",
            "--opt",
            "7",
            "--mode",
            "slow",
            "--speed",
            "slow",
            "--run-id",
            "9",
            "--alias",
            "11",
        ],
        look_for_args_file=False,
    )

    assert cfg.flag is True
    assert cfg.enabled is False
    assert cfg.count == 5
    assert cfg.opt == 7
    assert cfg.mode == "slow"
    assert cfg.speed == "slow"
    assert cfg.run_id == 9
    assert cfg.alias_value == 11


def test_parse_dict_and_files(tmp_path):
    parser = DataClassArgumentParser(Config)

    (cfg,) = parser.parse_dict({"count": 4, "mode": "fast"})
    assert cfg.count == 4
    assert cfg.mode == "fast"

    with pytest.raises(ValueError):
        parser.parse_dict({"count": 1, "extra": 2})

    json_path = tmp_path / "config.json"
    json_path.write_text('{"count": 8, "mode": "slow"}', encoding="utf-8")
    (cfg_json,) = parser.parse_json_file(json_path)
    assert cfg_json.count == 8
    assert cfg_json.mode == "slow"

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("count: 6\nmode: fast\n", encoding="utf-8")
    (cfg_yaml,) = parser.parse_yaml_file(yaml_path)
    assert cfg_yaml.count == 6
    assert cfg_yaml.mode == "fast"


def test_parse_args_file(tmp_path):
    parser = DataClassArgumentParser(Config)
    args_path = tmp_path / "config.args"
    args_path.write_text("--count 11 --mode slow", encoding="utf-8")

    (cfg,) = parser.parse_args_into_dataclasses(
        args=["--flag", "true"],
        args_filename=str(args_path),
        look_for_args_file=False,
    )

    assert cfg.count == 11
    assert cfg.mode == "slow"
    assert cfg.flag is True

"""Typed-config namespace methods, default copying and ``as_dict`` (pure Python)."""

import typing as tp

import pytest
from easydel.typings import ConfigDict, typed_config

pytestmark = pytest.mark.cpu_ok


def test_namespace_methods_and_default_copying_work_after_replace():
    def describe(self):
        return f"{self.name}: {','.join(self.tags)}"

    @typed_config(defaults={"tags": []}, namespace={"describe": describe})
    class Settings(tp.TypedDict):
        name: str
        tags: list[str]

    config = Settings.from_dict(name="first")
    config.tags.append("tag")
    other = Settings.from_dict(name="second")
    replaced = config.replace(name="updated")

    assert config.describe() == "first: tag"
    assert other.describe() == "second: "
    assert replaced.describe() == "updated: tag"
    assert replaced.to_dict() == {"name": "updated", "tags": ["tag"]}
    assert config.name == "first"


def test_as_dict_alias_deeply_unwraps_configs_and_returns_independent_containers():
    config = ConfigDict(nested=ConfigDict(values=[ConfigDict(number=1)]))
    result = config.as_dict()
    assert result == {"nested": {"values": [{"number": 1}]}}
    assert type(result) is dict
    assert type(result["nested"]) is dict
    assert type(result["nested"]["values"][0]) is dict
    result["nested"]["values"][0]["number"] = 2
    assert config.nested["values"][0].number == 1

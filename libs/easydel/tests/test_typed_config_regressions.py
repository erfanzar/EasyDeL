"""Pure-Python regressions for typed-config transport and schema composition."""

import pickle
import typing as tp

import pytest
from easydel.inference.esurge.config import eSurgeContextConfig, eSurgeRuntimeConfig
from easydel.typings import ConfigDict, typed_config

pytestmark = pytest.mark.cpu_ok


@typed_config
class _First(tp.TypedDict):
    first: int


@typed_config
class _Second(tp.TypedDict):
    second: int


@typed_config
class _UnionConfig(tp.TypedDict):
    child: _First | _Second


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_public_config_standard_pickle_roundtrip(protocol):
    original = eSurgeRuntimeConfig.from_dict(max_model_len=1234)
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is type(original)
    assert restored.to_dict() == original.to_dict()
    assert restored.max_model_len == 1234
    assert restored.replace(max_model_len=5678).max_model_len == 5678


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_public_config_with_explicit_token_budget_standard_pickle_roundtrip(protocol):
    original = eSurgeRuntimeConfig.from_dict(max_model_len=1234, max_num_batched_tokens=2048)
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is type(original)
    assert restored.to_dict() == original.to_dict()
    assert restored.max_model_len == 1234


def test_pickle_preserves_nested_config_types_and_shared_references():
    child = _Second.from_dict(second=7)
    original = ConfigDict(child=child, alias=child)
    original["self"] = original
    restored = pickle.loads(pickle.dumps(original))
    assert restored is restored["self"]
    assert restored.child is restored.alias
    assert type(restored.child) is type(child)
    assert restored.child.second == 7


def test_optional_child_preserves_required_base_fields():
    class Base(tp.TypedDict):
        required: int

    @typed_config
    class Child(Base, total=False):
        optional: str

    with pytest.raises(TypeError, match=r"missing required field.*required"):
        Child.from_dict()
    assert Child.from_dict(required=1).to_dict() == {"required": 1}


def test_required_child_preserves_optional_base_fields():
    class Base(tp.TypedDict, total=False):
        optional: int

    @typed_config
    class Child(Base):
        required: str

    assert Child.from_dict(required="ok").to_dict() == {"required": "ok"}
    with pytest.raises(TypeError, match=r"missing required field.*required"):
        Child.from_dict(optional=1)


def test_resolved_required_wrappers_override_postponed_metadata():
    # Quoted wrappers model postponed annotations without local-name resolution.
    @typed_config
    class Config(tp.TypedDict):
        required: "tp.Required[int]"
        optional: "tp.NotRequired[int]"

    assert Config.from_dict(required=1).to_dict() == {"required": 1}
    with pytest.raises(TypeError, match=r"missing required field.*required"):
        Config.from_dict()


@pytest.mark.parametrize("payload, spec", [({"first": 1}, _First), ({"second": 2}, _Second)])
def test_union_tries_each_config_schema(payload, spec):
    result = _UnionConfig.from_dict(child=payload)
    assert type(result.child) is spec.__typed_config_instance_cls__
    assert result.child.to_dict() == payload


def test_union_continues_after_missing_required_field():
    @typed_config
    class NeedsExtra(tp.TypedDict):
        first: int
        extra: str

    @typed_config
    class Config(tp.TypedDict):
        child: NeedsExtra | _First

    result = Config.from_dict(child={"first": 1})
    assert type(result.child) is _First.__typed_config_instance_cls__


def test_union_preserves_existing_branch_instance():
    child = _Second.from_dict(second=2)
    assert _UnionConfig.from_dict(child=child).child is child


def test_union_reports_failure_of_all_config_schemas():
    with pytest.raises(TypeError, match=r"child|union|Union"):
        _UnionConfig.from_dict(child={"other": 3})


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_union_does_not_swallow_post_init_errors(error_type):
    def validate(config):
        raise error_type("validator failure")

    @typed_config(post_init=validate)
    class Validated(tp.TypedDict):
        first: int

    @typed_config
    class Config(tp.TypedDict):
        child: Validated | _First

    with pytest.raises(error_type, match="validator failure"):
        Config.from_dict(child={"first": 1})


@pytest.mark.parametrize("updates", [{}, {"reserve_tokens": 20}])
def test_replace_always_copies_even_without_updates(updates):
    original = eSurgeContextConfig.from_dict(reserve_tokens=10)
    clone = original.replace(**updates)
    assert clone is not original
    assert clone.reserve_tokens == updates.get("reserve_tokens", 10)
    clone.reserve_tokens = 99
    assert original.reserve_tokens == 10
    assert eSurgeContextConfig.coerce_config(original) is original

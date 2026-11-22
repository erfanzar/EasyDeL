"""Pure-Python regression coverage for utility metadata and compatibility APIs."""

import importlib
import logging
import pickle
from collections import OrderedDict
from types import SimpleNamespace

import pytest
from easydel.utils import analyze_memory, flags, traversals
from easydel.utils.checkpoint_compat import rename_legacy_checkpoint_leaves
from easydel.utils.jit_context import CompilationContext, get_jit_context, jit_context, peek_jit_context
from easydel.utils.lazy_import import LazyModule
from easydel.utils.readme_generator import EASYDEL_TRAINER_README_TEMPLATE, ModelInfo, ReadmeGenerator
from jinja2 import Environment

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("legacy_first", [False, True])
def test_legacy_checkpoint_collision_has_documented_precedence(legacy_first):
    old_key = ("parameters", "projection", "kernel")
    new_key = (*old_key[:-1], "weight")
    pairs = [(old_key, "legacy"), (new_key, "modern")]
    if not legacy_first:
        pairs.reverse()
    assert rename_legacy_checkpoint_leaves(dict(pairs)) == {new_key: "legacy"}


def test_model_declared_scale_does_not_collide_with_weight():
    scale = ("parameters", "attn_hc", "scale")
    weight = (*scale[:-1], "weight")
    state = {scale: "scale value", weight: "weight value"}
    assert rename_legacy_checkpoint_leaves(state, known_keys={scale, weight}) == state


def test_memory_module_does_not_configure_application_logging(monkeypatch):
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level

    def forbidden(*args, **kwargs):
        pytest.fail("A library import must not call logging.basicConfig")

    monkeypatch.setattr(logging, "basicConfig", forbidden)
    importlib.reload(analyze_memory)
    assert root.handlers == handlers
    assert root.level == level


def test_tree_compatibility_roundtrip_preserves_empty_nodes_and_integer_paths():
    tree = {"block": {0: {"weight": 7}, 1: {}}, "scale": 3}
    flat = traversals.flatten_dict(tree, keep_empty_nodes=True)
    assert flat[("block", 1)] is traversals.empty_node
    assert traversals.unflatten_dict(flat) == tree
    joined = traversals.flatten_dict({"block": {0: {"weight": 7}}}, sep=".")
    assert joined == {"block.0.weight": 7}
    assert traversals.string_key_to_int(traversals.unflatten_dict(joined, sep=".")) == {"block": {0: {"weight": 7}}}


def test_documented_tree_helpers_remain_available():
    tree = OrderedDict([("a", {"b": 3})])
    assert traversals.flatten_dict(tree) == {("a", "b"): 3}
    assert traversals.named_tree_map(lambda path, value: (path, value), {"a": 3}, sep=".") == {"a": ("a", 3)}
    assert traversals.tree_apply({"a": lambda x: x + 1}, {"a": 3}) == {"a": 4}
    assert traversals.recursive_merge({"a": [1, 2]}, {"a": [9]}) == {"a": [9, 2]}
    assert traversals.flatten_tree({"a": (3, 4)}) == {("a", "0"): 3, ("a", "1"): 4}
    assert traversals.is_iterable("value")
    result = traversals.StateValidationResult(True, set(), {})
    assert result.is_valid and result.missing_keys == set()


def test_public_jit_context_still_supports_nested_contexts():
    before = peek_jit_context()
    prefill = CompilationContext("prefill", phase="prefill")
    decode = CompilationContext("decode", phase="decode")
    with jit_context(prefill):
        assert get_jit_context(CompilationContext) == prefill
        with jit_context(decode):
            assert get_jit_context() == decode
        assert get_jit_context() == prefill
    assert peek_jit_context() == before


def test_missing_backend_exports_remain_discoverable_but_raise_on_use():
    module = LazyModule(
        "optional_test_package",
        __file__,
        {frozenset({"_easydel_nonexistent_optional_backend_"}): {"not_importable": {"OptionalModel"}}},
    )
    assert "OptionalModel" in dir(module)
    placeholder = module.OptionalModel
    assert placeholder.__name__ == "OptionalModel"
    with pytest.raises(ImportError, match="_easydel_nonexistent_optional_backend_"):
        placeholder()
    with pytest.raises(ImportError, match="_easydel_nonexistent_optional_backend_"):
        _ = placeholder.from_pretrained
    restored = pickle.loads(pickle.dumps(module))
    with pytest.raises(ImportError, match="_easydel_nonexistent_optional_backend_"):
        restored.OptionalModel()


def test_available_backend_exports_import_normally(tmp_path, monkeypatch):
    package = tmp_path / "optional_test_package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "objects.py").write_text("class AvailableModel:\n    result = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = LazyModule(
        "optional_test_package",
        str(package / "__init__.py"),
        {frozenset({"json"}): {"objects": {"AvailableModel"}}},
    )
    assert module.AvailableModel.result == 42


def test_generated_cards_use_six_mesh_axes_and_shared_attention_mapping():
    info = ModelInfo("test", "test", "user/test", attn_mechanism="paged_flash_attention")
    card = ReadmeGenerator().generate_readme(info)
    assert "AttentionMechanisms.PAGED_FLASH_ATTENTION" in card
    assert "axis_dims: [1, 1, -1, 1, 1, 1]" in card
    model = SimpleNamespace(
        model_task_str="CausalLM",
        attn_mechanism_str="paged_flash_attention",
        dtype_str="float32",
        param_dtype_str="float32",
        device_info=SimpleNamespace(platform="UNKNOWN", device_count=8, host_device_count=8),
    )
    trainer_card = (
        Environment()
        .from_string(EASYDEL_TRAINER_README_TEMPLATE)
        .render(model=model, arguments=SimpleNamespace(), config=SimpleNamespace(), easydel_version="test")
    )
    assert "AttentionMechanisms.PAGED_FLASH_ATTENTION" in trainer_card
    assert '("pp","dp","fsdp","ep","tp","sp")' in trainer_card
    assert "8 devices: (1, 1, 8, 1, 1, 1)" in trainer_card
    assert "8 devices: (1, 2, 4, 1, 1, 1)" in trainer_card
    assert "8 devices: (1, 1, 4, 1, 2, 1)" in trainer_card


def test_model_repository_metadata_remains_public(tmp_path):
    info = ModelInfo("test", "test", str(tmp_path))
    assert info.repo_is_local
    assert info.repo_badge_message == tmp_path.name


def test_flag_catalog_and_documentation_are_public_capabilities(monkeypatch):
    name = flags.EASYDEL_WORKER_STARTUP_TIMEOUT
    monkeypatch.setenv(name, "12.5")
    assert flags.get_float(name) == 12.5
    assert any(spec.name == name for spec in flags.registered_flags())
    assert name in flags.render_markdown()


def test_existing_host_copy_aliases_keep_canonical_precedence(monkeypatch):
    monkeypatch.delenv(flags.EASYDEL_PREFERRED_HOST_COPY_INDEX, raising=False)
    monkeypatch.delenv(flags.EASYDEL_PREFERRED_HOST_COPY, raising=False)
    monkeypatch.setenv("EASYDEL_PERFRED_HOST_COPY_INDEX", "3")
    monkeypatch.setenv("EASYDEL_PERFRED_HOST_COPY", "cpu")
    assert flags.get_int(flags.EASYDEL_PREFERRED_HOST_COPY_INDEX) == 3
    assert flags.get_str(flags.EASYDEL_PREFERRED_HOST_COPY) == "cpu"
    monkeypatch.setenv(flags.EASYDEL_PREFERRED_HOST_COPY_INDEX, "1")
    monkeypatch.setenv(flags.EASYDEL_PREFERRED_HOST_COPY, "none")
    assert flags.get_int(flags.EASYDEL_PREFERRED_HOST_COPY_INDEX) == 1
    assert flags.get_str(flags.EASYDEL_PREFERRED_HOST_COPY) == "none"
    assert "EASYDEL_PERFRED_HOST_COPY_INDEX" in flags.render_markdown()

"""CPU-only regressions for shared parser helpers and legacy worker settings."""

import sys
from types import ModuleType

import pytest
from easydel.utils import flags
from easydel.utils._parser_utils import import_from_path, is_list_of

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("value", "typ", "check", "expected"),
    [
        (None, int, "first", False),
        ((1, 2), int, "all", False),
        ([], str, "first", True),
        ([], str, "all", True),
        ([1, "two"], int, "first", True),
        ([1, "two"], int, "all", False),
        (["one", 2], int, "first", False),
        ([1, 2], int, "all", True),
        ([1, "two"], (int, str), "all", True),
    ],
)
def test_list_check_modes(value, typ, check, expected):
    assert is_list_of(value, typ, check=check) is expected


@pytest.mark.parametrize("value", [[], [1]])
def test_list_invalid_mode_retains_both_legacy_policies(value):
    with pytest.raises(AssertionError):
        is_list_of(value, int, check="unknown")
    assert is_list_of(value, int, check="unknown", raise_on_invalid_check=False) is False
    assert is_list_of(None, int, check="unknown") is False


def test_list_checks_preserve_isinstance_exceptions_and_empty_short_circuit():
    assert is_list_of([], None)
    with pytest.raises(TypeError):
        is_list_of([1], None)


def test_plugin_registered_before_execution_and_replaces_old_module(tmp_path, monkeypatch):
    name = "_shared_parser_plugin"
    path = tmp_path / "plugin.py"
    path.write_text(
        "import sys\n"
        "from dataclasses import dataclass\n"
        "registered_during_execution = sys.modules[__name__].__dict__ is globals()\n"
        "@dataclass\n"
        "class Plugin:\n"
        "    value: int = 17\n",
        encoding="utf-8",
    )
    original = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, original)
    loaded = import_from_path(name, path)
    assert loaded is sys.modules[name]
    assert loaded is not original
    assert loaded.registered_during_execution
    assert loaded.Plugin().value == 17


def test_plugin_execution_exceptions_propagate_unchanged(tmp_path, monkeypatch):
    name = "_shared_parser_broken_plugin"
    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('broken plugin')\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    with pytest.raises(RuntimeError, match=r"^broken plugin$"):
        import_from_path(name, str(path))
    assert sys.modules[name].__file__ == str(path)


def test_plugin_missing_file_and_unsupported_extension_keep_import_errors(tmp_path, monkeypatch):
    name = "_shared_parser_missing_plugin"
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    with pytest.raises(FileNotFoundError):
        import_from_path(name, tmp_path / "missing.py")
    with pytest.raises(ModuleNotFoundError, match=name):
        import_from_path(name, tmp_path / "plugin.unsupported_extension")


def test_legacy_worker_settings_are_registered_and_documented(monkeypatch):
    specs = {spec.name: spec for spec in flags.registered_flags()}
    trust = flags.ESURGE_WORKER_TRUST_REMOTE_CODE
    context = flags.ESURGE_DETOKENIZER_CONTEXT_WINDOW
    monkeypatch.delenv(trust, raising=False)
    monkeypatch.delenv(context, raising=False)
    assert specs[trust].kind == "bool" and specs[trust].default is True
    assert specs[context].kind == "int" and specs[context].default == 4
    assert flags.get_bool(trust) is True
    assert flags.get_int(context) == 4
    rendered = flags.render_markdown()
    assert f"`{trust}`" in rendered
    assert f"`{context}`" in rendered
    assert "case-sensitive" in rendered


@pytest.mark.parametrize("raw", ["1", "on", "yes", "0", "off", "no", "true", "TRUE", "ON", "YES", "", " on "])
def test_worker_trust_flag_preserves_exact_legacy_truth_values(monkeypatch, raw):
    name = flags.ESURGE_WORKER_TRUST_REMOTE_CODE
    monkeypatch.setenv(name, raw)
    assert flags.get_bool(name) is (raw in ["1", "on", "yes"])
    monkeypatch.setenv(name, "on" if raw != "on" else "off")
    assert flags.get_bool(name) is (raw != "on")


@pytest.mark.parametrize(("raw", "expected"), [("7", 7), ("0", 0), ("-3", -3), ("", 4), ("invalid", 4)])
def test_worker_context_flag_parses_at_call_time_with_legacy_fallback(monkeypatch, raw, expected):
    name = flags.ESURGE_DETOKENIZER_CONTEXT_WINDOW
    monkeypatch.setenv(name, raw)
    assert flags.get_int(name) == expected
    monkeypatch.setenv(name, "9")
    assert flags.get_int(name) == 9


def test_worker_legacy_bool_policy_does_not_change_other_flags(monkeypatch):
    name = flags.EASYDEL_SYNC_INPUTS_FOR_TIMING
    for raw in ["true", "TRUE", "yes", "ok", "1", "easy"]:
        monkeypatch.setenv(name, raw)
        assert flags.get_bool(name) is True

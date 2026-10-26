"""Regression tests for ``define_flags_with_default`` tuple defaults."""

import sys

import pytest
from easydel.infra.etils import define_flags_with_default


@pytest.mark.cpu_ok
def test_tuple_default_is_kept_as_tuple_when_flag_is_omitted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args, defaults = define_flags_with_default(shape=(1, 2, 3), lr=1e-3)
    assert args.shape == (1, 2, 3)
    assert defaults["shape"] == (1, 2, 3)
    assert args.lr == 1e-3


@pytest.mark.cpu_ok
def test_tuple_flag_parses_comma_separated_integers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--shape", "4,5,6"])
    args, _ = define_flags_with_default(shape=(1, 2, 3))
    assert args.shape == (4, 5, 6)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("value,expected", [("False", False), ("TRUE", True), ("0", False), ("yes", True)])
def test_boolean_flags_parse_explicit_values(monkeypatch, value, expected):
    monkeypatch.setattr(sys, "argv", ["prog", "--enabled", value])
    args, defaults = define_flags_with_default(enabled=not expected)
    assert args.enabled is expected
    assert defaults["enabled"] is not expected


@pytest.mark.cpu_ok
def test_invalid_boolean_flag_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--enabled", "sometimes"])
    with pytest.raises(SystemExit) as exc:
        define_flags_with_default(enabled=True)
    assert exc.value.code == 2

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REQUIRES_PY311 = sys.version_info >= (3, 11)


def _load_trainer_types_module():
    path = Path(__file__).resolve().parents[1] / "easydel" / "infra" / "elarge_model" / "trainer_types.py"
    spec = importlib.util.spec_from_file_location("trainer_types_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_nash_md_alias_normalized():
    trainer_types = _load_trainer_types_module()
    config = trainer_types.normalize_trainer_config({"trainer_type": "nash_md"})
    assert config["trainer_type"] == "nash-md"


@pytest.mark.skipif(not _REQUIRES_PY311, reason="EasyDeL requires Python 3.11+")
def test_nash_md_alias_resolves_classes():
    pytest.importorskip("eformer.paths")
    from easydel.infra.elarge_model.trainer_types import get_trainer_class, get_training_arguments_class

    assert get_trainer_class("nash_md") is get_trainer_class("nash-md")
    assert get_training_arguments_class("nash_md") is get_training_arguments_class("nash-md")

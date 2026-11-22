"""CPU-only checks for the Qwen3-VL serving pixel-layout contract."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLConfig
from easydel.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import Qwen3VLMoeConfig

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("config_cls", [Qwen3VLConfig, Qwen3VLMoeConfig])
def test_flat_patch_capability_survives_config_roundtrip_and_family_rename(config_cls):
    config = config_cls.from_dict(config_cls().to_dict())
    # A derived family must not depend on the manager's legacy model_type table.
    config.model_type = "derived_qwen_vision_family"
    manager = SimpleNamespace(model=SimpleNamespace(config=config))
    assert MultiModalManager._supports_flat_patch_inputs(manager) is True
    # Serving callers can still explicitly override the class-level default.
    config.esurge_flat_patch_inputs = False
    assert MultiModalManager._supports_flat_patch_inputs(manager) is False

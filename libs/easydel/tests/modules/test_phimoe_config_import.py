"""Pure-Python configuration/import regressions; no array computation."""

import subprocess
import sys

import pytest


@pytest.mark.cpu_ok
def test_config_import_does_not_patch_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; "
            "from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding; "
            "original = PhimoeRotaryEmbedding.__init__; "
            "config_module = importlib.import_module('easydel.modules.phimoe.phimoe_configuration'); "
            "assert PhimoeRotaryEmbedding.__init__ is original, 'EasyDeL modified the HF constructor'; "
            "importlib.reload(config_module); "
            "assert PhimoeRotaryEmbedding.__init__ is original, 'Reloading EasyDeL modified the HF constructor'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.cpu_ok
def test_phimoe_longrope_preserves_legacy_scale_parameters():
    from easydel.modules.phimoe.phimoe_configuration import PhiMoeConfig

    scaling = {
        "type": "longrope",
        "short_factor": [1.0] * 4,
        "long_factor": [2.0] * 4,
        "short_mscale": 1.1,
        "long_mscale": 1.2,
        "original_max_position_embeddings": 16,
    }
    config = PhiMoeConfig(hidden_size=16, num_attention_heads=2, rope_scaling=scaling)
    assert config.rope_scaling["short_mscale"] == 1.1
    assert config.rope_scaling["long_mscale"] == 1.2
    assert config.original_max_position_embeddings == 16

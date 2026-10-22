"""Configuration imports must not modify a third-party model implementation."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.cpu_ok
def test_qwen3_next_config_import_leaves_transformers_unchanged():
    pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    # Reloading a decorated config replaces its registry entry. Keep that mutation
    # out of the pytest process, where model classes retain the original config.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import importlib

                hf = importlib.import_module("transformers.models.qwen3_next.modeling_qwen3_next")
                loss_function = hf.load_balancing_loss_func
                cache_class = getattr(hf, "Qwen3NextDynamicCache", None)
                cache_length = getattr(cache_class, "get_seq_length", None)
                config_module = importlib.import_module("easydel.modules.qwen3_next.qwen3_next_configuration")
                importlib.reload(config_module)
                assert hf.load_balancing_loss_func is loss_function
                assert getattr(cache_class, "get_seq_length", None) is cache_length
                """),
        ],
        env={**os.environ, "ENABLE_DISTRIBUTED_INIT": "0", "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cpu_ok
@pytest.mark.parametrize("head_dim,section", [(8, [2, 1, 1]), (128, [24, 20, 20])])
def test_thinker_default_config_supports_three_axis_positions(head_dim, section):
    from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import Qwen3OmniMoeTextConfig

    config = Qwen3OmniMoeTextConfig(hidden_size=head_dim * 2, num_attention_heads=2)
    assert config.rope_scaling["rope_type"] == "mrope"
    assert config.rope_scaling["mrope_section"] == section
    explicit = Qwen3OmniMoeTextConfig(
        hidden_size=head_dim * 2,
        num_attention_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": section, "mrope_interleaved": False},
    )
    assert explicit.rope_scaling["mrope_interleaved"] is False

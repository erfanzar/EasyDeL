# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""HF->EasyDeL converter contracts for DeepSeek-V4.1 found by the real-checkpoint run (no array compute).

- transformers 5.18 does not ship ``model_type="deepseek_v41"``; the converter must import the EasyDeL family
  (which registers the config with ``AutoConfig``) before ``AutoConfig.from_pretrained``.
- ``StateDictConverter.process_tensor`` transposes every 2-D key containing ``"weight"``; the engram
  ``q_weight`` / ``k_weight`` are plain ``[hc_mult, hidden]`` tensors and must keep the checkpoint layout, while
  the engram ``kv_proj`` linear is still transposed.
"""

import json

import pytest

pytestmark = pytest.mark.cpu_ok


def test_converter_imports_family_so_autoconfig_resolves_deepseek_v41(tmp_path):
    from easydel.scripts.convert_hf_to_easydel import _import_easydel_family
    from transformers import AutoConfig

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    _import_easydel_family(str(tmp_path))
    config = AutoConfig.from_pretrained(str(tmp_path))
    assert type(config).__name__ == "DeepseekV41Config"
    assert config.model_type == "deepseek_v41"


@pytest.mark.parametrize("payload", [None, "not json", json.dumps({"model_type": "no_such_family"})])
def test_import_family_ignores_missing_or_unknown_configs(tmp_path, payload):
    from easydel.scripts.convert_hf_to_easydel import _import_easydel_family

    if payload is not None:
        (tmp_path / "config.json").write_text(payload)
    _import_easydel_family(str(tmp_path))  # must not raise
    _import_easydel_family("some-org/some-hub-repo")


def test_engram_hc_weights_keep_checkpoint_layout_while_linears_transpose():
    import torch
    from easydel.layers.embeddings import GatedEngram
    from easydel.utils.parameters_transformation import StateDictConverter
    from jax import numpy as jnp

    prefix = "model.layers.1.engram"
    # reform_param is what GatedEngram contributes (base_module._get_reform_param prefixes the module path)
    rules = GatedEngram.reform_param.fget(None)
    reform_param = {f"{prefix}.{key}": dict(rule) for key, rule in rules.items()}
    StateDictConverter.validate_reform_param_schema(reform_param)
    config = {
        "embedding_layer_names": set(),
        "consolidated_moe_keys": set(),
        "dtype": jnp.bfloat16,
        "reform_param": reform_param,
        "lm_head_name": None,
        "uses_tie_word_embedding": False,
    }
    hc, hidden = 4, 64
    for leaf in ("q_weight", "k_weight"):
        out = StateDictConverter.process_tensor(f"{prefix}.{leaf}", torch.zeros(hc, hidden), config)
        assert [tuple(a.shape) for _, a in out] == [(hc, hidden)]
    out = StateDictConverter.process_tensor(f"{prefix}.kv_proj.weight", torch.zeros(96, 32), config)
    assert [tuple(a.shape) for _, a in out] == [(32, 96)]

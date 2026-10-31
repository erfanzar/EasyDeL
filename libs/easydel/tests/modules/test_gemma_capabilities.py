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

"""Family capability regressions without model initialization or array computation."""

from types import SimpleNamespace

import pytest
from easydel.modules.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from easydel.modules.gemma3.gemma3_configuration import Gemma3TextConfig

pytestmark = pytest.mark.cpu_ok


def test_gemma3_rope_layout_follows_capability_without_model_type_fallback(monkeypatch):
    monkeypatch.setattr(Gemma3TextConfig, "model_type", "review_gemma3_without_legacy_fallback")
    config = Gemma3TextConfig(num_hidden_layers=6)
    # A flat assignment must expand through the family's capability, not its registry name.
    config.rope_parameters = {"rope_type": "default", "rope_theta": 12345.0}
    assert set(config.rope_parameters) == {"sliding_attention", "full_attention"}
    assert all(params["rope_theta"] == 12345.0 for params in config.rope_parameters.values())


@pytest.mark.parametrize("scaling_key", ["type", "rope_type"])
def test_gemma3_serialized_rope_preserves_distinct_local_and_scaled_global_settings(scaling_key):
    config = Gemma3TextConfig(
        num_hidden_layers=6,
        rope_local_base_freq=20000.0,
        rope_theta=500000.0,
        rope_scaling={scaling_key: "linear", "factor": 8.0},
    )
    restored = Gemma3TextConfig.from_dict(config.to_dict())
    assert restored.rope_parameters == config.rope_parameters
    assert restored.rope_parameters["sliding_attention"]["rope_theta"] == 20000.0
    assert restored.rope_parameters["full_attention"]["rope_theta"] == 500000.0
    assert restored.rope_parameters["full_attention"]["rope_type"] == "linear"
    assert restored.rope_parameters["full_attention"]["factor"] == 8.0


@pytest.mark.parametrize("apply_lm_head", [True, False])
def test_gemma2_ce_only_declaration_matches_forward_output(apply_lm_head):
    """Exercise the head's actual output assembly using an array-free backbone stand-in."""
    assert Gemma2ForCausalLM.__dict__.get("_ce_only_forward") is True
    hidden, logits, cache = object(), object(), object()
    backbone_output = SimpleNamespace(
        last_hidden_state=hidden, hidden_states=(hidden,), attentions=None, past_key_values=cache
    )
    head = SimpleNamespace(
        model=lambda **kwargs: backbone_output,
        prepare_lm_head_inputs=lambda states: states,
        compute_lm_logits=lambda states: logits if states is hidden else None,
    )
    output = Gemma2ForCausalLM.forward(head, input_ids=object(), apply_lm_head=apply_lm_head)
    assert output.logits is (logits if apply_lm_head else None)
    assert output.last_hidden_state is hidden
    assert output.past_key_values is cache
    assert getattr(output, "aux_loss", None) is None
    assert getattr(output, "mtp_loss", None) is None

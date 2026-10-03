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

"""transformers 5.18 compatibility contracts that need no array compute.

- ``PreTrainedConfig.__post_init__`` runs ``remap_legacy_layer_types``, which assigns ``layer_types`` whenever
  the remapped list compares unequal; a read-only ``layer_types`` property must therefore return a list.
- transformers renamed ``GPTNeoXForCausalLM.embed_out`` to ``lm_head``; EasyDeL follows the new name and moves
  ``embed_out`` leaves of older native saves onto it.
"""

import pytest
from easydel.modules.falcon_h1.falcon_h1_configuration import FalconH1Config
from easydel.modules.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

pytestmark = pytest.mark.cpu_ok


def test_falcon_h1_config_builds_and_roundtrips():
    cfg = FalconH1Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    assert cfg.layer_types == ["hybrid", "hybrid", "hybrid"]
    again = FalconH1Config.from_dict(cfg.to_dict())
    assert again.layer_types == ["hybrid", "hybrid", "hybrid"]
    assert again.num_hidden_layers == 3


def test_gpt_neox_legacy_embed_out_moves_to_lm_head():
    legacy, current = object(), object()
    state = {
        ("parameters", "embed_out", "weight"): legacy,
        ("parameters", "gpt_neox", "embed_in", "weight"): current,
    }
    expected = {
        ("parameters", "lm_head", "weight"): None,
        ("parameters", "gpt_neox", "embed_in", "weight"): None,
    }
    out = GPTNeoXForCausalLM._upgrade_legacy_native_state(None, dict(state), expected)
    assert out == {
        ("parameters", "lm_head", "weight"): legacy,
        ("parameters", "gpt_neox", "embed_in", "weight"): current,
    }


def test_gpt_neox_upgrade_never_overwrites_current_head():
    legacy, current = object(), object()
    state = {("parameters", "embed_out", "weight"): legacy, ("parameters", "lm_head", "weight"): current}
    out = GPTNeoXForCausalLM._upgrade_legacy_native_state(None, dict(state), {("parameters", "lm_head", "weight"): None})
    assert out[("parameters", "lm_head", "weight")] is current
    # An undeclared target is left for the loader's unexpected-key handling.
    out = GPTNeoXForCausalLM._upgrade_legacy_native_state(None, {("parameters", "embed_out", "weight"): legacy}, {})
    assert out == {("parameters", "embed_out", "weight"): legacy}

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

"""CPU-safe config-default parity checks for SmolLM3 (regression for S17-3).

The EasyDeL SmolLM3 defaults must describe the same architecture as the
Hugging Face ``SmolLM3Config`` defaults: the checkpoint ties word
embeddings and uses 4 KV heads (GQA). A default that drifts produces an
untied / wrong-head model whenever a user builds the config directly
(e.g. from an eLarge YAML) instead of round-tripping through HF.
"""

import pytest

pytestmark = pytest.mark.cpu_ok


def test_smollm3_config_defaults_match_hf():
    """EasyDeL SmolLM3Config() defaults mirror HF SmolLM3Config()."""
    from easydel.modules.smollm3 import SmolLM3Config
    from transformers import SmolLM3Config as HFSmolLM3Config

    ed = SmolLM3Config()
    hf = HFSmolLM3Config()
    assert ed.tie_word_embeddings is bool(hf.tie_word_embeddings)
    assert ed.num_key_value_heads == hf.num_key_value_heads
    assert ed.num_attention_heads == hf.num_attention_heads
    assert ed.hidden_size == hf.hidden_size
    assert ed.intermediate_size == hf.intermediate_size
    assert ed.num_hidden_layers == hf.num_hidden_layers


def test_smollm3_kv_heads_fallback_never_exceeds_attention_heads():
    """The GQA fallback keeps a sane KV-head count for small test configs."""
    from easydel.modules.smollm3 import SmolLM3Config

    tiny = SmolLM3Config(num_attention_heads=2)
    assert tiny.num_key_value_heads == 2
    assert SmolLM3Config(num_attention_heads=16).num_key_value_heads == 4

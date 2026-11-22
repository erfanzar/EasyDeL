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

"""Default values of partially specified configs must match HF (no array compute)."""

import pytest
from easydel.modules.gemma2.gemma2_configuration import Gemma2Config
from easydel.modules.llama.llama_configuration import LlamaConfig
from easydel.modules.smollm3.smollm3_configuration import SmolLM3Config
from transformers import Gemma2Config as HFGemma2Config
from transformers import LlamaConfig as HFLlamaConfig
from transformers import SmolLM3Config as HFSmolLM3Config

pytestmark = pytest.mark.cpu_ok


def test_smollm3_kv_head_default_and_none_match_hf():
    """Omitted ``num_key_value_heads`` uses HF's default; an explicit ``None`` means MHA like HF."""
    for kwargs in (
        {},
        {"num_key_value_heads": None},
        {"num_attention_heads": 8},
        {"num_attention_heads": 8, "num_key_value_heads": None},
    ):
        ours = SmolLM3Config(**kwargs)
        theirs = HFSmolLM3Config(**kwargs)
        assert ours.num_key_value_heads == theirs.num_key_value_heads, kwargs
        assert ours.tie_word_embeddings == theirs.tie_word_embeddings


def test_gemma2_and_llama_defaults_match_hf():
    """Fields whose defaults changed track HF's defaults."""
    g, hg = Gemma2Config(), HFGemma2Config()
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "query_pre_attn_scalar",
    ):
        assert getattr(g, name) == getattr(hg, name), name
    llama, hf_llama = LlamaConfig(), HFLlamaConfig()
    assert llama.bos_token_id == hf_llama.bos_token_id
    assert llama.eos_token_id == hf_llama.eos_token_id

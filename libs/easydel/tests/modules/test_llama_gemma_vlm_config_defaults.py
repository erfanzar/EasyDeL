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

"""Config-only HF default compatibility for Llama/Gemma/VLM families."""

import pytest
from easydel.modules.gemma2.gemma2_configuration import Gemma2Config
from easydel.modules.gemma4_assistant.configuration_gemma4_assistant import Gemma4AssistantConfig
from easydel.modules.llama.llama_configuration import LlamaConfig
from easydel.modules.llava.llava_configuration import LlavaConfig
from easydel.modules.minimax_m3_vl.minimax_m3_vl_configuration import MiniMaxM3VLTextConfig
from transformers import Gemma2Config as HFGemma2Config
from transformers import Gemma4AssistantConfig as HFGemma4AssistantConfig
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlavaConfig as HFLlavaConfig
from transformers import MiniMaxM3VLTextConfig as HFMiniMaxM3VLTextConfig


@pytest.mark.cpu_ok
def test_gemma2_default_geometry_matches_hf():
    config = Gemma2Config()
    hf = HFGemma2Config()
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "query_pre_attn_scalar",
    ):
        assert getattr(config, name) == getattr(hf, name)


@pytest.mark.cpu_ok
def test_llama_special_token_defaults_match_hf():
    config = LlamaConfig()
    hf = HFLlamaConfig()
    assert (config.bos_token_id, config.eos_token_id) == (hf.bos_token_id, hf.eos_token_id)


@pytest.mark.cpu_ok
def test_llava_embedding_tie_default_and_override():
    assert LlavaConfig().tie_word_embeddings == HFLlavaConfig().tie_word_embeddings
    assert LlavaConfig(tie_word_embeddings=True).tie_word_embeddings is True


@pytest.mark.cpu_ok
def test_llava_inherits_tie_flag_from_text_config_like_hf():
    text = {"model_type": "llama", "tie_word_embeddings": True}
    hf = HFLlavaConfig(text_config=dict(text))
    assert LlavaConfig(text_config=dict(text)).tie_word_embeddings == hf.tie_word_embeddings is True


@pytest.mark.cpu_ok
def test_gemma4_assistant_default_geometry_and_head_match_hf():
    config = Gemma4AssistantConfig()
    hf = HFGemma4AssistantConfig()
    assert config.backbone_hidden_size == hf.backbone_hidden_size
    assert config.use_ordered_embeddings == hf.use_ordered_embeddings


@pytest.mark.cpu_ok
def test_minimax_m3_vl_text_special_token_defaults_and_override():
    config = MiniMaxM3VLTextConfig()
    hf = HFMiniMaxM3VLTextConfig()
    assert (config.bos_token_id, config.eos_token_id) == (hf.bos_token_id, hf.eos_token_id)
    overridden = MiniMaxM3VLTextConfig(bos_token_id=3, eos_token_id=4)
    assert (overridden.bos_token_id, overridden.eos_token_id) == (3, 4)

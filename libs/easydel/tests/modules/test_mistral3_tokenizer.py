# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Mistral3Tokenizer: no EOS on prompts and no re-encoding of templated chat (no array computation)."""

from types import SimpleNamespace

import pytest
from easydel.modules.mistral3 import mistral3_tokenizer as tokenizer_module
from easydel.modules.mistral3.mistral3_tokenizer import Mistral3Tokenizer

pytestmark = pytest.mark.cpu_ok

BOS, EOS, PAD = 1, 2, 0
TEMPLATED_TOKENS = [BOS, 10, 11, 12]


class _FakeInnerTokenizer:
    pad_id = PAD
    eos_id = EOS
    bos_id = BOS

    def encode(self, text, bos, eos):
        ids = [ord(ch) for ch in text]
        if bos:
            ids = [BOS, *ids]
        if eos:
            ids = [*ids, EOS]
        return ids


class _FakeMistralTokenizer:
    def encode_chat_completion(self, request):
        return SimpleNamespace(text="<s>[INST]hi[/INST]", tokens=list(TEMPLATED_TOKENS))


@pytest.fixture
def tokenizer(monkeypatch):
    monkeypatch.setattr(tokenizer_module, "ChatCompletionRequest", lambda messages: messages)
    tok = object.__new__(Mistral3Tokenizer)
    tok.mistral_tokenizer = _FakeMistralTokenizer()
    tok.tokenizer = _FakeInnerTokenizer()
    tok.pad_token_id = PAD
    tok.eos_token_id = EOS
    tok.bos_token_id = BOS
    tok.padding_side = "left"
    return tok


def test_prompt_gets_bos_but_never_eos(tokenizer):
    assert tokenizer("ab")["input_ids"] == [BOS, ord("a"), ord("b")]
    assert tokenizer("ab", add_special_tokens=False)["input_ids"] == [ord("a"), ord("b")]
    assert tokenizer.encode("ab")[-1] != EOS


def test_chat_template_returns_encoder_tokens_without_reencoding(tokenizer):
    out = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert out["input_ids"] == TEMPLATED_TOKENS
    assert out["attention_mask"] == [1, 1, 1, 1]
    assert out["input_ids"].count(BOS) == 1
    assert out["input_ids"][-1] != EOS


def test_chat_template_text_mode_and_special_token_flag(tokenizer):
    conversation = [{"role": "user", "content": "hi"}]
    assert tokenizer.apply_chat_template(conversation, tokenize=False) == "<s>[INST]hi[/INST]"
    assert tokenizer.apply_chat_template(conversation, add_special_tokens=False)["input_ids"] == TEMPLATED_TOKENS[1:]


def test_chat_template_honours_padding_and_truncation(tokenizer):
    conversation = [{"role": "user", "content": "hi"}]
    padded = tokenizer.apply_chat_template(conversation, padding="max_length", max_length=6)
    assert padded["input_ids"] == [PAD, PAD, *TEMPLATED_TOKENS]
    assert padded["attention_mask"] == [0, 0, 1, 1, 1, 1]
    truncated = tokenizer.apply_chat_template(conversation, truncation=True, max_length=2)
    assert truncated["input_ids"] == TEMPLATED_TOKENS[:2]


@pytest.mark.parametrize("chat", [False, True])
def test_disabled_string_strategies_do_not_truncate_or_require_a_pad_token(tokenizer, chat):
    tokenizer.pad_token_id = None
    kwargs = dict(padding="do_not_pad", truncation="do_not_truncate", max_length=2)
    if chat:
        output = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], **kwargs)
        expected = TEMPLATED_TOKENS
    else:
        output = tokenizer("abc", **kwargs)
        expected = [BOS, ord("a"), ord("b"), ord("c")]
    assert output["input_ids"] == expected
    assert output["attention_mask"] == [1] * len(expected)


@pytest.mark.parametrize("strategy", ["longest_first", "only_first", "only_second"])
def test_enabled_string_truncation_strategies(tokenizer, strategy):
    output = tokenizer("abc", truncation=strategy, max_length=2)
    assert output["input_ids"] == [BOS, ord("a")]

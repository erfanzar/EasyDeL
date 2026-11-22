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

"""Assistant-token masks produced by ``SFTPreprocessTransform`` with a real HF chat template.

Covers two regressions of the ``{% generation %}`` mask path:

- truncation must not zero the mask of a multi-turn row (HF stops at the first cut span), and
- concurrent renders must not corrupt each other through HF's process-global assistant tracker.
"""

from __future__ import annotations

import concurrent.futures
import sys
import threading

import pytest
from easydel.trainers._assistant_tracker import assistant_tracker_lock
from easydel.trainers.prompt_transforms import SFTPreprocessTransform

pytestmark = pytest.mark.cpu_ok

_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' %}<a> {% generation %}{{ m['content'] }} <e>{% endgeneration %} "
    "{% else %}<u> {{ m['content'] }} {% endif %}"
    "{% endfor %}"
)
_WORDS = ["[PAD]", "[UNK]", "<u>", "<a>", "<e>", *[f"w{i}" for i in range(60)]]


def _build_tokenizer(padding_side: str = "right"):
    """Return a word-level fast tokenizer whose chat template marks assistant turns."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(_WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="<e>",
    )
    tokenizer.padding_side = padding_side
    tokenizer.chat_template = _TEMPLATE
    return tokenizer


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.convert_tokens_to_ids(text.split())


_TWO_TURNS = {
    "messages": [
        {"role": "user", "content": "w1 w2"},
        {"role": "assistant", "content": "w3 w4"},
        {"role": "user", "content": "w5 w6"},
        {"role": "assistant", "content": "w7 w8"},
    ]
}
# Untruncated render: "<u> w1 w2 <a> w3 w4 <e> <u> w5 w6 <a> w7 w8 <e>" (14 tokens).
_FULL_TEXT = "<u> w1 w2 <a> w3 w4 <e> <u> w5 w6 <a> w7 w8 <e>"
_FULL_MASK = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]


def test_full_row_mask_marks_every_assistant_turn():
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(tokenizer, max_length=64, mask_prompt=True, padding=False)

    row = transform(_TWO_TURNS)

    assert row["input_ids"] == _ids(tokenizer, _FULL_TEXT)
    assert row["assistant_masks"] == _FULL_MASK


def test_keep_end_truncation_keeps_supervision_of_surviving_turns():
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(
        tokenizer, max_length=8, mask_prompt=True, padding=False, truncation_mode="keep_end"
    )

    row = transform(_TWO_TURNS)

    assert row["input_ids"] == _ids(tokenizer, "<e> <u> w5 w6 <a> w7 w8 <e>")
    assert row["attention_mask"] == [1] * 8
    # The cut-open first turn leaves its closing token supervised; the whole second turn is supervised.
    assert row["assistant_masks"] == [1, 0, 0, 0, 0, 1, 1, 1]


def test_keep_start_truncation_cuts_tail_of_ids_and_mask():
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(
        tokenizer, max_length=8, mask_prompt=True, padding=False, truncation_mode="keep_start"
    )

    row = transform(_TWO_TURNS)

    assert row["input_ids"] == _ids(tokenizer, "<u> w1 w2 <a> w3 w4 <e> <u>")
    assert row["assistant_masks"] == [0, 0, 0, 0, 1, 1, 1, 0]


@pytest.mark.parametrize(
    ("padding_side", "ids_text", "attention", "mask"),
    [
        (
            "right",
            "<u> w1 w2 <a> w3 w4 <e> [PAD] [PAD] [PAD]",
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ),
        (
            "left",
            "[PAD] [PAD] [PAD] <u> w1 w2 <a> w3 w4 <e>",
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ),
    ],
)
def test_short_row_is_padded_on_the_tokenizer_side(padding_side, ids_text, attention, mask):
    tokenizer = _build_tokenizer(padding_side)
    transform = SFTPreprocessTransform(tokenizer, max_length=10, mask_prompt=True)
    example = {"messages": _TWO_TURNS["messages"][:2]}

    row = transform(example)

    assert row["input_ids"] == _ids(tokenizer, ids_text)
    assert row["attention_mask"] == attention
    assert row["assistant_masks"] == mask


def test_truncation_disabled_keeps_the_full_row():
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(tokenizer, max_length=8, mask_prompt=True, padding=False, truncation=False)

    row = transform(_TWO_TURNS)

    assert row["input_ids"] == _ids(tokenizer, _FULL_TEXT)
    assert row["assistant_masks"] == _FULL_MASK


def test_unmasked_path_still_truncates_without_a_mask():
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(tokenizer, max_length=8, mask_prompt=False, padding=False)

    row = transform(_TWO_TURNS)

    assert row["input_ids"] == _ids(tokenizer, "<u> w1 w2 <a> w3 w4 <e> <u>")
    assert "assistant_masks" not in row


def _long_conversation(seed: int) -> dict:
    messages = []
    for turn in range(40):
        role = "user" if turn % 2 == 0 else "assistant"
        content = " ".join(f"w{(seed * 7 + turn * 3 + k) % 60}" for k in range(3 + (turn + seed) % 4))
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def test_concurrent_mask_renders_match_sequential_renders():
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(tokenizer, max_length=4096, mask_prompt=True, padding=False)
    examples = [_long_conversation(seed) for seed in range(12)]

    expected = [transform(example) for example in examples]
    assert all(sum(row["assistant_masks"]) > 0 for row in expected)

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(transform, example) for example in examples * 8]
            results = [future.result() for future in futures]
    finally:
        sys.setswitchinterval(previous_interval)

    assert results == expected * 8


def test_mask_render_waits_for_the_process_wide_tracker_lock(monkeypatch):
    tokenizer = _build_tokenizer()
    transform = SFTPreprocessTransform(tokenizer, max_length=64, mask_prompt=True, padding=False)
    original = tokenizer.apply_chat_template
    entered = threading.Event()

    def signalling_apply(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", signalling_apply)

    results: list[dict] = []
    with assistant_tracker_lock():
        worker = threading.Thread(target=lambda: results.append(transform(_TWO_TURNS)))
        worker.start()
        assert not entered.wait(timeout=0.3), "render ran while another thread held the assistant tracker lock"
    assert entered.wait(timeout=10)
    worker.join(timeout=10)

    assert results[0]["assistant_masks"] == _FULL_MASK


def test_tracker_lock_is_one_reentrant_lock():
    first = assistant_tracker_lock()
    with first:
        with assistant_tracker_lock():
            assert assistant_tracker_lock() is first

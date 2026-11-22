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

"""Speculative-trainer drafter calls only forward kwargs the drafter's ``forward`` accepts."""

from __future__ import annotations

import pytest
from easydel.trainers.speculative_decoding_trainer._fn import _call_module_or_method

pytestmark = pytest.mark.cpu_ok


class _PlainCausalLM:
    """Mimics EasyDeLBaseModule: a ``**kwargs`` ``__call__`` dispatching to a strict ``forward``."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=None):
        return {"input_ids": input_ids, "attention_mask": attention_mask, "output_hidden_states": output_hidden_states}


class _ContextDrafter(_PlainCausalLM):
    def forward(self, input_ids=None, attention_mask=None, target_hidden_states=None):
        return {"input_ids": input_ids, "target_hidden_states": target_hidden_states}


def test_plain_causal_lm_drafter_drops_target_hidden_states():
    out = _call_module_or_method(
        _PlainCausalLM(),
        {"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "labels": [[2, 3]]},
        method_name=None,
        extra_kwargs={"target_hidden_states": "ctx"},
    )
    assert out == {"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "output_hidden_states": None}


def test_context_drafter_still_receives_target_hidden_states():
    out = _call_module_or_method(
        _ContextDrafter(),
        {"input_ids": [[1, 2]]},
        method_name="forward",
        extra_kwargs={"target_hidden_states": "ctx"},
    )
    assert out == {"input_ids": [[1, 2]], "target_hidden_states": "ctx"}


def test_custom_method_is_filtered_against_itself():
    class _Custom(_PlainCausalLM):
        def draft(self, input_ids=None, target_logits=None):
            return {"input_ids": input_ids, "target_logits": target_logits}

    out = _call_module_or_method(
        _Custom(),
        {"input_ids": [[1]], "attention_mask": [[1]]},
        method_name="draft",
        extra_kwargs={"target_logits": "logits", "target_hidden_states": "ctx"},
    )
    assert out == {"input_ids": [[1]], "target_logits": "logits"}

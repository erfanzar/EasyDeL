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

"""Host-only contracts of the trainers' eSurge generation helpers; no array computation."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers._shared import esurge_generation

pytestmark = pytest.mark.cpu_ok


def _args(**kwargs):
    base = {
        "generation_max_new_tokens": None,
        "max_completion_length": None,
        "max_prompt_length": None,
        "max_length": 64,
        "generation_temperature": None,
        "generation_top_p": None,
        "generation_top_k": None,
        "generation_presence_penalty": None,
        "generation_frequency_penalty": None,
        "generation_repetition_penalty": None,
        "generation_num_return_sequences": None,
        "generation_do_sample": None,
        "total_batch_size": 2,
        "esurge_hbm_utilization": None,
        "esurge_max_num_seqs": None,
        "esurge_min_input_pad": None,
        "esurge_page_size": None,
        "esurge_max_cache_tokens": None,
        "esurge_max_num_batched_tokens": None,
        "esurge_enable_prefix_caching": None,
        "esurge_data_parallelism_axis": None,
        "esurge_max_num_seq_buckets": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "fields,overrides,expected",
    [
        ({}, None, {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.95, "top_k": 64, "n": 1}),
        ({"max_completion_length": 32}, None, {"max_tokens": 32}),
        ({"generation_max_new_tokens": 8, "max_completion_length": 32}, None, {"max_tokens": 8}),
        ({"generation_top_k": 0, "generation_top_p": 1.0}, None, {"top_k": 0, "top_p": 1.0}),
        ({}, {"max_new_tokens": "5", "top_k": 3.0, "num_return_sequences": 2}, {"max_tokens": 5, "top_k": 3, "n": 2}),
        ({}, {"temperature": None}, {"temperature": 0.7}),
        ({"generation_do_sample": False, "generation_temperature": 0.9}, None, {"temperature": 0.0}),
        ({"generation_do_sample": False}, {"do_sample": True, "temperature": 0.6}, {"temperature": 0.6}),
        ({}, {"do_sample": False, "temperature": 0.6}, {"temperature": 0.0}),
    ],
)
def test_resolve_sampling_fallbacks_overrides_and_greedy(fields, overrides, expected):
    params = esurge_generation.resolve_sampling(_args(**fields), overrides)
    for attr, value in expected.items():
        assert getattr(params, attr) == value, attr
    assert isinstance(params.max_tokens, int)
    assert isinstance(params.top_k, int)


def test_engine_length_is_stable_across_prompt_widths_and_budgets_within_the_bound():
    args = _args(max_prompt_length=32, max_completion_length=32, esurge_max_num_seqs=4)
    lengths = {
        esurge_generation.engine_kwargs(args, prompt_len=prompt_len, max_new_tokens=budget)["max_model_len"]
        for prompt_len, budget in ((32, 32), (12, 32), (20, 4), (1, 1))
    }
    # B = 32 + 32, plus the default reserve of max_num_seqs tokens.
    assert lengths == {68}
    # A call beyond the bound gets a length that fits it.
    assert esurge_generation.engine_kwargs(args, prompt_len=50, max_new_tokens=32)["max_model_len"] == 86


def test_engine_kwargs_without_call_shape_leaves_length_unset_and_sizes_pool():
    kwargs = esurge_generation.engine_kwargs(_args(generation_num_return_sequences=3), processor="tok")
    assert "max_model_len" not in kwargs
    assert kwargs["tokenizer"] == "tok"
    assert kwargs["max_num_seqs"] == 6
    assert kwargs["max_cache_tokens"] == 64 * 2 * 3
    assert kwargs["max_num_batched_tokens"] == 64


def test_prompt_token_id_rows_select_masked_tokens_in_row_order():
    ids = np.asarray([[0, 0, 5, 6], [7, 8, 9, 3]])
    mask = np.asarray([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert esurge_generation.prompt_token_id_rows(ids, mask) == [[5, 6], [7, 8, 9, 3]]
    assert esurge_generation.prompt_token_id_rows(ids, None, pad_token_id=0) == [[5, 6], [7, 8, 9, 3]]
    assert esurge_generation.prompt_token_id_rows(np.asarray([4, 5]), None) == [[4, 5]]
    with pytest.raises(ValueError, match="shape"):
        esurge_generation.prompt_token_id_rows(ids, mask[:, :3])


def test_outputs_to_arrays_rejects_output_count_mismatch():
    output = SimpleNamespace(prompt="p", prompt_token_ids=[[5]], outputs=[])
    with pytest.raises(ValueError, match="request order"):
        esurge_generation.outputs_to_arrays(
            [output],
            input_ids=[[0, 5], [0, 6]],
            attention_mask=None,
            return_prompts=None,
            pad_token_id=0,
            max_seq_len=2,
            max_new_tokens=2,
        )

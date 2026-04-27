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

"""Comprehensive tests for SamplingParams edge cases and validation."""

import copy

import pytest

from easydel.inference.sampling_params import SamplingParams, SamplingType


class TestSamplingParamsPostInit:
    def test_none_temperature_defaults_to_1(self):
        params = SamplingParams(temperature=None)
        assert params.temperature == 1

    def test_very_low_temperature_clamped(self):
        params = SamplingParams(temperature=0.005)
        assert params.temperature == 1e-2

    def test_seed_minus_one_becomes_none(self):
        params = SamplingParams(seed=-1)
        assert params.seed is None

    def test_negative_max_tokens_becomes_none(self):
        params = SamplingParams(max_tokens=-1)
        assert params.max_tokens is None

    def test_stop_string_coerced_to_list(self):
        params = SamplingParams(stop="END")
        assert params.stop == ["END"]

    def test_logprobs_true_becomes_1(self):
        params = SamplingParams(logprobs=True)
        assert params.logprobs == 1

    def test_prompt_logprobs_true_becomes_1(self):
        params = SamplingParams(prompt_logprobs=True)
        assert params.prompt_logprobs == 1

    def test_greedy_forces_top_p_top_k_min_p(self):
        params = SamplingParams(temperature=0.0)
        assert params.top_p == 1.0
        assert params.top_k == 0
        assert params.min_p == 0.0

    def test_stop_buffer_length_computed(self):
        params = SamplingParams(stop=["end", "stopping"])
        assert params._output_text_buffer_length == len("stopping") - 1

    def test_stop_buffer_length_with_include_stop(self):
        params = SamplingParams(stop=["end"], include_stop_str_in_output=True)
        assert params._output_text_buffer_length == 0

    def test_all_stop_token_ids_populated(self):
        params = SamplingParams(stop_token_ids=[1, 2, 3])
        assert params.all_stop_token_ids == {1, 2, 3}


class TestSamplingParamsValidation:
    def test_n_below_1_raises(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            SamplingParams(n=0)

    def test_presence_penalty_out_of_range(self):
        with pytest.raises(ValueError, match="presence_penalty"):
            SamplingParams(presence_penalty=3.0)

    def test_presence_penalty_negative_out_of_range(self):
        with pytest.raises(ValueError, match="presence_penalty"):
            SamplingParams(presence_penalty=-3.0)

    def test_repetition_penalty_zero_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=0.0)

    def test_repetition_penalty_negative_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=-1.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            SamplingParams(temperature=-0.5)

    def test_top_p_zero_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_top_p_above_1_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)

    def test_min_tokens_exceeds_max_tokens(self):
        with pytest.raises(ValueError, match="min_tokens"):
            SamplingParams(min_tokens=100, max_tokens=10)

    def test_stop_without_detokenize_raises(self):
        with pytest.raises(ValueError, match="stop strings require detokenize"):
            SamplingParams(stop=["end"], detokenize=False)

    def test_greedy_with_n_above_1_raises(self):
        with pytest.raises(ValueError, match="n must be 1 for greedy"):
            SamplingParams(temperature=0.0, n=2)

    def test_best_of_below_n_raises(self):
        with pytest.raises(ValueError, match="best_of"):
            SamplingParams(n=3, best_of=2, temperature=0.5)

    def test_valid_boundary_values(self):
        params = SamplingParams(
            presence_penalty=2.0,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=0.01,
        )
        assert params.presence_penalty == 2.0


class TestSamplingParamsUpdateMethods:
    def test_update_with_generation_config_single_eos(self):
        params = SamplingParams()
        params.update_with_generation_config({"eos_token_id": 50256})
        assert 50256 in params.all_stop_token_ids

    def test_update_with_generation_config_multiple_eos(self):
        params = SamplingParams()
        params.update_with_generation_config({"eos_token_id": [50256, 50257]})
        assert {50256, 50257} <= params.all_stop_token_ids

    def test_update_with_generation_config_deduplicates_model_eos(self):
        params = SamplingParams()
        params.update_with_generation_config(
            {"eos_token_id": [50256, 50257]},
            model_eos_token_id=50256,
        )
        assert 50256 in params.all_stop_token_ids
        assert 50257 in params.all_stop_token_ids

    def test_update_with_generation_config_ignore_eos(self):
        params = SamplingParams(ignore_eos=True)
        params.update_with_generation_config({"eos_token_id": 50256})
        assert 50256 not in set(params.stop_token_ids)

    def test_update_with_generation_config_empty_dict(self):
        params = SamplingParams(stop_token_ids=[1])
        params.update_with_generation_config({})
        assert params.stop_token_ids == [1]

    def test_update_with_model_eos_only(self):
        params = SamplingParams()
        params.update_with_generation_config({}, model_eos_token_id=42)
        assert 42 in params.all_stop_token_ids


class TestSamplingTypeProperty:
    def test_greedy_type(self):
        params = SamplingParams(temperature=0.0)
        assert params.sampling_type == SamplingType.GREEDY

    def test_random_type(self):
        params = SamplingParams(temperature=0.8)
        assert params.sampling_type == SamplingType.RANDOM

    def test_boundary_greedy(self):
        params = SamplingParams(temperature=1e-6)
        assert params.sampling_type == SamplingType.RANDOM

    def test_zero_temperature_is_greedy(self):
        params = SamplingParams(temperature=0.0)
        assert params.sampling_type == SamplingType.GREEDY


class TestSamplingParamsCopy:
    def test_deepcopy_preserves_values(self):
        params = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            stop=["end"],
            stop_token_ids=[42],
        )
        cloned = copy.deepcopy(params)
        assert cloned.temperature == params.temperature
        assert cloned.top_p == params.top_p
        assert cloned.stop == params.stop
        assert cloned.stop_token_ids == params.stop_token_ids

    def test_deepcopy_independent(self):
        params = SamplingParams(stop=["end"], stop_token_ids=[42])
        cloned = copy.deepcopy(params)
        cloned.stop.append("new")
        cloned.stop_token_ids.append(99)
        assert "new" not in params.stop
        assert 99 not in params.stop_token_ids


class TestBestOfHandling:
    def test_best_of_overrides_n(self):
        params = SamplingParams(n=2, best_of=5, temperature=0.5)
        assert params.n == 5
        assert params._real_n == 2

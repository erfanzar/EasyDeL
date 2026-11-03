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

"""Rank-major SPMD DP accepts KDA models and still rejects speculative decoding (pure Python, no array compute).

KDA linear-attention layers (glm5_next / kimi_linear) run rank-major steps through
``kda_ragged_rows_rank_major_forward`` (one shard_map over the attention data-parallel axis and the head axis around
the flat single-rank driver), so the startup validation no longer rejects them. Speculative decoding is not wired into
rank-major DP execution: it must still fail at startup, for KDA models too.
"""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.runners import model_runner as mr
from easydel.inference.esurge.runners.model_runner import MLA_RAGGED_ATTN_MECHANISM, eSurgeRunner

pytestmark = pytest.mark.cpu_ok


def _runner(text_config, *, drafter=None):
    runner = object.__new__(eSurgeRunner)
    runner.metadata = SimpleNamespace(data_parallel_size=2, get_max_num_seqs=lambda: 4)
    runner.pipeline_plan = SimpleNamespace(is_enabled=False)
    runner.drafter = drafter
    runner.max_num_reqs = 4
    runner.model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: text_config))
    return runner


def _glm5_next_config():
    import easydel as ed

    config = ed.Glm5NextTextConfig(num_hidden_layers=4, sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    assert "kda_linear_attention" in tuple(config.get_layer_types())
    return config


@pytest.fixture(autouse=True)
def _tpu_backend(monkeypatch):
    # Rank-major DP is TPU-only; the checks under test come after that gate.
    monkeypatch.setattr(mr.jax, "default_backend", lambda: "tpu")


def test_kda_model_is_accepted_for_rank_major_dp():
    text_config = SimpleNamespace(get_layer_types=lambda: ("kda_linear_attention", "full_attention"))
    _runner(text_config)._validate_spmd_dp_support(attn_mechanism="ragged_page_attention_v3")


def test_real_glm5_next_config_is_accepted_with_mla_ragged_attention():
    _runner(_glm5_next_config())._validate_spmd_dp_support(attn_mechanism=MLA_RAGGED_ATTN_MECHANISM)


def test_kda_model_with_speculative_decoding_is_still_rejected():
    with pytest.raises(ValueError, match="does not support speculative decoding"):
        _runner(_glm5_next_config(), drafter=object())._validate_spmd_dp_support(
            attn_mechanism=MLA_RAGGED_ATTN_MECHANISM
        )


def test_gdn_linear_attention_model_is_accepted_for_rank_major_dp():
    text_config = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    _runner(text_config)._validate_spmd_dp_support(attn_mechanism="ragged_page_attention_v3")


def test_kda_model_with_non_ragged_attention_is_still_rejected():
    with pytest.raises(ValueError, match="ragged TPU attention only"):
        _runner(_glm5_next_config())._validate_spmd_dp_support(attn_mechanism="vanilla")

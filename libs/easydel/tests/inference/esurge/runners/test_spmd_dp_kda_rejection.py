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

"""Rank-major SPMD DP rejects KDA models at startup (pure Python, no array compute).

The KDA packed-row driver has no per-rank state scatter for rank-major metadata. It raises
at trace time; the runner must reject the configuration up front instead. GDN-style linear
attention (``linear_attention``) is a different code path and must not be rejected here.
"""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.runners import model_runner as mr
from easydel.inference.esurge.runners.model_runner import eSurgeRunner

pytestmark = pytest.mark.cpu_ok


def _runner(text_config):
    runner = object.__new__(eSurgeRunner)
    runner.metadata = SimpleNamespace(data_parallel_size=2, get_max_num_seqs=lambda: 4)
    runner.pipeline_plan = SimpleNamespace(is_enabled=False)
    runner.drafter = None
    runner.max_num_reqs = 4
    runner.model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: text_config))
    return runner


@pytest.fixture(autouse=True)
def _tpu_backend(monkeypatch):
    # Rank-major DP is TPU-only; the checks under test come after that gate.
    monkeypatch.setattr(mr.jax, "default_backend", lambda: "tpu")


def test_kda_model_is_rejected_for_rank_major_dp():
    text_config = SimpleNamespace(get_layer_types=lambda: ("kda_linear_attention", "full_attention"))
    with pytest.raises(ValueError, match="KDA linear-attention"):
        _runner(text_config)._validate_spmd_dp_support(attn_mechanism="ragged_page_attention_v3")


def test_gdn_linear_attention_model_is_not_rejected_by_the_kda_check():
    text_config = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    _runner(text_config)._validate_spmd_dp_support(attn_mechanism="ragged_page_attention_v3")


def test_real_glm5_next_config_is_detected_as_kda():
    import easydel as ed

    config = ed.Glm5NextTextConfig(num_hidden_layers=4, sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    assert "kda_linear_attention" in tuple(config.get_layer_types())
    with pytest.raises(ValueError, match="KDA linear-attention"):
        _runner(config)._validate_spmd_dp_support(attn_mechanism="ragged_page_attention_v3")

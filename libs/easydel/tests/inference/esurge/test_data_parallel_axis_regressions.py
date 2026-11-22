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

from types import SimpleNamespace

import pytest
from easydel.axis import (
    register_attention_data_parallel_axis,
    reset_attention_data_parallel_axis,
    resolve_attention_data_parallel_axis,
)
from easydel.inference.esurge.esurge_engine import _resolve_cache_attention_axis, eSurge
from spectrax import PartitionAxis, PartitionManager

pytestmark = pytest.mark.cpu_ok


def test_attention_data_parallel_axis_resolves_independently_from_model_dp():
    pm = PartitionManager(PartitionAxis())
    register_attention_data_parallel_axis("ep")
    try:
        assert pm.paxis.data_parallel_axis == "dp"
        assert resolve_attention_data_parallel_axis(pm) == "ep"
    finally:
        reset_attention_data_parallel_axis()


@pytest.mark.cpu_ok
def test_cache_attention_axis_can_differ_from_request_dp():
    cfg = SimpleNamespace(data_parallelism_axis="dp", attention_data_parallel_axis="tp")
    assert _resolve_cache_attention_axis(cfg) == "tp"
    cfg.attention_data_parallel_axis = None
    assert _resolve_cache_attention_axis(cfg) == "dp"


def test_esurge_does_not_mutate_model_partition_axis():
    engine = object.__new__(eSurge)
    engine.data_parallelism_axis = "ep"
    engine._monitoring_initialized = False
    engine._scheduler_running = False

    cfg = SimpleNamespace(partition_axis=PartitionAxis())
    model = SimpleNamespace(config=cfg)

    eSurge._apply_data_parallel_axis_to_model(engine, model)

    assert cfg.partition_axis.data_parallel_axis == "dp"
    assert cfg.partition_axis.tensor_parallel_axis == "tp"
    assert cfg.partition_axis.mlp_intermediate_axis == "tp"
    assert cfg.partition_axis.vocab_axis == "tp"

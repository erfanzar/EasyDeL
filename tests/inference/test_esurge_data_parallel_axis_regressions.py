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

from eformer.escale import PartitionAxis, PartitionManager

from easydel.axis import (
    register_attention_data_parallel_axis,
    reset_attention_data_parallel_axis,
    resolve_attention_data_parallel_axis,
)
from easydel.inference.esurge.esurge_engine import eSurge


def test_attention_data_parallel_axis_resolves_independently_from_model_dp():
    pm = PartitionManager(PartitionAxis())
    register_attention_data_parallel_axis("ep")
    try:
        assert pm.paxis.data_parallel_axis == "dp"
        assert resolve_attention_data_parallel_axis(pm) == "ep"
    finally:
        reset_attention_data_parallel_axis()


def test_esurge_does_not_mutate_model_partition_axis():
    engine = object.__new__(eSurge)
    engine.data_parallelism_axis = "ep"
    engine._monitoring_initialized = False
    engine._scheduler_running = False

    cfg = SimpleNamespace(partition_axis=PartitionAxis())
    model = SimpleNamespace(config=cfg)

    eSurge._apply_data_parallel_axis_to_model(engine, model)

    assert cfg.partition_axis.data_parallel_axis == "dp"

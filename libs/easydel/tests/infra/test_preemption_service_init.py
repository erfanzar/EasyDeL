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

from __future__ import annotations

import jax

from easydel.infra import init_cluster
from easydel.trainers import utils as trainer_utils


def test_jax_distributed_config_enables_preemption_service_before_initialize(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_PREEMPTION_SERVICE", raising=False)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        trainer_utils.jax.config,
        "update",
        lambda name, value: calls.append(("config", (name, value))),
    )
    monkeypatch.setattr(
        trainer_utils.jax.distributed,
        "initialize",
        lambda *args, **kwargs: calls.append(("initialize", kwargs)),
    )

    trainer_utils.JaxDistributedConfig.initialize(
        {
            "initialize_jax_distributed": True,
            "coordinator_address": "127.0.0.1:1234",
            "num_processes": 2,
            "process_id": 1,
            "local_device_ids": "0,1",
        }
    )

    assert trainer_utils.os.environ["JAX_ENABLE_PREEMPTION_SERVICE"] == "true"
    assert calls[0] == ("config", ("jax_enable_preemption_service", True))
    assert calls[1][0] == "initialize"
    assert calls[1][1]["coordinator_address"] == "127.0.0.1:1234"
    assert calls[1][1]["num_processes"] == 2
    assert calls[1][1]["process_id"] == 1
    assert calls[1][1]["local_device_ids"] == [0, 1]


def test_init_cluster_enables_preemption_service_before_distributed_init(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_PREEMPTION_SERVICE", raising=False)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        jax.config,
        "update",
        lambda name, value: calls.append(("config", (name, value))),
    )
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)

    class _DistributedConfig:
        def initialize(self):
            calls.append(("initialize", None))

    monkeypatch.setattr("eformer.executor.DistributedConfig", _DistributedConfig)

    init_cluster()

    assert trainer_utils.os.environ["JAX_ENABLE_PREEMPTION_SERVICE"] == "true"
    assert calls == [
        ("config", ("jax_enable_preemption_service", True)),
        ("initialize", None),
    ]


def test_init_cluster_skips_when_jax_distributed_is_already_initialized(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(jax.config, "update", lambda name, value: calls.append(f"config:{name}={value}"))
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: True)

    class _DistributedConfig:
        def initialize(self):
            calls.append("initialize")

    monkeypatch.setattr("eformer.executor.DistributedConfig", _DistributedConfig)

    init_cluster()

    assert calls == ["config:jax_enable_preemption_service=True"]


def test_jax_distributed_config_skips_when_already_initialized(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        trainer_utils.jax.config,
        "update",
        lambda name, value: calls.append(f"config:{name}={value}"),
    )
    monkeypatch.setattr(trainer_utils.jax.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        trainer_utils.jax.distributed,
        "initialize",
        lambda *args, **kwargs: calls.append("initialize"),
    )

    trainer_utils.JaxDistributedConfig.initialize({"initialize_jax_distributed": True})

    assert calls == ["config:jax_enable_preemption_service=True"]

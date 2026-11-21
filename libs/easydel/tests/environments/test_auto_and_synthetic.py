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

"""CPU tests for AutoEnvironment, TaskVerificationGate, DifficultyTracker, SyntheticTaskStream, and AsyncGRPOConfig."""

from __future__ import annotations

import pytest
from easydel.data import MixedShardedSource
from easydel.data.core.protocols import ShardedDataSource
from easydel.environments import (
    AutoEnvironment,
    CodeJudgeEnvironment,
    ContainerConfig,
    DifficultyTracker,
    StatefulPythonEnvironment,
    SWERepoEnvironment,
    SyntheticTaskStream,
    TaskVerificationGate,
    TerminalEnvironment,
    make_environment_reward_fn,
)
from easydel.trainers.async_grpo_trainer.async_grpo_config import AsyncGRPOConfig

pytestmark = pytest.mark.cpu_ok


def test_auto_environment_inference_and_dataset_iteration():
    rows = [
        {"prompt": "Sum a+b", "test_code": "assert add(1, 2) == 3"},
        {"prompt": "What is 6*7?", "answer": "42"},
        {"problem_statement": "Fix bug", "fail_to_pass": ["pytest"], "test_patch": "diff"},
        {"task_prompt": "Check file", "verify_script": "test -f ok"},
    ]
    assert AutoEnvironment.infer_name(rows[0]) == "code_judge"
    assert AutoEnvironment.infer_name(rows[1]) == "python_tir"
    assert AutoEnvironment.infer_name(rows[2]) == "swe_repo"
    assert AutoEnvironment.infer_name(rows[3]) == "terminal"

    envs = list(AutoEnvironment.from_dataset(rows, backend="local", container_config=ContainerConfig(use_jail="off")))
    assert isinstance(envs[0], CodeJudgeEnvironment)
    assert isinstance(envs[1], StatefulPythonEnvironment)
    assert isinstance(envs[2], SWERepoEnvironment)
    assert isinstance(envs[3], TerminalEnvironment)


def test_make_environment_reward_fn_single_turn_and_passthrough():
    reward_fn = make_environment_reward_fn(
        "code_judge",
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    # 1) Passthrough when AsyncGRPO/GRPO already stepped environments during rollout.
    assert reward_fn(environment_rewards=[1.0, 0.0, 0.5]) == [1.0, 0.0, 0.5]

    # 2) Direct RLVR grading from completions + dataset columns.
    scores = reward_fn(
        prompts=["Write mul(a, b)", "Write mul(a, b)"],
        completions=[
            "```python\ndef mul(a, b):\n    return a * b\n```",
            "```python\ndef mul(a, b):\n    return 0\n```",
        ],
        test_code=["assert mul(3, 4) == 12", "assert mul(3, 4) == 12"],
    )
    assert scores == [1.0, 0.0]


def test_task_verification_gate_rejects_trivial_and_unsolvable_tasks():
    gate = TaskVerificationGate(backend="local", container_config=ContainerConfig(use_jail="off"))

    # Trivial task: empty baseline still passes `assert True` -> rejected!
    trivial = {
        "prompt": "noop",
        "test_code": "assert True",
        "reference_solution": "```python\nx = 1\n```",
    }
    r_triv = gate.verify(trivial)
    assert r_triv.accepted is False
    assert "trivial" in r_triv.reason

    # Unsolvable / wrong reference task -> rejected!
    unsolvable = {
        "prompt": "Write sq(x)",
        "test_code": "assert sq(5) == 25",
        "reference_solution": "```python\ndef sq(x):\n    return x\n```",
    }
    r_bad = gate.verify(unsolvable)
    assert r_bad.accepted is False
    assert "reference solution scored" in r_bad.reason

    # Valid task -> accepted!
    valid = {
        "prompt": "Write sq(x)",
        "test_code": "assert sq(5) == 25",
        "reference_solution": "```python\ndef sq(x):\n    return x * x\n```",
    }
    r_ok = gate.verify(valid)
    assert r_ok.accepted is True
    assert gate.stats()["accepted"] == 1
    assert gate.stats()["rejected_trivial"] == 1
    assert gate.stats()["rejected_unsolvable"] == 1


def test_difficulty_tracker_and_synthetic_task_stream_with_real_mixture():
    tracker = DifficultyTracker(window_size=4, min_solve_rate=0.0, max_solve_rate=1.0)
    assert tracker.is_admissible("t1") is True
    assert tracker.record("t1", [1.0, 0.0, 1.0, 0.0]) == 0.5
    assert tracker.is_admissible("t1", strict_band=True) is True
    assert tracker.record("t1", [1.0, 1.0, 1.0, 1.0]) == 1.0
    assert tracker.is_admissible("t1") is False  # mastered -> retired!

    def _gen_task(seed: int) -> dict[str, object]:
        # Even seeds intentionally emit a broken reference to prove the gate filters them out!
        if seed % 2 == 0:
            return {
                "task_id": f"bad_{seed}",
                "prompt": f"Compute {seed} * 2",
                "test_code": f"assert fn() == {seed * 2}",
                "reference_solution": "```python\ndef fn():\n    return -1\n```",
            }
        return {
            "task_id": f"good_{seed}",
            "prompt": f"Compute {seed} * 2",
            "test_code": f"assert fn() == {seed * 2}",
            "reference_solution": f"```python\ndef fn():\n    return {seed * 2}\n```",
        }

    gate = TaskVerificationGate(backend="local", container_config=ContainerConfig(use_jail="off"))
    stream = SyntheticTaskStream(
        _gen_task,
        verification_gate=gate,
        difficulty_tracker=tracker,
        num_workers=2,
        buffer_size=8,
        max_tasks=4,
        strip_reference=True,
    )
    assert isinstance(stream, ShardedDataSource)

    with stream:
        synthesized = list(stream)
        assert len(synthesized) == 4
        assert all(t["task_id"].startswith("good_") for t in synthesized)
        assert all("reference_solution" not in t for t in synthesized)
        assert stream.stats()["gate"]["rejected_unsolvable"] >= 1

    # Mix with a real hard ShardedDataSource using EasyDeL's MixedShardedSource.
    class _ListShardedSource(ShardedDataSource[dict[str, object]]):
        def __init__(self, items: list[dict[str, object]]):
            self._items = items

        @property
        def shard_names(self):
            return ("shard_0",)

        def num_shards(self) -> int:
            return 1

        def open_shard(self, shard_name: str):
            return iter(self._items)

    real_hard = _ListShardedSource([{"task_id": f"real_{i}", "prompt": f"Real {i}", "answer": str(i)} for i in range(4)])
    stream2 = SyntheticTaskStream(
        _gen_task,
        verification_gate=gate,
        num_workers=2,
        buffer_size=8,
        max_tasks=4,
    )
    with stream2:
        mixed = MixedShardedSource(
            sources={"real": real_hard, "synth": stream2},
            weights={"real": 0.5, "synth": 0.5},
            block_size=4,
            seed=7,
            stop_strategy="first_exhausted",
        )
        rows = list(mixed.open_shard(mixed.shard_names[0]))
        assert len(rows) >= 4
        sources = {r["__source__"] for r in rows}
        assert "real" in sources and "synth" in sources


def test_async_grpo_config_environment_validation():
    cfg = AsyncGRPOConfig(
        environment="swe_repo",
        environment_backend="kiri",
        environment_endpoints=["http://cpu-node-1:2375", "http://cpu-node-2:2375"],
    )
    assert cfg.environment == "swe_repo"
    assert cfg.environment_backend == "kiri"
    assert len(cfg.environment_endpoints) == 2
    assert AsyncGRPOConfig().environment_backend == "kiri"

    with pytest.raises(ValueError, match="environment_backend"):
        AsyncGRPOConfig(environment_backend="invalid_backend")


def test_elarge_trainer_async_mode_and_environment_config():
    from easydel.infra.elarge.types.training import BASE_TRAINER_DEFAULTS, normalize_trainer_config

    assert BASE_TRAINER_DEFAULTS["async_mode"] is False
    assert BASE_TRAINER_DEFAULTS["environment_backend"] == "kiri"

    cfg = normalize_trainer_config(
        {
            "trainer_type": "gspo",
            "async_mode": True,
            "max_inflight_tasks": 16,
            "max_staleness": 2,
            "environment": "swe_repo",
            "environment_backend": "kiri",
            "environment_endpoints": ["http://127.0.0.1:2375"],
        }
    )
    assert cfg["async_mode"] is True
    assert cfg["max_inflight_tasks"] == 16
    assert cfg["max_staleness"] == 2
    assert cfg["environment"] == "swe_repo"
    assert cfg["environment_backend"] == "kiri"
    assert cfg["environment_endpoints"] == ["http://127.0.0.1:2375"]

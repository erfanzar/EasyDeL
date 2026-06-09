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

import json
from pathlib import Path

import pytest

import easydel as ed
from easydel.trainers.nemo_gym_trainer._fn import _environment_reward_func


def test_nemo_gym_family_is_public_and_esurge_default():
    cfg = ed.NeMoGymConfig(num_generations=2)

    assert hasattr(ed, "NeMoGymConfig")
    assert hasattr(ed, "NeMoGymTrainer")
    assert cfg.use_esurge_generation is True
    assert cfg.num_generations_eval == 1
    assert cfg.shuffle_dataset is False
    assert cfg.num_return_sequences == 2

    with pytest.raises(ValueError, match="request_timeout"):
        ed.NeMoGymConfig(request_timeout=0)


def test_nemo_gym_environment_feedback_routes_metadata_and_agent_refs():
    calls = []

    class Env:
        def __init__(self, metadata, agent_ref):
            self.metadata = metadata
            self.agent_ref = agent_ref

        def reset(self, metadata=None, agent_ref=None):
            calls.append(("reset", metadata, agent_ref))

        def step(self, action):
            calls.append(("step", self.metadata, self.agent_ref, action))
            return {
                "observation": {"text": f"obs:{action}"},
                "reward": self.metadata["reward"],
                "terminated": True,
                "info": {"agent": self.agent_ref["name"]},
            }

        def close(self):
            calls.append(("close", self.agent_ref["name"]))

    def factory(metadata, agent_ref, request_timeout):
        calls.append(("factory", metadata, agent_ref, request_timeout))
        return Env(metadata, agent_ref)

    trainer = object.__new__(ed.NeMoGymTrainer)
    trainer.arguments = ed.NeMoGymConfig(environment_reward_weight=2.0)
    trainer.environment_factory = factory
    trainer._nemo_active_metadata = [{"reward": 1.5}, {"reward": 2.0}]
    trainer._nemo_active_agent_refs = [{"name": "agent-a"}, {"name": "agent-b"}]

    feedback = trainer._run_environment_feedback(action_texts=["a0", "a1", "b0", "b1"], tool_calls=[None] * 4)

    assert feedback["environment_rewards"] == [3.0, 3.0, 4.0, 4.0]
    assert feedback["environment_infos"][0]["agent"] == "agent-a"
    assert feedback["environment_infos"][2]["agent"] == "agent-b"
    assert calls[0][0] == "factory"


def test_nemo_gym_default_reward_reads_environment_rewards():
    rewards = _environment_reward_func(["a", "b"], environment_rewards=[1, 2.5])

    assert rewards == [1.0, 2.5]


def test_load_nemo_gym_jsonl_preserves_metadata(tmp_path: Path):
    path = tmp_path / "nemo.jsonl"
    item = {
        "responses_create_params": {"input": [{"role": "user", "content": "Solve"}]},
        "agent_ref": {"name": "math"},
        "ground_truth": "42",
    }
    path.write_text(json.dumps(item) + "\n")

    dataset = ed.load_nemo_gym_jsonl(path)
    row = dataset[0]

    assert json.loads(row["metadata"])["ground_truth"] == "42"
    assert row["agent_ref"]["name"] == "math"

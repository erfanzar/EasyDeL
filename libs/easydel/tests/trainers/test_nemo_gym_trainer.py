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
from types import SimpleNamespace

import easydel as ed
import numpy as np
import pytest
from easydel.trainers._shared.tools import FunctionTool
from easydel.trainers.nemo_gym_trainer._fn import _environment_reward_func, _normalize_step_result
from easydel.trainers.prompt_transforms import GRPOPreprocessTransform
from easydel.trainers.utils import GRPODataCollatorGrain, GRPODataCollatorTFDS

pytestmark = pytest.mark.cpu_ok


def test_nemo_gym_family_is_public_and_esurge_default():
    cfg = ed.NeMoGymConfig(num_generations=2)

    assert hasattr(ed, "NeMoGymConfig")
    assert hasattr(ed, "NeMoGymTrainer")
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


def test_extract_nemo_sidechannels_supports_numpy_pytree_list_and_decoupled_agent_refs():
    trainer = object.__new__(ed.NeMoGymTrainer)
    trainer.arguments = ed.NeMoGymConfig()

    # 1) 1D np.ndarray string metadata + PyTree-transposed nested agent_ref (grain.Batch format).
    grain_batch = {
        "input_ids": np.zeros((2, 4), dtype=np.int32),
        "metadata": np.array(['{"reward": 0.1}', '{"reward": 0.2}']),
        "agent_ref": {
            "name": np.array(["agent-a", "agent-b"]),
            "cfg": {"temp": np.array([0.2, 0.8])},
        },
    }
    meta, refs = trainer._extract_nemo_sidechannels(grain_batch)
    assert meta == [{"reward": 0.1}, {"reward": 0.2}]
    assert refs == [
        {"name": "agent-a", "cfg": {"temp": 0.2}},
        {"name": "agent-b", "cfg": {"temp": 0.8}},
    ]

    # 2) PyTree-transposed numeric-only dict metadata (e.g. {"reward": np.array([0.1, 0.2])}).
    numeric_grain_batch = {
        "input_ids": np.zeros((2, 4), dtype=np.int32),
        "metadata": {"reward": np.array([0.1, 0.2])},
    }
    meta_num, refs_num = trainer._extract_nemo_sidechannels(numeric_grain_batch)
    assert meta_num == [{"reward": pytest.approx(0.1)}, {"reward": pytest.approx(0.2)}]
    assert refs_num == [None, None]

    # 3) Uncollated list[dict] batch.
    uncollated = [
        {"input_ids": [1, 2], "metadata": {"reward": 0.3}, "agent_ref": {"name": "a"}},
        {"input_ids": [3, 4], "metadata": {"reward": 0.4}, "agent_ref": {"name": "b"}},
    ]
    meta_list, refs_list = trainer._extract_nemo_sidechannels(uncollated)
    assert meta_list == [{"reward": 0.3}, {"reward": 0.4}]
    assert refs_list == [{"name": "a"}, {"name": "b"}]

    # 4) Decoupled agent_refs when metadata is absent (must NOT discard agent_refs).
    refs_only = {
        "input_ids": np.zeros((2, 4), dtype=np.int32),
        "agent_ref": {"name": np.array(["only-a", "only-b"])},
    }
    meta_empty, refs_present = trainer._extract_nemo_sidechannels(refs_only)
    assert meta_empty == []
    assert refs_present == [{"name": "only-a"}, {"name": "only-b"}]

    # 5) 1D jax.Array column support.
    import jax.numpy as jnp

    jax_batch = {
        "input_ids": np.zeros((2, 4), dtype=np.int32),
        "metadata": jnp.asarray([1.0, 2.0]),
        "agent_ref": jnp.asarray([10, 20]),
    }
    meta_jax, refs_jax = trainer._extract_nemo_sidechannels(jax_batch)
    assert meta_jax == [{"value": 1.0}, {"value": 2.0}]
    assert refs_jax == [10, 20]


def test_nemo_preprocess_and_grain_tfds_collators_preserve_numeric_only_dict_metadata():
    class _MiniTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kwargs):
            del kwargs
            ids = [ord(c) for c in text][:8]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    trainer = object.__new__(ed.NeMoGymTrainer)
    trainer.arguments = ed.NeMoGymConfig(max_prompt_length=8)
    trainer.processing_class = _MiniTokenizer()
    trainer.padding_value = 0
    trainer._train_source = None
    trainer._environment_tools = []

    transform = trainer._get_preprocess_transform()
    assert transform is not None
    # Verify plain GRPOPreprocessTransform WOULD drop numeric-only dict metadata, while NeMoGym preserves it.
    raw_examples = [
        {"prompt": "hi", "metadata": {"reward": 0.1}, "agent_ref": {"name": "a"}},
        {"prompt": "yo", "metadata": {"reward": 0.2}, "agent_ref": {"name": "b"}},
    ]
    base_grpo = GRPOPreprocessTransform(tokenizer=trainer.processing_class, max_prompt_length=8)
    assert "metadata" not in base_grpo(raw_examples[0])

    transformed = [transform(ex) for ex in raw_examples]
    assert [row["metadata"] for row in transformed] == [{"reward": 0.1}, {"reward": 0.2}]
    assert [row["agent_ref"] for row in transformed] == [{"name": "a"}, {"name": "b"}]

    # End-to-end Grain collation + PyTree batching -> _extract_nemo_sidechannels.
    grain_collator = trainer.create_grain_collect_function(max_sequence_length=8)
    assert isinstance(grain_collator._base_collator, GRPODataCollatorGrain)
    per_item = [grain_collator(row) for row in transformed]
    # Simulate grain.Batch PyTree stacking across the two examples.
    batched_grain = {
        "input_ids": np.stack([item["input_ids"] for item in per_item]),
        "attention_mask": np.stack([item["attention_mask"] for item in per_item]),
        "metadata": {"reward": np.array([item["metadata"]["reward"] for item in per_item])},
        "agent_ref": {"name": np.array([item["agent_ref"]["name"] for item in per_item])},
    }
    meta_out, refs_out = trainer._extract_nemo_sidechannels(batched_grain)
    assert meta_out == [{"reward": pytest.approx(0.1)}, {"reward": pytest.approx(0.2)}]
    assert refs_out == [{"name": "a"}, {"name": "b"}]

    # End-to-end TFDS collation -> _extract_nemo_sidechannels.
    tfds_collator = trainer.create_tfds_collect_function(max_sequence_length=8)
    assert isinstance(tfds_collator._base_collator, GRPODataCollatorTFDS)
    batched_tfds = tfds_collator(transformed)
    meta_tfds, refs_tfds = trainer._extract_nemo_sidechannels(batched_tfds)
    assert meta_tfds == [{"reward": 0.1}, {"reward": 0.2}]
    assert refs_tfds == [{"name": "a"}, {"name": "b"}]


def test_nemo_expand_sidechannel_validates_divisibility_and_normalize_step_result_handles_none_and_bad_tuples():
    assert ed.NeMoGymTrainer._expand_sidechannel(["a", "b"], 4) == ["a", "a", "b", "b"]
    assert ed.NeMoGymTrainer._expand_sidechannel([], 3) == [None, None, None]
    with pytest.raises(ValueError, match="divisible"):
        ed.NeMoGymTrainer._expand_sidechannel(["a", "b"], 3)
    with pytest.raises(ValueError, match="divisible"):
        ed.NeMoGymTrainer._expand_sidechannel(["a", "b", "c"], 2)

    # None rewards must default safely to 0.0 (not crash float(None)).
    assert _normalize_step_result(("obs", None, True, False, None)) == ("obs", 0.0, True, False, {})
    assert _normalize_step_result(("obs", None, True, None)) == ("obs", 0.0, True, False, {})
    assert _normalize_step_result({"observation": "x", "reward": None}) == ("x", 0.0, False, False, {})
    assert _normalize_step_result({"observation": "x", "reward": None, "env_reward": 0.75}) == (
        "x",
        0.75,
        False,
        False,
        {},
    )
    assert _normalize_step_result(SimpleNamespace(observation="x", reward=None, terminated=True, truncated=False, info=None)) == (
        "x",
        0.0,
        True,
        False,
        {},
    )
    assert _environment_reward_func(["a", "b"], environment_rewards=[None, 2.5]) == [0.0, 2.5]

    # Unexpected tuple lengths must raise ValueError instead of silently returning 0.0.
    for bad_tuple in ((), ("obs",), ("obs", 1.0), ("obs", 1.0, True), ("obs", 1.0, True, False, {}, "extra")):
        with pytest.raises(ValueError, match="4-tuple or 5-tuple"):
            _normalize_step_result(bad_tuple)


def test_nemo_run_environment_feedback_wraps_tool_env_and_coordinates_multihost(monkeypatch):
    import jax

    inner_steps: list[str] = []

    class _Env:
        def reset(self, metadata=None, agent_ref=None):
            self.metadata = metadata
            self.agent_ref = agent_ref

        def step(self, action):
            inner_steps.append(action)
            return {"observation": f"inner:{action}", "reward": 1.0, "terminated": True, "info": {}}

        def close(self):
            return None

    tool = FunctionTool(lambda text: f"ECHO:{text}", tool_name="echo")
    trainer = object.__new__(ed.NeMoGymTrainer)
    trainer.arguments = ed.NeMoGymConfig(environment_reward_weight=1.0)
    trainer.environment_factory = lambda metadata, agent_ref: _Env()
    trainer._environment_tools = [tool]
    trainer._environment_tool_parser = None
    trainer._nemo_active_metadata = [{"id": 1}]
    trainer._nemo_active_agent_refs = [{"name": "a"}]

    feedback = trainer._run_environment_feedback(
        action_texts=['<tool_call>{"name":"echo","arguments":{"text":"hi"}}</tool_call>', "plain"],
        tool_calls=[None, None],
    )
    assert feedback["environment_observations"][0] == "[echo]: ECHO:hi"
    assert feedback["environment_observations"][1] == "inner:plain"
    assert inner_steps == ["plain"]

    # Multi-host coordination: rank 0 runs envs once and broadcasts; rank 1 never invokes factory.
    broadcast_box: list[object] = []
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        "easydel.trainers.nemo_gym_trainer.nemo_gym_trainer._broadcast_object_from_host0",
        lambda obj: (broadcast_box.append(obj), obj)[1],
    )
    env_count = 0

    def counting_factory(metadata, agent_ref):
        nonlocal env_count
        env_count += 1
        return _Env()

    trainer.environment_factory = counting_factory
    fb_rank0 = trainer._run_environment_feedback(action_texts=["plain"], tool_calls=[None])
    assert env_count == 1

    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        "easydel.trainers.nemo_gym_trainer.nemo_gym_trainer._broadcast_object_from_host0",
        lambda obj: broadcast_box[0],
    )
    fb_rank1 = trainer._run_environment_feedback(action_texts=["plain"], tool_calls=[None])
    assert env_count == 1  # Rank 1 did NOT re-execute environments!
    assert fb_rank1 == fb_rank0


def test_load_nemo_gym_jsonl_handles_heterogeneous_schemas(tmp_path: Path):
    path = tmp_path / "heterogeneous.jsonl"
    items = [
        {"prompt": "plain string prompt", "agent_ref": {"name": "math", "id": 1}},
        {
            "responses_create_params": {"input": [{"role": "user", "content": "chat prompt"}]},
            "agent_ref": {"name": "code", "id": "abc", "nested": {"k": True}},
        },
        {"prompt": "third prompt", "agent_ref": "string-agent-ref"},
    ]
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")

    dataset = ed.load_nemo_gym_jsonl(path)
    assert len(dataset) == 3

    trainer = object.__new__(ed.NeMoGymTrainer)
    trainer.arguments = ed.NeMoGymConfig()
    rows = [dataset[i] for i in range(len(dataset))]
    meta, refs = trainer._extract_nemo_sidechannels(rows)
    assert len(meta) == 3
    assert refs[0] == {"name": "math", "id": 1}
    assert refs[1] == {"name": "code", "id": "abc", "nested": {"k": True}}
    assert refs[2] == "string-agent-ref"

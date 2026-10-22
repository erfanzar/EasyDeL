"""Host-only replay and shared-tool regressions; no JAX array execution."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.grpo_replay_buffer_trainer import grpo_replay_buffer_trainer as replay_module
from easydel.trainers.grpo_replay_buffer_trainer.replay_buffer import _ReplayBuffer

pytestmark = pytest.mark.cpu_ok


class HostArray(np.ndarray):
    """NumPy-only array double supporting the immutable indexed-update API."""

    @property
    def at(self):
        return HostUpdate(self)


class HostUpdate:
    def __init__(self, array, index=None):
        self.array = array
        self.index = index

    def __getitem__(self, index):
        return HostUpdate(self.array, index)

    def set(self, value):
        result = self.array.copy()
        result[self.index] = value
        return result


def host_array(value):
    return np.asarray(value).view(HostArray)


@pytest.fixture
def host_replay(monkeypatch):
    # Exercise replay's storage/replacement state machine without initializing
    # a JAX backend. Actual accelerator-array coverage lives in the numerical file.
    monkeypatch.setattr(
        replay_module,
        "jnp",
        SimpleNamespace(
            abs=np.abs,
            max=np.max,
            sum=np.sum,
            std=np.std,
            float32=np.float32,
            pad=lambda *args, **kwargs: host_array(np.pad(*args, **kwargs)),
        ),
    )
    monkeypatch.setattr(replay_module.jax, "device_get", lambda value: value)
    trainer = object.__new__(replay_module.GRPOWithReplayBufferTrainer)
    trainer.arguments = SimpleNamespace(num_generations=2)
    trainer._pad_token_id = 0
    trainer.replay_buffer = _ReplayBuffer(2, seed=0)
    return trainer


def rollout_batch(length, *, useful, separate_attention=True):
    loss_mask = np.ones((2, length), dtype=np.int32)
    loss_mask[:, 1:2] = 0
    batch = {
        "prompt_ids": host_array([[1, 2]]),
        "prompt_mask": host_array([[1, 1]]),
        "completion_ids": host_array(np.arange(10, 10 + 2 * length).reshape(2, length)),
        "completion_mask": host_array(loss_mask),
        "advantages": host_array([1.0, -1.0] if useful else [0.0, 0.0]),
        # GRPO includes this key even when beta=0 disables reference scoring.
        "ref_per_token_logps": None,
        "num_items_in_batch": int(loss_mask.sum()),
    }
    if separate_attention:
        batch["completion_attention_mask"] = host_array(np.ones_like(loss_mask))
    return batch


@pytest.mark.parametrize("stored_length,active_length", [(3, 5), (5, 3)])
@pytest.mark.parametrize("legacy", [False, True])
def test_replay_stores_pads_and_replaces_grpo_attention_mask(host_replay, stored_length, active_length, legacy):
    useful = rollout_batch(stored_length, useful=True, separate_attention=not legacy)
    _, inserted = host_replay._apply_replay_buffer(useful)
    assert inserted["replay_groups_added"] == 1
    assert inserted["replay_groups_replaced"] == 0
    flat = rollout_batch(active_length, useful=False)
    actual, replaced = host_replay._apply_replay_buffer(flat)
    assert replaced["replay_groups_replaced"] == 1
    assert replaced["replay_buffer_size"] == 1
    assert "replay_skipped_unsupported_keys" not in replaced
    target_length = max(stored_length, active_length)
    for key in ("completion_ids", "completion_mask", "completion_attention_mask"):
        source_key = "completion_mask" if legacy and key == "completion_attention_mask" else key
        expected = np.pad(useful[source_key], ((0, 0), (0, target_length - stored_length)))
        np.testing.assert_array_equal(actual[key], expected)
    np.testing.assert_array_equal(actual["advantages"], [1.0, -1.0])
    assert actual["num_items_in_batch"] == useful["num_items_in_batch"]
    assert actual["ref_per_token_logps"] is None
    assert flat["completion_ids"].shape == (2, active_length)
    np.testing.assert_array_equal(flat["advantages"], [0.0, 0.0])


@pytest.mark.parametrize("stored_length,active_length", [(3, 5), (5, 3), (1, 5), (5, 1)])
@pytest.mark.parametrize("mode", ["sequence_truncate", "sequence_mask", "token_truncate", "token_mask"])
def test_esurge_correction_survives_replay_for_replaced_and_live_groups(
    host_replay, monkeypatch, stored_length, active_length, mode
):
    from easydel.trainers.group_relative_policy_optimization import grpo_trainer as grpo_module

    # Use the actual correction constructor and replay state machine, but only
    # NumPy arrays. Never initialize a JAX backend in this host contract test.
    monkeypatch.setattr(
        grpo_module,
        "jnp",
        SimpleNamespace(float32=np.float32, sum=np.sum, where=np.where, exp=np.exp, minimum=np.minimum, log=np.log),
    )
    monkeypatch.setattr(grpo_module.jax.lax, "stop_gradient", lambda value: value)
    host_replay.arguments.esurge_importance_sampling_mode = mode
    host_replay.arguments.esurge_importance_sampling_cap = 10.0
    host_replay.replay_buffer = _ReplayBuffer(1, seed=0)

    def attach_correction(batch, ratios):
        mask = batch["completion_mask"]
        log_ratio = np.log(np.asarray(ratios, dtype=np.float32))[:, None]
        if mode.startswith("sequence_"):
            log_ratio = log_ratio / mask.sum(axis=1, keepdims=True)
        policy_logps = np.broadcast_to(log_ratio, mask.shape)
        ratio = host_replay._esurge_importance_sampling_ratio(policy_logps, np.zeros_like(policy_logps), mask)
        batch["importance_sampling_ratio"] = host_array(ratio)
        return batch

    stored = attach_correction(rollout_batch(stored_length, useful=True), [1.25, 0.5])
    # Ensure the original stored group wins the capacity-one priority buffer,
    # so replay differs from the useful group retained in the current batch.
    stored["advantages"] = host_array([2.0, -2.0])
    host_replay._apply_replay_buffer(stored)
    live = attach_correction(rollout_batch(active_length, useful=True), [1.5, 0.75])
    flat = attach_correction(rollout_batch(active_length, useful=False), [0.25, 0.25])
    active = {
        key: host_array(np.concatenate([live[key], flat[key]], axis=0))
        if isinstance(live[key], np.ndarray)
        else live[key]
        for key in live
    }
    actual, metrics = host_replay._apply_replay_buffer(active)
    assert metrics["replay_groups_replaced"] == 1
    target_length = max(stored_length, active_length)
    if mode.startswith("sequence_"):
        assert actual["importance_sampling_ratio"].shape == (4, 1)
        expected = [[1.5], [0.75], [1.25], [0.5]]
        # This is how grpo_step consumes sequence-wide ratios: every valid
        # token, including tokens after position zero, receives the same weight.
        weighted_tokens = actual["completion_mask"] * actual["importance_sampling_ratio"]
        np.testing.assert_allclose(weighted_tokens, actual["completion_mask"] * np.asarray(expected), rtol=1e-6)
        assert np.all(weighted_tokens[actual["completion_mask"] > 0] > 0)
        assert np.any(weighted_tokens[:, 2] > 0)
    else:
        assert actual["importance_sampling_ratio"].shape == (4, target_length)
        expected = np.concatenate(
            [
                np.pad(live["importance_sampling_ratio"], ((0, 0), (0, target_length - active_length))),
                np.pad(stored["importance_sampling_ratio"], ((0, 0), (0, target_length - stored_length))),
            ]
        )
    np.testing.assert_allclose(actual["importance_sampling_ratio"], expected, rtol=1e-6)
    np.testing.assert_array_equal(actual["advantages"], [1.0, -1.0, 2.0, -2.0])
    np.testing.assert_array_equal(active["importance_sampling_ratio"][:2], live["importance_sampling_ratio"])


def test_grpo_shared_tool_schema_and_execution():
    def echo(message: str) -> str:
        """Return a visible tool observation."""
        return "seen:" + message

    callable_tool, registered_tool = GRPOTrainer._normalize_environment_tools([echo, "regex"])
    schema = callable_tool.chat_schema
    assert schema["name"] == "echo"
    assert schema["parameters"]["properties"]["message"]["type"] == "string"
    assert schema["parameters"]["required"] == ["message"]
    assert callable_tool.execute('{"message": "hello"}') == "seen:hello"
    assert registered_tool.execute('{"pattern": "[a-z]+", "text": "one 2 two"}') == '["one", "two"]'


def test_grpo_shared_tool_environment_dispatches_and_preserves_plain_actions():
    from easydel.trainers.group_relative_policy_optimization import grpo_trainer as module

    tools = GRPOTrainer._normalize_environment_tools(["regex"])
    inner = SimpleNamespace(
        reset=lambda seed=None: SimpleNamespace(observation="question"),
        step=lambda action: SimpleNamespace(observation="final:" + action),
    )
    wrapper = module.ToolEnvWrapper(inner, tools, tool_call_parser=module.create_tool_call_parser(None))
    assert wrapper.reset(seed=7).observation == "question"
    result = wrapper.step('<tool_call>{"name":"regex","arguments":{"pattern":"[a-z]+","text":"one 2 two"}}</tool_call>')
    assert result.observation == '[regex]: ["one", "two"]'
    assert not result.terminated
    assert wrapper.step("answer").observation == "final:answer"

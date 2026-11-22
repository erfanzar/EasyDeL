"""Host-only trainer runtime tests: scheduled KTO batches, SDPO lengths, async GRPO snapshots, rollouts."""

from types import SimpleNamespace

import pandas as pd
import pytest
from easydel.trainers._shared.environment import ToolEnvWrapper
from easydel.trainers.agentic_moshpit.env_manager import RolloutManager
from easydel.trainers.async_grpo_trainer.async_grpo_trainer import AsyncGRPOTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from easydel.trainers.kto_trainer._fn import _prepare_kto_scheduled_batch
from easydel.trainers.self_distillation_policy_optimization import SDPOConfig

pytestmark = pytest.mark.cpu_ok


def test_scheduled_kto_accepts_explicit_runtime_pair_with_legacy_reference_key():
    reference = [-0.1, -0.2]
    batch = {
        "reference_logps": [-1.0, -2.0],
        "reference_KL_logps": reference,
        "_policy_kl_logps": [-0.3, -0.4],
    }
    call = SimpleNamespace(batch=batch, get={"calculate_kl": True}.get)
    prepared = _prepare_kto_scheduled_batch(call)
    assert prepared["_reference_kl_logps"] is reference
    assert prepared["_policy_kl_logps"] is batch["_policy_kl_logps"]
    assert "_reference_kl_logps" not in batch
    # A dataset reference cache alone cannot establish the shuffled runtime pair.
    del batch["_policy_kl_logps"]
    with pytest.raises(RuntimeError, match="requires reference_state and forward_fn"):
        _prepare_kto_scheduled_batch(call)


def test_sdpo_keeps_valid_inferred_context_cap_and_completion_trim():
    config = SDPOConfig(max_prompt_length=512, max_completion_length=512)
    assert config.max_length == 1024
    assert config.max_completion_length == 256
    assert config.max_prompt_length + config.max_feedback_length + config.max_completion_length == 1024


def test_forced_snapshot_refresh_keeps_busy_outgoing_owner_alive():
    trainer = object.__new__(AsyncGRPOTrainer)
    trainer.arguments = SimpleNamespace(weight_sync_steps=4)
    outgoing = SimpleNamespace(policy_step=4, released=False)
    outgoing.release = lambda: setattr(outgoing, "released", True)
    replacement = SimpleNamespace(policy_step=5, released=False)
    trainer._take_rollout_snapshot = lambda *args, **kwargs: replacement
    current, elapsed = trainer._sync_rollout_snapshot(
        outgoing, object(), 5, force=True, can_release=False, cache_scope_key="test"
    )
    assert current is replacement
    assert current.policy_step == 5
    assert not outgoing.released
    assert elapsed >= 0.0


def test_completion_logging_without_checkpoint_model_name(monkeypatch, tmp_path):
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        log_completions=True, log_completions_hub_repo="user/grpo-completions", save_directory=str(tmp_path)
    )
    trainer._completion_log_dir = None
    trainer._completion_commit_scheduler = None
    repos = []
    triggers = []

    class Scheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def trigger(self):
            triggers.append(True)

    monkeypatch.setattr("easydel.trainers.group_relative_policy_optimization.grpo_trainer.jax.process_index", lambda: 0)
    monkeypatch.setattr("easydel.trainers.group_relative_policy_optimization.grpo_trainer.jax.device_get", lambda x: x)
    monkeypatch.setattr("huggingface_hub.create_repo", lambda *args, **kwargs: repos.append((args, kwargs)))
    monkeypatch.setattr("huggingface_hub.CommitScheduler", Scheduler)
    rows = trainer._maybe_log_grpo_completions(
        prompts=["p0", "p1"], completions=["c0", "c1"], completion_lengths=[2, 3], step=12
    )
    assert len(rows) == 2
    assert repos == [(("user/grpo-completions",), {"repo_type": "dataset", "exist_ok": True})]
    assert triggers == [True]
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    frame = pd.read_parquet(files[0])
    assert frame["step"].tolist() == [12, 12]
    assert frame["completion"].tolist() == ["c0", "c1"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_rollout_accepts_reset_step_close_structural_environment(wrapped):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(message["content"] for message in messages)

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    class Environment:
        def reset(self, seed=None):
            return SimpleNamespace(observation="solve", info={"seed": seed})

        def step(self, action):
            return SimpleNamespace(observation="", reward=1.0, terminated=True, truncated=False, info={})

        def close(self):
            pass

    def factory():
        env = Environment()
        return ToolEnvWrapper(env, tools=[]) if wrapped else env

    manager = RolloutManager(tokenizer=Tokenizer(), max_steps=2, max_seq_length=256)
    trajectories = manager.run_grouped_episodes(
        env_factory=factory, generate_fn=lambda prompts, **kwargs: ["answer"] * len(prompts), group_size=1
    )
    assert len(trajectories) == 1
    assert trajectories[0].episode_reward == 1.0
    assert trajectories[0].turns[-1].content == "answer"

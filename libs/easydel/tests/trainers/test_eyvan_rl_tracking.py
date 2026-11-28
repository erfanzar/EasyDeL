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

"""CPU-only end-to-end tests for RL-native Eyvan telemetry across core EasyDeL trainers."""

from __future__ import annotations

import math
import threading
import types
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers import training_configurations as tc
from easydel.trainers._online_rl import trainer as online_mod
from easydel.trainers._online_rl.trainer import OnlineActorCriticTrainer
from easydel.trainers.async_rollout_mixin import AsyncRolloutMixin
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.metrics import MetricsTracker, StepMetrics
from easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer import PPOTrainer
from easydel.trainers.trainer import Trainer
from easydel.trainers.training_configurations import TrainingArguments
from eyvan.track._store import store_for_uri
from eyvan.track.tracker.index import TrackingIndex

# Pure-Python + NumPy + local Eyvan store + TrackingIndex tests; no JAX device computation runs.
pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def _clean_env_and_no_dist(monkeypatch):
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    for key in (
        "EYVAN_TRACKING_URI",
        "EYVAN_WORKLOAD_ID",
        "EYVAN_PROGRESS_URL",
        "EYVAN_MANAGED_RESULT_TOKEN",
        "EYVAN_DISTRIBUTED_PROCESS_ID",
        "JAX_PROCESS_ID",
        "EYVAN_ATTEMPT",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_args(store_dir: Path | None = None, **kwargs) -> TrainingArguments:
    defaults: dict[str, object] = {
        "model_name": "acme/rl-model",
        "trainer_prefix": "GRPO",
        "use_wandb": False,
        "save_directory": None,
        "eyvan_track_flush_interval_s": 0.1,
    }
    if store_dir is not None:
        project = str(kwargs.get("eyvan_track_project", "rl-proj"))
        defaults["use_eyvan_track"] = True
        defaults["eyvan_track_uri"] = str(store_dir / "default" / project)
        defaults.setdefault("eyvan_track_project", project)
    defaults.update(kwargs)
    return TrainingArguments(**defaults)


def _scan_store(store_dir: Path, tmp_path: Path, name: str = "index.db") -> tuple[TrackingIndex, object]:
    store = store_for_uri(str(store_dir))
    index = TrackingIndex(tmp_path / name)
    scan_res = index.scan(store)
    return index, scan_res


class _DummyPbar:
    def reset(self):
        pass

    def set_postfix(self, **kwargs):
        pass

    def update(self, n=1):
        pass


def test_training_arguments_rl_primitives_and_tracking_index_scan(tmp_path):
    store_dir = tmp_path / "rl_store"
    args = _make_args(store_dir, eyvan_track_name="rl-e2e")
    run = args.get_eyvan_track()
    assert run is not None
    run_id = run.id

    # 1. Scalars with custom x-axes (tokens, wall_s, sanitized key, non-finite ignored)
    args.log_metrics(
        {"train/loss": 0.8, "rl/reward_mean": 0.4},
        step=1,
        axes={"tokens": 100, "wall_s": 1.25, "bad axis!": 7.0, "nan_axis": float("nan")},
    )
    args.log_metrics(
        {"train/loss": 0.5, "rl/reward_mean": 0.9},
        step=2,
        axes={"tokens": 250, "wall_s": 2.75},
    )

    # 2. Completions with decomposed reward components (including unsanitized key + np.bool_ truncated)
    ok_comp_1 = args.log_eyvan_completions(
        prompt="Compute 2 + 2.",
        responses=["4", "5", "4 (verified)", "4"],
        rewards={
            "accuracy (exact)": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
            "format score": np.asarray([0.5, 0.0, 0.5, 0.5], dtype=np.float32),
        },
        advantages=np.asarray([0.5, -1.5, 0.5, 0.5], dtype=np.float32),
        group="g-step1-0",
        answer="4",
        tokens=np.asarray([4.0, 8.0, 6.0, 5.0], dtype=np.float32),
        kl=np.asarray([0.02, 0.10, 0.03, 0.01], dtype=np.float32),
        truncated=np.asarray([False, True, False, False], dtype=np.bool_),
        policy_version=1,
        step=1,
    )
    assert ok_comp_1 is True

    # Zero-variance group at step 2
    ok_comp_2 = args.log_eyvan_completions(
        prompt="Compute 3 + 3.",
        responses=["6", "6"],
        rewards={"accuracy (exact)": [1.0, 1.0], "format score": [0.5, 0.5]},
        tokens=[3, 4],
        truncated=[False, False],
        policy_version=2,
        step=2,
    )
    assert ok_comp_2 is True

    # 3. Episodes with policy lag + >32 extra keys clamped + long actor/env/checkpoint truncated safely
    oversized_extra = {f"metric ({i})": float(i) for i in range(40)}
    oversized_extra["nan_val"] = float("nan")
    oversized_extra["complex_val"] = np.complex64(1.0 + 2.0j)
    ok_ep_1 = args.log_eyvan_episode(
        ret=ml_dtypes.bfloat16(1.5),
        length=6,
        success=np.bool_(True),
        kind="train",
        actor="a" * 100,
        env="e" * 200,
        seed=42,
        policy_version=1,
        learner_version=4,
        checkpoint="c" * 300,
        step=4,
        axes={"tokens": 250, "env_steps": 12},
        extra=oversized_extra,
    )
    ok_ep_2 = args.log_eyvan_episode(
        ret=0.0,
        length=10,
        success=False,
        kind="train",
        actor="worker-0",
        env="math_env",
        policy_version=3,
        learner_version=4,
        step=4,
        extra={"turns": 10.0},
    )
    ok_ep_eval = args.log_eyvan_episode(
        ret=2.0,
        length=4,
        success=True,
        kind="eval",
        env="math_env",
        checkpoint="ckpt-step-4",
        policy_version=4,
        step=4,
    )
    assert ok_ep_1 is True and ok_ep_2 is True and ok_ep_eval is True

    # 4. Rollouts: (a) tracks + int actions + action_names, (b) steps + string actions (auto-indexed!)
    ok_ro_1 = args.log_eyvan_rollout(
        tracks={"reward": [0.2, 0.8], "value (est)": [0.1, 0.7]},
        actions=[0, 1],
        action_names=["search", "answer"],
        episode_id="ep-tracks-1",
        actor="worker-0",
        env="math_env",
        policy_version=1,
        step=1,
        axes={"tokens": 100},
    )
    ok_ro_2 = args.log_eyvan_rollout(
        steps=[
            {"reward": 0.0, "action_tokens": 12.0},
            {"reward": 1.0, "action_tokens": 8.0},
        ],
        actions=["calc(2+2)", "submit(4)"],
        episode_id="ep-steps-2",
        actor="worker-0",
        env="math_env",
        policy_version=2,
        step=2,
        axes={"tokens": 250},
    )
    assert ok_ro_1 is True and ok_ro_2 is True

    # 5. Histograms (float32, bfloat16, complex rejected, empty rejected)
    assert (
        args.log_eyvan_histogram(
            "rl/reward_dist", np.asarray([0.0, 0.5, 1.5], dtype=np.float32), step=1, axes={"tokens": 100}
        )
        is True
    )
    assert (
        args.log_eyvan_histogram(
            "rl/bf16_dist", np.asarray([-0.5, 0.25, 1.0], dtype=tc.jnp.bfloat16), step=2, axes={"tokens": 250}
        )
        is True
    )
    assert (
        args.log_eyvan_histogram("rl/complex_skip", np.asarray([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64), step=2)
        is False
    )
    assert args.log_eyvan_histogram("rl/empty_skip", np.asarray([], dtype=np.float32), step=2) is False

    args.close_eyvan_track(status="finished")

    # Scan with real Eyvan TrackingIndex and assert EVERY stream indexes cleanly with ZERO errors/drops
    index, scan_res = _scan_store(store_dir, tmp_path)
    assert scan_res.errors == 0
    assert scan_res.ingested_segments > 0

    detail = index.run_detail("default", "rl-proj", run_id)
    assert detail is not None
    assert detail["errors"] == []
    assert detail["stats"]["dropped"] == {"keys": 0, "rate": 0, "invalid": 0, "queue": 0}
    assert set(detail["axes"]) >= {"tokens", "wall_s", "bad_axis"}
    assert detail["completion_groups"] == 2
    assert detail["rollouts"] == 2
    assert detail["episodes"] == {"train": 2, "eval": 1}
    assert set(detail["hist_keys"]) == {"rl/reward_dist", "rl/bf16_dist"}

    # Scalars queried against custom x-axes: tokens and wall_s
    tok_series = index.series("default", "rl-proj", run_id, ["train/loss"], x="tokens")
    assert tok_series is not None
    assert [(pt[0], pt[1]) for pt in tok_series["series"][0]["points"]] == [
        (100.0, pytest.approx(0.8)),
        (250.0, pytest.approx(0.5)),
    ]

    wall_series = index.series("default", "rl-proj", run_id, ["rl/reward_mean"], x="wall_s")
    assert wall_series is not None
    assert [(pt[0], pt[1]) for pt in wall_series["series"][0]["points"]] == [
        (1.25, pytest.approx(0.4)),
        (2.75, pytest.approx(0.9)),
    ]

    # Histograms indexed with custom x-axis
    hist_view = index.histograms("default", "rl-proj", run_id, "rl/bf16_dist", x="tokens")
    assert hist_view is not None
    assert len(hist_view["columns"]) == 1
    assert hist_view["columns"][0]["x"] == pytest.approx(250.0)
    assert hist_view["columns"][0]["n"] == 3

    # Completions: verify reward decomposition, pass_rate, std_reward, zero_variance
    comp_list = index.completions("default", "rl-proj", run_id)
    assert comp_list is not None
    assert comp_list["total"] == 2
    by_group = {grp["group"]: grp for grp in comp_list["groups"]}
    assert by_group["g-step1-0"]["pass_rate"] == pytest.approx(0.75)
    assert by_group["g-step1-0"]["std_reward"] > 0.0
    assert by_group["g-step1-0"]["zero_variance"] is False
    assert by_group["g-step1-0"]["truncated"] == 1

    comp_stats = index.completion_stats("default", "rl-proj", run_id)
    assert comp_stats is not None
    step1_stats = next(row for row in comp_stats["steps"] if row["step"] == 1)
    assert step1_stats["components"]["accuracy_exact"] == pytest.approx(0.75)
    assert step1_stats["components"]["format_score"] == pytest.approx(0.375)
    assert step1_stats["truncated_frac"] == pytest.approx(0.25)

    # Episodes: verify policy lag (learner_version - policy_version = 3 and 1) & extra cap <= 32
    ep_page = index.episodes("default", "rl-proj", run_id, kind="train")
    assert ep_page is not None
    assert ep_page["total"] == 2
    for ep in ep_page["episodes"]:
        assert len(ep["extra"]) <= 32
    breakdown = index.episode_breakdown("default", "rl-proj", run_id, kind="train")
    assert breakdown is not None
    assert breakdown["lag"]["count"] == 2
    assert breakdown["lag"]["max"] == 3
    assert breakdown["lag"]["p50"] == pytest.approx(2.0)
    assert len(breakdown["evals"]) == 1
    assert breakdown["evals"][0]["checkpoint"] == "ckpt-step-4"

    # Rollouts: verify tracks, integer actions, and auto-indexed action_names
    ro_list = index.rollouts("default", "rl-proj", run_id)
    assert ro_list is not None
    assert ro_list["total"] == 2
    ro_by_ep = {row["episode_id"]: row for row in ro_list["rollouts"]}
    ro2_detail = index.rollout_detail("default", "rl-proj", run_id, ro_by_ep["ep-steps-2"]["id"])
    assert ro2_detail is not None
    assert ro2_detail["actions"] == [0, 1]
    assert ro2_detail["action_names"] == ["calc(2+2)", "submit(4)"]
    assert ro2_detail["ret"] == pytest.approx(1.0)
    assert ro2_detail["tracks"]["action_tokens"] == [12.0, 8.0]


def test_base_trainer_log_metrics_merges_wall_s_tokens_and_caller_axes(tmp_path):
    store_dir = tmp_path / "base_axes_store"
    args = _make_args(store_dir, eyvan_track_name="base-axes", report_steps=1, log_steps=1)
    trainer = object.__new__(Trainer)
    trainer.arguments = args
    trainer.eyvan_run = None
    pbar = _DummyPbar()

    # Step 1: per-step num_items_in_batch accumulates
    trainer.log_metrics({"train/loss": 1.0, "num_items_in_batch": 64}, pbar, step=1)
    # Step 2: per-step agent_flow/action_tokens accumulates on top + merges caller axes
    trainer.log_metrics(
        {"train/loss": 0.7, "agent_flow/action_tokens": 36},
        pbar,
        step=2,
        axes={"env_turns": 12},
    )
    # Step 3: explicit cumulative train/tokens_seen advances monotonic counter
    trainer.log_metrics({"train/loss": 0.4, "train/tokens_seen": 500}, pbar, step=3)
    # Eval step must not inflate train tokens_seen
    trainer.log_metrics({"eval/loss": 0.5, "num_items_in_batch": 999}, pbar, step=3, mode="eval")

    assert trainer._eyvan_tokens_seen == pytest.approx(500.0)
    run_id = args.get_eyvan_track().id
    args.close_eyvan_track()

    index, scan_res = _scan_store(store_dir, tmp_path, name="base_axes.db")
    assert scan_res.errors == 0
    detail = index.run_detail("default", "rl-proj", run_id)
    assert detail is not None
    assert set(detail["axes"]) >= {"wall_s", "tokens", "env_turns"}

    series_tok = index.series("default", "rl-proj", run_id, ["train/loss"], x="tokens")
    assert series_tok is not None
    pts = [(pt[0], pt[1]) for pt in series_tok["series"][0]["points"]]
    assert pts == [
        (64.0, pytest.approx(1.0)),
        (100.0, pytest.approx(0.7)),
        (500.0, pytest.approx(0.4)),
    ]

    series_wall = index.series("default", "rl-proj", run_id, ["train/loss"], x="wall_s")
    assert series_wall is not None
    wall_xs = [pt[0] for pt in series_wall["series"][0]["points"]]
    assert len(wall_xs) == 3
    assert all(x >= 0.0 for x in wall_xs)
    assert wall_xs == sorted(wall_xs)


def test_grpo_rl_health_icepop_diagnostics_and_eyvan_logging(tmp_path):
    store_dir = tmp_path / "grpo_store"
    args = _make_args(store_dir, eyvan_track_name="grpo-icepop")
    args.max_completion_length = 4

    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = args
    trainer._eos_token_id = np.asarray([2], dtype=np.int64)
    trainer.kl_estimator = "k3"
    trainer.reward_func_names = ["accuracy", "format"]
    trainer._async_rollout_policy_version = 7

    # 2 prompt groups x G=2 completions = 4 completions, seq len = 4
    # Row 0: ends with EOS (not truncated), pass (reward 1.5)
    # Row 1: full length 4 without EOS (truncated), fail (reward 0.0)
    # Row 2 & 3: zero-variance group (both reward 1.0, EOS present)
    completion_ids = np.asarray(
        [
            [10, 11, 2, 0],
            [10, 11, 12, 13],
            [14, 2, 0, 0],
            [15, 2, 0, 0],
        ],
        dtype=np.int64,
    )
    completion_attention_mask = np.asarray(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    completion_mask = completion_attention_mask.copy()
    rewards_per_func = np.asarray(
        [
            [1.0, 0.5, 0.25],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rewards = rewards_per_func.sum(axis=1)
    advantages = np.asarray([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    is_std_zero = np.asarray([False, True], dtype=np.bool_)
    # 11 active tokens total; 3 of them masked to 0.0 by IcePop; padded positions are 1.0 (must be ignored!)
    importance_sampling_ratio = np.asarray(
        [
            [1.0, 1.5, 0.0, 1.0],
            [0.0, 0.8, 2.0, 0.0],
            [1.2, 1.0, 1.0, 1.0],
            [0.9, 1.1, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    ref_logps = np.full((4, 4), -1.0, dtype=np.float32)
    old_logps = np.full((4, 4), -1.2, dtype=np.float32)

    health, truncated_flags, active_is_ratios = trainer._compute_grpo_rl_health_metrics(
        rewards=rewards,
        is_std_zero=is_std_zero,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_attention_mask=completion_attention_mask,
        importance_sampling_ratio=importance_sampling_ratio,
    )
    assert truncated_flags == [False, True, False, False]
    assert health["rl/pass_rate"] == pytest.approx(0.75)
    assert health["rl/truncation_rate"] == pytest.approx(0.25)
    assert health["rl/zero_var_group_frac"] == pytest.approx(0.5)
    assert health["rl/icepop_masked_frac"] == pytest.approx(3.0 / 11.0)
    assert health["rl/icepop_ratio_max"] == pytest.approx(2.0)
    expected_mean = (1.0 + 1.5 + 0.0 + 0.0 + 0.8 + 2.0 + 0.0 + 1.2 + 1.0 + 0.9 + 1.1) / 11.0
    assert health["rl/icepop_ratio_mean"] == pytest.approx(expected_mean)
    assert active_is_ratios is not None and active_is_ratios.size == 11

    # Also verify sequence-level [N, 1] IcePop ratio broadcasting
    seq_ratio = np.asarray([[1.2], [0.0], [1.0], [1.0]], dtype=np.float32)
    seq_health, _, _ = trainer._compute_grpo_rl_health_metrics(
        rewards=rewards,
        is_std_zero=is_std_zero,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_attention_mask=completion_attention_mask,
        importance_sampling_ratio=seq_ratio,
    )
    assert seq_health["rl/icepop_masked_frac"] == pytest.approx(4.0 / 11.0)

    trainer._log_eyvan_grpo_rl_metrics(
        cur_step=10,
        prompts=["Solve A", "Solve A", "Solve B", "Solve B"],
        completions=["ans A1", "ans A2", "ans B1", "ans B2"],
        rewards=rewards,
        rewards_per_func=rewards_per_func,
        advantages=advantages,
        completion_length=completion_mask.sum(axis=-1),
        completion_mask=completion_mask,
        ref_per_token_logps=ref_logps,
        old_per_token_logps=old_logps,
        truncated_flags=truncated_flags,
        generation_factor=2,
        has_environment_feedback=True,
        active_is_ratios=active_is_ratios,
    )
    run_id = args.get_eyvan_track().id
    args.close_eyvan_track()

    index, scan_res = _scan_store(store_dir, tmp_path, name="grpo.db")
    assert scan_res.errors == 0
    detail = index.run_detail("default", "rl-proj", run_id)
    assert detail is not None
    assert detail["stats"]["dropped"] == {"keys": 0, "rate": 0, "invalid": 0, "queue": 0}
    assert detail["completion_groups"] == 2
    assert set(detail["hist_keys"]) == {
        "rl/reward_dist",
        "rl/advantage_dist",
        "rl/completion_length_dist",
        "rl/kl_dist",
        "rl/icepop_ratio_dist",
    }

    comp_list = index.completions("default", "rl-proj", run_id)
    assert comp_list is not None
    by_group = {g["group"]: g for g in comp_list["groups"]}
    g0_detail = index.completion_detail("default", "rl-proj", run_id, by_group["s10-g0"]["id"])
    assert g0_detail is not None
    assert g0_detail["policy_version"] == 7
    assert g0_detail["responses"][0]["rewards"] == {
        "accuracy": pytest.approx(1.0),
        "format": pytest.approx(0.5),
        "environment": pytest.approx(0.25),
    }
    assert g0_detail["responses"][1]["truncated"] is True
    assert g0_detail["responses"][0]["kl"] > 0.0

    # Verify group cap <= 8 per step
    store_cap = tmp_path / "grpo_cap_store"
    args_cap = _make_args(store_cap, eyvan_track_name="grpo-cap")
    trainer.arguments = args_cap
    n_groups = 12
    trainer._log_eyvan_grpo_rl_metrics(
        cur_step=1,
        prompts=[f"p{i}" for i in range(n_groups * 2)],
        completions=[f"c{i}" for i in range(n_groups * 2)],
        rewards=np.ones(n_groups * 2, dtype=np.float32),
        rewards_per_func=np.ones((n_groups * 2, 2), dtype=np.float32),
        advantages=np.zeros(n_groups * 2, dtype=np.float32),
        completion_length=np.full(n_groups * 2, 4.0, dtype=np.float32),
        completion_mask=np.ones((n_groups * 2, 4), dtype=np.float32),
        truncated_flags=[False] * (n_groups * 2),
        generation_factor=2,
    )
    cap_run_id = args_cap.get_eyvan_track().id
    args_cap.close_eyvan_track()
    cap_idx, cap_scan = _scan_store(store_cap, tmp_path, name="grpo_cap.db")
    assert cap_scan.errors == 0
    assert cap_idx.run_detail("default", "rl-proj", cap_run_id)["completion_groups"] == 8


def test_ppo_rl_health_completions_and_histograms(tmp_path):
    store_dir = tmp_path / "ppo_store"
    args = _make_args(store_dir, trainer_prefix="PPO", eyvan_track_name="ppo-rl")
    args.max_completion_length = 4

    trainer = object.__new__(PPOTrainer)
    trainer.arguments = args
    trainer._eos_token_id = np.asarray([2], dtype=np.int64)
    trainer.reward_func_names = ["rm_score"]
    trainer._async_rollout_policy_version = 3

    completion_ids = np.asarray([[5, 6, 2, 0], [5, 6, 7, 8]], dtype=np.int64)
    completion_mask = np.asarray([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.float32)
    scores = np.asarray([1.2, -0.4], dtype=np.float32)
    rewards_per_func = np.asarray([[1.2], [-0.4]], dtype=np.float32)
    advantages = np.asarray([[0.6, 0.3, 0.3, 0.0], [-0.2, -0.4, -0.2, -0.4]], dtype=np.float32)
    kl = np.asarray([[0.03, 0.06, 0.09, 0.0], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    old_values = np.asarray([[0.5, 0.6, 0.7, 0.0], [0.2, 0.1, 0.0, -0.1]], dtype=np.float32)

    truncated_flags = trainer._compute_ppo_truncated_flags(
        completion_ids=completion_ids,
        raw_completion_mask=completion_mask,
    )
    assert truncated_flags == [False, True]

    trainer._log_eyvan_ppo_rl_metrics(
        cur_step=5,
        prompts=["Prompt 1", "Prompt 2"],
        completions=["Good reply", "Cutoff reply"],
        scores=scores,
        rewards_per_func=rewards_per_func,
        advantages=advantages,
        kl=kl,
        old_values=old_values,
        completion_mask=completion_mask,
        truncated_flags=truncated_flags,
        generation_factor=1,
    )
    run_id = args.get_eyvan_track().id
    args.close_eyvan_track()

    index, scan_res = _scan_store(store_dir, tmp_path, name="ppo.db")
    assert scan_res.errors == 0
    detail = index.run_detail("default", "rl-proj", run_id)
    assert detail is not None
    assert detail["stats"]["dropped"] == {"keys": 0, "rate": 0, "invalid": 0, "queue": 0}
    assert detail["completion_groups"] == 2
    assert set(detail["hist_keys"]) == {
        "rl/score_dist",
        "rl/advantage_dist",
        "rl/kl_dist",
        "rl/value_dist",
        "rl/completion_length_dist",
    }

    comp_list = index.completions("default", "rl-proj", run_id)
    by_group = {g["group"]: g for g in comp_list["groups"]}
    g0 = index.completion_detail("default", "rl-proj", run_id, by_group["s5-g0"]["id"])
    assert g0 is not None
    assert g0["policy_version"] == 3
    assert g0["responses"][0]["advantage"] == pytest.approx(0.4)
    assert g0["responses"][0]["kl"] == pytest.approx(0.06)
    assert g0["responses"][0]["tokens"] == 3
    assert g0["responses"][0]["truncated"] is False


def test_async_rollout_mixin_policy_version_thread_local_and_policy_lag(monkeypatch):
    class _FakeState:
        def __init__(self, step: int):
            self.step = step
            self.esurge_cache_scope_key = "test-scope"

    class _FakeSnapshot:
        def __init__(self, policy_step: int):
            self.policy_step = policy_step
            self.state = _FakeState(policy_step)
            self.nbytes = 0

        def release(self):
            pass

    observed_worker_versions: list[int | None] = []
    logged_train_metrics: list[dict[str, object]] = []

    class _MinimalAsyncTrainer(AsyncRolloutMixin):
        supports_async_rollout = True

        def __init__(self):
            self.arguments = types.SimpleNamespace(
                async_mode=True,
                max_inflight_tasks=2,
                queue_maxsize=2,
                weight_sync_steps=3,
                max_staleness=2,
                request_timeout=10.0,
                first_rollout_timeout=10.0,
                heartbeat_stale_after_s=10.0,
                esurge_async_scheduling=True,
                esurge_overlap_execution=True,
                max_length=16,
                learning_rate=1e-4,
                performance_mode=False,
                merge_lora_before_save=False,
                sync_ref_model=False,
                generation_interval=None,
                benchmark_interval=None,
                total_batch_size=1,
            )
            self.data_collator = lambda x: x
            self.max_training_steps = 3
            self.training_batch_size = 1
            self.scheduler = None
            self.pruning_module = None
            self._backward_flops_per_token = 1.0
            self._extra_backward_flops_per_token = 0.0
            self._has_buckets = False
            self.train_tracker = types.SimpleNamespace(trace_compilation=lambda: _nullctx())
            self._async_grpo_train_compiled_once = True
            self._batches = [{"id": i} for i in range(6)]
            self._batch_idx = 0

        def _get_epoch_step_bounds(self, epoch):
            return 0, 3

        def _get_next_batch(self, train_iter, train_dataset):
            idx = self._batch_idx
            self._batch_idx += 1
            return self._batches[idx % len(self._batches)], train_iter

        def _take_rollout_snapshot(self, state, *, policy_step, cache_scope_key):
            return _FakeSnapshot(policy_step)

        def _preprocess_batch_input(self, *, state, batch, is_train):
            observed_worker_versions.append(self._async_rollout_policy_version)
            return {"input_ids": np.zeros((1, 4), dtype=np.int32)}, {}

        def _execute_preprocessed_train_step(self, state, batch, informations):
            next_state = _FakeState(int(state.step) + 1)
            return next_state, LossMetrics(loss=0.5, other_metrics=dict(informations)), None

        def on_step_start(self, *, state, step):
            return state

        def on_step_end(self, *, state, metrics, step):
            return state, metrics

        def apply_training_hooks(self, *, metrics):
            return metrics

        def _maybe_start_profiler(self, step):
            pass

        def _should_save_tpu_preemption_checkpoint(self, step):
            return False

        def _save_checkpoint_for_step(self, **kwargs):
            pass

        def _should_run_evaluation(self, step):
            return False

        def log_metrics(self, *, metrics, pbar, step, mode="train", update_progress=True, **kwargs):
            if update_progress:
                logged_train_metrics.append(dict(metrics))

        def log_weight_distribution(self, **kwargs):
            pass

        def log_watchers(self, **kwargs):
            pass

        def maybe_generate(self, **kwargs):
            pass

        def maybe_benchmark(self, **kwargs):
            pass

    class _nullctx:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("easydel.trainers.async_rollout_mixin.jax.device_get", lambda x: x)
    monkeypatch.setattr("easydel.trainers.async_rollout_mixin.jax.process_count", lambda: 1)

    trainer = _MinimalAsyncTrainer()
    # Direct thread-local test: main thread never sees worker's policy_version
    assert trainer._async_rollout_policy_version is None
    res = trainer._preprocess_async_rollout(state=_FakeState(4), batch={"id": 0}, produced_at_step=4)
    assert res.produced_at_step == 4
    assert observed_worker_versions == [4]
    assert trainer._async_rollout_policy_version is None

    # Full async epoch emits rl/policy_lag == async_rollout/rollout_staleness
    observed_worker_versions.clear()
    state, err, _ = trainer._train_epoch_async(
        state=_FakeState(0),
        train_dataset=None,
        train_iter=None,
        metrics_tracker=MetricsTracker(),
        step_metrics=StepMetrics(trainer.arguments),
        pbar=_DummyPbar(),
        epoch=0,
        epoch_start_step=0,
        epoch_end_step=3,
    )
    assert err is None
    assert state.step == 3
    assert len(logged_train_metrics) == 3
    for step_m in logged_train_metrics:
        assert "train/rl/policy_lag" in step_m
        assert step_m["train/rl/policy_lag"] == step_m["train/async_rollout/rollout_staleness"]
    # With weight_sync_steps=3 and max_staleness=2, steps 0, 1, 2 consume snapshot 0 -> lags [0, 1, 2]
    assert [m["train/rl/policy_lag"] for m in logged_train_metrics] == [0, 1, 2]


def test_online_actor_critic_policy_lag_and_eyvan_episodes(monkeypatch, tmp_path):
    store_dir = tmp_path / "online_rl_store"
    args = _make_args(store_dir, trainer_prefix="OnlineRL", eyvan_track_name="online-ac")
    args.weight_sync_steps = 3
    args.max_policy_staleness = 4
    args.max_inflight_rollouts = 2
    args.rollout_queue_size = 2
    args.asynchronous_rollouts = False
    args.rollout_timeout = 10.0

    trainer = object.__new__(OnlineActorCriticTrainer)
    trainer.arguments = args
    trainer._online_rollout_idle = True
    trainer._online_rollout_snapshot = None

    # 1. Verify _log_eyvan_provider_episodes writes episodes with policy_version / learner_version
    trainer._log_eyvan_provider_episodes(
        host_versions=np.asarray([1, 3], dtype=np.int64),
        host_rewards=np.asarray([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        host_action_mask=np.asarray([[True, True, False], [True, True, True]], dtype=np.bool_),
        host_scores=np.asarray([1.0, -0.5], dtype=np.float32),
        host_traj_ids=np.asarray([10, 11], dtype=np.int64),
        learner_version=5,
        is_train=True,
    )
    run_id = args.get_eyvan_track().id
    args.close_eyvan_track()

    index, scan_res = _scan_store(store_dir, tmp_path, name="online_rl.db")
    assert scan_res.errors == 0
    breakdown = index.episode_breakdown("default", "rl-proj", run_id, kind="train")
    assert breakdown is not None
    # Lags: 5 - 1 = 4, 5 - 3 = 2 -> max=4, p50=3.0
    assert breakdown["lag"]["count"] == 2
    assert breakdown["lag"]["max"] == 4
    assert breakdown["lag"]["p50"] == pytest.approx(3.0)
    ep_page = index.episodes("default", "rl-proj", run_id, kind="train")
    assert ep_page is not None
    rets = sorted(ep["ret"] for ep in ep_page["episodes"])
    assert rets == [pytest.approx(-0.5), pytest.approx(1.5)]

    # 2. Verify collect_rollouts emits rl/policy_lag (both provider-lag and snapshot-lag paths)
    class _FakeOwnedSnapshot:
        def __init__(self, policy_step: int, state):
            self.policy_step = policy_step
            self.state = state

        @classmethod
        def from_training_state(cls, state, *, policy_step, cache_scope_key):
            return cls(policy_step=policy_step, state=state)

        def release(self):
            pass

    monkeypatch.setattr(online_mod, "OwnedPolicySnapshot", _FakeOwnedSnapshot)
    monkeypatch.setattr(online_mod.jax, "device_get", lambda x: x)
    monkeypatch.setattr(OnlineActorCriticTrainer, "_process_count", staticmethod(lambda: 1))

    def _provider(*, trainer, state, batch, is_train):
        return batch

    trainer.rollout_provider = _provider
    trainer._normalize_provider_batch = types.MethodType(
        lambda self, batch, *, scoring_state, learner_step=None, is_train=True: (
            setattr(self, "_last_provider_policy_lag", batch.get("lag")) or batch
        ),
        trainer,
    )
    live_state = types.SimpleNamespace(step=0, esurge_cache_scope_key="scope")
    batches = [{"id": 0, "lag": None}, {"id": 1, "lag": None}, {"id": 2, "lag": 2.5}]
    lags = []
    for _, info in trainer.collect_rollouts(live_state, batches, is_train=True, learner_state=lambda: live_state):
        lags.append(info["rl/policy_lag"])
        live_state.step += 1
    # Step 0: snap=0 -> lag 0; step 1: snap=0 -> lag 1; step 2: explicit provider lag 2.5
    assert lags == [0, 1, pytest.approx(2.5)]


def test_eyvan_rl_safety_bfloat16_nan_non_rank0_and_errors(monkeypatch, tmp_path):
    store_dir = tmp_path / "safety_store"

    # 1. Non-rank-0 with log_all_workers=False: all 4 RL helpers return False and create no thread
    monkeypatch.setattr(tc.jax, "process_index", lambda: 1)
    args_r1 = _make_args(store_dir, log_all_workers=False)
    assert args_r1.log_eyvan_completions("p", ["r"], rewards=[1.0], step=1) is False
    assert args_r1.log_eyvan_episode(1.0, 5, step=1) is False
    assert args_r1.log_eyvan_rollout(tracks={"reward": [1.0]}, step=1) is False
    assert args_r1.log_eyvan_histogram("rl/h", [1.0, 2.0], step=1) is False
    assert args_r1.get_eyvan_track() is None

    # 2. Rank-0 with bfloat16 & NaN/Inf inputs: cleanly handled, never raises
    monkeypatch.setattr(tc.jax, "process_index", lambda: 0)
    args_r0 = _make_args(store_dir, eyvan_track_name="safety-bf16")
    assert (
        args_r0.log_eyvan_completions(
            "bf16 prompt",
            ["r0", "r1"],
            rewards=np.asarray([ml_dtypes.bfloat16(1.5), ml_dtypes.bfloat16(float("nan"))], dtype=tc.jnp.bfloat16),
            advantages=np.asarray([ml_dtypes.bfloat16(0.5), ml_dtypes.bfloat16(-0.5)], dtype=tc.jnp.bfloat16),
            kl=[0.1, float("nan")],
            step=1,
        )
        is True
    )
    assert args_r0.log_eyvan_episode(ml_dtypes.bfloat16(float("nan")), length=3, step=1) is True
    assert (
        args_r0.log_eyvan_histogram(
            "rl/nan_bf16",
            np.asarray(
                [ml_dtypes.bfloat16(1.0), ml_dtypes.bfloat16(float("nan")), ml_dtypes.bfloat16(2.0)],
                dtype=tc.jnp.bfloat16,
            ),
            step=1,
        )
        is True
    )
    # Invalid/empty/complex inputs safely return False without warning
    assert args_r0.log_eyvan_completions("p", [], step=1) is False
    assert args_r0.log_eyvan_episode(np.complex64(1.0 + 1.0j), step=1) is False
    assert args_r0.log_eyvan_rollout(steps=[], step=1) is False
    assert args_r0.log_eyvan_histogram("rl/c", np.asarray([1.0 + 1.0j], dtype=np.complex64), step=1) is False

    run_id = args_r0.get_eyvan_track().id
    args_r0.close_eyvan_track()
    assert not any(t.name == f"eyvan-track-{run_id}" and t.is_alive() for t in threading.enumerate())

    index, scan_res = _scan_store(store_dir, tmp_path, name="safety.db")
    assert scan_res.errors == 0
    assert index.run_detail("default", "rl-proj", run_id)["stats"]["dropped"] == {
        "keys": 0,
        "rate": 0,
        "invalid": 0,
        "queue": 0,
    }

    # 3. Broken Run methods warn and return False without crashing training
    class _BrokenRun:
        finished = False
        active = True
        episodes_active = True

        def completions(self, *a, **kw):
            raise RuntimeError("boom completions")

        def episode(self, *a, **kw):
            raise RuntimeError("boom episode")

        def rollout(self, *a, **kw):
            raise RuntimeError("boom rollout")

        def histogram(self, *a, **kw):
            raise RuntimeError("boom histogram")

    args_broken = _make_args(use_eyvan_track=True)
    args_broken.__dict__["_eyvan_track_run"] = _BrokenRun()
    with pytest.warns(UserWarning, match="completions"):
        assert args_broken.log_eyvan_completions("p", ["r"], step=1) is False
    with pytest.warns(UserWarning, match="episode"):
        assert args_broken.log_eyvan_episode(1.0, step=1) is False
    with pytest.warns(UserWarning, match="rollout"):
        assert args_broken.log_eyvan_rollout(tracks={"reward": [1.0]}, step=1) is False
    with pytest.warns(UserWarning, match="histogram"):
        assert args_broken.log_eyvan_histogram("rl/h", [1.0, 2.0], step=1) is False
    args_broken.__dict__.pop("_eyvan_track_run", None)
    assert math.isfinite(0.0)


def test_grpo_and_ppo_weighted_nan_rewards_clipping_and_1d_icepop(tmp_path):
    # 1. GRPO with non-unit reward_weights + NaN component + clipped/modified group + 1D [N] IS ratio
    store_grpo = tmp_path / "grpo_weighted"
    args_grpo = _make_args(store_grpo, eyvan_track_name="grpo-weighted")
    args_grpo.max_completion_length = 4
    grpo = object.__new__(GRPOTrainer)
    grpo.arguments = args_grpo
    grpo._eos_token_id = np.asarray([2], dtype=np.int64)
    grpo.kl_estimator = "k3"
    grpo.reward_func_names = ["acc", "fmt"]

    comp_ids = np.asarray([[10, 2, 0, 0], [11, 2, 0, 0], [12, 2, 0, 0], [13, 2, 0, 0]], dtype=np.int64)
    comp_mask = np.asarray([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    # Group 0: weighted nansum with weights=[2.0, 0.5]:
    #   row 0: 1.0*2.0 + nan*0.5 -> 2.0 (pass!)
    #   row 1: 0.0*2.0 + 1.0*0.5 -> 0.5 (pass!)
    # Group 1: clipped totals [1.0, -0.5] diverge from weighted sum [2.0, 0.0] -> falls back to total rewards
    rewards_pf = np.asarray(
        [
            [1.0, np.nan],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    reward_weights = np.asarray([2.0, 0.5], dtype=np.float32)
    rewards = np.asarray([2.0, 0.5, 1.0, -0.5], dtype=np.float32)
    advantages = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    is_std_zero = np.asarray([False, False], dtype=np.bool_)

    # Verify 1D [N] importance_sampling_ratio broadcasts cleanly over [N, L]
    ratio_1d = np.asarray([1.5, 0.0, 1.0, 0.0], dtype=np.float32)
    health_1d, trunc_flags, active_1d = grpo._compute_grpo_rl_health_metrics(
        rewards=rewards,
        is_std_zero=is_std_zero,
        completion_ids=comp_ids,
        completion_mask=comp_mask,
        importance_sampling_ratio=ratio_1d,
    )
    assert health_1d["rl/icepop_masked_frac"] == pytest.approx(0.5)
    assert health_1d["rl/icepop_ratio_max"] == pytest.approx(1.5)
    assert active_1d is not None and active_1d.size == 8

    grpo._log_eyvan_grpo_rl_metrics(
        cur_step=3,
        prompts=["p0", "p0", "p1", "p1"],
        completions=["c0", "c1", "c2", "c3"],
        rewards=rewards,
        rewards_per_func=rewards_pf,
        reward_weights=reward_weights,
        advantages=advantages,
        completion_length=comp_mask.sum(axis=-1),
        completion_mask=comp_mask,
        truncated_flags=trunc_flags,
        generation_factor=2,
    )
    grpo_run_id = args_grpo.get_eyvan_track().id
    args_grpo.close_eyvan_track()

    grpo_idx, grpo_scan = _scan_store(store_grpo, tmp_path, name="grpo_w.db")
    assert grpo_scan.errors == 0
    g_list = grpo_idx.completions("default", "rl-proj", grpo_run_id)
    by_grp = {g["group"]: g for g in g_list["groups"]}
    # Group 0: weighted & NaN-safe -> totals [2.0, 0.5], mean=1.25, pass_rate=1.0, decomposed rewards preserved
    assert by_grp["s3-g0"]["mean_reward"] == pytest.approx(1.25)
    assert by_grp["s3-g0"]["pass_rate"] == pytest.approx(1.0)
    g0_det = grpo_idx.completion_detail("default", "rl-proj", grpo_run_id, by_grp["s3-g0"]["id"])
    assert g0_det["responses"][0]["rewards"] == {"acc": pytest.approx(2.0), "fmt": pytest.approx(0.0)}
    assert g0_det["responses"][0]["reward"] == pytest.approx(2.0)
    assert g0_det["responses"][1]["rewards"] == {"acc": pytest.approx(0.0), "fmt": pytest.approx(0.5)}
    assert g0_det["responses"][1]["reward"] == pytest.approx(0.5)
    # Group 1: totals diverged from weighted sum (e.g. reward clipping) -> fell back to exact totals [1.0, -0.5]
    assert by_grp["s3-g1"]["mean_reward"] == pytest.approx(0.25)
    assert by_grp["s3-g1"]["pass_rate"] == pytest.approx(0.5)

    # 2. PPO with non-unit reward_weights + NaN component (group 0) and missing_eos_penalty divergence (group 1)
    store_ppo = tmp_path / "ppo_weighted"
    args_ppo = _make_args(store_ppo, trainer_prefix="PPO", eyvan_track_name="ppo-weighted")
    ppo = object.__new__(PPOTrainer)
    ppo.arguments = args_ppo
    ppo.reward_func_names = ["rm_a", "rm_b"]
    ppo.reward_weights = np.asarray([3.0, 0.5], dtype=np.float32)

    ppo_pf = np.asarray([[1.0, np.nan], [1.0, 1.0]], dtype=np.float32)
    # Row 0 matches weighted nansum (3.0); row 1 had missing_eos_penalty (-1.0) -> 2.5 != 3.5 -> fallback to score
    ppo_scores = np.asarray([3.0, 2.5], dtype=np.float32)
    ppo._log_eyvan_ppo_rl_metrics(
        cur_step=4,
        prompts=["pp0", "pp1"],
        completions=["pc0", "pc1"],
        scores=ppo_scores,
        rewards_per_func=ppo_pf,
        advantages=np.ones((2, 2), dtype=np.float32),
        kl=np.zeros((2, 2), dtype=np.float32),
        old_values=np.zeros((2, 2), dtype=np.float32),
        completion_mask=np.ones((2, 2), dtype=np.float32),
        truncated_flags=[False, True],
        generation_factor=1,
    )
    ppo_run_id = args_ppo.get_eyvan_track().id
    args_ppo.close_eyvan_track()

    ppo_idx, ppo_scan = _scan_store(store_ppo, tmp_path, name="ppo_w.db")
    assert ppo_scan.errors == 0
    p_by_grp = {g["group"]: g for g in ppo_idx.completions("default", "rl-proj", ppo_run_id)["groups"]}
    assert p_by_grp["s4-g0"]["mean_reward"] == pytest.approx(3.0)
    assert p_by_grp["s4-g1"]["mean_reward"] == pytest.approx(2.5)
    p0_det = ppo_idx.completion_detail("default", "rl-proj", ppo_run_id, p_by_grp["s4-g0"]["id"])
    assert p0_det["responses"][0]["rewards"] == {"rm_a": pytest.approx(3.0), "rm_b": pytest.approx(0.0)}


def test_base_trainer_axes_step_metrics_fallback_visited_and_total_tokens(tmp_path):
    store_dir = tmp_path / "step_metrics_axes"
    args = _make_args(store_dir, eyvan_track_name="step-metrics-axes", report_steps=1, log_steps=1)
    trainer = object.__new__(Trainer)
    trainer.arguments = args
    trainer.eyvan_run = None
    pbar = _DummyPbar()

    # Standard StepMetrics.calculate payload emits train/visited_tokens + performance/total_tokens
    trainer.log_metrics(
        {"train/loss": 1.2, "train/visited_tokens": 1024, "performance/total_tokens": 1024},
        pbar,
        step=1,
    )
    assert trainer._eyvan_tokens_seen == pytest.approx(1024.0)
    trainer.log_metrics(
        {"train/loss": 0.9, "train/visited_tokens": 2048, "performance/total_tokens": 1024},
        pbar,
        step=2,
    )
    assert trainer._eyvan_tokens_seen == pytest.approx(2048.0)

    # Pure performance/total_tokens fallback when visited_tokens is absent
    trainer2 = object.__new__(Trainer)
    trainer2.arguments = args
    trainer2.eyvan_run = None
    axes_p = trainer2._build_eyvan_axes({"performance/total_tokens": 256}, mode="train")
    assert axes_p["tokens"] == pytest.approx(256.0)
    axes_p2 = trainer2._build_eyvan_axes({"performance/total_tokens": 256}, mode="train")
    assert axes_p2["tokens"] == pytest.approx(512.0)

    run_id = args.get_eyvan_track().id
    args.close_eyvan_track()
    idx, scan_res = _scan_store(store_dir, tmp_path, name="step_axes.db")
    assert scan_res.errors == 0
    ser = idx.series("default", "rl-proj", run_id, ["train/loss"], x="tokens")
    assert [(pt[0], pt[1]) for pt in ser["series"][0]["points"]] == [
        (1024.0, pytest.approx(1.2)),
        (2048.0, pytest.approx(0.9)),
    ]


def test_multihost_local_jax_guard_and_utf8_surrogate_sanitization(monkeypatch, tmp_path):
    # Fake JAX array stubs to verify local-read guard without invoking XLA
    class _FakeJaxArray:
        def __init__(self, data, *, addressable: bool, replicated: bool, has_local: bool = True):
            self._data = np.asarray(data)
            self.is_fully_addressable = addressable
            self.is_fully_replicated = replicated
            self.sharding = types.SimpleNamespace(has_addressable_devices=has_local)

    device_get_calls: list[object] = []

    def _fake_device_get(x):
        if isinstance(x, _FakeJaxArray):
            device_get_calls.append(x)
            return x._data
        return x

    monkeypatch.setattr(tc.jax, "Array", _FakeJaxArray)
    monkeypatch.setattr(tc.jax, "device_get", _fake_device_get)

    replicated_remote = _FakeJaxArray(3.5, addressable=False, replicated=True, has_local=True)
    sharded_remote = _FakeJaxArray(7.5, addressable=False, replicated=False, has_local=True)
    replicated_no_local = _FakeJaxArray(9.5, addressable=False, replicated=True, has_local=False)

    # Replicated global array with addressable devices succeeds locally
    assert tc._to_eyvan_scalar(replicated_remote) == (True, pytest.approx(3.5))
    assert tc._coerce_eyvan_int(_FakeJaxArray(12, addressable=False, replicated=True)) == 12
    assert tc._coerce_eyvan_sequence(_FakeJaxArray([1.0, 2.0], addressable=False, replicated=True)) == [1.0, 2.0]

    # Sharded non-replicated global array and no-local-device array are refused WITHOUT calling device_get
    calls_before = len(device_get_calls)
    assert tc._to_eyvan_scalar(sharded_remote) == (False, 0.0)
    assert tc._to_eyvan_scalar(replicated_no_local) == (False, 0.0)
    assert tc._coerce_eyvan_int(sharded_remote) is None
    assert tc._coerce_eyvan_sequence(sharded_remote) is None
    assert tc._sanitize_eyvan_axes({"good": replicated_remote, "bad_sharded": sharded_remote}) == {
        "good": pytest.approx(3.5)
    }
    assert len(device_get_calls) == calls_before + 1  # only replicated_remote in _sanitize_eyvan_axes

    # Lone UTF-8 surrogates in completions & rollouts must be sanitized so TrackingIndex scans cleanly
    store_dir = tmp_path / "utf8_store"
    args = _make_args(store_dir, eyvan_track_name="utf8-safe")
    assert args.log_eyvan_histogram("rl/sharded_skip", sharded_remote, step=1) is False
    assert (
        args.log_eyvan_histogram("rl/repl_ok", _FakeJaxArray([1.0, 2.0], addressable=False, replicated=True), step=1)
        is True
    )
    assert (
        args.log_eyvan_completions(
            "bad prompt \ud800 end",
            ["bad resp \udfff ok"],
            rewards=[1.0],
            answer="ans \ud83d",
            group="grp\ud800-1",
            step=1,
        )
        is True
    )
    assert (
        args.log_eyvan_rollout(
            steps=[{"reward": 1.0}],
            actions=["act \ud800 name"],
            env="env\udfff",
            step=1,
        )
        is True
    )
    run_id = args.get_eyvan_track().id
    args.close_eyvan_track()

    idx, scan_res = _scan_store(store_dir, tmp_path, name="utf8.db")
    assert scan_res.errors == 0
    detail = idx.run_detail("default", "rl-proj", run_id)
    assert detail is not None
    assert detail["errors"] == []
    assert detail["completion_groups"] == 1
    assert detail["rollouts"] == 1
    assert detail["hist_keys"] == ["rl/repl_ok"]


def test_rl_trainers_lazy_init_before_log_metrics_and_clean_finish(tmp_path):
    # Verify GRPOTrainer, PPOTrainer, and OnlineActorCriticTrainer ALL start with eyvan_run=None
    # (zero files/threads) and trigger lazy Eyvan init cleanly when their VERY FIRST telemetry call
    # is log_eyvan_completions / log_eyvan_histogram / log_eyvan_episode BEFORE log_metrics().
    baseline = set(threading.enumerate())

    # 1. GRPOTrainer lazy init via _log_eyvan_grpo_rl_metrics
    grpo_store = tmp_path / "lazy_grpo"
    grpo_args = _make_args(grpo_store, trainer_prefix="GRPO", eyvan_track_name="lazy-grpo")
    grpo = object.__new__(GRPOTrainer)
    grpo.arguments = grpo_args
    grpo._eos_token_id = np.asarray([2], dtype=np.int64)
    grpo.kl_estimator = "k3"
    grpo.reward_func_names = ["acc"]
    grpo._bind_eyvan_context()
    assert grpo.eyvan_run is None
    assert not grpo_store.exists()

    grpo._log_eyvan_grpo_rl_metrics(
        cur_step=1,
        prompts=["p0", "p0"],
        completions=["c0", "c1"],
        rewards=np.asarray([1.0, 0.0], dtype=np.float32),
        rewards_per_func=np.asarray([[1.0], [0.0]], dtype=np.float32),
        advantages=np.asarray([1.0, -1.0], dtype=np.float32),
        completion_length=np.asarray([2.0, 2.0], dtype=np.float32),
        completion_mask=np.ones((2, 2), dtype=np.float32),
        truncated_flags=[False, False],
        generation_factor=2,
    )
    assert grpo.eyvan_run is not None
    grpo_run_id = grpo.eyvan_run.id
    grpo.finish()
    assert grpo.eyvan_run is None

    # 2. PPOTrainer lazy init via _log_eyvan_ppo_rl_metrics
    ppo_store = tmp_path / "lazy_ppo"
    ppo_args = _make_args(ppo_store, trainer_prefix="PPO", eyvan_track_name="lazy-ppo")
    ppo = object.__new__(PPOTrainer)
    ppo.arguments = ppo_args
    ppo.reward_func_names = ["rm"]
    ppo._bind_eyvan_context()
    assert ppo.eyvan_run is None
    assert not ppo_store.exists()

    ppo._log_eyvan_ppo_rl_metrics(
        cur_step=2,
        prompts=["p0"],
        completions=["c0"],
        scores=np.asarray([0.8], dtype=np.float32),
        rewards_per_func=np.asarray([[0.8]], dtype=np.float32),
        advantages=np.ones((1, 2), dtype=np.float32),
        kl=np.zeros((1, 2), dtype=np.float32),
        old_values=np.zeros((1, 2), dtype=np.float32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        truncated_flags=[False],
        generation_factor=1,
    )
    assert ppo.eyvan_run is not None
    ppo_run_id = ppo.eyvan_run.id
    ppo.finish()
    assert ppo.eyvan_run is None

    # 3. OnlineActorCriticTrainer lazy init via _log_eyvan_provider_episodes
    online_store = tmp_path / "lazy_online"
    online_args = _make_args(online_store, trainer_prefix="OnlineRL", eyvan_track_name="lazy-online")
    online = object.__new__(OnlineActorCriticTrainer)
    online.arguments = online_args
    online._bind_eyvan_context()
    assert online.eyvan_run is None
    assert not online_store.exists()

    online._log_eyvan_provider_episodes(
        host_versions=np.asarray([1], dtype=np.int64),
        host_rewards=np.asarray([[0.5, 0.5]], dtype=np.float32),
        host_action_mask=np.asarray([[True, True]], dtype=np.bool_),
        learner_version=2,
        is_train=True,
    )
    assert online.eyvan_run is not None
    online_run_id = online.eyvan_run.id
    online.finish()
    assert online.eyvan_run is None

    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []

    # Verify all 3 lazily-initialized runs indexed cleanly with real RL records.
    idx_g, scan_g = _scan_store(grpo_store, tmp_path, name="lazy_g.db")
    assert scan_g.errors == 0
    assert idx_g.run_detail("default", "rl-proj", grpo_run_id)["completion_groups"] == 1

    idx_p, scan_p = _scan_store(ppo_store, tmp_path, name="lazy_p.db")
    assert scan_p.errors == 0
    assert idx_p.run_detail("default", "rl-proj", ppo_run_id)["completion_groups"] == 1

    idx_o, scan_o = _scan_store(online_store, tmp_path, name="lazy_o.db")
    assert scan_o.errors == 0
    assert idx_o.run_detail("default", "rl-proj", online_run_id)["episodes"] == {"train": 1, "eval": 0}

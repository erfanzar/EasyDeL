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
"""CPU-only end-to-end tests for Eyvan RL telemetry in AgentFlowTrainer and AgenticMoshPitTrainer."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.agent_flow_trainer import (
    AgentFlowConfig,
    AgentFlowEpisode,
    AgentFlowTrainer,
    AgentFlowTurn,
)
from easydel.trainers.agent_flow_trainer.agent_flow_trainer import (
    _extract_nonneg_step,
    _resolve_eyvan_env_name,
    _to_host_numpy,
    _to_host_numpy_local_only,
)
from easydel.trainers.agentic_moshpit import AgenticMoshPitConfig, AgenticMoshPitTrainer
from easydel.trainers.agentic_moshpit.env_manager import RolloutManager, TrajectoryResult, TurnRecord
from easydel.trainers.group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from easydel.trainers.training_configurations import TrainingArguments
from eyvan.track._store import store_for_uri
from eyvan.track.tracker.index import TrackingIndex

# Pure host-side NumPy + Eyvan local store tests; no accelerator compilation is validated here.
pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def _clean_eyvan_env(monkeypatch):
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


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = ""

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        ids = [ord(c) for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text: str, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if 32 <= int(i) < 127)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True, **kwargs):
        rendered = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        return rendered + ("assistant:" if add_generation_prompt else "")


class _SyntheticCodeEnv:
    def reset(self, seed=None):
        return SimpleNamespace(observation="solve", info={})

    def step(self, action):
        return SimpleNamespace(observation="", reward=1.0, terminated=True, truncated=False, info={})


def _make_stub_grpo(args: GRPOConfig) -> GRPOTrainer:
    inner = object.__new__(GRPOTrainer)
    inner.arguments = args
    inner.processing_class = _CharTokenizer()
    inner._preprocess_batch_input = lambda state, batch, is_train: ({}, {})
    inner._all_gather = lambda arr: arr
    return inner


def _make_episode(
    *,
    prompt_index: int,
    group_id: int,
    prompt: str,
    turns_spec: list[tuple[str, str, float, bool, bool, list[float] | None]],
    episode_reward: float,
    env_error: bool = False,
    prompt_clipped: bool = False,
) -> AgentFlowEpisode:
    prompt_ids = [ord(c) for c in prompt]
    comp_ids: list[int] = []
    action_mask: list[int] = []
    all_lps: list[float] = []
    has_lps = True
    turns: list[AgentFlowTurn] = []
    step_rewards: list[float] = []
    for t_idx, (act_txt, obs_txt, r, term, trunc, lps) in enumerate(turns_spec):
        act_ids = [ord(c) for c in act_txt]
        obs_ids = [ord(c) for c in obs_txt]
        comp_ids.extend(act_ids)
        action_mask.extend([0 if env_error else 1] * len(act_ids))
        if lps is not None and len(lps) == len(act_ids):
            all_lps.extend(lps)
        else:
            has_lps = False
            all_lps.extend([0.0] * len(act_ids))
        if obs_ids:
            comp_ids.extend(obs_ids)
            action_mask.extend([0] * len(obs_ids))
            all_lps.extend([0.0] * len(obs_ids))
        step_rewards.append(r)
        turns.append(
            AgentFlowTurn(
                turn_index=t_idx,
                prompt_text=prompt,
                action_text=act_txt,
                action_token_ids=act_ids,
                observation_text=obs_txt,
                observation_token_ids=obs_ids,
                reward=r,
                terminated=term,
                truncated=trunc,
                logprobs=lps,
            )
        )
    return AgentFlowEpisode(
        prompt_index=prompt_index,
        group_id=group_id,
        prompt_token_ids=prompt_ids,
        completion_token_ids=comp_ids,
        action_mask=action_mask,
        step_rewards=step_rewards,
        episode_reward=episode_reward,
        turns=tuple(turns),
        logprobs=all_lps if has_lps else None,
        env_error=env_error,
        prompt_clipped=prompt_clipped,
    )


def test_agent_flow_adapt_episodes_eyvan_end_to_end_and_index_scan(tmp_path: Path):
    store_root = tmp_path / "tracking_root"
    scope_uri = store_root / "default" / "agentflow-e2e"
    grpo_args = GRPOConfig(
        model_name="acme/agent-model",
        use_wandb=False,
        save_directory=None,
        num_generations=2,
        max_prompt_length=16,
        max_completion_length=12,
        beta=0.04,
        kl_estimator="k3",
        esurge_importance_sampling_correction=True,
        esurge_importance_sampling_mode="token_icepop",
        esurge_importance_sampling_cap=2.0,
        esurge_importance_sampling_cap_low=0.5,
        use_eyvan_track=True,
        eyvan_track_uri=str(scope_uri),
        eyvan_track_project="agentflow-e2e",
        eyvan_track_name="af-icepop-run",
        eyvan_track_flush_interval_s=0.1,
    )
    inner = _make_stub_grpo(grpo_args)
    # Simulate async rollout policy lag: rollout produced at policy version 4, learner at step 7.
    inner._async_rollout_policy_version = 4

    # Construct policy logps vs sampling logps so exact IcePop masking occurs on active tokens:
    # Sampling logps are -1.0 on all active tokens.
    # Ep 0 active tokens (4 tokens: "ab" + "cd", obs "xy" masked out at positions 2,3):
    #   policy logps = [-1.0, -1.0 + ln(1.5), (obs), (obs), -1.0 + ln(3.0) (masked!), -1.0 + ln(0.2) (masked!)]
    # Ep 1 active tokens (2 tokens: "ef"): policy logps = [-1.0, -1.0] (ratios 1.0, 1.0)
    # Ep 2 active tokens (2 tokens: "gh"): policy logps = [-1.0 + ln(1.2), -1.0 + ln(4.0) (masked!)]
    # Ep 3: env_error=True (0 turns, 0 active tokens -> masked out of comp_mask)
    # Total active tokens = 4 + 2 + 2 = 8; masked (ratio == 0.0) = 3 -> masked_frac = 3/8 = 0.375!
    # Active ratios = [1.0, 1.5, 0.0, 0.0, 1.0, 1.0, 1.2, 0.0] -> mean = 5.7/8 = 0.7125, max = 1.5!
    policy_lps = np.full((4, 12), -1.0, dtype=np.float32)
    policy_lps[0, 1] = -1.0 + math.log(1.5)
    policy_lps[0, 4] = -1.0 + math.log(3.0)  # > cap 2.0 -> IcePop masks to 0.0
    policy_lps[0, 5] = -1.0 + math.log(0.2)  # < cap_low 0.5 -> IcePop masks to 0.0
    policy_lps[2, 0] = -1.0 + math.log(1.2)
    policy_lps[2, 1] = -1.0 + math.log(4.0)  # > cap 2.0 -> IcePop masks to 0.0

    inner.compute_state_logps = lambda state, full_ids, full_mask, kw, **opts: policy_lps
    inner.ref_state = SimpleNamespace(graphstate={}, graphother={})
    ref_lps = np.full((4, 12), -1.5, dtype=np.float32)
    inner.compute_refmodel_logps = lambda gs, go, full_ids, full_mask, kw: ref_lps

    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=3, group_size=2, max_prompt_length=16, max_completion_length=12),
        trainer=inner,
        environment=_SyntheticCodeEnv,
    )

    episodes = [
        _make_episode(
            prompt_index=0,
            group_id=0,
            prompt="calc 2+2",
            turns_spec=[
                ("ab", "xy", 0.5, False, False, [-1.0, -1.0]),
                ("cd", "", 1.0, True, False, [-1.0, -1.0]),
            ],
            episode_reward=1.5,
        ),
        _make_episode(
            prompt_index=0,
            group_id=1,
            prompt="calc 2+2",
            turns_spec=[
                ("ef", "zz", 0.0, False, True, [-1.0, -1.0]),
            ],
            episode_reward=0.0,
        ),
        _make_episode(
            prompt_index=1,
            group_id=2,
            prompt="reverse hi",
            turns_spec=[
                ("gh", "", 1.0, True, False, [-1.0, -1.0]),
            ],
            episode_reward=1.0,
        ),
        _make_episode(
            prompt_index=1,
            group_id=3,
            prompt="reverse hi",
            turns_spec=[],
            episode_reward=0.0,
            env_error=True,
            prompt_clipped=True,
        ),
    ]

    state = SimpleNamespace(step=7)
    batch, metrics = flow._adapt_episodes_to_trainer(episodes, group_size=2, state=state, is_train=True)

    # Verify RL-native health metrics + IcePop diagnostics
    assert metrics["agent_flow/pass_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["rl/pass_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["agent_flow/zero_var_group_frac"] == pytest.approx(0.5)
    assert metrics["rl/zero_var_group_frac"] == pytest.approx(0.5)
    assert metrics["agent_flow/truncation_rate"] == pytest.approx(0.25)
    assert metrics["rl/truncation_rate"] == pytest.approx(0.25)
    assert metrics["agent_flow/mean_action_tokens"] == pytest.approx(2.0)  # (4 + 2 + 2 + 0) / 4
    assert metrics["agent_flow/mean_obs_tokens"] == pytest.approx(1.0)  # (2 + 2 + 0 + 0) / 4
    assert metrics["agent_flow/icepop_masked_frac"] == pytest.approx(3.0 / 8.0)
    assert metrics["rl/icepop_masked_frac"] == pytest.approx(3.0 / 8.0)
    assert metrics["agent_flow/icepop_ratio_mean"] == pytest.approx(5.7 / 8.0, abs=1e-4)
    assert metrics["rl/icepop_ratio_mean"] == pytest.approx(5.7 / 8.0, abs=1e-4)
    assert metrics["agent_flow/icepop_ratio_max"] == pytest.approx(1.5, abs=1e-4)
    assert metrics["rl/icepop_ratio_max"] == pytest.approx(1.5, abs=1e-4)
    assert "importance_sampling_ratio" in batch
    assert flow._eyvan_total_turns == 4  # 2 + 1 + 1 + 0
    assert flow._eyvan_total_action_tokens == 8

    # Log scalar metrics through TrainingArguments too, then close and scan with TrackingIndex
    run = grpo_args.get_eyvan_track()
    assert run is not None
    run_id = run.id
    grpo_args.log_metrics(metrics, step=7)
    grpo_args.close_eyvan_track(status="finished")

    index = TrackingIndex(tmp_path / "index.db")
    scan = index.scan(store_for_uri(str(store_root)))
    assert scan.errors == 0
    assert scan.ingested_segments > 0

    detail = index.run_detail("default", "agentflow-e2e", run_id)
    assert detail is not None
    assert detail["errors"] == []
    assert detail["stats"]["dropped"]["invalid"] == 0
    assert detail["envs"] == ["_SyntheticCodeEnv"]
    assert detail["episodes"] == {"train": 4, "eval": 0}
    assert detail["rollouts"] == 3  # 0-turn env_error episode cleanly skipped
    assert detail["completion_groups"] == 2
    assert set(detail["hist_keys"]) >= {
        "rl/reward_dist",
        "rl/advantage_dist",
        "rl/episode_turns_dist",
        "rl/action_tokens_dist",
        "rl/icepop_ratio_dist",
    }

    # Verify indexed episodes
    eps_page = index.episodes("default", "agentflow-e2e", run_id)
    assert eps_page is not None and eps_page["total"] == 4
    eps_by_ret_err = {(r["ret"], r["extra"]["env_error"]): r for r in eps_page["episodes"]}
    best_ep = eps_by_ret_err[(1.5, 0.0)]
    assert best_ep["env"] == "_SyntheticCodeEnv"
    assert best_ep["success"] is True
    assert best_ep["length"] == 2
    assert best_ep["policy_version"] == 4
    assert best_ep["learner_version"] == 7
    assert best_ep["extra"]["action_tokens"] == 4.0
    assert best_ep["extra"]["obs_tokens"] == 2.0
    assert best_ep["extra"]["truncated"] == 0.0
    assert best_ep["extra"]["mean_turn_reward"] == pytest.approx(0.75)
    assert "env_turns" in best_ep["x"] and "action_tokens" in best_ep["x"]

    err_ep = eps_by_ret_err[(0.0, 1.0)]
    assert err_ep["success"] is False
    assert err_ep["length"] == 0
    assert err_ep["extra"]["prompt_clipped"] == 1.0

    # Verify policy lag breakdown
    breakdown = index.episode_breakdown("default", "agentflow-e2e", run_id, kind="train")
    assert breakdown is not None
    assert breakdown["lag"]["count"] == 4
    assert breakdown["lag"]["p50"] == pytest.approx(3.0)
    assert breakdown["lag"]["max"] == 3

    # Verify indexed rollouts (steps/tracks + integer actions + action_names)
    rollouts_page = index.rollouts("default", "agentflow-e2e", run_id)
    assert rollouts_page is not None and rollouts_page["total"] == 3
    by_ep_id = {r["episode_id"]: r for r in rollouts_page["rollouts"]}
    assert set(by_ep_id) == {"s7-p0-g0", "s7-p0-g1", "s7-p1-g2"}
    r_detail = index.rollout_detail("default", "agentflow-e2e", run_id, by_ep_id["s7-p0-g0"]["id"])
    assert r_detail is not None
    assert r_detail["length"] == 2
    assert r_detail["tracks"]["reward"] == [0.5, 1.0]
    assert r_detail["tracks"]["action_tokens"] == [2.0, 2.0]
    assert r_detail["tracks"]["obs_tokens"] == [2.0, 0.0]
    assert r_detail["tracks"]["terminated"] == [0.0, 1.0]
    assert r_detail["tracks"]["truncated"] == [0.0, 0.0]
    assert r_detail["tracks"]["mean_logprob"] == [-1.0, -1.0]
    assert r_detail["actions"] == [0, 1]
    assert r_detail["action_names"] == ["ab", "cd"]

    # Verify indexed grouped completions (mean_reward, std_reward, pass_rate, transcripts, kl, truncated)
    comps_page = index.completions("default", "agentflow-e2e", run_id)
    assert comps_page is not None and comps_page["total"] == 2
    by_group = {g["group"]: g for g in comps_page["groups"]}
    g0 = by_group["s7-p0"]
    assert g0["k"] == 2
    assert g0["mean_reward"] == pytest.approx(0.75)
    assert g0["std_reward"] == pytest.approx(0.75)
    assert g0["pass_rate"] == pytest.approx(0.5)
    assert g0["zero_variance"] is False
    assert g0["truncated"] == 1
    g0_detail = index.completion_detail("default", "agentflow-e2e", run_id, g0["id"])
    assert g0_detail is not None
    assert g0_detail["prompt"] == "calc 2+2"
    assert g0_detail["policy_version"] == 4
    assert "[Turn 0" in g0_detail["responses"][0]["text"]
    assert "Observation: xy" in g0_detail["responses"][0]["text"]
    assert g0_detail["responses"][0]["tokens"] == 4
    assert g0_detail["responses"][0]["kl"] is not None and g0_detail["responses"][0]["kl"] > 0.0
    assert g0_detail["responses"][1]["truncated"] is True

    # Verify IcePop histogram
    icepop_hist = index.histograms("default", "agentflow-e2e", run_id, "rl/icepop_ratio_dist")
    assert icepop_hist is not None and icepop_hist["total"] == 1
    assert icepop_hist["columns"][0]["n"] == 8
    assert icepop_hist["columns"][0]["max"] == pytest.approx(1.5, abs=1e-4)


def test_agent_flow_build_inner_trainer_forwards_eyvan_config(monkeypatch):
    captured_cfgs: list[GRPOConfig] = []

    class _RecordingGRPO(GRPOTrainer):
        def __init__(self, arguments, **kwargs):
            captured_cfgs.append(arguments)
            self.arguments = arguments
            self.processing_class = kwargs.get("processing_class")
            self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    monkeypatch.setattr(
        "easydel.utils.Registry.get_or_raise", lambda kind, name: _RecordingGRPO if kind == "trainer" else GRPOConfig
    )

    # 1. base_config=None forwards all 8 Eyvan fields from AgentFlowConfig.
    tags = ["agentic", "icepop"]
    cfg_none = AgentFlowConfig(
        base_trainer_type="grpo",
        base_config=None,
        environment="synthetic_tool",
        use_eyvan_track=True,
        eyvan_track_uri="/tmp/eyvan-fwd-none",
        eyvan_track_project="fwd-proj",
        eyvan_track_name="fwd-run",
        eyvan_track_group="fwd-grp",
        eyvan_track_tags=tags,
        eyvan_track_flush_interval_s=2.5,
        eyvan_report_progress=False,
    )
    flow_none = AgentFlowTrainer(arguments=cfg_none, environment=_SyntheticCodeEnv)
    inner_none = captured_cfgs[-1]
    assert flow_none.trainer.arguments is inner_none
    assert inner_none.use_eyvan_track is True
    assert inner_none.eyvan_track_uri == "/tmp/eyvan-fwd-none"
    assert inner_none.eyvan_track_project == "fwd-proj"
    assert inner_none.eyvan_track_name == "fwd-run"
    assert inner_none.eyvan_track_group == "fwd-grp"
    assert inner_none.eyvan_track_tags == ["agentic", "icepop"]
    assert inner_none.eyvan_track_tags is not cfg_none.eyvan_track_tags
    assert inner_none.eyvan_track_flush_interval_s == pytest.approx(2.5)
    assert inner_none.eyvan_report_progress is False

    # 2. dict base_config inherits unspecified Eyvan fields while preserving explicit overrides.
    cfg_dict = AgentFlowConfig(
        base_trainer_type="grpo",
        base_config={"num_generations": 4, "eyvan_track_project": "override-proj", "eyvan_report_progress": True},
        use_eyvan_track=True,
        eyvan_track_uri="/tmp/eyvan-fwd-dict",
        eyvan_track_project="ignored-proj",
        eyvan_track_name="dict-run",
        eyvan_track_group="dict-grp",
        eyvan_track_tags=["t1"],
        eyvan_track_flush_interval_s=1.25,
        eyvan_report_progress=False,
    )
    AgentFlowTrainer(arguments=cfg_dict, environment=_SyntheticCodeEnv)
    inner_dict = captured_cfgs[-1]
    assert inner_dict.num_generations == 4
    assert inner_dict.use_eyvan_track is True
    assert inner_dict.eyvan_track_uri == "/tmp/eyvan-fwd-dict"
    assert inner_dict.eyvan_track_project == "override-proj"
    assert inner_dict.eyvan_track_name == "dict-run"
    assert inner_dict.eyvan_track_group == "dict-grp"
    assert inner_dict.eyvan_track_tags == ["t1"]
    assert inner_dict.eyvan_track_flush_interval_s == pytest.approx(1.25)
    assert inner_dict.eyvan_report_progress is True


def test_agent_flow_eyvan_interop_caps_eval_multihost_and_fault_tolerance(monkeypatch, tmp_path: Path):
    # 1. Env name resolution across str, mapping, class, callable, instance, fallback.
    assert _resolve_eyvan_env_name("python_tir") == "python_tir"
    assert _resolve_eyvan_env_name({"name": "swe_bench"}) == "swe_bench"
    assert _resolve_eyvan_env_name({"type": "terminal"}) == "terminal"
    assert _resolve_eyvan_env_name(_SyntheticCodeEnv) == "_SyntheticCodeEnv"
    assert _resolve_eyvan_env_name(lambda: None) == "agent_flow"
    assert _resolve_eyvan_env_name(_SyntheticCodeEnv()) == "_SyntheticCodeEnv"

    # 2. Interop with TrainingArguments.log_eyvan_* helpers + 8-item rollout/completion caps + eval mode.
    recorded: dict[str, list[dict[str, object]]] = {"episode": [], "rollout": [], "completions": [], "histogram": []}

    class _InteropArgs(GRPOConfig):
        def log_eyvan_episode(self, ret, length=None, **kw):
            recorded["episode"].append({"ret": ret, "length": length, **kw})
            return True

        def log_eyvan_rollout(self, steps=None, **kw):
            recorded["rollout"].append({"steps": steps, **kw})
            return True

        def log_eyvan_completions(self, prompt, responses, **kw):
            recorded["completions"].append({"prompt": prompt, "responses": responses, **kw})
            return True

        def log_eyvan_histogram(self, key, values, **kw):
            recorded["histogram"].append({"key": key, "len": len(values), **kw})
            return True

    args = _InteropArgs(
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(tmp_path / "unused"),
        max_prompt_length=16,
        max_completion_length=16,
        beta=0.0,
    )
    inner = _make_stub_grpo(args)
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=2, group_size=1, max_prompt_length=16, max_completion_length=16),
        trainer=inner,
        environment="math_env",
    )

    # 12 prompt groups with rewards 0..11 -> rollouts capped at 8 (must include best=11 & worst=0), completions capped at 8.
    twelve_eps = [
        _make_episode(
            prompt_index=i,
            group_id=i,
            prompt=f"p{i}",
            turns_spec=[(f"a{i}", "", float(i), True, False, [-0.1, -0.1])],
            episode_reward=float(i),
        )
        for i in range(12)
    ]
    flow._adapt_episodes_to_trainer(twelve_eps, group_size=1, state=SimpleNamespace(step=3), is_train=True)
    assert len(recorded["episode"]) == 12
    assert len(recorded["rollout"]) == 8
    rollout_ids = {r["episode_id"] for r in recorded["rollout"]}
    assert "s3-p11-g11" in rollout_ids  # best reward
    assert "s3-p0-g0" in rollout_ids  # worst reward
    assert len(recorded["completions"]) == 8
    assert flow._eyvan_total_turns == 12

    # Eval mode emits kind="eval" without advancing training _eyvan_total_turns.
    recorded["episode"].clear()
    recorded["rollout"].clear()
    flow._adapt_episodes_to_trainer(twelve_eps[:2], group_size=1, state=SimpleNamespace(step=3), is_train=False)
    assert all(r["kind"] == "eval" for r in recorded["episode"])
    assert all(r["kind"] == "eval" for r in recorded["rollout"])
    assert flow._eyvan_total_turns == 12

    # 3. Tracking errors must NEVER crash training.
    class _ExplodingRun:
        finished = False

        def episode(self, *a, **kw):
            raise RuntimeError("boom episode")

        def rollout(self, *a, **kw):
            raise RuntimeError("boom rollout")

        def completions(self, *a, **kw):
            raise RuntimeError("boom completions")

        def histogram(self, *a, **kw):
            raise RuntimeError("boom histogram")

    broken_args = GRPOConfig(use_wandb=False, save_directory=None, use_eyvan_track=True, beta=0.0)
    broken_args.__dict__["_eyvan_track_run"] = _ExplodingRun()
    broken_inner = _make_stub_grpo(broken_args)
    broken_flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=2, group_size=1, max_prompt_length=16, max_completion_length=16),
        trainer=broken_inner,
        environment="math_env",
    )
    batch_out, metrics_out = broken_flow._adapt_episodes_to_trainer(
        twelve_eps[:2], group_size=1, state=SimpleNamespace(step=1), is_train=True
    )
    assert "completion_ids" in batch_out
    assert metrics_out["rl/pass_rate"] == pytest.approx(0.5)

    # 4. Multi-host safety: rank > 0 (can_log_metrics=False) skips Eyvan logging; _to_host_numpy_local_only never gathers.
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    rank1_args = GRPOConfig(
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(tmp_path / "rank1_store"),
        log_all_workers=False,
        beta=0.0,
    )
    assert rank1_args.eyvan_tracking_enabled is False
    rank1_inner = _make_stub_grpo(rank1_args)
    rank1_flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=2, group_size=1, max_prompt_length=16, max_completion_length=16),
        trainer=rank1_inner,
        environment="math_env",
    )
    _, r1_metrics = rank1_flow._adapt_episodes_to_trainer(
        twelve_eps[:2], group_size=1, state=SimpleNamespace(step=1), is_train=True
    )
    assert r1_metrics["rl/pass_rate"] == pytest.approx(0.5)
    assert not (tmp_path / "rank1_store").exists()

    # Verify multi-host replicated vs sharded JAX array handling (zero-collective local reads).
    from jax.experimental import multihost_utils

    class _FakeGlobalArray:
        def __init__(self, value, *, is_fully_addressable: bool, is_fully_replicated: bool, has_local: bool = True):
            self.value = value
            self.is_fully_addressable = is_fully_addressable
            self.is_fully_replicated = is_fully_replicated
            self.sharding = SimpleNamespace(has_addressable_devices=has_local)

    allgather_calls: list[object] = []
    monkeypatch.setattr("easydel.trainers.agent_flow_trainer.agent_flow_trainer.jax.Array", _FakeGlobalArray)
    monkeypatch.setattr(
        "easydel.trainers.agent_flow_trainer.agent_flow_trainer.jax.device_get",
        lambda x: x.value if isinstance(x, _FakeGlobalArray) else x,
    )
    monkeypatch.setattr(
        multihost_utils,
        "process_allgather",
        lambda arr, tiled=True: (allgather_calls.append(arr), arr.value)[1],
    )

    # 4a. Multi-host replicated array (is_fully_addressable=False, is_fully_replicated=True) succeeds locally.
    replicated_step = _FakeGlobalArray(
        np.asarray(19, dtype=np.int32), is_fully_addressable=False, is_fully_replicated=True
    )
    replicated_vals = _FakeGlobalArray(
        np.asarray([1.25, -0.5], dtype=np.float32), is_fully_addressable=False, is_fully_replicated=True
    )
    assert _extract_nonneg_step(SimpleNamespace(step=replicated_step), fallback=0) == 19
    np.testing.assert_allclose(_to_host_numpy_local_only(replicated_vals), [1.25, -0.5])
    np.testing.assert_allclose(_to_host_numpy(replicated_vals), [1.25, -0.5])
    assert allgather_calls == []

    # 4b. Multi-host replicated array on a host with no addressable devices is refused locally.
    remote_only_rep = _FakeGlobalArray(
        np.asarray(21, dtype=np.int32), is_fully_addressable=False, is_fully_replicated=True, has_local=False
    )
    assert _extract_nonneg_step(SimpleNamespace(step=remote_only_rep), fallback=7) == 7
    assert _to_host_numpy_local_only(remote_only_rep) is None
    assert allgather_calls == []

    # 4c. Genuinely sharded multi-host array (is_fully_addressable=False, is_fully_replicated=False) is refused locally.
    sharded_step = _FakeGlobalArray(
        np.asarray(23, dtype=np.int32), is_fully_addressable=False, is_fully_replicated=False
    )
    sharded_vals = _FakeGlobalArray(
        np.asarray([0.1, 0.2], dtype=np.float32), is_fully_addressable=False, is_fully_replicated=False
    )
    assert _extract_nonneg_step(SimpleNamespace(step=sharded_step), fallback=9) == 9
    assert _to_host_numpy_local_only(sharded_vals) is None
    assert allgather_calls == []

    # _to_host_numpy (only called unconditionally across all hosts) still gathers genuinely sharded arrays.
    np.testing.assert_allclose(_to_host_numpy(sharded_vals), [0.1, 0.2])
    assert len(allgather_calls) == 1


def test_agentic_moshpit_eyvan_telemetry_and_icepop_end_to_end(tmp_path: Path):
    store_root = tmp_path / "moshpit_root"
    scope_uri = store_root / "default" / "moshpit-e2e"
    cfg = AgenticMoshPitConfig(
        model_name="acme/moshpit-model",
        use_wandb=False,
        save_directory=None,
        max_steps=3,
        group_size=2,
        num_env_groups=2,
        max_prompt_length=12,
        max_completion_length=8,
        advantage_estimator="agentic_reinforce",
        esurge_importance_sampling_correction=True,
        esurge_importance_sampling_mode="token_icepop",
        esurge_importance_sampling_cap=2.0,
        esurge_importance_sampling_cap_low=0.5,
        use_eyvan_track=True,
        eyvan_track_uri=str(scope_uri),
        eyvan_track_project="moshpit-e2e",
        eyvan_track_name="moshpit-run",
        eyvan_track_flush_interval_s=0.1,
    )

    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = cfg
    trainer.env_factory = _SyntheticCodeEnv
    trainer._eyvan_env_name = _resolve_eyvan_env_name(_SyntheticCodeEnv, fallback="agentic_moshpit")
    trainer._eyvan_total_turns = 0
    trainer._eyvan_total_action_tokens = 0
    trainer.scale_rewards = "group"
    trainer._rollout_step = 2
    trainer._eval_rollout_step = 0
    trainer._tools = []
    trainer._tool_call_parser = None
    trainer._tool_schemas = None
    trainer.processing_class = _CharTokenizer()
    trainer.ref_state = SimpleNamespace(graphstate={}, graphother={}, model=SimpleNamespace(forward=lambda **kw: None))
    trainer.ref_logps_chunk_size = None
    trainer.compute_refmodel_logps = lambda gs, go, seqs, mask, kw, *, prompt_length: np.full(
        (seqs.shape[0], seqs.shape[1] - prompt_length), -1.5, dtype=np.float32
    )
    trainer.reward_funcs = [lambda prompts, completions, **kw: [0.25, 0.0, 0.5, 0.5]]
    trainer.reward_func_names = ["format_bonus"]
    trainer.reward_weights = np.array([1.0], dtype=np.float32)
    trainer.reward_processing_classes = [None]
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer._log_trajectories_to_wandb = lambda *a, **kw: None
    trainer._rollout_manager = RolloutManager(_CharTokenizer(), max_steps=3)

    # 4 trajectories (2 groups of 2):
    # Each has 2 prompt tokens [10, 11] + 4 completion tokens [65, 66, 67, 68] with response_mask [0, 0, 1, 1, 0, 1]
    # (3 active tokens per traj -> 12 active tokens total).
    # Policy logps vs sampling logps (-1.0 on all active):
    #   3 active tokens have log_ratio = ln(4.0) > ln(2.0) -> masked to 0.0 -> masked_frac = 3/12 = 0.25!
    #   Remaining 9 active tokens have log_ratio = 0.0 -> ratio = 1.0 -> mean = 9/12 = 0.75, max = 1.0!
    policy_lps = np.full((4, 8), -1.0, dtype=np.float32)
    policy_lps[0, 0] = -1.0 + math.log(4.0)
    policy_lps[1, 1] = -1.0 + math.log(4.0)
    policy_lps[2, 3] = -1.0 + math.log(4.0)
    trainer.compute_state_logps = lambda state, seqs, mask, kw, **opts: policy_lps

    def _make_traj(env_r: float, step_rs: list[float], truncated: bool = False) -> TrajectoryResult:
        return TrajectoryResult(
            input_ids=np.array([10, 11, 65, 66, 67, 68], dtype=np.int64),
            attention_mask=np.ones(6, dtype=np.int32),
            prompt_mask=np.array([1, 1, 0, 0, 1, 0], dtype=np.int32),
            response_mask=np.array([0, 0, 1, 1, 0, 1], dtype=np.int32),
            episode_reward=env_r,
            step_rewards=list(step_rs),
            num_steps=len(step_rs),
            turns=[
                TurnRecord(role="user", content="solve task"),
                TurnRecord(
                    role="assistant",
                    content="step1",
                    token_ids=np.array([65, 66]),
                    is_response=True,
                    reward=step_rs[0],
                    sampling_logps=np.array([-1.0, -1.0], dtype=np.float32),
                ),
                TurnRecord(role="tool", content="obs1"),
                TurnRecord(
                    role="assistant",
                    content="final",
                    token_ids=np.array([68]),
                    is_response=True,
                    reward=step_rs[-1],
                    sampling_logps=np.array([-1.0], dtype=np.float32),
                ),
            ],
            info={"truncated": truncated},
            turn_ids=np.array([-1, -1, 0, 0, -1, 1], dtype=np.int32),
            sampling_logps=np.array([0.0, 0.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
        )

    # Group 0 rewards after aux: [1.0+0.25=1.25, 0.0+0.0=0.0] (non-zero std)
    # Group 1 rewards after aux: [0.5+0.5=1.0, 0.5+0.5=1.0] (zero std)
    def _fresh_trajs(*, state, base_seed):
        del state, base_seed
        return [
            _make_traj(1.0, [0.5, 0.5], truncated=False),
            _make_traj(0.0, [0.0, 0.0], truncated=True),
            _make_traj(0.5, [0.0, 0.5], truncated=False),
            _make_traj(0.5, [0.0, 0.5], truncated=False),
        ]

    trainer._run_grouped_episodes_coordinated = _fresh_trajs
    state = SimpleNamespace(step=5, model=SimpleNamespace(forward=lambda **kw: None, pause_esurge=lambda **kw: None))

    batch, metrics = AgenticMoshPitTrainer._preprocess_batch_input(trainer, state=state, batch={}, is_train=True)
    assert "importance_sampling_ratio" in batch
    assert metrics["rl/pass_rate"] == pytest.approx(0.75)
    assert metrics["rl/truncation_rate"] == pytest.approx(0.25)
    assert metrics["rl/zero_var_group_frac"] == pytest.approx(0.5)
    assert metrics["rl/icepop_masked_frac"] == pytest.approx(0.25)
    assert metrics["rl/icepop_ratio_mean"] == pytest.approx(0.75, abs=1e-4)
    assert metrics["rl/icepop_ratio_max"] == pytest.approx(1.0, abs=1e-4)

    # Also test eval step emits kind="eval" episodes
    AgenticMoshPitTrainer._preprocess_batch_input(trainer, state=state, batch={}, is_train=False)

    run = cfg.get_eyvan_track()
    assert run is not None
    run_id = run.id
    cfg.log_metrics(metrics, step=5)
    cfg.close_eyvan_track(status="finished")

    index = TrackingIndex(tmp_path / "moshpit.db")
    scan = index.scan(store_for_uri(str(store_root)))
    assert scan.errors == 0

    detail = index.run_detail("default", "moshpit-e2e", run_id)
    assert detail is not None
    assert detail["errors"] == []
    assert detail["stats"]["dropped"]["invalid"] == 0
    assert detail["episodes"] == {"train": 4, "eval": 4}
    assert detail["rollouts"] == 8
    assert detail["completion_groups"] == 4
    assert detail["envs"] == ["_SyntheticCodeEnv"]
    assert set(detail["hist_keys"]) >= {
        "rl/reward_dist",
        "rl/advantage_dist",
        "rl/episode_turns_dist",
        "rl/completion_length_dist",
        "rl/icepop_ratio_dist",
    }

    # Verify completion group reward decomposition (env + aux/format_bonus) and KL
    comps = index.completions("default", "moshpit-e2e", run_id)
    assert comps is not None and comps["total"] == 4
    g0 = next(g for g in comps["groups"] if g["group"] == "s5-g0")
    assert g0["mean_reward"] == pytest.approx(0.625)
    assert g0["pass_rate"] == pytest.approx(0.5)
    g0_detail = index.completion_detail("default", "moshpit-e2e", run_id, g0["id"])
    assert g0_detail is not None
    assert g0_detail["responses"][0]["rewards"]["env"] == pytest.approx(1.0)
    assert g0_detail["responses"][0]["rewards"]["aux/format_bonus"] == pytest.approx(0.25)
    assert g0_detail["responses"][0]["kl"] is not None and g0_detail["responses"][0]["kl"] > 0.0

    # Verify rollout detail tracks & action_names
    rollouts = index.rollouts("default", "moshpit-e2e", run_id, kind="train")
    assert rollouts is not None and rollouts["total"] == 4
    r_detail = index.rollout_detail("default", "moshpit-e2e", run_id, rollouts["rollouts"][0]["id"])
    assert r_detail is not None
    assert r_detail["length"] == 2
    assert r_detail["actions"] == [0, 1]
    assert r_detail["action_names"] == ["step1", "final"]
    assert "mean_logprob" in r_detail["tracks"]


def test_agent_flow_and_agentic_moshpit_lazy_init_before_log_metrics_and_status_teardown(tmp_path: Path):
    import threading

    baseline = set(threading.enumerate())

    # 1. AgentFlowTrainer: __init__ creates ZERO files/threads; first episode adaptation lazily starts
    #    Eyvan run (exposing flow.eyvan_run), and finish(status="failed") closes cleanly through _wrapped_teardown.
    af_store = tmp_path / "lazy_af_root"
    af_scope = af_store / "default" / "lazy-af"
    grpo_args = GRPOConfig(
        model_name="Qwen/Qwen3-0.6B",
        use_wandb=False,
        save_directory=None,
        max_prompt_length=16,
        max_completion_length=12,
        beta=0.0,
        use_eyvan_track=True,
        eyvan_track_uri=str(af_scope),
        eyvan_track_project="lazy-af",
        eyvan_track_flush_interval_s=0.1,
    )
    inner = _make_stub_grpo(grpo_args)
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=2, group_size=1, max_prompt_length=16, max_completion_length=12),
        trainer=inner,
        environment=_SyntheticCodeEnv,
    )
    assert flow.eyvan_run is None
    assert not af_scope.exists()
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []

    ep = _make_episode(
        prompt_index=0,
        group_id=0,
        prompt="hi",
        turns_spec=[("ok", "", 1.0, True, False, [-0.5, -0.5])],
        episode_reward=1.0,
    )
    flow._adapt_episodes_to_trainer([ep], group_size=1, state=SimpleNamespace(step=1), is_train=True)
    assert flow.eyvan_run is not None
    af_run_id = flow.eyvan_run.id
    assert {"agentflow", "env:synthetic_code_env"} <= set(flow.eyvan_run.tags)
    flow.finish(status="failed")
    assert flow.eyvan_run is None
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []

    af_idx = TrackingIndex(tmp_path / "lazy_af.db")
    assert af_idx.scan(store_for_uri(str(af_store))).errors == 0
    af_detail = af_idx.run_detail("default", "lazy-af", af_run_id)
    assert af_detail is not None
    assert af_detail["status"] == "failed"
    assert af_detail["episodes"] == {"train": 1, "eval": 0}
    assert af_detail["rollouts"] == 1

    # 2. AgenticMoshPitTrainer: first _log_trajectories_to_eyvan call triggers lazy init before log_metrics().
    mp_store = tmp_path / "lazy_mp_root"
    mp_scope = mp_store / "default" / "lazy-mp"
    mp_cfg = AgenticMoshPitConfig(
        model_name="Qwen/Qwen3-0.6B",
        use_wandb=False,
        save_directory=None,
        max_steps=2,
        group_size=1,
        num_env_groups=1,
        use_eyvan_track=True,
        eyvan_track_uri=str(mp_scope),
        eyvan_track_project="lazy-mp",
        eyvan_track_flush_interval_s=0.1,
    )
    mp = object.__new__(AgenticMoshPitTrainer)
    mp.arguments = mp_cfg
    mp.env_factory = _SyntheticCodeEnv
    mp._eyvan_env_name = _resolve_eyvan_env_name(_SyntheticCodeEnv, fallback="agentic_moshpit")
    mp._eyvan_total_turns = 0
    mp._eyvan_total_action_tokens = 0
    mp.processing_class = _CharTokenizer()
    assert mp.eyvan_run is None
    assert not mp_scope.exists()

    traj = TrajectoryResult(
        input_ids=np.array([10, 65], dtype=np.int64),
        attention_mask=np.ones(2, dtype=np.int32),
        prompt_mask=np.array([1, 0], dtype=np.int32),
        response_mask=np.array([0, 1], dtype=np.int32),
        episode_reward=1.0,
        step_rewards=[1.0],
        num_steps=1,
        turns=[
            TurnRecord(role="user", content="go"),
            TurnRecord(role="assistant", content="done", token_ids=np.array([65]), is_response=True, reward=1.0),
        ],
        info={},
    )
    mp._log_trajectories_to_eyvan(
        trajectories=[traj],
        state=SimpleNamespace(step=1),
        is_train=True,
        env_rewards_np=np.array([1.0], dtype=np.float32),
        advantages=np.array([0.5], dtype=np.float32),
        comp_mask_np=np.array([[1]], dtype=np.int32),
        comp_lengths_np=np.array([1], dtype=np.int64),
        traj_truncated=[False],
    )
    assert mp.eyvan_run is not None
    mp_run_id = mp.eyvan_run.id
    assert "agentic_moshpit" in mp.eyvan_run.tags
    mp.finish()
    assert mp.eyvan_run is None
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []

    mp_idx = TrackingIndex(tmp_path / "lazy_mp.db")
    assert mp_idx.scan(store_for_uri(str(mp_store))).errors == 0
    mp_detail = mp_idx.run_detail("default", "lazy-mp", mp_run_id)
    assert mp_detail is not None
    assert mp_detail["status"] == "finished"
    assert mp_detail["episodes"] == {"train": 1, "eval": 0}
    assert mp_detail["rollouts"] == 1


def test_agent_flow_tito_turn_reconstruction_transcript_and_factory_env_name():
    from easydel.environments import AutoEnvironment
    from easydel.trainers.agent_flow_trainer.agent_flow_trainer import (
        AgentFlowEpisode,
        _format_agent_flow_transcript,
        _reconstruct_turns_from_tito,
        _resolve_eyvan_env_name,
    )

    tok = _CharTokenizer()
    decode = tok.decode
    prompt_ids = tok.encode("Fix bug")
    act0 = tok.encode("run pytest")
    obs0 = tok.encode("1 failed")
    act1 = tok.encode("patch line 3")
    comp_ids = act0 + obs0 + act1
    comp_mask = [1] * len(act0) + [0] * len(obs0) + [1] * len(act1)
    comp_lps = [-0.2] * len(act0) + [0.0] * len(obs0) + [-0.1] * len(act1)

    turns = _reconstruct_turns_from_tito(
        prompt_ids=prompt_ids,
        comp_ids=comp_ids,
        comp_amask=comp_mask,
        comp_lps=comp_lps,
        final_reward=1.0,
        decode_ids=decode,
        truncated=False,
    )
    assert len(turns) == 2
    assert turns[0].action_text == "run pytest"
    assert turns[0].observation_text == "1 failed"
    assert turns[0].reward == 0.0
    assert turns[0].terminated is False
    assert turns[0].logprobs == pytest.approx([-0.2] * len(act0))
    assert turns[1].action_text == "patch line 3"
    assert turns[1].observation_text == ""
    assert turns[1].reward == 1.0
    assert turns[1].terminated is True
    assert turns[1].logprobs == pytest.approx([-0.1] * len(act1))

    # Even if an episode reaches transcript formatting without pre-populated turns,
    # decoding completion_token_ids produces the multi-turn transcript rather than "[empty episode]".
    ep_no_turns = AgentFlowEpisode(
        prompt_index=0,
        group_id=0,
        prompt_token_ids=prompt_ids,
        completion_token_ids=comp_ids,
        action_mask=comp_mask,
        step_rewards=[1.0],
        episode_reward=1.0,
        turns=(),
        logprobs=comp_lps,
        env_error=False,
    )
    transcript = _format_agent_flow_transcript(ep_no_turns, decode_ids=decode)
    assert transcript != "[empty episode]"
    assert "run pytest" in transcript
    assert "1 failed" in transcript
    assert "patch line 3" in transcript

    # AutoEnvironment.as_factory should expose a meaningful env label rather than "_factory".
    factory = AutoEnvironment.as_factory("harness", harness="qwen_code")
    assert _resolve_eyvan_env_name(factory) == "harness:qwen_code"
    assert _resolve_eyvan_env_name(lambda: None, fallback="harness") == "harness"

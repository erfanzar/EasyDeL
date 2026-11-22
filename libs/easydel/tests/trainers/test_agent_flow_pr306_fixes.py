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
"""Regression tests for PR #306 AgentFlow, eLarge, AsyncGRPO, and DPO `_fn.py` fixes."""

from __future__ import annotations

import functools
import json
import queue
import threading
import typing as tp
from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest
from easydel.environments._base import AgenticEnvironment, ResetResult, StepResult
from easydel.infra.elarge.model import eLargeModel
from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
from easydel.trainers.agent_flow_trainer.agent_flow_trainer import (
    AgentFlowEpisode,
    _to_host_numpy,
    agent_flow_supplied_reward,
)
from easydel.trainers.direct_preference_optimization_trainer._fn import (
    _reduce_pair_losses,
    compute_dpo_losses,
    concatenated_forward,
    evaluation_step,
    training_step,
)
from easydel.trainers.generalized_knowledge_distillation_trainer import GKDTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import normalize_generation_model_kwargs
from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationTrainer
from easydel.trainers.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer
from easydel.trainers.proximal_policy_optimization_trainer._fn import ppo_step
from easydel.trainers.seq_kd_trainer import SeqKDTrainer
from easydel.trainers.ssd_trainer import SSDTrainer
from easydel.trainers.ssd_trainer._fn import ssd_step
from easydel.trainers.trainer._fn import base_step
from easydel.trainers.vapo_trainer import VAPOConfig, VAPOTrainer
from easydel.utils import Registry
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    eos_token = "<E>"
    bos_token = "<s>"

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        ids = [ord(c) for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) > 0)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "".join(f"<{m['role']}>{m['content']}<E>" for m in messages) + (
            "<assistant>" if add_generation_prompt else ""
        )


class _StrictAlternatingTokenizer(_DummyTokenizer):
    """Chat template enforcing strict user/assistant alternation (like Llama-3/Mistral/Gemma)."""

    eos_token_id: tp.ClassVar[list[int]] = [2, 3]

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        for i in range(1, len(messages)):
            if messages[i]["role"] == messages[i - 1]["role"]:
                raise ValueError(f"Consecutive {messages[i]['role']!r} messages violate alternating template.")
        return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)


class _ProcessorWrapper:
    """ProcessorMixin-like wrapper where pad/eos live on ``.tokenizer`` rather than top-level."""

    def __init__(self, tok):
        self.tokenizer = tok

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class _ScriptedEnv(AgenticEnvironment):
    def __init__(
        self,
        prompt: str = "hi",
        observations: tuple[str, ...] = ("obs1", ""),
        rewards: tuple[float, ...] = (0.5, 1.0),
        fail_step: bool = False,
        close_raises: bool = False,
        reset_raises: Exception | None = None,
    ):
        self._prompt = prompt
        self._observations = observations
        self._rewards = rewards
        self._fail_step = fail_step
        self._close_raises = close_raises
        self._reset_raises = reset_raises
        self._turn = 0
        self.closed = False

    def reset(self, seed: int | None = None) -> ResetResult:
        if self._reset_raises is not None:
            raise self._reset_raises
        self._turn = 0
        return ResetResult(observation=self._prompt, info={})

    def step(self, action: str) -> StepResult:
        idx = self._turn
        self._turn += 1
        if self._fail_step:
            return StepResult(observation="err", reward=0.0, terminated=True, truncated=False, info={"env_error": True})
        done = self._turn >= len(self._observations)
        obs = self._observations[min(idx, len(self._observations) - 1)]
        rew = self._rewards[min(idx, len(self._rewards) - 1)]
        return StepResult(observation=obs, reward=float(rew), terminated=done, truncated=False, info={})

    def close(self) -> None:
        self.closed = True
        if self._close_raises:
            raise OSError("broken pipe on close")


def _stub_trainer(cls, args=None, tok=None):
    inst = object.__new__(cls)
    inst.arguments = args or SimpleNamespace(
        num_generations=2,
        scale_rewards="group",
        async_mode=False,
        max_prompt_length=16,
        max_completion_length=32,
        beta=0.04,
    )
    inst.processing_class = tok or _DummyTokenizer()
    inst._preprocess_batch_input = lambda state, batch, is_train: ({}, {})
    return inst


def _make_1d_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]), ("dp",))


# ---------------------------------------------------------------------------
# Finding #1: GRPO compute_refmodel_logps pytree in_shardings & prompt_length
# ---------------------------------------------------------------------------


def test_pr306_f1_grpo_ref_logps_passes_normalized_model_kwargs_dict_and_prompt_length():
    from easydel.trainers.training_utils import compile_trainer_step

    mesh = _make_1d_mesh()
    rep = NamedSharding(mesh, PartitionSpec())
    model_kwargs_shardings = {key: None for key in normalize_generation_model_kwargs(None).keys()}

    # Case 1: Exact mirror of GRPOTrainer.configure_functions (in_shardings + static_argnames).
    # Passing None for model_kwargs previously crashed JAX pytree validation, and passing
    # prompt_length= as a kwarg triggers JAX's "pjit does not support kwargs when in_shardings is specified".
    def _ref_fn(gs, go, ids, mask, model_kwargs=None, *, graphdef=None, prompt_length: int | None = None):
        assert isinstance(model_kwargs, dict)
        assert set(model_kwargs.keys()) == set(model_kwargs_shardings.keys())
        pl = 12 if prompt_length is None else int(prompt_length)
        return -0.25 * jnp.ones((ids.shape[0], ids.shape[1] - pl), dtype=jnp.float32)

    _compiled_ref_logps = compile_trainer_step(
        functools.partial(_ref_fn, graphdef="gd"),
        mesh=mesh,
        static_argnames=("graphdef", "prompt_length"),
        in_shardings=(rep, rep, rep, rep, model_kwargs_shardings),
        out_shardings=rep,
    )

    inner = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=2, beta=0.04, max_prompt_length=12, max_completion_length=16),
    )
    inner.model_state = SimpleNamespace(model=SimpleNamespace(mesh=mesh))
    inner.ref_state = SimpleNamespace(
        graphstate=jnp.asarray(1.0, dtype=jnp.float32),
        graphother=jnp.asarray(0.0, dtype=jnp.float32),
    )
    inner.compute_refmodel_logps = _compiled_ref_logps

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=2,
            max_prompt_length=12,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="hi", observations=("",), rewards=(1.0,)),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a"), ord("b")], "ab", [-0.1, -0.2]),
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["hi"]}, True)
    assert out["ref_per_token_logps"].shape == (2, 16)
    np.testing.assert_allclose(np.asarray(out["ref_per_token_logps"]), -0.25)

    # Case 2: Wrapper callable accepting prompt_length= (e.g. positional static_argnums wrapper)
    # receives both the normalized model_kwargs dict and explicit prompt_length=12 via _call_compute_refmodel_logps.
    seen: dict[str, object] = {}

    def _ref_wrapper(gs, go, ids, mask, model_kwargs, prompt_length=None):
        seen["model_kwargs"] = model_kwargs
        seen["prompt_length"] = prompt_length
        pl = 12 if prompt_length is None else int(prompt_length)
        return -0.5 * jnp.ones((ids.shape[0], ids.shape[1] - pl), dtype=jnp.float32)

    inner.compute_refmodel_logps = _ref_wrapper
    out2, _ = inner._preprocess_batch_input(SimpleNamespace(step=1), {"prompts": ["hi"]}, True)
    assert isinstance(seen["model_kwargs"], dict)
    assert set(seen["model_kwargs"].keys()) == set(model_kwargs_shardings.keys())
    assert seen["prompt_length"] == 12
    assert out2["ref_per_token_logps"].shape == (2, 16)
    np.testing.assert_allclose(np.asarray(out2["ref_per_token_logps"]), -0.5)


# ---------------------------------------------------------------------------
# Finding #2: Distillation / preference / unsupported inner trainers
# ---------------------------------------------------------------------------


def _install_init_recorder(monkeypatch, cls) -> dict[str, object]:
    captured: dict[str, object] = {}
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def _rec(self, **kwargs):
        captured.clear()
        captured.update(kwargs)
        self.arguments = kwargs["arguments"]
        self.processing_class = kwargs.get("processing_class")
        self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    monkeypatch.setattr(cls, "__init__", _rec)
    return captured


def test_pr306_f2_student_model_bound_for_seq_kd_and_opd_and_no_reward_for_ssd(monkeypatch):
    for base_type, expected_cls in (
        ("seq_kd", SeqKDTrainer),
        ("on_policy_distillation", OnPolicyDistillationTrainer),
        ("ssd", SSDTrainer),
    ):
        captured = _install_init_recorder(monkeypatch, expected_cls)
        model_sentinel = object()
        teacher_sentinel = object()
        extra = {"teacher_model": teacher_sentinel} if base_type != "ssd" else {}
        flow = AgentFlowTrainer(
            arguments=AgentFlowConfig(base_trainer_type=base_type, max_turns=1),
            model=model_sentinel,
            processing_class=_DummyTokenizer(),
            environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
            **extra,
        )
        assert isinstance(flow.trainer, expected_cls)
        if base_type in {"seq_kd", "on_policy_distillation"}:
            assert captured["student_model"] is model_sentinel
            assert "model" not in captured
            assert captured["teacher_model"] is teacher_sentinel
        else:
            assert captured["model"] is model_sentinel
            assert "reward_funcs" not in captured


def test_pr306_f2_seq_kd_masked_labels_only_and_dft_and_ce_steps_work():
    inner = _stub_trainer(
        SeqKDTrainer,
        args=SimpleNamespace(async_mode=False, max_prompt_length=10, max_completion_length=24),
    )
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            group_size=1,
            max_prompt_length=10,
            max_completion_length=24,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="PR", observations=("OBS", ""), rewards=(0.0, 1.0)),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("A")] if t == 0 else [ord("B")], "A" if t == 0 else "B", None),
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["PR"]}, True)
    # SeqKD must emit only input_ids, attention_mask, labels (no extra keys that crash DFT/compute_loss).
    assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}
    labels = np.asarray(out["labels"])[0]
    ids = np.asarray(out["input_ids"])[0]
    # Prompt positions [0..9] must all be -100; only the 2 action tokens ('A' and 'B') are unmasked.
    assert np.all(labels[:10] == -100)
    unmasked = labels[labels != -100].tolist()
    assert unmasked == [ord("A"), ord("B")]
    assert np.array_equal(ids[labels != -100], labels[labels != -100])

    # Run real base_step with loss_type="dft" to verify zero TypeError on kwargs and zero grad on masked tokens.
    mesh = _make_1d_mesh()
    vocab = 128

    class _Output:
        def __init__(self, logits, loss=None):
            self.logits = logits
            self.loss = loss

        def replace(self, **kwargs):
            return _Output(logits=kwargs.get("logits", self.logits), loss=kwargs.get("loss", self.loss))

    class _BiasLM:
        training = False

        def __init__(self, bias):
            self.bias = bias
            self.mesh = mesh

        def prepare_inputs_for_call(self, **kwargs):
            return kwargs

        def __call__(self, input_ids, attention_mask=None):
            return _Output(logits=self.bias)

    bias = jnp.zeros((1, 34, vocab), dtype=jnp.float32)
    def _dft_loss(b):
        st = SimpleNamespace(
            model=_BiasLM(b),
            graphstate={"bias": b},
            step=jnp.asarray(0, dtype=jnp.int32),
            merge=lambda t: _BiasLM(t["bias"]),
        )
        return base_step(
            state=st,
            batch=out,
            loss_config=None,
            learning_rate_fn=lambda _: 1e-3,
            partition_spec=None,
            gradient_accumulation_steps=1,
            is_training=False,
            loss_type="dft",
        ).loss

    loss_val, grad_bias = jax.value_and_grad(_dft_loss)(bias)
    assert np.isfinite(float(loss_val))
    # Only the 2 shifted logit positions predicting the 2 action tokens may carry non-zero gradient!
    grad_norms = np.abs(np.asarray(grad_bias[0])).sum(axis=-1)
    action_positions = np.flatnonzero(labels != -100)
    expected_grad_pos = action_positions - 1
    assert np.all(grad_norms[expected_grad_pos] > 0.0)
    other_mask = np.ones(grad_norms.shape, dtype=bool)
    other_mask[expected_grad_pos] = False
    np.testing.assert_array_equal(grad_norms[other_mask], 0.0)


def test_pr306_f2_ssd_and_opd_adapters_match_step_contracts():
    # SSD expects [B, C] completion_mask (not [B, P+C]).
    ssd_inner = _stub_trainer(
        SSDTrainer,
        args=SimpleNamespace(async_mode=False, max_prompt_length=8, max_completion_length=16),
    )
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            group_size=1,
            max_prompt_length=8,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=ssd_inner,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="P", observations=("O", ""), rewards=(0.0, 1.0)),
        generate_turn_fn=lambda st, ids, txt, t: ([1, 2], "ab", None),
    )
    ssd_out, _ = ssd_inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)
    assert ssd_out["prompt_ids"].shape == (1, 8)
    assert ssd_out["completion_ids"].shape == (1, 16)
    assert ssd_out["completion_mask"].shape == (1, 16)
    assert ssd_out["completion_attention_mask"].shape == (1, 16)

    # Run real ssd_step in eval mode to verify zero shape mismatch.
    mesh = _make_1d_mesh()

    class _TinyLM:
        def __init__(self, scale):
            self.scale = scale
            self.mesh = mesh

        def __call__(self, input_ids, attention_mask=None):
            b, s = input_ids.shape
            return SimpleNamespace(logits=jnp.zeros((b, s, 128), dtype=jnp.float32) + self.scale)

    state = SimpleNamespace(
        model=_TinyLM(0.1),
        graphstate={"scale": jnp.array(0.1, dtype=jnp.float32)},
        merge=lambda t: _TinyLM(t["scale"]),
    )
    ssd_metrics = ssd_step(state=state, batch=ssd_out, is_training=False)
    assert np.isfinite(float(ssd_metrics.loss))

    # OnPolicyDistillation expects [B, P+C] input_ids, attention_mask, completion_mask.
    opd_inner = _stub_trainer(
        OnPolicyDistillationTrainer,
        args=SimpleNamespace(async_mode=False, max_prompt_length=8, max_completion_length=16),
    )
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=1,
            max_prompt_length=8,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=opd_inner,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="P", observations=("",), rewards=(1.0,)),
        generate_turn_fn=lambda st, ids, txt, t: ([5, 6], "xy", None),
    )
    opd_out, _ = opd_inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)
    assert set(opd_out.keys()) == {"input_ids", "attention_mask", "completion_mask"}
    assert opd_out["completion_mask"].shape == (1, 24)
    assert int(np.sum(opd_out["completion_mask"][:, :8])) == 0
    assert int(np.sum(opd_out["completion_mask"][:, 8:])) == 2


@pytest.mark.parametrize(
    "bad_name",
    ["sdpo", "sdft", "self_distillation", "nash-md", "nash_md", "xpo", "sao", "compaction_rl", "agentic-moshpit"],
)
def test_pr306_f2_incompatible_inner_trainers_rejected_by_config_and_prebuilt(bad_name):
    with pytest.raises(ValueError, match="not supported"):
        AgentFlowConfig(base_trainer_type=bad_name)

    trainer_cls = Registry.get("trainer", bad_name)
    if isinstance(trainer_cls, type):
        stub = _stub_trainer(trainer_cls)
        with pytest.raises(ValueError, match="not supported"):
            AgentFlowTrainer(
                arguments=AgentFlowConfig(max_turns=1),
                trainer=stub,
                environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
            )


# ---------------------------------------------------------------------------
# Finding #3: Multi-host sharding, _all_gather, _decode_prompt_rows, lockstep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trainer_cls",
    [GRPOTrainer, PPOTrainer, OnlineDPOTrainer, GKDTrainer, OnPolicyDistillationTrainer, SeqKDTrainer, SSDTrainer],
)
def test_pr306_f3_all_adapters_route_outputs_through_all_gather(trainer_cls):
    inner = _stub_trainer(trainer_cls)
    inner.arguments.beta = 0.0
    gathered_shapes: list[tuple[int, ...]] = []

    def _spy_gather(arr):
        out = jnp.asarray(arr)
        gathered_shapes.append(tuple(out.shape))
        return out

    inner._all_gather = _spy_gather
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=2,
            max_prompt_length=8,
            max_completion_length=12,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="p", observations=("",), rewards=(1.0,)),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a")], "a", [-0.1]),
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
    assert len(gathered_shapes) >= len(out)


def test_pr306_f3_decode_prompt_rows_uses_process_allgather_for_non_addressable_array(monkeypatch):
    from jax.experimental import multihost_utils

    inner = _stub_trainer(GRPOTrainer)
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, skip_apply_chat_template=True),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        generate_turn_fn=lambda *a: ([ord("x")], "x", [-0.1]),
    )

    real_arr = jnp.asarray([[ord("h"), ord("i")]], dtype=jnp.int32)
    called = {"allgather": 0}

    class _FakeNonAddressableArray:
        is_fully_addressable = False

    monkeypatch.setattr(
        "easydel.trainers.agent_flow_trainer.agent_flow_trainer.jax.Array",
        (_FakeNonAddressableArray, jax.Array),
    )

    def _fake_allgather(arr, tiled=False):
        assert tiled is True
        called["allgather"] += 1
        return np.asarray(real_arr)

    monkeypatch.setattr(multihost_utils, "process_allgather", _fake_allgather)
    rows = flow._decode_prompt_rows(_FakeNonAddressableArray(), None)
    assert called["allgather"] == 1
    assert rows == ["hi"]
    assert np.array_equal(_to_host_numpy(real_arr), np.asarray(real_arr))


def test_pr306_f3_multihost_lockstep_syncs_hosts_deterministically_with_workers_gt_1(monkeypatch):
    """Simulate a 2-host cluster running with max_workers=2 and verify deterministic lockstep rollout."""
    from jax.experimental import multihost_utils

    host_local = threading.local()
    ch_0_to_1: queue.Queue[np.ndarray] = queue.Queue()

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: host_local.pid)

    def _sim_bcast(arr):
        np_arr = np.asarray(arr)
        if host_local.pid == 0:
            ch_0_to_1.put(np_arr.copy())
            return np_arr.copy()
        recv = ch_0_to_1.get(timeout=5.0)
        assert recv.shape == np_arr.shape and recv.dtype == np_arr.dtype
        return recv

    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", _sim_bcast)

    trace_by_host: dict[int, list[tuple[int, list[int]]]] = {0: [], 1: []}
    env_calls_by_host: dict[int, int] = {0: 0, 1: 0}
    results_by_host: dict[int, dict] = {}
    errors: list[BaseException] = []

    def _run_host(pid: int) -> None:
        host_local.pid = pid
        try:
            inner = _stub_trainer(
                GRPOTrainer,
                args=GRPOConfig(num_generations=2, beta=0.0, max_prompt_length=16, max_completion_length=32),
            )

            def _make_env(**kw):
                env_calls_by_host[pid] += 1
                # Different turn counts across prompt tasks to stress termination sync.
                obs = ("o1", "") if kw.get("prompt") == "p0" else ("",)
                rews = (0.2, 0.8) if kw.get("prompt") == "p0" else (1.5,)
                return _ScriptedEnv(prompt=str(kw.get("prompt", "")), observations=obs, rewards=rews)

            def _gen_turn(state, prompt_ids, prompt_text, turn_idx):
                trace_by_host[pid].append((turn_idx, list(prompt_ids)))
                toks = [ord("x"), ord("y")]
                return toks, "xy", [-0.1, -0.2]

            AgentFlowTrainer(
                arguments=AgentFlowConfig(
                    max_turns=2,
                    group_size=2,
                    max_workers=2,
                    max_prompt_length=16,
                    max_completion_length=32,
                    skip_apply_chat_template=True,
                ),
                trainer=inner,
                environment_factory=_make_env,
                generate_turn_fn=_gen_turn,
            )
            out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p0", "p1"]}, True)
            results_by_host[pid] = out
        except BaseException as exc:
            errors.append(exc)

    t0 = threading.Thread(target=_run_host, args=(0,))
    t1 = threading.Thread(target=_run_host, args=(1,))
    t0.start()
    t1.start()
    t0.join(timeout=10.0)
    t1.join(timeout=10.0)
    assert not errors, errors

    # Worker host 1 never instantiates or steps environments, yet executes identical generation calls in lockstep.
    assert env_calls_by_host[0] == 4
    assert env_calls_by_host[1] == 0
    assert trace_by_host[0] == trace_by_host[1]
    assert len(trace_by_host[0]) == 6  # p0 x 2 episodes x 2 turns + p1 x 2 episodes x 1 turn
    for key in results_by_host[0]:
        np.testing.assert_array_equal(np.asarray(results_by_host[0][key]), np.asarray(results_by_host[1][key]))


def test_pr306_f3_multihost_host0_error_aborts_worker_without_deadlock(monkeypatch):
    from jax.experimental import multihost_utils

    host_local = threading.local()
    ch_0_to_1: queue.Queue[np.ndarray] = queue.Queue()
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: host_local.pid)

    def _sim_bcast(arr):
        np_arr = np.asarray(arr)
        if host_local.pid == 0:
            ch_0_to_1.put(np_arr.copy())
            return np_arr.copy()
        return ch_0_to_1.get(timeout=5.0)

    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", _sim_bcast)
    caught: dict[int, BaseException] = {}

    def _run_host(pid: int) -> None:
        host_local.pid = pid
        inner = _stub_trainer(GRPOTrainer)
        AgentFlowTrainer(
            arguments=AgentFlowConfig(max_turns=1, group_size=1, max_workers=2, skip_apply_chat_template=True),
            trainer=inner,
            environment_factory=lambda **kw: (_ for _ in ()).throw(ValueError("boom on host0")),
            generate_turn_fn=lambda *a: ([ord("a")], "a", [-0.1]),
        )
        try:
            inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
        except BaseException as exc:
            caught[pid] = exc

    t0 = threading.Thread(target=_run_host, args=(0,))
    t1 = threading.Thread(target=_run_host, args=(1,))
    t0.start()
    t1.start()
    t0.join(timeout=5.0)
    t1.join(timeout=5.0)
    assert isinstance(caught[0], ValueError)
    assert isinstance(caught[1], RuntimeError)
    assert "boom on host0" in str(caught[1])


# ---------------------------------------------------------------------------
# Finding #11: eSurge HBM leak — release_generation_runtime in sync mode
# ---------------------------------------------------------------------------


def test_pr306_f11_sync_mode_releases_generation_runtime_on_success_and_error():
    released: list[object] = []

    def _make_inner(async_mode: bool):
        inner = _stub_trainer(
            GRPOTrainer,
            args=GRPOConfig(num_generations=1, beta=0.0, async_mode=async_mode, max_prompt_length=8, max_completion_length=8),
        )
        inner.rollout = lambda **kw: SimpleNamespace(
            completion_ids=np.asarray([[ord("x")]]),
            completion_mask=np.asarray([[1]]),
            sampling_per_token_logps=np.asarray([[-0.1]]),
        )
        inner.release_generation_runtime = lambda *, state=None, **kw: released.append(state)
        return inner

    # 1) Sync mode success: releases runtime with exact state.
    inner_sync = _make_inner(async_mode=False)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, skip_apply_chat_template=True),
        trainer=inner_sync,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
    )
    st = SimpleNamespace(step=1)
    inner_sync._preprocess_batch_input(st, {"prompts": ["p"]}, True)
    assert released == [st]

    # 2) Sync mode episode failure (all env_error -> RuntimeError): still releases runtime in finally!
    released.clear()
    inner_err = _make_inner(async_mode=False)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, skip_apply_chat_template=True),
        trainer=inner_err,
        environment_factory=lambda **kw: _ScriptedEnv(fail_step=True),
    )
    with pytest.raises(RuntimeError, match="All 1 AgentFlow episodes failed"):
        inner_err._preprocess_batch_input(st, {"prompts": ["p"]}, True)
    assert released == [st]

    # 3) Async mode: must NOT release generation runtime mid-step.
    released.clear()
    inner_async = _make_inner(async_mode=True)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, async_mode=True, skip_apply_chat_template=True),
        trainer=inner_async,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
    )
    inner_async._preprocess_batch_input(st, {"prompts": ["p"]}, True)
    assert released == []


# ---------------------------------------------------------------------------
# Finding #12: SandboxPool / KiriRuntime lifecycle & no eager unused boot
# ---------------------------------------------------------------------------


def test_pr306_f12_agent_flow_runtime_lifecycle_and_no_unused_boot(monkeypatch):
    import easydel.environments as env_mod

    created: list[SimpleNamespace] = []

    def _make_rt(*args, **kwargs):
        rt = SimpleNamespace(closed=0, args=args, kwargs=kwargs)
        rt.close = lambda: setattr(rt, "closed", rt.closed + 1)
        created.append(rt)
        return rt

    monkeypatch.setattr(env_mod, "SandboxPool", _make_rt)
    monkeypatch.setattr(env_mod, "KiriRuntime", _make_rt)
    monkeypatch.setattr(env_mod, "make_environment_factory", lambda *a, **kw: (lambda **ek: _ScriptedEnv()))

    # Callable factory, env instance, and mapping with explicit runtime must NOT boot a runtime.
    explicit_rt = SimpleNamespace(close=lambda: None)
    for spec in (lambda **kw: _ScriptedEnv(), _ScriptedEnv(), {"name": "code_judge", "runtime": explicit_rt}):
        inner = _stub_trainer(GRPOTrainer)
        AgentFlowTrainer(arguments=AgentFlowConfig(max_turns=1), trainer=inner, environment=spec)
    assert created == []

    # String spec boots 1 runtime, closes on _teardown_resources() / close() / finish() (idempotently once).
    inner = _stub_trainer(GRPOTrainer)
    teardown_ran = {"n": 0}
    inner._teardown_resources = lambda: teardown_ran.__setitem__("n", teardown_ran["n"] + 1)
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, environment_endpoints=["unix:///tmp/a.sock"]),
        trainer=inner,
        environment="code_judge",
    )
    assert len(created) == 1
    assert created[0].closed == 0
    inner._teardown_resources()
    assert teardown_ran["n"] == 1
    assert created[0].closed == 1
    flow.close()
    assert created[0].closed == 1

    # Factory failure closes newly booted runtime immediately before re-raising.
    monkeypatch.setattr(
        env_mod,
        "make_environment_factory",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad env spec")),
    )
    with pytest.raises(RuntimeError, match="bad env spec"):
        AgentFlowTrainer(
            arguments=AgentFlowConfig(max_turns=1),
            trainer=_stub_trainer(GRPOTrainer),
            environment="bad_env",
        )
    assert len(created) == 2
    assert created[1].closed == 1


def test_pr306_f12_async_grpo_and_elarge_runtime_lifecycle_and_no_unused_boot(monkeypatch):
    import easydel.environments as env_mod
    from easydel.trainers.async_grpo_trainer import AsyncGRPOConfig, AsyncGRPOTrainer
    from easydel.trainers.group_relative_policy_optimization import GRPOTrainer as _GRPO

    created: list[SimpleNamespace] = []

    def _make_rt(*args, **kwargs):
        rt = SimpleNamespace(closed=0)
        rt.close = lambda: setattr(rt, "closed", rt.closed + 1)
        created.append(rt)
        return rt

    monkeypatch.setattr(env_mod, "SandboxPool", _make_rt)
    monkeypatch.setattr(env_mod, "KiriRuntime", _make_rt)
    monkeypatch.setattr(env_mod.AutoEnvironment, "as_factory", staticmethod(lambda spec, runtime=None: (lambda **k: None)))

    # 1) AsyncGRPO with callable environment does NOT boot runtime.
    monkeypatch.setattr(_GRPO, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(_GRPO, "_teardown_resources", lambda self: None)
    t_call = AsyncGRPOTrainer(
        arguments=AsyncGRPOConfig(environment_endpoints=["unix:///tmp/s.sock"]),
        model=object(),
        environment=lambda **kw: _ScriptedEnv(),
    )
    assert created == []
    t_call._teardown_resources()

    # 2) AsyncGRPO with string environment stores runtime and closes it on _teardown_resources.
    t_str = AsyncGRPOTrainer(
        arguments=AsyncGRPOConfig(environment_endpoints=["unix:///tmp/s.sock"]),
        model=object(),
        environment="code_judge",
    )
    assert len(created) == 1
    assert created[0].closed == 0
    t_str._teardown_resources()
    assert created[0].closed == 1

    # 3) AsyncGRPO closes runtime if super().__init__ fails.
    monkeypatch.setattr(_GRPO, "__init__", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("init boom")))
    with pytest.raises(RuntimeError, match="init boom"):
        AsyncGRPOTrainer(arguments=AsyncGRPOConfig(), model=object(), environment="code_judge")
    assert len(created) == 2
    assert created[1].closed == 1

    # 4) eLarge GRPO with explicit environment_factory + reward_funcs=None must NOT boot runtime.
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy"},
        "trainer": {"trainer_type": "grpo", "environment": "code_judge", "environment_endpoints": ["unix:///tmp/x"]},
    }
    elm._model = object()
    elm._state = None
    elm._tokenizer = _DummyTokenizer()
    elm.build_reward_model = lambda: None

    class _StubCtor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    created_before = len(created)
    elm.build_trainer(trainer_class=_StubCtor, environment_factory=lambda **kw: _ScriptedEnv())
    assert len(created) == created_before

    # 5) eLarge GRPO without environment_factory boots runtime and hooks _teardown_resources.
    built = elm.build_trainer(trainer_class=_StubCtor)
    assert len(created) == created_before + 1
    assert created[-1].closed == 0
    built._teardown_resources()
    assert created[-1].closed == 1

    # 6) eLarge GRPO closes runtime if trainer_class raises during init.
    class _FailCtor:
        def __init__(self, **kwargs):
            raise RuntimeError("trainer ctor failed")

    with pytest.raises(RuntimeError, match="trainer ctor failed"):
        elm.build_trainer(trainer_class=_FailCtor)
    assert len(created) == created_before + 2
    assert created[-1].closed == 1


# ---------------------------------------------------------------------------
# Finding #13: PPO / VAPO constructor, old_values, KL shaping, GAE, token counts
# ---------------------------------------------------------------------------


def test_pr306_f13_ppo_and_vapo_constructor_gets_placeholder_reward(monkeypatch):
    for base_type, cls in (("ppo", PPOTrainer), ("vapo", VAPOTrainer)):
        captured = _install_init_recorder(monkeypatch, cls)
        AgentFlowTrainer(
            arguments=AgentFlowConfig(base_trainer_type=base_type, max_turns=1),
            model=object(),
            processing_class=_DummyTokenizer(),
            environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        )
        assert captured["reward_funcs"] is agent_flow_supplied_reward


def test_pr306_f13_ppo_vapo_scores_old_values_applies_kl_gae_and_vapo_token_counts():
    vapo_args = VAPOConfig(
        kl_coef=0.1,
        kl_estimator="k1",
        gamma=1.0,
        lam=1.0,
        gae_lambda_critic=1.0,
        length_adaptive_gae=False,
        whiten_rewards=False,
        whiten_advantages=False,
        token_level_loss=True,
        positive_nll_coef=0.1,
        positive_reward_threshold=0.0,
        max_prompt_length=8,
        max_completion_length=24,
    )
    inner = _stub_trainer(VAPOTrainer, args=vapo_args)
    scorer_calls = {"n": 0}
    ref_calls = {"n": 0}

    def _score(state, input_ids, attention_mask):
        scorer_calls["n"] += 1
        b = input_ids.shape[0]
        return (
            jnp.full((b, 24), -0.3, dtype=jnp.float32),
            jnp.full((b, 24), 0.5, dtype=jnp.float32),
        )

    inner._score_rollout_policy_and_values = _score
    inner.ref_state = SimpleNamespace(graphstate=jnp.array(0.0), graphother=jnp.array(0.0))

    def _ref_logps(gs, go, ids, mask):
        ref_calls["n"] += 1
        return jnp.full((ids.shape[0], 24), -0.5, dtype=jnp.float32)

    inner.compute_refmodel_logps = _ref_logps

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            group_size=1,
            max_prompt_length=8,
            max_completion_length=24,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="P", observations=("O1", ""), rewards=(0.0, 2.0)),
        # Even when sampling logprobs ARE present, old_values must still be scored!
        generate_turn_fn=lambda st, ids, txt, t: ([1, 2], "ab", [-0.2, -0.2]),
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)

    assert scorer_calls["n"] == 1
    assert ref_calls["n"] == 1
    assert "ref_logps" not in out
    np.testing.assert_allclose(np.asarray(out["old_values"]), 0.5)
    assert float(out["num_items_in_batch"]) == 4.0
    assert float(out["num_positive_tokens"]) == 4.0

    # Verify KL shaping + multi-turn GAE across the observation gap:
    # Each of the 4 action tokens has old_lp=-0.2, ref_lp=-0.5 -> k1 KL = old_lp - ref_lp = +0.3
    # -> non_score_reward = -0.1 * 0.3 = -0.03 per action token; terminal action gets +2.0.
    # With gamma=1, lam=1, V=0.5:
    # returns at packed positions [0, 1, 2, 3] = [2.0 - 4*0.03, 2.0 - 3*0.03, 2.0 - 2*0.03, 2.0 - 0.03]
    #                                         = [1.88, 1.91, 1.94, 1.97].
    mask = np.asarray(out["completion_mask"][0])
    active_idx = np.flatnonzero(mask > 0)
    obs_idx = np.flatnonzero(mask == 0)
    assert active_idx.size == 4
    np.testing.assert_allclose(np.asarray(out["returns"][0])[active_idx], [1.88, 1.91, 1.94, 1.97], atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["advantages"][0])[active_idx], [1.38, 1.41, 1.44, 1.47], atol=1e-5)
    # Observation and pad tokens must have exact 0.0 advantages and returns.
    np.testing.assert_array_equal(np.asarray(out["advantages"][0])[obs_idx], 0.0)
    np.testing.assert_array_equal(np.asarray(out["returns"][0])[obs_idx], 0.0)

    # Verify adapted batch runs cleanly through real ppo_step with VAPO knobs!
    mesh = _make_1d_mesh()

    class _TinyValueLM:
        def __init__(self, w):
            self.w = w
            self.mesh = mesh

        def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
            b, s = input_ids.shape
            hidden = jnp.ones((b, s, 4), dtype=jnp.float32) * self.w
            logits = jnp.zeros((b, s, 128), dtype=jnp.float32).at[:, :, :4].set(self.w)
            return SimpleNamespace(logits=logits, last_hidden_state=hidden)

        def value_head(self, hidden):
            return jnp.mean(hidden, axis=-1, keepdims=True)

    state = SimpleNamespace(
        model=_TinyValueLM(jnp.array(0.2, dtype=jnp.float32)),
        graphstate={"w": jnp.array(0.2, dtype=jnp.float32)},
        step=jnp.asarray(0, dtype=jnp.int32),
        merge=lambda t: _TinyValueLM(t["w"]),
    )
    metrics = ppo_step(
        state=state,
        batch=out,
        prompt_length=8,
        cliprange=0.2,
        vf_coef=0.5,
        cliprange_value=0.2,
        entropy_coef=0.0,
        loss_config=None,
        learning_rate_fn=lambda _: 1e-3,
        partition_spec=None,
        gradient_accumulation_steps=1,
        is_training=False,
        logprob_vocab_chunk_size=None,
        token_level_loss=True,
        positive_nll_coef=0.1,
    )
    assert np.isfinite(float(metrics.loss))


# ---------------------------------------------------------------------------
# Finding #14: OnlineDPO max_completion_length / max_length forwarding
# ---------------------------------------------------------------------------


def test_pr306_f14_online_dpo_preserves_completion_and_total_lengths_direct_and_elarge(monkeypatch):
    captured: dict[str, object] = {}
    orig_init = OnlineDPOTrainer.__init__

    @functools.wraps(orig_init)
    def _rec(self, **kwargs):
        captured.clear()
        captured.update(kwargs)
        self.arguments = kwargs["arguments"]
        self.processing_class = kwargs.get("processing_class")
        self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    monkeypatch.setattr(OnlineDPOTrainer, "__init__", _rec)

    # 1) Direct AgentFlowTrainer without base_config: must NOT clobber max_completion_length to 64 or max_length to 512.
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            base_trainer_type="online_dpo",
            max_prompt_length=400,
            max_completion_length=300,
        ),
        model=object(),
        processing_class=_DummyTokenizer(),
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
    )
    inner_args = captured["arguments"]
    assert type(inner_args) is OnlineDPOConfig
    assert inner_args.max_new_tokens == 300
    assert inner_args.max_completion_length == 300
    assert inner_args.max_length == 700

    # 2) eLarge YAML without max_new_tokens or max_length.
    def _elm(tcfg):
        elm = object.__new__(eLargeModel)
        elm._config = {"model": {"name_or_path": "dummy"}, "trainer": dict(tcfg)}
        elm._model = object()
        elm._state = None
        elm._tokenizer = _DummyTokenizer()
        elm.build_reward_model = lambda: None
        elm.build_reference_model = lambda: None
        return elm

    _elm(
        {
            "trainer_type": "agent_flow",
            "base_trainer_type": "online_dpo",
            "environment": "code_judge",
            "environment_backend": "local",
            "max_prompt_length": 400,
            "max_completion_length": 300,
        }
    ).build_trainer()
    inner_args = captured["arguments"]
    assert inner_args.max_new_tokens == 300
    assert inner_args.max_completion_length == 300
    assert inner_args.max_length == 700

    # 3) eLarge YAML with explicit max_new_tokens > 512 - max_prompt_length bumps max_length automatically.
    _elm(
        {
            "trainer_type": "agent_flow",
            "base_trainer_type": "online_dpo",
            "environment": "code_judge",
            "environment_backend": "local",
            "max_prompt_length": 400,
            "max_new_tokens": 250,
        }
    ).build_trainer()
    inner_args = captured["arguments"]
    assert inner_args.max_new_tokens == 250
    assert inner_args.max_completion_length == 250
    assert inner_args.max_length == 650

    # 4) Explicit max_length in YAML is respected verbatim.
    _elm(
        {
            "trainer_type": "agent_flow",
            "base_trainer_type": "online_dpo",
            "environment": "code_judge",
            "environment_backend": "local",
            "max_prompt_length": 400,
            "max_completion_length": 300,
            "max_length": 1024,
        }
    ).build_trainer()
    assert captured["arguments"].max_length == 1024


# ---------------------------------------------------------------------------
# Finding #15: AgentFlowTrainer.eval() / evaluate() delegation
# ---------------------------------------------------------------------------


def test_pr306_f15_eval_and_evaluate_delegate_to_inner_eval():
    inner = _stub_trainer(GRPOTrainer)
    seen: list[object] = []
    inner.eval = lambda model_state: (seen.append(model_state), [{"eval/loss": 0.25}])[1]
    assert not hasattr(inner, "evaluate")

    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
    )
    st = SimpleNamespace(step=5)
    assert flow.eval(st) == [{"eval/loss": 0.25}]
    assert flow.evaluate(st) == [{"eval/loss": 0.25}]
    assert seen == [st, st]


# ---------------------------------------------------------------------------
# Finding #16: DPO _fn.py pair_valid math under GA, aot*, sft + 878f9f1 checks
# ---------------------------------------------------------------------------


class _RowPosModel:
    def __init__(self, bias, mesh, aux_loss=None):
        self.bias = bias
        self.mesh = mesh
        self._aux_loss = aux_loss

    def forward(self, input_ids, attention_mask, **kwargs):
        _b, s = input_ids.shape
        logits = self.bias[None, :s, :] + 0.05 * input_ids[:, :, None].astype(jnp.float32)
        return SimpleNamespace(logits=logits, last_hidden_state=None, aux_loss=self._aux_loss)

    __call__ = forward


class _PerRowBiasModel:
    def __init__(self, bias):
        self.bias = bias

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(logits=self.bias, last_hidden_state=None, aux_loss=None)

    __call__ = forward


def _make_dpo_batch_4pairs(valid_flags: tuple[bool, bool, bool, bool], *, chosen_lens=(3, 4, 2, 5)):
    # Prompt length 4, completion length 6 -> total length 10.
    prompt_ids = np.asarray([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]], dtype=np.int32)
    prompt_mask = np.ones_like(prompt_ids, dtype=np.int32)
    chosen_ids = np.asarray(
        [[10, 11, 12, 0, 0, 0], [13, 14, 15, 16, 0, 0], [17, 18, 0, 0, 0, 0], [19, 20, 21, 22, 23, 0]],
        dtype=np.int32,
    )
    rejected_ids = np.asarray(
        [[25, 26, 27, 0, 0, 0], [28, 29, 30, 0, 0, 0], [31, 32, 33, 0, 0, 0], [34, 35, 36, 0, 0, 0]],
        dtype=np.int32,
    )
    chosen_attn = np.zeros((4, 6), dtype=np.int32)
    chosen_loss = np.zeros((4, 6), dtype=np.int32)
    rejected_attn = np.zeros((4, 6), dtype=np.int32)
    rejected_loss = np.zeros((4, 6), dtype=np.int32)
    for i in range(4):
        cl = chosen_lens[i]
        chosen_attn[i, :cl] = 1
        rejected_attn[i, :3] = 1
        if valid_flags[i]:
            chosen_loss[i, :cl] = 1
            rejected_loss[i, :3] = 1
    return {
        "prompt_input_ids": jnp.asarray(prompt_ids),
        "prompt_attention_mask": jnp.asarray(prompt_mask),
        "chosen_input_ids": jnp.asarray(chosen_ids),
        "chosen_attention_mask": jnp.asarray(chosen_attn),
        "chosen_loss_mask": jnp.asarray(chosen_loss),
        "rejected_input_ids": jnp.asarray(rejected_ids),
        "rejected_attention_mask": jnp.asarray(rejected_attn),
        "rejected_loss_mask": jnp.asarray(rejected_loss),
    }


def _make_dpo_state(bias_init, mesh, aux_loss=None):
    tx = optax.sgd(learning_rate=0.5)
    tree = {"bias": bias_init}
    opt_state = tx.init(tree)

    class _FakeState:
        def __init__(self, graphstate, opt_state, step):
            self.graphstate = graphstate
            self.opt_state = opt_state
            self.step = step
            self.tx = tx
            self.model = _RowPosModel(graphstate["bias"], mesh, aux_loss=aux_loss)

        def merge(self, tree):
            return _RowPosModel(tree["bias"], mesh, aux_loss=aux_loss)

        def apply_gradients(self, grads):
            updates, new_opt = self.tx.update(grads, self.opt_state, self.graphstate)
            new_tree = optax.apply_updates(self.graphstate, updates)
            return _FakeState(new_tree, new_opt, self.step + 1)

    return _FakeState(tree, opt_state, jnp.asarray(0, dtype=jnp.int32))


@pytest.mark.parametrize(
    ("valid_flags", "loss_type"),
    [
        # Microbatch 0 has 2 valid pairs, microbatch 1 has 1 valid + 1 placeholder -> uniform GA would overweight pair 2.
        ((True, True, True, False), "sigmoid"),
        # Microbatch 0 has 2 valid pairs, microbatch 1 has 0 valid pairs -> uniform GA would dilute by 2x!
        ((True, True, False, False), "sigmoid"),
        # All invalid pairs -> both GA=1 and GA=2 must produce exact 0 loss and 0 update!
        ((False, False, False, False), "sigmoid"),
        # Pure SFT with unequal valid chosen lengths (3, 4, 2) across microbatches -> exact token weighting!
        ((True, True, True, False), "sft"),
    ],
)
def test_pr306_f16_dpo_ga_valid_pair_weighting_matches_ga1_exact(valid_flags, loss_type):
    mesh = _make_1d_mesh()
    batch = _make_dpo_batch_4pairs(valid_flags)
    bias0 = jax.random.normal(jax.random.PRNGKey(11), (10, 40), dtype=jnp.float32)
    fwd = functools.partial(
        concatenated_forward,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=None,
        truncation_mode="keep_end",
        return_mean_logits=False,
    )
    st1, m1 = training_step(
        state=_make_dpo_state(bias0, mesh),
        batch=batch,
        reference_state=None,
        learning_rate_fn=lambda _: 0.5,
        concatenated_forward=fwd,
        beta=0.1,
        loss_type=loss_type,
        reference_free=True,
        gradient_accumulation_steps=1,
    )
    st2, m2 = training_step(
        state=_make_dpo_state(bias0, mesh),
        batch=batch,
        reference_state=None,
        learning_rate_fn=lambda _: 0.5,
        concatenated_forward=fwd,
        beta=0.1,
        loss_type=loss_type,
        reference_free=True,
        gradient_accumulation_steps=2,
    )
    np.testing.assert_allclose(float(m2.loss), float(m1.loss), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(st2.graphstate["bias"]),
        np.asarray(st1.graphstate["bias"]),
        rtol=1e-5,
        atol=1e-6,
    )
    if not any(valid_flags):
        assert float(m2.loss) == 0.0
        np.testing.assert_array_equal(np.asarray(st2.graphstate["bias"]), np.asarray(bias0))


@pytest.mark.parametrize("loss_type", ["aot", "aot_pair", "aot_unpaired"])
@pytest.mark.parametrize("f_div", ["reverse_kl", "js_divergence"])
def test_pr306_f16_dpo_aot_excludes_placeholders_and_aligns_quantiles(loss_type, f_div):
    # 3 pairs: row 0 valid, row 1 placeholder (0.0), row 2 valid.
    chosen = jnp.asarray([-4.0, 0.0, -1.5], dtype=jnp.float32)
    rejected = jnp.asarray([-2.0, 0.0, -3.5], dtype=jnp.float32)
    ref_chosen = jnp.asarray([-3.0, 0.0, -2.0], dtype=jnp.float32)
    ref_rejected = jnp.asarray([-2.5, 0.0, -3.0], dtype=jnp.float32)
    pair_valid = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)

    def _loss_with_placeholder(c, r):
        losses = compute_dpo_losses(
            c,
            r,
            ref_chosen,
            ref_rejected,
            beta=0.2,
            label_smoothing=0.05,
            loss_type=loss_type,
            f_divergence_type=f_div,
            pair_valid=pair_valid,
        )
        return _reduce_pair_losses(losses, {"pair_valid": pair_valid})

    idx = jnp.asarray([0, 2])

    def _loss_valid_only(c, r):
        losses = compute_dpo_losses(
            c,
            r,
            ref_chosen[idx],
            ref_rejected[idx],
            beta=0.2,
            label_smoothing=0.05,
            loss_type=loss_type,
            f_divergence_type=f_div,
        )
        return losses.mean()

    val_ph, (gc_ph, gr_ph) = jax.value_and_grad(_loss_with_placeholder, argnums=(0, 1))(chosen, rejected)
    val_only, (gc_only, gr_only) = jax.value_and_grad(_loss_valid_only, argnums=(0, 1))(chosen[idx], rejected[idx])

    np.testing.assert_allclose(float(val_ph), float(val_only), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gc_ph[idx]), np.asarray(gc_only), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gr_ph[idx]), np.asarray(gr_only), rtol=1e-5, atol=1e-6)
    assert float(gc_ph[1]) == 0.0
    assert float(gr_ph[1]) == 0.0

    # All-invalid AOT batch must yield exact 0.0 loss and finite 0.0 grads (no NaN).
    zeros_valid = jnp.zeros((3,), dtype=jnp.float32)

    def _all_invalid(c, r):
        losses = compute_dpo_losses(
            c,
            r,
            ref_chosen,
            ref_rejected,
            beta=0.2,
            label_smoothing=0.0,
            loss_type=loss_type,
            f_divergence_type=f_div,
            pair_valid=zeros_valid,
        )
        return _reduce_pair_losses(losses, {"pair_valid": zeros_valid})

    val0, (gc0, gr0) = jax.value_and_grad(_all_invalid, argnums=(0, 1))(chosen, rejected)
    assert float(val0) == 0.0
    np.testing.assert_array_equal(np.asarray(gc0), 0.0)
    np.testing.assert_array_equal(np.asarray(gr0), 0.0)


def test_pr306_f16_dpo_sft_excludes_placeholders_and_half_empty_invalid_pairs():
    # Row 0: valid (len 4), Row 1: placeholder (len 1 clamped, pair_valid=0),
    # Row 2: half-empty invalid (chosen non-empty len 6, rejected empty -> pair_valid=0), Row 3: valid (len 2).
    chosen = jnp.asarray([-8.0, 0.0, -50.0, -3.0], dtype=jnp.float32)
    rejected = jnp.asarray([-5.0, 0.0, 0.0, -4.0], dtype=jnp.float32)
    chosen_lens = jnp.asarray([4.0, 1.0, 6.0, 2.0], dtype=jnp.float32)
    rejected_lens = jnp.asarray([3.0, 1.0, 1.0, 3.0], dtype=jnp.float32)
    pair_valid = jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)

    def _sft_masked(c):
        losses = compute_dpo_losses(
            c,
            rejected,
            jnp.zeros_like(c),
            jnp.zeros_like(rejected),
            beta=0.1,
            label_smoothing=0.0,
            loss_type="sft",
            chosen_lengths=chosen_lens,
            rejected_lengths=rejected_lens,
            pair_valid=pair_valid,
        )
        return _reduce_pair_losses(losses, {"pair_valid": pair_valid})

    val, grad = jax.value_and_grad(_sft_masked)(chosen)
    # Exact valid-only SFT: -(-8.0 + -3.0) / (4.0 + 2.0) = 11.0 / 6.0
    np.testing.assert_allclose(float(val), 11.0 / 6.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(grad), [-1.0 / 6.0, 0.0, 0.0, -1.0 / 6.0], rtol=1e-6)


def test_pr306_f16_dpo_preserves_878f9f1_last_col_mask_clear_and_aux_loss_and_eval():
    # Verify jnp.roll last-col mask clear when keep_end truncates the entire prompt.
    batch = {
        "prompt_input_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.asarray([[1, 1]], dtype=jnp.int32),
        "chosen_input_ids": jnp.asarray([[3, 4, 5]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
        "chosen_loss_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
        "rejected_input_ids": jnp.asarray([[6, 7, 8]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
        "rejected_loss_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
    }
    bias = jax.random.normal(jax.random.PRNGKey(3), (2, 3, 16), dtype=jnp.float32)
    out = concatenated_forward(
        _PerRowBiasModel(bias),
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=3,  # keep_end drops the entire prompt, so loss_mask[:, 0] == True before roll!
        truncation_mode="keep_end",
    )
    # After roll(-1) + .at[:, -1].set(False), only 2 positions remain scored (not 3 wrapped onto last col!).
    np.testing.assert_array_equal(np.asarray(out["chosen_lengths"]), [2.0])
    np.testing.assert_array_equal(np.asarray(out["rejected_lengths"]), [2.0])

    # Verify evaluation_step also forwards pair_valid into compute_dpo_losses (e.g. for sft).
    mesh = _make_1d_mesh()
    eval_batch = _make_dpo_batch_4pairs((True, False, False, True), chosen_lens=(4, 3, 5, 2))
    bias_e = jax.random.normal(jax.random.PRNGKey(9), (10, 40), dtype=jnp.float32)
    fwd = functools.partial(
        concatenated_forward,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        return_mean_logits=False,
    )
    m_full = evaluation_step(
        state=_make_dpo_state(bias_e, mesh),
        batch=eval_batch,
        reference_state=None,
        concatenated_forward=fwd,
        beta=0.1,
        loss_type="sft",
        reference_free=True,
    )
    valid_only_batch = {k: v[jnp.asarray([0, 3])] for k, v in eval_batch.items()}
    m_only = evaluation_step(
        state=_make_dpo_state(bias_e, mesh),
        batch=valid_only_batch,
        reference_state=None,
        concatenated_forward=fwd,
        beta=0.1,
        loss_type="sft",
        reference_free=True,
    )
    np.testing.assert_allclose(float(m_full.loss), float(m_only.loss), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Finding #25 (AgentFlow part): env_error zeroes difficulty_weights in GRPO
# ---------------------------------------------------------------------------


def test_pr306_f25_agent_flow_env_error_zeros_difficulty_weights_no_dilution():
    inner = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=2, beta=0.0, max_prompt_length=8, max_completion_length=8),
    )
    n = {"i": 0}

    def _make_env(**kw):
        n["i"] += 1
        return _ScriptedEnv(fail_step=(n["i"] == 1), observations=("",), rewards=(1.0,))

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=2,
            max_workers=1,
            max_prompt_length=8,
            max_completion_length=8,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=_make_env,
        generate_turn_fn=lambda st, ids, txt, t: ([1, 2], "ab", [-0.5, -0.5]),
    )
    batch, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
    np.testing.assert_array_equal(np.asarray(batch["difficulty_weights"]), [0.0, 1.0])
    batch["advantages"] = jnp.asarray([0.0, 1.0], dtype=jnp.float32)

    valid_only = {k: (v[1:2] if hasattr(v, "ndim") and v.ndim >= 1 else v) for k, v in batch.items()}
    valid_only["num_items_in_batch"] = jnp.sum(valid_only["completion_mask"])

    class _TinyLM:
        mesh = None

        def __init__(self, scale):
            self.scale = scale

        def __call__(self, input_ids, attention_mask=None):
            b, s = input_ids.shape
            logits = jnp.zeros((b, s, 64), dtype=jnp.float32).at[:, :, 1:3].set(self.scale)
            return SimpleNamespace(logits=logits)

    def _loss(scale, b):
        m = grpo_step(
            state=SimpleNamespace(
                model=_TinyLM(scale),
                graphstate=scale,
                merge=lambda s: _TinyLM(s),
            ),
            batch=b,
            num_generations=1,
            beta=0.0,
            loss_config=None,
            learning_rate_fn=lambda _: 1e-4,
            partition_spec=None,
            gradient_accumulation_steps=1,
            is_training=False,
            loss_type="grpo",
        )
        return m.loss

    l_both, g_both = jax.value_and_grad(_loss)(jnp.array(0.3, dtype=jnp.float32), batch)
    l_only, g_only = jax.value_and_grad(_loss)(jnp.array(0.3, dtype=jnp.float32), valid_only)
    np.testing.assert_allclose(float(l_both), float(l_only), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(g_both), float(g_only), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Finding #32: Multi-EOS, ProcessorMixin, empty obs, prompt truncation TITO
# ---------------------------------------------------------------------------


def test_pr306_f32_multi_eos_processor_empty_obs_and_prompt_truncation_tito():
    # 1) Multi-EOS list [2, 3] behind ProcessorMixin + empty observation + small obs_cap <= 24.
    proc = _ProcessorWrapper(_StrictAlternatingTokenizer())
    inner = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=1, beta=0.0, max_prompt_length=64, max_completion_length=128),
        tok=proc,
    )
    seen_prompts: list[list[int]] = []

    def _gen_turn(state, prompt_ids, prompt_text, turn_idx):
        seen_prompts.append(list(prompt_ids))
        # Turn 0 ends with secondary EOS id 2 (from list [2, 3]); turn 1 ends with primary EOS id 3.
        eos = 2 if turn_idx == 0 else 3
        return [ord("a"), eos], "a", [-0.1, -0.1]

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=3,
            group_size=1,
            max_prompt_length=64,
            max_completion_length=128,
            max_obs_tokens_per_turn=16,  # <= 24 forces raw_limit=0 -> empty raw_obs on non-terminal turn
            skip_apply_chat_template=False,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(
            prompt="P",
            observations=("   ", "long observation text clipped to empty", ""),
            rewards=(0.1, 0.2, 0.3),
        ),
        generate_turn_fn=_gen_turn,
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)
    assert len(seen_prompts) == 3
    # Every non-terminal empty observation must still append turn-closing + <user><E><assistant> header!
    turn1_text = proc.decode(seen_prompts[1], skip_special_tokens=False)
    assert "<user><E><assistant>" in turn1_text
    assert int(np.sum(out["completion_mask"])) == 6

    # 2) Long prompt exceeding max_prompt_length is left-truncated BEFORE turn 0 for TITO alignment.
    inner_clip = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=1, beta=0.0, max_prompt_length=6, max_completion_length=16),
    )
    turn0_ids_seen: list[list[int]] = []
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=1,
            max_prompt_length=6,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=inner_clip,
        environment_factory=lambda **kw: _ScriptedEnv(prompt="0123456789", observations=("",), rewards=(1.0,)),
        generate_turn_fn=lambda st, ids, txt, t: (turn0_ids_seen.append(list(ids)), ([ord("z")], "z", [-0.2]))[1],
    )
    out_clip, metrics_clip = inner_clip._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["0123456789"]}, True)
    expected_suffix = [ord(c) for c in "456789"]
    assert turn0_ids_seen == [expected_suffix]
    assert np.asarray(out_clip["prompt_ids"][0]).tolist() == expected_suffix
    assert metrics_clip["agent_flow/clipped_episodes"] == 1.0


# ---------------------------------------------------------------------------
# Finding #33: Post-init async_mode validation, eSurge flags, force_old_logps
# ---------------------------------------------------------------------------


def test_pr306_f33_async_mode_guard_esurge_flags_and_force_old_logps():
    # 1) Trainer without supports_async_rollout=True must reject post-init async_mode=True.
    class _NoAsyncTrainer:
        supports_async_rollout = False

        def __init__(self):
            self.arguments = SimpleNamespace(async_mode=False)
            self.processing_class = _DummyTokenizer()
            self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    with pytest.raises(ValueError, match="does not support `async_mode=True`"):
        AgentFlowTrainer(
            arguments=AgentFlowConfig(async_mode=True, max_turns=1),
            trainer=_NoAsyncTrainer(),
            environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        )

    # 2) Async-capable trainer gets esurge_async_scheduling=True and esurge_overlap_execution=True enforced.
    inner = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(
            num_generations=1,
            beta=0.0,
            async_mode=False,
            max_prompt_length=8,
            max_completion_length=12,
        ),
    )
    assert inner.arguments.esurge_async_scheduling is not True
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            async_mode=True,
            max_turns=1,
            group_size=1,
            max_prompt_length=8,
            max_completion_length=12,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a")], "a", None),
    )
    assert inner.arguments.async_mode is True
    assert inner.arguments.esurge_async_scheduling is True
    assert inner.arguments.esurge_overlap_execution is True

    # 3) _async_grpo_force_old_logps scores old_per_token_logps via compute_state_logps when lp_arr is None.
    inner._async_grpo_force_old_logps = True
    state_logp_calls = {"n": 0}

    def _compute_state_logps(state, full_ids, full_mask, model_kwargs, *, prompt_length, logprob_vocab_chunk_size=None):
        state_logp_calls["n"] += 1
        assert isinstance(model_kwargs, dict)
        assert prompt_length == 8
        return jnp.full((full_ids.shape[0], full_ids.shape[1] - prompt_length), -0.42, dtype=jnp.float32)

    inner.compute_state_logps = _compute_state_logps
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=1), {"prompts": ["hi"]}, True)
    assert state_logp_calls["n"] == 1
    assert "old_per_token_logps" in out
    np.testing.assert_allclose(np.asarray(out["old_per_token_logps"]), -0.42)


# ---------------------------------------------------------------------------
# Finding #34: eLarge forwarding (gamma, teacher_model, runtime overrides, aliases)
# ---------------------------------------------------------------------------


def test_pr306_f34_elarge_forwards_gamma_teacher_model_runtime_overrides_and_aliases(monkeypatch):
    teacher_sentinel = object()

    def _elm(tcfg):
        elm = object.__new__(eLargeModel)
        elm._config = {"model": {"name_or_path": "dummy"}, "trainer": dict(tcfg)}
        elm._model = object()
        elm._state = None
        elm._tokenizer = _DummyTokenizer()
        elm.build_reward_model = lambda: None
        elm.build_reference_model = lambda: None
        elm.build_teacher_model = lambda: teacher_sentinel
        return elm

    # 1) YAML gamma reaches PPOConfig / VAPOConfig.
    for bt, cls in (("ppo", PPOTrainer), ("vapo", VAPOTrainer)):
        captured = _install_init_recorder(monkeypatch, cls)
        _elm(
            {
                "trainer_type": "agent_flow",
                "base_trainer_type": bt,
                "environment": "code_judge",
                "environment_backend": "local",
                "gamma": 0.97,
            }
        ).build_trainer()
        assert captured["arguments"].gamma == pytest.approx(0.97)
        assert captured["reward_funcs"] is agent_flow_supplied_reward

    # 2) Distillation inners build & forward teacher_model (and bind student_model where declared).
    for bt, cls in (
        ("gkd", GKDTrainer),
        ("on_policy_distillation", OnPolicyDistillationTrainer),
        ("seq_kd", SeqKDTrainer),
    ):
        captured = _install_init_recorder(monkeypatch, cls)
        _elm(
            {
                "trainer_type": "agent_flow",
                "base_trainer_type": bt,
                "environment": "code_judge",
                "environment_backend": "local",
            }
        ).build_trainer()
        assert captured["teacher_model"] is teacher_sentinel

    # 3) Runtime inner-config kwargs (e.g. beta=0.33) route to base_config, not inner trainer __init__.
    captured = {}
    orig_grpo = GRPOTrainer.__init__

    @functools.wraps(orig_grpo)
    def _rec_grpo(self, **kwargs):
        captured.update(kwargs)
        self.arguments = kwargs["arguments"]
        self.processing_class = kwargs.get("processing_class")
        self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    monkeypatch.setattr(GRPOTrainer, "__init__", _rec_grpo)
    _elm(
        {
            "trainer_type": "agent_flow",
            "base_trainer_type": "reinforce++",  # alias normalized cleanly!
            "environment": "code_judge",
            "environment_backend": "local",
        }
    ).build_trainer(beta=0.33)
    assert "beta" not in captured
    assert captured["arguments"].beta == pytest.approx(0.33)

    # 4) AgentFlowConfig normalizes _TRAINER_TYPE_ALIASES.
    assert AgentFlowConfig(base_trainer_type="reinforce++").base_trainer_type == "reinforce_plus_plus"
    assert AgentFlowConfig(base_trainer_type="online-dpo").base_trainer_type == "online_dpo"
    assert AgentFlowConfig(base_trainer_type="async-grpo").base_trainer_type == "async_grpo"
    assert AgentFlowConfig(base_trainer_type="grpo_replay_buffer").base_trainer_type == "grpo_with_replay_buffer"


# ---------------------------------------------------------------------------
# Finding #35: Save/load roundtrip for base_config and non-serializable env
# ---------------------------------------------------------------------------


def test_pr306_f35_save_load_roundtrip_base_config_and_runtime_environment(monkeypatch):
    # 1) Runtime callable environment must NOT stringify into "<function ...>"; base_config roundtrips via from_dict.
    cfg = AgentFlowConfig(
        base_trainer_type="grpo",
        base_config=GRPOConfig(beta=0.15, tx_mu_dtype=jnp.bfloat16, num_generations=4),
        environment=lambda **kw: _ScriptedEnv(),
    )
    serialized = cfg.to_dict()
    assert "environment" not in serialized
    raw_json = json.dumps(serialized)
    assert "<function" not in raw_json

    reloaded = AgentFlowConfig.from_dict(json.loads(raw_json))
    assert reloaded.environment is None
    assert isinstance(reloaded.base_config, GRPOConfig)
    assert reloaded.base_config.beta == pytest.approx(0.15)
    assert reloaded.base_config.tx_mu_dtype == jnp.bfloat16
    assert reloaded.base_config.num_generations == 4

    # 2) Passing a raw serialized base_config dict directly into AgentFlowTrainer uses from_dict cleanly.
    captured: dict[str, object] = {}
    orig_grpo = GRPOTrainer.__init__

    @functools.wraps(orig_grpo)
    def _rec(self, **kwargs):
        captured.update(kwargs)
        self.arguments = kwargs["arguments"]
        self.processing_class = kwargs.get("processing_class")
        self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    monkeypatch.setattr(GRPOTrainer, "__init__", _rec)
    raw_dict_cfg = AgentFlowConfig(
        base_trainer_type="grpo",
        base_config=serialized["base_config"],  # raw JSON dict with tx_mu_dtype="bfloat16" string
    )
    AgentFlowTrainer(
        arguments=raw_dict_cfg,
        model=object(),
        processing_class=_DummyTokenizer(),
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
    )
    assert isinstance(captured["arguments"], GRPOConfig)
    assert captured["arguments"].tx_mu_dtype == jnp.bfloat16

    # 3) JSON-serializable dict/string environment IS preserved through to_dict/from_dict.
    cfg_dict_env = AgentFlowConfig(environment={"name": "code_judge", "max_steps": 2})
    reloaded_dict_env = AgentFlowConfig.from_dict(json.loads(json.dumps(cfg_dict_env.to_dict())))
    assert reloaded_dict_env.environment == {"name": "code_judge", "max_steps": 2}


# ---------------------------------------------------------------------------
# Finding #36: Non-finite (NaN / inf) rewards in _compute_advantages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_rewards", ["group", "batch", "batch_advantage", "none"])
def test_pr306_f36_non_finite_rewards_do_not_poison_advantages(scale_rewards):
    holder = SimpleNamespace(trainer=SimpleNamespace(arguments=SimpleNamespace(scale_rewards=scale_rewards)))
    rewards = np.asarray([1.0, 3.0, float("nan"), 5.0, float("inf"), float("-inf")], dtype=np.float32)
    valid = np.ones_like(rewards)

    adv = AgentFlowTrainer._compute_advantages(holder, rewards, valid, group_size=3)
    assert np.all(np.isfinite(adv))
    assert adv[2] == 0.0
    assert adv[4] == 0.0
    assert adv[5] == 0.0
    if scale_rewards == "group":
        np.testing.assert_allclose(adv[:2], [-1.0 / 1.0001, 1.0 / 1.0001], rtol=1e-5)
        assert adv[3] == 0.0  # single finite row in group 1
    elif scale_rewards == "none":
        np.testing.assert_allclose(adv[:2], [-1.0, 1.0], rtol=1e-5)
        assert adv[3] == 0.0

    # group_size == 1 must also zero non-finite rewards instead of leaking NaN/inf.
    adv_single = AgentFlowTrainer._compute_advantages(holder, rewards, valid, group_size=1)
    assert np.all(np.isfinite(adv_single))
    assert adv_single[2] == 0.0 and adv_single[4] == 0.0 and adv_single[5] == 0.0


# ---------------------------------------------------------------------------
# Finding #46: env.close() failure in finally: must not mask errors or abort step
# ---------------------------------------------------------------------------


def test_pr306_f46_env_close_exception_does_not_mask_errors_or_abort_episode():
    inner = _stub_trainer(GRPOTrainer)
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, skip_apply_chat_template=True),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",), rewards=(2.0,), close_raises=True),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a")], "a", [-0.1]),
    )
    ep = flow._run_episode(SimpleNamespace(step=0), {"prompt": "p"}, 0, 0)
    assert isinstance(ep, AgentFlowEpisode)
    assert ep.episode_reward == pytest.approx(2.0)

    # When episode itself raises, close() error must NOT replace the original exception.
    flow_err = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, skip_apply_chat_template=True),
        trainer=_stub_trainer(GRPOTrainer),
        environment_factory=lambda **kw: _ScriptedEnv(
            reset_raises=ValueError("original reset error"),
            close_raises=True,
        ),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a")], "a", [-0.1]),
    )
    with pytest.raises(ValueError, match="original reset error"):
        flow_err._run_episode(SimpleNamespace(step=0), {"prompt": "p"}, 0, 0)


# ---------------------------------------------------------------------------
# Verification-round regressions (#303): multihost symmetric failure, NaN mask,
# mapping base_config async_mode pre-init, DPO None loss_mask values.
# ---------------------------------------------------------------------------


def test_pr306_f3_multihost_symmetric_generation_error_leaves_zero_unmatched_broadcasts(monkeypatch):
    from jax.experimental import multihost_utils

    # Case A: Symmetric exception inside _default_generate_turn on both hosts must propagate
    # cleanly on both hosts with ZERO unmatched abort broadcasts left in the collective stream.
    q_to_worker: queue.Queue = queue.Queue()
    host_local = threading.local()

    def _fake_process_count():
        return 2

    def _fake_process_index():
        return int(getattr(host_local, "rank", 0))

    def _fake_bcast(arr):
        rank = _fake_process_index()
        arr_np = np.asarray(arr)
        if rank == 0:
            q_to_worker.put(arr_np.copy())
            return arr_np
        return q_to_worker.get(timeout=5.0)

    monkeypatch.setattr(jax, "process_count", _fake_process_count)
    monkeypatch.setattr(jax, "process_index", _fake_process_index)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", _fake_bcast)

    def _make_flow(reset_raises=None, gen_raises=False):
        inner = _stub_trainer(
            GRPOTrainer,
            args=GRPOConfig(num_generations=1, beta=0.0, max_prompt_length=8, max_completion_length=8),
        )

        def _gen(st, ids, txt, t):
            if gen_raises:
                raise RuntimeError("symmetric rollout OOM")
            return [ord("x")], "x", [-0.1]

        return AgentFlowTrainer(
            arguments=AgentFlowConfig(
                max_turns=1,
                group_size=1,
                max_workers=2,
                max_prompt_length=8,
                max_completion_length=8,
                skip_apply_chat_template=True,
            ),
            trainer=inner,
            environment_factory=lambda **kw: _ScriptedEnv(prompt="hi", observations=("",), reset_raises=reset_raises),
            generate_turn_fn=_gen,
        )

    flow0 = _make_flow(gen_raises=True)
    flow1 = _make_flow(gen_raises=True)
    errors: dict[int, BaseException | None] = {0: None, 1: None}

    def _run_rank(rank: int, flow: AgentFlowTrainer):
        host_local.rank = rank
        try:
            flow._collect_episodes_multihost(SimpleNamespace(step=0), [(0, 0, {"prompt": "hi"})], workers=1)
        except BaseException as exc:
            errors[rank] = exc

    t0 = threading.Thread(target=_run_rank, args=(0, flow0))
    t1 = threading.Thread(target=_run_rank, args=(1, flow1))
    t0.start()
    t1.start()
    t0.join(timeout=5.0)
    t1.join(timeout=5.0)
    assert not t0.is_alive() and not t1.is_alive()
    assert isinstance(errors[0], RuntimeError) and "symmetric rollout OOM" in str(errors[0])
    assert isinstance(errors[1], RuntimeError) and "symmetric rollout OOM" in str(errors[1])
    assert q_to_worker.empty(), "Host 0 must not emit unmatched abort broadcasts after a symmetric rollout error."

    # Case B: Host-0-only env.reset() error MUST broadcast abort (-1) so worker exits cleanly.
    flow0_reset = _make_flow(reset_raises=ValueError("host0 env down"), gen_raises=False)
    flow1_reset = _make_flow(reset_raises=None, gen_raises=False)
    errors_b: dict[int, BaseException | None] = {0: None, 1: None}

    def _run_rank_b(rank: int, flow: AgentFlowTrainer):
        host_local.rank = rank
        try:
            flow._collect_episodes_multihost(SimpleNamespace(step=0), [(0, 0, {"prompt": "hi"})], workers=1)
        except BaseException as exc:
            errors_b[rank] = exc

    t0 = threading.Thread(target=_run_rank_b, args=(0, flow0_reset))
    t1 = threading.Thread(target=_run_rank_b, args=(1, flow1_reset))
    t0.start()
    t1.start()
    t0.join(timeout=5.0)
    t1.join(timeout=5.0)
    assert not t0.is_alive() and not t1.is_alive()
    assert isinstance(errors_b[0], ValueError) and "host0 env down" in str(errors_b[0])
    assert isinstance(errors_b[1], RuntimeError) and "host0 env down" in str(errors_b[1])
    assert q_to_worker.empty()


def test_pr306_f36_f13_nonfinite_rewards_zero_completion_mask_and_vapo_grpo_targets():
    # 1) VAPO / PPO path: NaN reward episode must have completion_mask=0, advantages=0, returns=0,
    #    and be excluded from num_items_in_batch / num_positive_tokens.
    vapo_args = VAPOConfig(
        max_prompt_length=6,
        max_completion_length=6,
        kl_coef=0.0,
        whiten_rewards=False,
        whiten_advantages=False,
        positive_reward_threshold=0.5,
    )
    vapo_inner = _stub_trainer(VAPOTrainer, args=vapo_args)
    vapo_inner.compute_values = lambda st, ids, mask: jnp.full(ids.shape, 0.7, dtype=jnp.float32)
    vapo_inner.compute_old_logps = lambda st, ids, mask: jnp.full(
        (ids.shape[0], vapo_args.max_completion_length), -0.3, dtype=jnp.float32
    )
    vapo_inner._compute_advantages_and_returns = VAPOTrainer._compute_advantages_and_returns.__get__(
        vapo_inner, VAPOTrainer
    )

    flow_vapo = AgentFlowTrainer(
        arguments=AgentFlowConfig(
            base_trainer_type="vapo",
            max_turns=1,
            group_size=1,
            max_prompt_length=6,
            max_completion_length=6,
            skip_apply_chat_template=True,
        ),
        trainer=vapo_inner,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a"), ord("b")], "ab", [-0.1, -0.2]),
    )
    eps_vapo = [
        AgentFlowEpisode(
            prompt_index=0,
            group_id=0,
            prompt_token_ids=[10, 11],
            completion_token_ids=[20, 21],
            action_mask=[1, 1],
            step_rewards=[1.0],
            episode_reward=1.0,
            turns=(),
            logprobs=[-0.1, -0.2],
        ),
        AgentFlowEpisode(
            prompt_index=1,
            group_id=1,
            prompt_token_ids=[10, 11],
            completion_token_ids=[22, 23],
            action_mask=[1, 1],
            step_rewards=[float("nan")],
            episode_reward=float("nan"),
            turns=(),
            logprobs=[-0.1, -0.2],
        ),
    ]
    out_vapo, _ = flow_vapo._adapt_episodes_to_trainer(eps_vapo, group_size=1, state=SimpleNamespace(step=0), is_train=True)
    np.testing.assert_array_equal(np.asarray(out_vapo["completion_mask"])[1], 0)
    np.testing.assert_array_equal(np.asarray(out_vapo["advantages"])[1], 0.0)
    np.testing.assert_array_equal(np.asarray(out_vapo["returns"])[1], 0.0)
    assert np.all(np.isfinite(np.asarray(out_vapo["advantages"])))
    assert np.all(np.isfinite(np.asarray(out_vapo["returns"])))
    assert float(out_vapo["num_items_in_batch"]) == pytest.approx(2.0)
    assert float(out_vapo["num_positive_tokens"]) == pytest.approx(2.0)

    # 2) GRPO path: NaN / +inf rows must have completion_mask=0, completion_attention_mask=0,
    #    advantages=0, difficulty_weights=0, and be excluded from num_items_in_batch.
    grpo_inner = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=3, beta=0.0, max_prompt_length=6, max_completion_length=6),
    )
    flow_grpo = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=3, max_prompt_length=6, max_completion_length=6),
        trainer=grpo_inner,
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        generate_turn_fn=lambda st, ids, txt, t: ([20, 21], "ab", [-0.1, -0.2]),
    )
    eps_grpo = [
        AgentFlowEpisode(0, 0, [10], [20, 21], [1, 1], [1.0], 1.0, (), [-0.1, -0.2]),
        AgentFlowEpisode(0, 1, [10], [22, 23], [1, 1], [3.0], 3.0, (), [-0.1, -0.2]),
        AgentFlowEpisode(0, 2, [10], [24, 25], [1, 1], [float("inf")], float("inf"), (), [-0.1, -0.2]),
    ]
    out_grpo, _ = flow_grpo._adapt_episodes_to_trainer(eps_grpo, group_size=3, state=SimpleNamespace(step=0), is_train=True)
    np.testing.assert_array_equal(np.asarray(out_grpo["completion_mask"])[2], 0)
    np.testing.assert_array_equal(np.asarray(out_grpo["completion_attention_mask"])[2], 0)
    assert float(out_grpo["advantages"][2]) == 0.0
    assert float(out_grpo["difficulty_weights"][2]) == 0.0
    assert float(out_grpo["num_items_in_batch"]) == pytest.approx(4.0)


def test_pr306_f33_mapping_base_config_injects_async_mode_before_post_init(monkeypatch):
    # OnPolicyDistillationConfig rejects distillation_mode="trust_region" + async_mode=True in __post_init__.
    with pytest.raises(ValueError, match="async_mode"):
        AgentFlowTrainer(
            arguments=AgentFlowConfig(
                base_trainer_type="on_policy_distillation",
                async_mode=True,
                base_config={"distillation_mode": "trust_region"},
            ),
            model=object(),
            teacher_model=object(),
            processing_class=_DummyTokenizer(),
            environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
        )

    # For a compatible mapping base_config omitting async_mode, async_mode=True is injected pre-init.
    captured = _install_init_recorder(monkeypatch, GRPOTrainer)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            base_trainer_type="grpo",
            async_mode=True,
            base_config={"beta": 0.02},
        ),
        model=object(),
        processing_class=_DummyTokenizer(),
        environment_factory=lambda **kw: _ScriptedEnv(observations=("",)),
    )
    assert isinstance(captured["arguments"], GRPOConfig)
    assert captured["arguments"].async_mode is True


def test_pr306_f16_none_loss_mask_values_treated_as_absent_for_microbatch_weight():
    from easydel.trainers.direct_preference_optimization_trainer._fn import _make_dpo_microbatch_weight_fn

    assert (
        _make_dpo_microbatch_weight_fn(
            {"chosen_loss_mask": None, "rejected_loss_mask": None},
            concatenated_forward_fn=None,
            loss_type="sigmoid",
            use_weighting=False,
            rpo_alpha=None,
        )
        is None
    )

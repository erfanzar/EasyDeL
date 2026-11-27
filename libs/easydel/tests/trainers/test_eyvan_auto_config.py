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
"""CPU-only tests for zero-config Eyvan URI, W&B-style auto run names, and rich auto-tags."""

from __future__ import annotations

import random
import re
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from easydel.inference.esurge import EyvanMetricsExporter, monitoring
from easydel.inference.esurge.engine.monitoring_stack import MonitoringStack
from easydel.trainers import training_configurations as tc
from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
from easydel.trainers.agentic_moshpit import AgenticMoshPitConfig, AgenticMoshPitTrainer
from easydel.trainers.direct_preference_optimization_trainer import DPOConfig
from easydel.trainers.group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from easydel.trainers.group_sequence_policy_optimization import GSPOConfig
from easydel.trainers.proximal_policy_optimization_trainer import PPOConfig
from easydel.trainers.supervised_fine_tuning_trainer import SFTConfig
from easydel.trainers.trainer import Trainer
from easydel.trainers.training_configurations import (
    TrainingArguments,
    _auto_detect_eyvan_tags,
    _clear_eyvan_auto_caches,
    _detect_eyvan_hardware_tags,
    _normalize_eyvan_chip_generation,
    _resolve_eyvan_default_tracking_uri,
    _resolve_eyvan_run_name,
    is_eyvan_auto_track_enabled,
    set_eyvan_auto_track_enabled,
)
from eyvan.track._format import META_START, STREAM_META, decode_segment
from eyvan.track._store import LocalTrackingStore

# Pure host-side metadata, config, and local Eyvan store tests; no JAX computation runs.
pytestmark = pytest.mark.cpu_ok

_WANDB_SUFFIX_RE = re.compile(r"^[a-z]+-[a-z]+-\d{1,3}$")


@dataclass(frozen=True)
class _FakeGcpContext:
    on_gcp: bool = True
    project: str | None = "demo-proj"
    zone: str | None = "us-central2-b"
    region: str | None = "us-central2"
    accelerator_type: str | None = "v6e-8"


@pytest.fixture(autouse=True)
def _clean_env_and_caches(monkeypatch):
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    _clear_eyvan_auto_caches()
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
    monitoring.stop_monitoring()
    try:
        yield
    finally:
        monitoring.stop_monitoring()
        _clear_eyvan_auto_caches()


def _read_meta_start(store_dir: Path, run_id: str) -> dict[str, object]:
    store = LocalTrackingStore(store_dir)
    for key in store.list(f"{run_id}/{STREAM_META}"):
        for rec in decode_segment(store.get(key)):
            if rec.get("op") == META_START:
                return rec
    raise AssertionError(f"META_START record not found for run {run_id!r} in {store_dir}")


def _mock_jax_topology(
    monkeypatch,
    *,
    backend: str,
    device_kind: str,
    local_count: int,
    process_count: int = 1,
) -> None:
    total = local_count * process_count
    fake_devs = [types.SimpleNamespace(device_kind=device_kind) for _ in range(local_count)]
    monkeypatch.setattr(tc.jax, "default_backend", lambda: backend)
    monkeypatch.setattr(tc.jax, "device_count", lambda: total)
    monkeypatch.setattr(tc.jax, "local_device_count", lambda: local_count)
    monkeypatch.setattr(tc.jax, "process_count", lambda: process_count)
    monkeypatch.setattr(tc.jax, "local_devices", lambda: fake_devs)


def test_zero_config_tracking_uri_precedence_and_fallbacks(monkeypatch, tmp_path: Path):
    default_bucket_dir = tmp_path / "default-monitoring-bucket" / "default" / "my-proj"
    env_store_dir = tmp_path / "env-store"
    explicit_dir = tmp_path / "explicit-store"
    calls: list[tuple[str, str, bool]] = []

    def _fake_resolve_default_tracking_uri(*, project: str, org: str = "default", allow_create: bool = False):
        calls.append((project, org, allow_create))
        return str(default_bucket_dir)

    fake_storage = types.SimpleNamespace(resolve_default_tracking_uri=_fake_resolve_default_tracking_uri)
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: fake_storage)

    # 1. Explicit eyvan_track_uri wins over env and default storage resolver.
    monkeypatch.setenv("EYVAN_TRACKING_URI", str(env_store_dir))
    assert _resolve_eyvan_default_tracking_uri(explicit_uri=str(explicit_dir), project="my-proj") == str(explicit_dir)
    assert calls == []

    # 2. EYVAN_TRACKING_URI wins over default storage resolver when explicit_uri is None.
    assert _resolve_eyvan_default_tracking_uri(explicit_uri=None, project="my-proj") == str(env_store_dir)
    assert calls == []

    # 3. With use_eyvan_track=True and no explicit URI or env var, resolves default monitoring bucket.
    monkeypatch.delenv("EYVAN_TRACKING_URI", raising=False)
    args = TrainingArguments(
        model_name="Qwen/Qwen3-8B",
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_project="my-proj",
        eyvan_track_flush_interval_s=0.1,
    )
    run = args.get_eyvan_track()
    assert run is not None
    assert calls == [("my-proj", "default", False)]
    assert run.store_root == default_bucket_dir.resolve().as_uri()
    args.close_eyvan_track()

    # 4. Under pytest isolation, use_eyvan_track=None stays disabled without calling storage resolver;
    #    outside pytest isolation (auto-track enabled), use_eyvan_track=None auto-enables while
    #    explicit use_eyvan_track=False stays disabled without calling storage resolver.
    calls.clear()
    assert is_eyvan_auto_track_enabled() is False
    args_off = TrainingArguments(use_wandb=False, save_directory=None)
    assert args_off.eyvan_tracking_enabled is False
    assert args_off.get_eyvan_track() is None
    assert calls == []

    with set_eyvan_auto_track_enabled(True):
        args_auto_on = TrainingArguments(
            model_name="Qwen/Qwen3-8B",
            use_wandb=False,
            save_directory=None,
            eyvan_track_project="auto-proj",
            eyvan_track_flush_interval_s=0.1,
        )
        assert args_auto_on.use_eyvan_track is None
        assert args_auto_on.eyvan_tracking_enabled is True
        run_auto = args_auto_on.get_eyvan_track()
        assert run_auto is not None
        assert calls == [("auto-proj", "default", False)]
        assert run_auto.store_root == default_bucket_dir.resolve().as_uri()
        args_auto_on.close_eyvan_track()

        calls.clear()
        args_explicit_off = TrainingArguments(use_wandb=False, save_directory=None, use_eyvan_track=False)
        assert args_explicit_off.eyvan_tracking_enabled is False
        assert args_explicit_off.get_eyvan_track() is None
        assert calls == []

    # 5. When resolve_default_tracking_uri returns None, falls back cleanly to ~/.eyvan/tracking/default/<project>.
    fake_home = tmp_path / "fake-home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(
        tc,
        "_import_eyvan_storage",
        lambda: types.SimpleNamespace(resolve_default_tracking_uri=lambda **kw: None),
    )
    args_local = TrainingArguments(
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_project="local-fallback",
        eyvan_track_flush_interval_s=0.1,
    )
    run_local = args_local.get_eyvan_track()
    assert run_local is not None
    expected_local = (fake_home / ".eyvan" / "tracking" / "default" / "local-fallback").resolve().as_uri()
    assert run_local.store_root == expected_local
    args_local.close_eyvan_track()

    # 6. Storage resolver raising or eyvan.storage absent never crashes a training run.
    def _boom(**_kw):
        raise RuntimeError("metadata timeout")

    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: types.SimpleNamespace(resolve_default_tracking_uri=_boom))
    assert _resolve_eyvan_default_tracking_uri(explicit_uri=None, project="p") is None
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)
    assert _resolve_eyvan_default_tracking_uri(explicit_uri=None, project="p") is None


def test_wandb_style_auto_run_name_generation_interop_caching_and_schema(monkeypatch, tmp_path: Path):
    # 1. Explicit eyvan_track_name is preserved verbatim.
    args_explicit = TrainingArguments(
        model_name="Qwen/Qwen3-8B",
        trainer_prefix="GRPO",
        use_wandb=False,
        save_directory=None,
        eyvan_track_name="  custom-exp / v1  ",
    )
    assert _resolve_eyvan_run_name(args_explicit) == "custom-exp / v1"

    # 2. Explicit wandb_name is respected when eyvan_track_name is None.
    args_wb = TrainingArguments(
        model_name="Qwen/Qwen3-8B",
        use_wandb=False,
        save_directory=None,
        wandb_name="wb-run-7",
    )
    assert _resolve_eyvan_run_name(args_wb) == "wb-run-7"

    # 3. Interop with eyvan.track.generate_run_name(prefix=...) when available.
    seen_prefixes: list[str | None] = []

    def _fake_generate_run_name(*, prefix: str | None = None, seed: int | None = None) -> str:
        del seed
        seen_prefixes.append(prefix)
        return f"{prefix}-amber-falcon-42" if prefix else "amber-falcon-42"

    fake_track = types.SimpleNamespace(generate_run_name=_fake_generate_run_name)
    monkeypatch.setattr(tc, "_import_eyvan_track", lambda: fake_track)

    args_ctx = GRPOConfig(model_name="Qwen/Qwen3-8B", use_wandb=False, save_directory=None)
    name_1 = args_ctx.resolve_eyvan_run_name()
    assert name_1 == "qwen3-8b-grpo-amber-falcon-42"
    assert len(name_1) <= 64
    assert seen_prefixes == ["qwen3-8b-grpo"]

    # Repeated calls on the same TrainingArguments instance are deterministic (cached!) and do NOT regenerate.
    assert args_ctx.resolve_eyvan_run_name() == name_1
    assert len(seen_prefixes) == 1
    # Caching must NOT pollute dataclass fields or to_dict()!
    assert args_ctx.eyvan_track_name is None
    assert args_ctx.to_dict()["eyvan_track_name"] is None

    # Generic placeholders ("BaseTrainer", "model", "module", None) omit model prefix -> pure W&B name.
    for placeholder in (None, "BaseTrainer", "model", "module"):
        args_plain = TrainingArguments(
            model_name=placeholder,
            trainer_prefix=None,
            use_wandb=False,
            save_directory=None,
        )
        assert _resolve_eyvan_run_name(args_plain) == "amber-falcon-42"

    # Very long model name still keeps total <= 64 chars and retains the random suffix.
    args_long = GRPOConfig(
        model_name="org/super-ultra-long-experimental-model-checkpoint-name-v99-70b-instruct",
        use_wandb=False,
        save_directory=None,
    )
    long_name = _resolve_eyvan_run_name(args_long)
    assert len(long_name) <= 64
    assert long_name.endswith("amber-falcon-42")

    # 4. Fallback stdlib adjective-noun-number generator when older eyvan lacks generate_run_name or raises.
    monkeypatch.setattr(tc, "_import_eyvan_track", lambda: types.SimpleNamespace())
    args_fallback = TrainingArguments(model_name="model", use_wandb=False, save_directory=None)
    fb_name = _resolve_eyvan_run_name(args_fallback)
    assert _WANDB_SUFFIX_RE.fullmatch(fb_name), f"Unexpected fallback name: {fb_name!r}"
    assert len(fb_name) <= 64
    assert _resolve_eyvan_run_name(args_fallback) == fb_name

    monkeypatch.setattr(
        tc,
        "_import_eyvan_track",
        lambda: types.SimpleNamespace(generate_run_name=lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    err_fb_name = _resolve_eyvan_run_name(model_name="Qwen/Qwen3-8B", trainer_slug="grpo")
    assert err_fb_name.startswith("qwen3-8b-grpo-")
    assert len(err_fb_name) <= 64

    # Fallback stays independent of Python's global random.seed() when seed=None.
    random.seed(12345)
    generated_set = {_resolve_eyvan_run_name(model_name=None) for _ in range(12)}
    assert len(generated_set) > 1

    # 5. Same cached auto name persists across get_eyvan_track() -> close -> get_eyvan_track().
    monkeypatch.undo()
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    store_dir = tmp_path / "cached-name-store"
    args_live = GRPOConfig(
        model_name="Qwen/Qwen3-0.6B",
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        eyvan_track_flush_interval_s=0.1,
    )
    run_a = args_live.get_eyvan_track()
    assert run_a is not None
    expected_name = run_a.name
    assert expected_name.startswith("qwen3-0-6b-grpo-")
    assert len(expected_name) <= 64
    args_live.close_eyvan_track()

    run_b = args_live.get_eyvan_track()
    assert run_b is not None
    assert run_b.name == expected_name
    args_live.close_eyvan_track()


def test_auto_detect_tags_trainers_features_model_and_user_merge(monkeypatch):
    _mock_jax_topology(monkeypatch, backend="cpu", device_kind="cpu", local_count=8, process_count=1)
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)

    # SFT: non-RL trainer, no icepop/async_rl/kl
    sft_args = SFTConfig(model_name="meta-llama/Llama-3.1-8B", use_wandb=False, save_directory=None)
    sft_tags = _auto_detect_eyvan_tags(sft_args)
    assert {"easydel", "jax", "sft", "llama-3.1-8b", "cpu"} <= set(sft_tags)
    assert "rl" not in sft_tags
    assert "icepop" not in sft_tags
    assert "async_rl" not in sft_tags
    assert "kl" not in sft_tags

    # DPO: preference trainer with positive beta -> "dpo", "kl", no "rl"
    dpo_args = DPOConfig(model_name="Qwen/Qwen3-4B", beta=0.1, use_wandb=False, save_directory=None)
    dpo_tags = _auto_detect_eyvan_tags(dpo_args)
    assert {"easydel", "jax", "dpo", "kl", "qwen3-4b"} <= set(dpo_tags)
    assert "rl" not in dpo_tags

    # GRPO + IcePop + Async RL + KL + LoRA + Quant + Model family + Env + User tags
    grpo_args = GRPOConfig(
        model_name="Qwen/Qwen3-8B",
        beta=0.04,
        async_mode=True,
        esurge_importance_sampling_correction=True,
        esurge_importance_sampling_mode="token_icepop",
        quantization_mode="nf4",
        trainable_selector="lora",
        eyvan_track_tags=["my-exp", "grpo", "  TeamTag  "],
        use_wandb=False,
        save_directory=None,
    )
    fake_model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="qwen3"),
        lora_is_enabled=True,
    )
    grpo_tags = _auto_detect_eyvan_tags(grpo_args, model=fake_model, env_name="WarehouseAuditToolEnv")
    assert {
        "easydel",
        "jax",
        "grpo",
        "rl",
        "icepop",
        "token_icepop",
        "async_rl",
        "env:warehouse_audit_tool_env",
        "kl",
        "lora",
        "peft",
        "quant:nf4",
        "qwen3",
        "cpu",
        "my-exp",
        "TeamTag",
    } <= set(grpo_tags)
    assert len(grpo_tags) == len(set(grpo_tags))

    # GSPO & PPO trainer identifiers + RL mode tag
    gspo_tags = _auto_detect_eyvan_tags(GSPOConfig(use_wandb=False, save_directory=None))
    assert {"easydel", "jax", "gspo", "rl"} <= set(gspo_tags)
    ppo_tags = _auto_detect_eyvan_tags(PPOConfig(use_wandb=False, save_directory=None))
    assert {"easydel", "jax", "ppo", "rl"} <= set(ppo_tags)

    # beta=0.0 does NOT emit "kl"; model-level quantization_config emits "quant:<method>"
    no_kl_args = GRPOConfig(model_name="BaseTrainer", beta=0.0, use_wandb=False, save_directory=None)
    quant_model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            model_type="deepseek_v3",
            quantization_config=types.SimpleNamespace(quant_type="mxfp4"),
        )
    )
    no_kl_tags = _auto_detect_eyvan_tags(no_kl_args, model=quant_model)
    assert "kl" not in no_kl_tags
    assert "basetrainer" not in no_kl_tags
    assert {"quant:mxfp4", "deepseek_v3"} <= set(no_kl_tags)

    # User tag preservation & bounds under 64-tag / 64-char limits:
    # Even with 60 unique user tags, ALL 60 user tags are preserved and total <= 64!
    many_user_tags = [f"user-tag-{i:02d}" for i in range(60)]
    many_user_tags.append("z" * 90)
    capped_args = GRPOConfig(
        model_name="Qwen/Qwen3-8B",
        eyvan_track_tags=many_user_tags,
        use_wandb=False,
        save_directory=None,
    )
    capped = _auto_detect_eyvan_tags(capped_args)
    assert len(capped) == 64
    assert all(1 <= len(t) <= 64 for t in capped)
    assert len(set(capped)) == 64
    for i in range(60):
        assert f"user-tag-{i:02d}" in capped
    assert ("z" * 64) in capped


def test_hardware_topology_tags_metadata_only_and_gcp_caching(monkeypatch):
    # Strictly forbid JAX compilation or collective/device reads during hardware tag detection.
    monkeypatch.setattr(tc.jax, "jit", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("jax.jit forbidden")))
    monkeypatch.setattr(
        tc.jax,
        "device_get",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("jax.device_get forbidden")),
    )

    # Chip generation normalization matrix
    assert _normalize_eyvan_chip_generation("TPU v6 lite") == "v6e"
    assert _normalize_eyvan_chip_generation("TPU v5 lite") == "v5e"
    assert _normalize_eyvan_chip_generation(None, "v5litepod-16") == "v5e"
    assert _normalize_eyvan_chip_generation("TPU v5") == "v5p"
    assert _normalize_eyvan_chip_generation("TPU v4") == "v4"
    assert _normalize_eyvan_chip_generation("NVIDIA H100 80GB HBM3") == "h100"
    assert _normalize_eyvan_chip_generation("NVIDIA A100-SXM4-80GB") == "a100"
    assert _normalize_eyvan_chip_generation("NVIDIA B200") == "b200"
    assert _normalize_eyvan_chip_generation("cpu") is None

    # 1. Multi-host TPU v6e-8x4 + GCP region caching (detect_gcp_context called only once!)
    gcp_calls = 0

    def _detect_gcp():
        nonlocal gcp_calls
        gcp_calls += 1
        return _FakeGcpContext(on_gcp=True, region="us-central2", zone="us-central2-b", accelerator_type="v6e-8")

    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: types.SimpleNamespace(detect_gcp_context=_detect_gcp))
    _mock_jax_topology(monkeypatch, backend="tpu", device_kind="TPU v6 lite", local_count=8, process_count=4)
    tpu_multi = _detect_eyvan_hardware_tags()
    assert tpu_multi == ("tpu", "v6e", "v6e-8x4", "32-chips", "us-central2")
    assert _detect_eyvan_hardware_tags() == tpu_multi
    assert gcp_calls == 1

    # 2. Single-host TPU v6e-8
    _mock_jax_topology(monkeypatch, backend="tpu", device_kind="TPU v6 lite", local_count=8, process_count=1)
    assert _detect_eyvan_hardware_tags() == ("tpu", "v6e", "v6e-8", "8-chips", "us-central2")

    # 3. Single-host 8x H100 GPU (cuda backend normalized to "gpu")
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)
    _mock_jax_topology(monkeypatch, backend="cuda", device_kind="NVIDIA H100 80GB HBM3", local_count=8, process_count=1)
    assert _detect_eyvan_hardware_tags() == ("gpu", "h100", "h100-8", "8-chips")

    # 4. Multi-host 2x8 B200 GPU
    _mock_jax_topology(monkeypatch, backend="gpu", device_kind="NVIDIA B200", local_count=8, process_count=2)
    assert _detect_eyvan_hardware_tags() == ("gpu", "b200", "b200-8x2", "16-chips")

    # 5. Zone-only GCP context derives region cleanly; raising GCP detector degrades gracefully.
    monkeypatch.setattr(
        tc,
        "_import_eyvan_storage",
        lambda: types.SimpleNamespace(
            detect_gcp_context=lambda: _FakeGcpContext(region=None, zone="europe-west4-a", accelerator_type=None)
        ),
    )
    _mock_jax_topology(monkeypatch, backend="tpu", device_kind="TPU v5 lite", local_count=4, process_count=1)
    assert _detect_eyvan_hardware_tags() == ("tpu", "v5e", "v5e-4", "4-chips", "europe-west4")

    monkeypatch.setattr(
        tc,
        "_import_eyvan_storage",
        lambda: types.SimpleNamespace(detect_gcp_context=lambda: (_ for _ in ()).throw(RuntimeError("gcp down"))),
    )
    assert _detect_eyvan_hardware_tags() == ("tpu", "v5e", "v5e-4", "4-chips")

    # 6. CPU host with 8 virtual devices emits only "cpu" (no bogus "8-chips" or "cpu-8")
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)
    _mock_jax_topology(monkeypatch, backend="cpu", device_kind="cpu", local_count=8, process_count=1)
    assert _detect_eyvan_hardware_tags() == ("cpu",)

    # 6. Backend init failure degrades gracefully without raising
    monkeypatch.setattr(tc.jax, "default_backend", lambda: (_ for _ in ()).throw(RuntimeError("no backend")))
    monkeypatch.setattr(tc.jax, "device_count", lambda: (_ for _ in ()).throw(RuntimeError("no backend")))
    monkeypatch.setattr(tc.jax, "local_device_count", lambda: (_ for _ in ()).throw(RuntimeError("no backend")))
    monkeypatch.setattr(tc.jax, "process_count", lambda: (_ for _ in ()).throw(RuntimeError("no backend")))
    monkeypatch.setattr(tc.jax, "local_devices", lambda: (_ for _ in ()).throw(RuntimeError("no backend")))
    assert _detect_eyvan_hardware_tags() == ()


def test_agent_flow_and_agentic_moshpit_preserve_auto_name_and_emit_rich_tags(monkeypatch, tmp_path: Path):
    _mock_jax_topology(monkeypatch, backend="tpu", device_kind="TPU v6 lite", local_count=8, process_count=4)
    monkeypatch.setattr(
        tc,
        "_import_eyvan_storage",
        lambda: types.SimpleNamespace(
            detect_gcp_context=lambda: _FakeGcpContext(region="us-central2", accelerator_type="v6e-8"),
            resolve_default_tracking_uri=lambda **kw: None,
        ),
    )

    class _WarehouseAuditToolEnv:
        pass

    captured_runs: list[tuple[GRPOConfig, object]] = []

    class _RecordingInnerGRPO(GRPOTrainer):
        def __init__(self, arguments, **kwargs):
            self.arguments = arguments
            self._model = kwargs.get("model")
            self.processing_class = kwargs.get("processing_class")
            self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})
            self._initialize_eyvan_track()
            captured_runs.append((arguments, self.eyvan_run))

    monkeypatch.setattr(
        "easydel.utils.Registry.get_or_raise",
        lambda kind, name: _RecordingInnerGRPO if kind == "trainer" else GRPOConfig,
    )

    # 1. AgentFlowTrainer with base_config=None: outer and inner configs share the EXACT same auto run name!
    store_af = tmp_path / "agentflow-auto"
    fake_model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3"))
    af_cfg = AgentFlowConfig(
        model_name="Qwen/Qwen3-1.7B",
        base_trainer_type="grpo",
        base_config={
            "beta": 0.04,
            "esurge_importance_sampling_correction": True,
            "esurge_importance_sampling_mode": "token_icepop",
        },
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_af),
        eyvan_track_tags=["experiment-alpha"],
        eyvan_track_flush_interval_s=0.1,
    )
    flow = AgentFlowTrainer(
        arguments=af_cfg,
        model=fake_model,
        environment=_WarehouseAuditToolEnv,
    )
    inner_cfg, inner_run = captured_runs[-1]
    assert inner_run is not None
    # Auto-generated name matches across AgentFlowConfig, inner GRPOConfig, and live Eyvan Run!
    assert af_cfg.resolve_eyvan_run_name() == inner_cfg.resolve_eyvan_run_name() == inner_run.name
    assert inner_run.name.startswith("qwen3-1-7b-agentflow-")
    assert len(inner_run.name) <= 64
    assert {
        "easydel",
        "jax",
        "agentflow",
        "grpo",
        "rl",
        "icepop",
        "token_icepop",
        "kl",
        "env:warehouse_audit_tool_env",
        "qwen3",
        "tpu",
        "v6e",
        "v6e-8x4",
        "32-chips",
        "us-central2",
        "experiment-alpha",
    } <= set(inner_run.tags)
    inner_cfg.close_eyvan_track()
    assert flow.trainer.arguments is inner_cfg

    # 2. AgentFlowTrainer with pre-cached auto name on AgentFlowConfig preserves it verbatim into base_config=None.
    store_af2 = tmp_path / "agentflow-precached"
    af_cfg2 = AgentFlowConfig(
        model_name="Qwen/Qwen3-0.6B",
        base_trainer_type="grpo",
        base_config=None,
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_af2),
        eyvan_track_flush_interval_s=0.1,
    )
    pre_name = af_cfg2.resolve_eyvan_run_name()
    AgentFlowTrainer(arguments=af_cfg2, environment="python_tir")
    inner_cfg2, inner_run2 = captured_runs[-1]
    assert inner_run2 is not None
    assert inner_cfg2.resolve_eyvan_run_name() == pre_name == inner_run2.name
    assert "env:python_tir" in inner_run2.tags
    inner_cfg2.close_eyvan_track()

    # 2b. AgentFlowTrainer wrapping a pre-built inner trainer syncs auto run name and agentflow/env tags.
    store_af3 = tmp_path / "agentflow-prebuilt-trainer"
    inner_pre_args = GRPOConfig(
        model_name="Qwen/Qwen3-4B",
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_af3),
        eyvan_track_flush_interval_s=0.1,
    )
    stub_inner = object.__new__(GRPOTrainer)
    stub_inner.arguments = inner_pre_args
    stub_inner._model = fake_model
    stub_inner._preprocess_batch_input = lambda state, batch, is_train: ({}, {})
    outer_pre_cfg = AgentFlowConfig(use_wandb=False, save_directory=None)
    AgentFlowTrainer(arguments=outer_pre_cfg, trainer=stub_inner, environment="swe_bench")
    pre_run = inner_pre_args.get_eyvan_track()
    assert pre_run is not None
    assert outer_pre_cfg.resolve_eyvan_run_name() == inner_pre_args.resolve_eyvan_run_name() == pre_run.name
    assert {"agentflow", "grpo", "rl", "env:swe_bench", "qwen3"} <= set(pre_run.tags)
    inner_pre_args.close_eyvan_track()

    # 3. AgenticMoshPitTrainer preserves auto-generated name and attaches agentic_moshpit + env + hardware tags.
    store_mp = tmp_path / "moshpit-auto"
    mp_cfg = AgenticMoshPitConfig(
        model_name="Qwen/Qwen3-4B",
        beta=0.02,
        esurge_importance_sampling_correction=True,
        esurge_importance_sampling_mode="token_icepop",
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_mp),
        eyvan_track_flush_interval_s=0.1,
    )
    mp_pre_name = mp_cfg.resolve_eyvan_run_name()

    def _fake_grpo_super_init(self, *, arguments, model, **kwargs):
        self.arguments = arguments
        self._model = model
        self.processing_class = kwargs.get("processing_class")
        self._initialize_eyvan_track()

    monkeypatch.setattr(GRPOTrainer, "__init__", _fake_grpo_super_init)
    mp_trainer = AgenticMoshPitTrainer(
        arguments=mp_cfg,
        model=fake_model,
        env_factory=_WarehouseAuditToolEnv,
    )
    assert mp_trainer.eyvan_run is not None
    assert mp_trainer.eyvan_run.name == mp_pre_name == mp_cfg.resolve_eyvan_run_name()
    assert {
        "easydel",
        "jax",
        "agentic_moshpit",
        "rl",
        "icepop",
        "token_icepop",
        "kl",
        "env:warehouse_audit_tool_env",
        "qwen3",
        "tpu",
        "v6e",
        "v6e-8x4",
        "32-chips",
        "us-central2",
    } <= set(mp_trainer.eyvan_run.tags)
    mp_cfg.close_eyvan_track()

    # 4. BaseTrainer._initialize_eyvan_track attaches model family & LoRA tags automatically.
    store_base = tmp_path / "base-trainer-auto"
    base_args = SFTConfig(
        model_name="BaseTrainer",
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_base),
        eyvan_track_flush_interval_s=0.1,
    )
    base_trainer = object.__new__(Trainer)
    base_trainer.arguments = base_args
    base_trainer._model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="gemma3"),
        lora_is_enabled=True,
    )
    base_trainer.eyvan_run = None
    base_trainer._initialize_eyvan_track()
    assert base_trainer.eyvan_run is not None
    assert {"easydel", "jax", "sft", "gemma3", "lora", "peft", "tpu", "v6e", "v6e-8x4", "32-chips"} <= set(
        base_trainer.eyvan_run.tags
    )
    assert base_trainer.eyvan_run.name.startswith("gemma3-sft-")
    base_args.close_eyvan_track()


def test_esurge_zero_config_uri_auto_name_and_rich_tags(monkeypatch, tmp_path: Path):
    default_esurge_store = tmp_path / "default-monitoring-bucket" / "default" / "esurge"
    resolver_calls: list[tuple[str, str, bool]] = []

    def _fake_resolve(*, project: str, org: str = "default", allow_create: bool = False):
        resolver_calls.append((project, org, allow_create))
        return str(default_esurge_store)

    _mock_jax_topology(monkeypatch, backend="tpu", device_kind="TPU v6 lite", local_count=8, process_count=1)
    monkeypatch.setattr(
        tc,
        "_import_eyvan_storage",
        lambda: types.SimpleNamespace(
            resolve_default_tracking_uri=_fake_resolve,
            detect_gcp_context=lambda: _FakeGcpContext(region="us-central2", accelerator_type="v6e-8"),
        ),
    )

    # 1. MonitoringStack with enable_eyvan_track=True and eyvan_track_uri=None resolves default monitoring bucket!
    stack = MonitoringStack(info=lambda *a, **kw: None, model_name="Qwen/Qwen3-8B")
    try:
        urls = stack.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=True,
            eyvan_track_tags=("serve-prod",),
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        assert resolver_calls == [("esurge", "default", False)]
        assert urls["eyvan_track"] == default_esurge_store.resolve().as_uri()
        run_id = urls["eyvan_run_id"]
        assert run_id
    finally:
        stack.stop_monitoring()

    meta = _read_meta_start(default_esurge_store, run_id)
    assert isinstance(meta["name"], str)
    assert meta["name"].startswith("qwen3-8b-esurge-")
    assert len(meta["name"]) <= 64
    assert {
        "easydel",
        "jax",
        "esurge",
        "inference",
        "qwen3-8b",
        "tpu",
        "v6e",
        "v6e-8",
        "8-chips",
        "us-central2",
        "serve-prod",
    } <= set(meta["tags"])

    # 2. Direct EyvanMetricsExporter with explicit name=None & no model_name generates "esurge-<adj>-<noun>-<num>".
    direct_store = tmp_path / "direct-esurge-store"
    exporter = EyvanMetricsExporter(
        name=None,
        uri=str(direct_store),
        flush_interval_s=0.1,
    )
    direct_run_id = exporter.run_id
    exporter.close()
    assert direct_run_id is not None
    direct_meta = _read_meta_start(direct_store, direct_run_id)
    assert isinstance(direct_meta["name"], str)
    assert direct_meta["name"].startswith("esurge-")
    assert len(direct_meta["name"]) <= 64
    assert {"easydel", "jax", "esurge", "inference", "tpu", "v6e", "v6e-8", "8-chips", "us-central2"} <= set(
        direct_meta["tags"]
    )

    # 3. Explicit eyvan_track_name is preserved verbatim; EYVAN_TRACKING_URI beats resolve_default_tracking_uri.
    resolver_calls.clear()
    env_esurge_store = tmp_path / "env-esurge-store"
    monkeypatch.setenv("EYVAN_TRACKING_URI", str(env_esurge_store))
    stack_explicit = MonitoringStack(info=lambda *a, **kw: None, model_name="Qwen/Qwen3-8B")
    try:
        urls_exp = stack_explicit.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=True,
            eyvan_track_name="explicit-serve-bench",
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        assert resolver_calls == []
        assert urls_exp["eyvan_track"] == env_esurge_store.resolve().as_uri()
        exp_run_id = urls_exp["eyvan_run_id"]
    finally:
        stack_explicit.stop_monitoring()
    assert _read_meta_start(env_esurge_store, exp_run_id)["name"] == "explicit-serve-bench"

    # 4. When resolve_default_tracking_uri returns None, eSurge falls back cleanly to ~/.eyvan/tracking/default/esurge.
    monkeypatch.delenv("EYVAN_TRACKING_URI", raising=False)
    fake_home = tmp_path / "esurge-home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(
        tc,
        "_import_eyvan_storage",
        lambda: types.SimpleNamespace(
            resolve_default_tracking_uri=lambda **kw: None,
            detect_gcp_context=lambda: None,
        ),
    )
    stack_local = MonitoringStack(info=lambda *a, **kw: None)
    try:
        urls_local = stack_local.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=True,
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        expected_local = (fake_home / ".eyvan" / "tracking" / "default" / "esurge").resolve().as_uri()
        assert urls_local["eyvan_track"] == expected_local
    finally:
        stack_local.stop_monitoring()


def test_resolve_eyvan_storage_path_and_elarge_agent_flow_eyvan_forwarding(monkeypatch):
    from easydel.infra.elarge.model import eLargeModel
    from easydel.utils import eyvan_helpers

    seen_storage_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def _fake_storage_path(*parts: str, region=None, zone=None, create=None, **kwargs):
        seen_storage_calls.append((parts, {"region": region, "zone": zone, "create": create, **kwargs}))
        suffix = "/".join(p.strip("/") for p in parts if p)
        bucket = f"gs://eyvan_{region or 'us-central2'}"
        return f"{bucket}/{suffix}" if suffix else bucket

    monkeypatch.setattr(eyvan_helpers, "storage_path", _fake_storage_path)

    ckpt_uri = TrainingArguments.resolve_eyvan_storage_path("checkpoints", "qwen3-8b", region="us-east5", create=False)
    assert ckpt_uri == "gs://eyvan_us-east5/checkpoints/qwen3-8b"
    data_uri = eLargeModel.resolve_eyvan_storage_path("datasets", "math", zone="us-central2-b", create=False)
    assert data_uri == "gs://eyvan_us-central2/datasets/math"
    assert seen_storage_calls == [
        (("checkpoints", "qwen3-8b"), {"region": "us-east5", "zone": None, "create": False}),
        (("datasets", "math"), {"region": None, "zone": "us-central2-b", "create": False}),
    ]

    # Verify eLargeModel._agent_flow_base_config forwards model_name and all 8 Eyvan fields from training_args
    # while letting explicit keys in raw_trainer_cfg override them cleanly.
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "Qwen/Qwen3-8B"},
        "trainer": {
            "trainer_type": "agent_flow",
            "base_trainer_type": "grpo",
            "eyvan_track_project": "override-proj",
        },
    }
    af_args = AgentFlowConfig(
        model_name="Qwen/Qwen3-8B",
        base_trainer_type="grpo",
        use_wandb=False,
        save_directory=None,
        use_eyvan_track=True,
        eyvan_track_uri="/tmp/elarge-af-store",
        eyvan_track_project="ignored-proj",
        eyvan_track_name="elarge-af-run",
        eyvan_track_group="elarge-af-grp",
        eyvan_track_tags=["elarge", "af"],
        eyvan_track_flush_interval_s=2.0,
        eyvan_report_progress=False,
    )
    merged = elm._agent_flow_base_config(af_args, elm._config["trainer"])
    assert isinstance(merged, GRPOConfig)
    assert merged.model_name == "Qwen/Qwen3-8B"
    assert merged.use_eyvan_track is True
    assert merged.eyvan_track_uri == "/tmp/elarge-af-store"
    assert merged.eyvan_track_project == "override-proj"
    assert merged.eyvan_track_name == "elarge-af-run"
    assert merged.eyvan_track_group == "elarge-af-grp"
    assert merged.eyvan_track_tags == ["elarge", "af"]
    assert merged.eyvan_track_flush_interval_s == pytest.approx(2.0)
    assert merged.eyvan_report_progress is False

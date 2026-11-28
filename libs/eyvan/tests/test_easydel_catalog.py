# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shipped EasyDeL trainer catalog and the assistant functions built on it."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from eyvan import assistant_knowledge as ak

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "scripts" / "generate_easydel_catalog.py"
_MONOREPO_EASYDEL = REPO_ROOT.parent / "easydel" / "easydel" / "trainers"
EASYDEL_SOURCE = (
    _MONOREPO_EASYDEL
    if _MONOREPO_EASYDEL.is_dir()
    else REPO_ROOT.parent / "EasyDeL" / "libs" / "easydel" / "easydel" / "trainers"
)


def _run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    # -I: isolated mode, so neither the checkout nor a stray easydel install can be imported implicitly.
    return subprocess.run(
        [sys.executable, "-I", str(GENERATOR), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=REPO_ROOT,
    )


def _field(described: dict, name: str) -> dict:
    return next(item for item in described["fields"] if item["name"] == name)


def test_catalog_ships_core_trainers_with_inherited_base_fields() -> None:
    ids = {entry["id"] for entry in ak.list_trainers()}
    assert {"dpo", "sft", "grpo", "orpo", "kto", "ppo", "reward", "distillation", "gkd"} <= ids
    info = ak.catalog_info()
    assert info["easydel_revision"] and info["trainer_count"] == len(ids)

    for trainer in ("dpo", "sft", "grpo"):
        described = ak.describe_trainer(trainer, query="learning_rate")
        assert any(item["name"] == "learning_rate" for item in described["fields"]), trainer
        assert described["mro"][-1] == "TrainingArguments"

    # An inherited base field is marked with its declaring class.
    total = _field(ak.describe_trainer("dpo", query="total_batch_size"), "total_batch_size")
    assert total["inherited_from"] == "TrainingArguments"
    # DPO redeclares learning_rate with its own default, so it is not inherited.
    lr = _field(ak.describe_trainer("dpo", query="learning_rate"), "learning_rate")
    assert lr["default"] == 1e-6 and "inherited_from" not in lr


def test_dpo_beta_and_loss_type_defaults_match_source() -> None:
    described = ak.describe_trainer("dpo", include_inherited=False)
    beta = _field(described, "beta")
    loss = _field(described, "loss_type")
    assert beta["default"] == 0.1
    assert loss["default"] == "sigmoid"
    assert {"sigmoid", "ipo", "hinge", "robust"} <= set(loss["choices"])
    assert loss["choices_allow_list"] is True
    assert "beta" in {item["field"] for item in described["essential"]}
    assert not any("inherited_from" in item for item in described["fields"])


def test_describe_query_filters_and_output_is_bounded() -> None:
    narrowed = ak.describe_trainer("DPO", query="reference")
    names = {item["name"] for item in narrowed["fields"]}
    assert {"reference_free", "force_use_ref_model"} <= names
    assert "learning_rate" not in names
    assert narrowed["matched_fields"] == len(narrowed["fields"]) < narrowed["total_fields"]

    # A model asks for several fields at once; any term matching keeps the field.
    several = {item["name"] for item in ak.describe_trainer("dpo", query="beta loss_type")["fields"]}
    assert {"beta", "loss_type"} <= several

    full = ak.describe_trainer("grpo")
    assert len(json.dumps(full, ensure_ascii=False)) <= ak.DEFAULT_MAX_CHARS
    assert full["truncated"] is True and "query=" in full["note"]
    # Essentials come first so truncation never hides them.
    essential = {item["field"] for item in full["essential"]}
    leading = [item["name"] for item in full["fields"][: len(essential)]]
    assert set(leading) == essential


def test_unknown_trainer_suggests_close_ids() -> None:
    with pytest.raises(ak.UnknownTrainerError) as caught:
        ak.describe_trainer("dpoo")
    assert "dpo" in caught.value.suggestions
    assert ak.resolve_trainer_id("nash_md") == "nash-md"
    assert ak.resolve_trainer_id("spec") == "speculative_decoding"


def test_validate_flags_unknowns_missing_essentials_and_invalid_choices() -> None:
    result = ak.validate_trainer_arguments("dpo", {"betta": 0.2, "loss_type": "sigmoidd", "lr": 1e-6})
    assert result["ok"] is False
    assert set(result["unknown"]) == {"betta", "lr"}
    assert result["suggestions"] == {"betta": "beta", "lr": "learning_rate"}
    missing = {item.get("field") for item in result["missing_essential"]}
    assert {"beta", "max_length", "max_prompt_length", "learning_rate", "total_batch_size"} <= missing
    assert any(item.get("any_of") == ["num_train_epochs", "max_training_steps"] for item in result["missing_essential"])
    invalid = result["invalid_choices"][0]
    assert invalid["field"] == "loss_type" and "sigmoid" in invalid["allowed"]
    assert invalid["did_you_mean"] == ["sigmoid"]


def test_validate_types_lists_and_a_clean_draft() -> None:
    bad = ak.validate_trainer_arguments("dpo", {"beta": "high", "loss_type": ["sigmoid", "nope"]})
    assert [item["field"] for item in bad["type_warnings"]] == ["beta"]
    assert bad["invalid_choices"][0]["field"] == "loss_type"
    # Metadata-declared choices (KTO) are enforced too.
    kto = ak.validate_trainer_arguments("kto", {"loss_type": "sigmoid"})
    assert kto["invalid_choices"][0]["allowed"] == ["kto", "apo_zero_unpaired"]

    clean = {
        "beta": 0.1,
        "loss_type": ["sigmoid", "sft"],
        "loss_weights": [1.0, 0.5],
        "max_length": 4096,
        "max_prompt_length": 1024,
        "reference_free": False,
        "learning_rate": 5e-7,
        "total_batch_size": 64,
        "gradient_accumulation_steps": 2,
        "save_directory": "gs://bucket/dpo",
        "save_steps": 200,
        "max_training_steps": 2000,
    }
    # A NaN or infinite number is never a valid argument: YAML writes it as .nan and
    # EasyDeL accepts it, so it must block the draft here.
    for value in (float("nan"), float("inf"), [1.0, float("-inf")]):
        flagged = ak.validate_trainer_arguments("dpo", {"learning_rate": value})
        assert [item["field"] for item in flagged["type_warnings"]] == ["learning_rate"], value
    blocked = ak.render_launch("sft", {"learning_rate": float("nan")}, model="m", dataset="d")
    assert blocked["runnable"] is False and any("learning_rate" in reason for reason in blocked["blocking"])

    result = ak.validate_trainer_arguments("dpo", clean)
    assert result["ok"] is True, result
    assert any(item["input"] == "reference_model" for item in result["launch_inputs_to_confirm"])
    assert result["constraints"], "post_init checks on loss_weights/beta should be surfaced"


def test_render_launch_produces_elarge_yaml_and_command() -> None:
    launch = ak.render_launch(
        "dpo",
        {"beta": 0.2, "loss_type": "ipo", "max_length": 2048, "max_prompt_length": 512},
        model="meta-llama/Llama-3.1-8B-Instruct",
        dataset="trl-lib/ultrafeedback_binarized",
        num_devices=8,
    )
    assert launch["runnable"] is True, launch["blocking"]
    assert launch["command"] == "python -m easydel.scripts.elarge --config easydel-launch.yaml"
    config = yaml.safe_load(launch["config_yaml"])
    assert config == launch["config"]
    assert config["trainer"] == {
        "trainer_type": "dpo",
        "beta": 0.2,
        "loss_type": "ipo",
        "max_length": 2048,
        "max_prompt_length": 512,
    }
    assert config["model"]["name_or_path"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["mixture"]["informs"][0] == {
        "type": "huggingface",
        "data_files": "trl-lib/ultrafeedback_binarized",
        "split": "train",
    }
    assert config["actions"] == ["validate", "train"]
    assert launch["script"].endswith(launch["command"] + "\n")
    assert "beta: 0.2" in launch["script"]


def test_render_launch_declares_one_cpu_torch_wheel_per_worker_python_and_machine() -> None:
    launch = ak.render_launch("sft", {"learning_rate": 1e-4}, model="m", dataset="d")
    assert launch["dependencies"] == ak.torch_cpu_requirements()
    pairs = set()
    for item in launch["dependencies"]:
        assert ak.requirement_name(item) == "torch"
        url, marker = item.removeprefix("torch @ ").split(" ; ")
        assert url.startswith("https://download.pytorch.org/whl/cpu/torch-") and "%2Bcpu-" in url
        python = marker.split('python_version == "')[1].split('"')[0]
        machine = marker.split('platform_machine == "')[1].split('"')[0]
        tag = "cp" + python.replace(".", "")
        assert f"-{tag}-{tag}-manylinux_2_28_{machine}.whl" in url and 'sys_platform == "linux"' in marker
        pairs.add((python, machine))
    assert pairs == {(p, m) for p in ("3.11", "3.12", "3.13") for m in ("x86_64", "aarch64")}
    assert ak.requirement_name("Torch_Vision>=1") == "torch-vision" and ak.requirement_name("@x") is None


def test_render_launch_blocks_missing_reward_model_and_bad_mesh() -> None:
    launch = ak.render_launch(
        "grpo",
        {"num_generations": 8, "betta": 0.0},
        model="Qwen/Qwen3-8B",
        dataset="data/prompts.jsonl",
        sharding={"axis_dims": [1, 3, 1, 1, 1]},
        num_devices=8,
    )
    assert launch["runnable"] is False
    text = " ".join(launch["blocking"])
    assert "reward_model" in text and "betta" in text and "mesh product" in text
    assert launch["config"]["mixture"]["informs"][0]["type"] == "json"

    ok = ak.render_launch(
        "grpo",
        {"num_generations": 8},
        model="Qwen/Qwen3-8B",
        dataset="data/prompts.jsonl",
        reward_model="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    )
    assert ok["runnable"] is True, ok["blocking"]
    assert ok["config"]["reward_model"]["task"] == "sequence-classification"


def test_essentials_name_only_real_fields() -> None:
    spec = json.loads((REPO_ROOT / "eyvan" / "assistant_knowledge" / "easydel_essentials.json").read_text())
    for trainer_id, entry in spec["trainers"].items():
        described = ak.describe_trainer(trainer_id, query=None, max_chars=60_000)
        known = {item["name"] for item in described["fields"]} | set(described.get("more_fields", []))
        assert set(entry["fields"]) <= known, trainer_id


def test_generator_never_imports_easydel() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"easydel", "jax", "eformer", "spectrax"}

    probe = (
        "import runpy, sys; sys.argv = ['gen', '--check']\n"
        "try:\n    runpy.run_path(sys.argv_path, run_name='__main__')\nexcept SystemExit:\n    pass\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'easydel', 'jax', 'eformer'}))"
    ).replace("sys.argv_path", repr(str(GENERATOR)))
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=120, check=False, cwd=REPO_ROOT
    )
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout + result.stderr


@pytest.mark.skipif(not EASYDEL_SOURCE.is_dir(), reason="EasyDeL source checkout not present")
def test_generator_check_matches_committed_catalog() -> None:
    result = _run_generator("--check", "--source", str(EASYDEL_SOURCE))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok:")


def test_generator_check_skips_without_source(tmp_path: Path) -> None:
    result = _run_generator("--check", "--source", str(tmp_path / "missing"))
    assert result.returncode == 0
    assert result.stdout.startswith("skip:")


def test_default_sharding_uses_six_easydel_axes_and_torch_version_matches() -> None:
    assert ak.DEFAULT_SHARDING == {
        "axis_dims": [1, 1, -1, 1, 1, 1],
        "axis_names": ["pp", "dp", "fsdp", "ep", "tp", "sp"],
        "auto_shard_model": True,
    }
    assert ak.TORCH_CPU_VERSION == "2.8.0"

    launch = ak.render_launch("sft", {"learning_rate": 1e-4}, model="m", dataset="d", num_devices=8)
    assert launch["runnable"] is True, launch["blocking"]
    assert launch["config"]["sharding"] == {
        "axis_dims": [1, 1, -1, 1, 1, 1],
        "axis_names": ["pp", "dp", "fsdp", "ep", "tp", "sp"],
        "auto_shard_model": True,
    }

    legacy = ak.render_launch(
        "sft",
        {"learning_rate": 1e-4},
        model="m",
        dataset="d",
        sharding={"axis_dims": [1, -1, 1, 1, 1]},
        num_devices=8,
    )
    assert legacy["runnable"] is True, legacy["blocking"]
    assert legacy["config"]["sharding"]["axis_names"] == ["dp", "fsdp", "ep", "tp", "sp"]

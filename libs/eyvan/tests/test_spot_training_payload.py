# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CPU-only payload boundary tests; these do not qualify TPU computation."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def payload():
    path = Path(__file__).resolve().parents[1] / "examples" / "spot_training_payload.py"
    assert path.is_file(), "the portable remote payload is not integrated"
    spec = importlib.util.spec_from_file_location("spot_payload_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(module):
    return {
        "schema": module.SCHEMA,
        "run_id": "a" * 32,
        "step": 20,
        "weights": [0.5] * 32,
        "initial_loss": 10.0,
        "checkpoint_loss": 1.0,
        "instance": "owned-spot-vm",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "foreign"),
        ("step", 19),
        ("weights", [0.0]),
        ("weights", [float("nan")] * 32),
        ("weights", [True] * 32),
        ("initial_loss", float("inf")),
        ("checkpoint_loss", -1),
        ("instance", ""),
    ],
)
def test_invalid_checkpoint_is_rejected(payload, field, value):
    with pytest.raises(ValueError):
        payload.validate_checkpoint(_checkpoint(payload) | {field: value}, run_id="a" * 32)


def test_valid_checkpoint_preserves_identity_and_weights(payload):
    checkpoint = _checkpoint(payload)
    assert payload.validate_checkpoint(checkpoint, run_id="a" * 32) == checkpoint


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/path",
        "gs://user:pass@bucket/path",  # pragma: allowlist secret -- rejected dummy credentials
        "gs://bucket/path?token=x",
        "gs://bucket/path#fragment",
        "gs://bucket/../path",
        "gs://bucket",
        "gs://bucket/path\n",
    ],
)
def test_invalid_checkpoint_uri_is_rejected(payload, uri):
    with pytest.raises(ValueError):
        payload.validate_checkpoint_uri(uri)


def test_hook_uses_no_accelerator_and_checks_uri_before_network(payload, monkeypatch, capsys):
    import builtins
    import json

    uri = "gs://example-bucket/demo/" + "a" * 32 + "/checkpoint.json"
    args = ["hook", "--gcp-project", "demo-project-123", "--run-id", "a" * 32, "--checkpoint-uri", uri]
    original_import = builtins.__import__
    calls = []

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"jax", "numpy"}, "hook must not acquire the trainer's TPU"
        return original_import(name, *args, **kwargs)

    def metadata(path, **kwargs):
        calls.append(path)
        return {
            "project/project-id": "demo-project-123",
            "instance/scheduling/preemptible": "true",
            "instance/name": "owned-spot-vm",
        }[path]

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(payload, "metadata_value", metadata)
    monkeypatch.setattr(payload, "_read_checkpoint", lambda uri: _checkpoint(payload))
    monkeypatch.setenv("EYVAN_CHECKPOINT_URI", "gs://other-bucket/foreign")
    with pytest.raises(ValueError):
        payload.main(args)
    assert not calls
    monkeypatch.setenv("EYVAN_CHECKPOINT_URI", uri)
    assert payload.main(args) == 0
    assert "instance/scheduling/preemptible" in calls
    assert json.loads(capsys.readouterr().out)["marker"] == "checkpoint_verified"


def test_metadata_refuses_non_spot_and_wrong_project(payload, monkeypatch):
    values = {"project/project-id": "foreign", "instance/scheduling/preemptible": "false"}
    monkeypatch.setattr(payload, "metadata_value", lambda path: values[path])
    with pytest.raises(ValueError):
        payload.validate_instance(project="demo-project-123")
    values["project/project-id"] = "demo-project-123"
    with pytest.raises(ValueError):
        payload.validate_instance(project="demo-project-123")

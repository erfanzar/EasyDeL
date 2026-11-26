# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import stat

import pytest

from eyvan.sdk.contexts import (
    ContextConfigError,
    EyvanContext,
    FileContextStore,
    context_config_path,
    load_context_config,
    set_active_context,
)


def test_context_path_honors_environment_override(monkeypatch, tmp_path) -> None:
    configured = tmp_path / "client.yaml"
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(configured))

    assert context_config_path() == configured


def test_named_contexts_round_trip_atomically_with_owner_only_permissions(tmp_path) -> None:
    path = tmp_path / "config.json"
    first = EyvanContext(
        fleet="research",
        org_id="acme",
        project_id="training",
        base_url="https://eyvan.example.com/",
    )
    second = EyvanContext(
        fleet="serving",
        org_id="acme",
        project_id="inference",
        base_url="https://eyvan.example.com",
    )

    set_active_context(first, path=path)
    updated = set_active_context(second, path=path)
    loaded = load_context_config(path)

    assert updated == loaded
    assert loaded.active_context == "serving"
    assert loaded.active == second
    assert loaded.contexts["research"] == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".config.json.*")) == []


def test_file_context_store_owns_load_save_and_activation(tmp_path) -> None:
    store = FileContextStore(tmp_path / "contexts.json")
    context = EyvanContext(fleet="research", org_id="acme", project_id="training")

    updated = store.activate(context)

    assert store.resolved_path == tmp_path / "contexts.json"
    assert updated.active == context
    assert store.active() == context
    assert store.load() == updated


def test_yaml_context_file_is_supported(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    context = EyvanContext(fleet="research", org_id="acme", project_id="training")

    set_active_context(context, path=path)

    assert load_context_config(path).active == context
    assert "active_context: research" in path.read_text(encoding="utf-8")


def test_context_normalizes_and_persists_an_autoscale_cache_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.json"
    context = EyvanContext(
        fleet="research",
        org_id="acme",
        project_id="training",
        autoscale_cache_dir="autoscale-cache",
    )

    set_active_context(context, path=path)

    expected = str((tmp_path / "autoscale-cache").resolve())
    assert context.autoscale_cache_dir == expected
    assert load_context_config(path).active is not None
    assert load_context_config(path).active.autoscale_cache_dir == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "active_context": None, "contexts": {}},
        {
            "version": 1,
            "active_context": "research",
            "contexts": {
                "research": {
                    "fleet": "research",
                    "org_id": "acme",
                    "project_id": "training",
                    "base_url": "https://eyvan.example.com",
                    "token": "must-not-be-stored",
                }
            },
        },
        {"version": 1, "active_context": "missing", "contexts": {}},
    ],
)
def test_context_schema_rejects_unknown_versions_secrets_and_missing_active_context(tmp_path, payload) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContextConfigError):
        load_context_config(path)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://token@eyvan.example.com",
        "https://eyvan.example.com?token=secret",
        "file:///tmp/eyvan.sock",
    ],
)
def test_context_rejects_base_urls_that_can_persist_credentials(base_url: str) -> None:
    with pytest.raises(ValueError):
        EyvanContext(fleet="research", org_id="acme", project_id="training", base_url=base_url)

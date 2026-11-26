# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Environment parsing, validation, and repr hygiene for EyvanSettings."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator

import pytest

from eyvan.config import EyvanSettings, RepositoryBackend


@pytest.fixture(autouse=True)
def _clean_eyvan_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop every inherited ``EYVAN_*`` variable so from_env sees a clean process."""

    for name in tuple(os.environ):
        if name.startswith("EYVAN_"):
            monkeypatch.delenv(name, raising=False)
    yield


def test_from_env_defaults_apply_without_any_variables() -> None:
    settings = EyvanSettings.from_env()

    assert settings.repository_backend is RepositoryBackend.MEMORY
    assert settings.scheduler_interval_s == 2.0
    assert settings.cell_reconcile_interval_s == 15.0
    assert settings.cell_reconcile_concurrency == 8
    assert settings.lease_ttl_s == 120
    assert settings.checkpoint_grace_s == 300
    assert settings.preemption_stop_grace_s == 300
    assert settings.outbox_claim_ttl_s == 90
    assert settings.controller_enabled is True
    assert settings.dev_auth_enabled is False
    assert settings.trusted_callable_projects == ()
    assert settings.agent_shared_secret is None


def test_from_env_parses_intervals_and_ttls_with_documented_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_SCHEDULER_INTERVAL_S", "0.25")
    monkeypatch.setenv("EYVAN_CELL_RECONCILE_INTERVAL_S", "7.5")
    monkeypatch.setenv("EYVAN_CELL_RECONCILE_CONCURRENCY", "3")
    monkeypatch.setenv("EYVAN_LEASE_TTL_S", "45")
    monkeypatch.setenv("EYVAN_CHECKPOINT_GRACE_S", "60")
    monkeypatch.setenv("EYVAN_PREEMPTION_STOP_GRACE_S", "90")
    monkeypatch.setenv("EYVAN_OUTBOX_CLAIM_TTL_S", "30")
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_TTL_S", "3600")

    settings = EyvanSettings.from_env()

    assert settings.scheduler_interval_s == 0.25
    assert isinstance(settings.scheduler_interval_s, float)
    assert settings.cell_reconcile_interval_s == 7.5
    assert isinstance(settings.cell_reconcile_interval_s, float)
    assert settings.cell_reconcile_concurrency == 3
    assert settings.lease_ttl_s == 45
    assert isinstance(settings.lease_ttl_s, int)
    assert settings.checkpoint_grace_s == 60
    assert settings.preemption_stop_grace_s == 90
    assert settings.outbox_claim_ttl_s == 30
    assert settings.artifact_token_ttl_s == 3600


@pytest.mark.parametrize("ttl", ("0", "-15"))
def test_from_env_rejects_a_non_positive_outbox_claim_ttl(monkeypatch: pytest.MonkeyPatch, ttl: str) -> None:
    monkeypatch.setenv("EYVAN_OUTBOX_CLAIM_TTL_S", ttl)

    with pytest.raises(ValueError, match="outbox_claim_ttl_s must be positive"):
        EyvanSettings.from_env()


def test_direct_construction_rejects_a_non_positive_outbox_claim_ttl() -> None:
    with pytest.raises(ValueError, match="outbox_claim_ttl_s must be positive"):
        EyvanSettings(outbox_claim_ttl_s=0)


def test_from_env_rejects_an_unknown_repository_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "sqlite")

    with pytest.raises(ValueError, match=r"EYVAN_REPOSITORY_BACKEND must be one of: memory, local, postgres"):
        EyvanSettings.from_env()


def test_from_env_accepts_each_documented_repository_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    for backend in RepositoryBackend:
        monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", backend.value)
        assert EyvanSettings.from_env().repository_backend is backend


def test_from_env_carries_native_bootstrap_integrity_and_generation_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "a" * 64
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_PACKAGE_SHA256", digest)
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_PACKAGE_GENERATION", "123")
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_WHEELHOUSE_SHA256", digest)
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_WHEELHOUSE_GENERATION", "456")

    settings = EyvanSettings.from_env()

    assert settings.native_worker_package_sha256 == digest
    assert settings.native_worker_package_generation == "123"
    assert settings.native_worker_wheelhouse_sha256 == digest
    assert settings.native_worker_wheelhouse_generation == "456"


def test_from_env_rejects_a_non_numeric_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_LEASE_TTL_S", "soon")

    with pytest.raises(ValueError, match="invalid literal"):
        EyvanSettings.from_env()


def test_from_env_parses_truthy_flags_and_csv_project_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_DEV_AUTH_ENABLED", "yes")
    monkeypatch.setenv("EYVAN_AUTO_CREATE_SCHEMA", "1")
    monkeypatch.setenv("EYVAN_CONTROLLER_ENABLED", "off")
    monkeypatch.setenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", " acme/research ,, bf/serving ,")

    settings = EyvanSettings.from_env()

    assert settings.dev_auth_enabled is True
    assert settings.auto_create_schema is True
    assert settings.controller_enabled is False
    assert settings.trusted_callable_projects == ("acme/research", "bf/serving")


def test_secret_fields_load_from_env_but_never_appear_in_the_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_AGENT_SHARED_SECRET", "agent-secret-value")
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_ROOT_SECRET", "native-root-value")

    settings = EyvanSettings.from_env()
    rendered = repr(settings)

    assert settings.agent_shared_secret == "agent-secret-value"
    assert settings.native_worker_root_secret == "native-root-value"
    assert "agent-secret-value" not in rendered
    assert "native-root-value" not in rendered
    assert "agent_shared_secret" not in rendered
    assert "native_worker_root_secret" not in rendered
    assert "database_url" in rendered


def test_from_env_reads_every_native_worker_setting_the_gcp_factory_consumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_gcp_factory` reads these; only `from_gcp_native_runtime_profile` used to set them.

    So a locally-profiled agent had an artifact root and an ingress tag while an
    env-configured production agent silently got ``None`` for both — no artifact root for
    its native workers, and no firewall tag for public ingress.
    """

    from eyvan.capacity.drivers import registry as capacity_registry

    monkeypatch.setenv("EYVAN_NATIVE_WORKER_ARTIFACT_ROOT_URI", "gs://eyvan-artifacts/native")
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_PUBLIC_INGRESS_TAG", "eyvan-native-worker")

    settings = EyvanSettings.from_env()

    assert settings.native_worker_artifact_root_uri == "gs://eyvan-artifacts/native"
    assert settings.native_worker_public_ingress_tag == "eyvan-native-worker"
    # Every field the factory reads off settings must be reachable from the environment,
    # or an env-only deployment cannot configure it at all.
    factory_source = inspect.getsource(capacity_registry._gcp_factory)
    for field in ("native_worker_artifact_root_uri", "native_worker_public_ingress_tag"):
        assert f"settings.{field}" in factory_source
        assert getattr(settings, field) is not None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_collection_interval_s", 0.0),
        ("artifact_collection_interval_s", float("inf")),
        ("artifact_recovery_grace_s", -1.0),
        ("artifact_recovery_grace_s", float("nan")),
        ("artifact_retention_s", -1),
        ("log_retention_s", -1),
    ),
)
def test_artifact_lifecycle_settings_reject_unbounded_or_negative_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        EyvanSettings(**{field: value})


def test_artifact_lifecycle_settings_load_safe_operational_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_ARTIFACT_COLLECTION_INTERVAL_S", "12.5")
    monkeypatch.setenv("EYVAN_ARTIFACT_RECOVERY_GRACE_S", "0")
    monkeypatch.setenv("EYVAN_ARTIFACT_RETENTION_S", "0")
    monkeypatch.setenv("EYVAN_LOG_RETENTION_S", "42")
    settings = EyvanSettings.from_env()
    assert settings.artifact_collection_interval_s == 12.5
    assert settings.artifact_recovery_grace_s == 0
    assert settings.artifact_retention_s == 0
    assert settings.log_retention_s == 42

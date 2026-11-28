# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``eyvan.storage`` regional buckets, monitoring store, GCP detection, and CLI."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from click.testing import CliRunner

import eyvan
from eyvan.sdk.cli import main as root_cli
from eyvan.sdk.commands import storage as storage_commands
from eyvan.storage import (
    BucketInfo,
    BucketNameConflictError,
    BucketNamingPolicy,
    GcpContext,
    GcsBucketBackend,
    MonitoringBucketConfig,
    RegionalBucketConfig,
    StorageAuthError,
    StorageBackendError,
    StorageConfig,
    StorageConfigError,
    StorageConfigStore,
    StorageConsentRequiredError,
    StorageHandle,
    StorageNotConfiguredError,
    clear_gcp_context_cache,
    configure_storage,
    default_gcp_token_provider,
    detect_gcp_context,
    get_monitoring_storage,
    get_storage,
    join_storage_path,
    monitoring_path,
    resolve_default_tracking_uri,
    select_preferred_gcp_project,
    storage_path,
    storage_status,
    validate_bucket_name,
    validate_location,
    validate_region,
    zone_to_region,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeBucketBackend:
    """In-memory ``StorageBackend`` simulating owned, foreign, and raced buckets."""

    def __init__(
        self,
        *,
        existing: dict[str, BucketInfo] | None = None,
        foreign_get: set[str] | None = None,
        foreign_create: set[str] | None = None,
    ) -> None:
        self.buckets: dict[str, BucketInfo] = dict(existing or {})
        self.foreign_get: set[str] = set(foreign_get or ())
        self.foreign_create: set[str] = set(foreign_create or ())
        self.get_calls: list[str] = []
        self.create_calls: list[tuple[str, str, str, str, bool]] = []

    def bucket_exists(self, bucket: str) -> bool:
        return self.get_bucket(bucket) is not None

    def get_bucket(self, bucket: str) -> BucketInfo | None:
        self.get_calls.append(bucket)
        if bucket in self.foreign_get:
            raise BucketNameConflictError(f"bucket {bucket!r} owned by another project (HTTP 403)")
        return self.buckets.get(bucket)

    def create_bucket(
        self,
        bucket: str,
        *,
        project: str,
        location: str,
        storage_class: str = "STANDARD",
        uniform_access: bool = True,
    ) -> BucketInfo:
        self.create_calls.append((bucket, project, location, storage_class, uniform_access))
        if bucket in self.foreign_create or bucket in self.foreign_get:
            raise BucketNameConflictError(f"bucket {bucket!r} is globally taken (HTTP 409)")
        if bucket in self.buckets:
            return self.buckets[bucket]
        info = BucketInfo(
            name=bucket,
            location=location.lower(),
            storage_class=storage_class,
            uniform_access=uniform_access,
            created=True,
        )
        self.buckets[bucket] = BucketInfo(
            name=bucket,
            location=location.lower(),
            storage_class=storage_class,
            uniform_access=uniform_access,
            created=False,
        )
        return info


class FakeGcsHttpServer:
    """Offline HTTP transport implementing the GCS JSON bucket API."""

    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, object]] = {}
        self.foreign: set[str] = set()
        self.race_owned: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.force_status: int | None = None
        self.auth_fail_once = False
        self.valid_tokens = {"secret-token-1", "secret-token-2"}

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((method, url, headers, body))
        auth = headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ")
        if self.auth_fail_once:
            self.auth_fail_once = False
            return 401, {}, b'{"error":"expired secret-token-1"}'
        if token not in self.valid_tokens:
            return 401, {}, b'{"error":"invalid token"}'
        if self.force_status is not None:
            return self.force_status, {}, b'{"error":"contains secret-token-1"}'

        parsed = urlsplit(url)
        if method == "GET" and "/storage/v1/b/" in parsed.path:
            bucket = unquote(parsed.path.split("/storage/v1/b/", 1)[1])
            if bucket in self.foreign:
                return 403, {}, b'{"error":"forbidden secret-token-1"}'
            if bucket not in self.buckets:
                return 404, {}, b'{"error":"not found"}'
            return 200, {}, json.dumps(self.buckets[bucket]).encode("utf-8")

        if method == "POST" and parsed.path == "/storage/v1/b":
            query = parse_qs(parsed.query)
            assert "project" in query and len(query["project"]) == 1
            assert body is not None
            payload = json.loads(body.decode("utf-8"))
            name = payload["name"]
            if name in self.race_owned:
                self.buckets[name] = self.race_owned.pop(name)
                return 409, {}, b'{"error":"already exists"}'
            if name in self.foreign or name in self.buckets:
                return 409, {}, b'{"error":"already exists secret-token-1"}'
            record = {
                "name": name,
                "location": payload["location"],
                "projectNumber": "1234567890",
                "storageClass": payload["storageClass"],
                "iamConfiguration": payload["iamConfiguration"],
            }
            self.buckets[name] = record
            return 200, {}, json.dumps(record).encode("utf-8")

        return 400, {}, b'{"error":"unexpected route"}'


# ------------------------------------------------------------------ naming & validation


def test_zone_region_location_and_bucket_validation() -> None:
    assert zone_to_region("us-central2-b") == "us-central2"
    assert zone_to_region("europe-west4-a") == "europe-west4"
    assert zone_to_region("us-central1-ai1a") == "us-central1"
    assert zone_to_region("projects/123456/zones/us-east5-c") == "us-east5"

    assert validate_region("US-CENTRAL2") == "us-central2"
    with pytest.raises(ValueError, match="zone"):
        validate_region("us-central2-b")

    assert validate_location("us") == "us"
    assert validate_location("us-central1") == "us-central1"
    with pytest.raises(ValueError, match="zone"):
        validate_location("us-central1-a")

    assert validate_bucket_name("eyvan_us-central2") == "eyvan_us-central2"
    for bad in ("", "ab", "UPPER", "gs://bucket", "-leading", "trailing-", "a..b", "a" * 64):
        with pytest.raises(ValueError):
            validate_bucket_name(bad)


def test_bucket_naming_policy_candidates_and_length_bounds() -> None:
    policy = BucketNamingPolicy()
    regional = policy.regional_candidates("us-central2", project="example-project-a")
    assert regional[0] == "eyvan_us-central2"
    assert regional[1] == "eyvan_us-central2_example-project-a"
    assert all(3 <= len(name) <= 63 for name in regional)

    # Very long custom prefix + max-length project still fits within 63 chars.
    long_policy = BucketNamingPolicy(
        regional_prefix="custom_org_storage_prefix",
        monitoring_prefix="custom_org_monitoring_prefix",
    )
    long_regional = long_policy.regional_candidates("northamerica-northeast1", project="a" * 28 + "1")
    long_monitoring = long_policy.monitoring_candidates(project="a" * 28 + "1")
    assert all(3 <= len(name) <= 63 for name in (*long_regional, *long_monitoring))
    assert long_regional[0].startswith("custom_org_storage_prefix_")
    assert long_monitoring[0].startswith("custom_org_monitoring_prefix_")


def test_join_storage_path_and_handle_truediv_reject_traversal() -> None:
    assert join_storage_path("gs://eyvan_us-central2", "datasets", "gsm8k/train.parquet") == (
        "gs://eyvan_us-central2/datasets/gsm8k/train.parquet"
    )
    handle = StorageHandle(uri="gs://eyvan_us-central2", bucket="eyvan_us-central2", region="us-central2")
    chained = handle / "checkpoints" / "step-100"
    assert isinstance(chained, str)
    assert chained == "gs://eyvan_us-central2/checkpoints/step-100"
    assert str(handle) == "gs://eyvan_us-central2"
    assert handle.tracking_uri(org="acme", project="rl") == "gs://eyvan_us-central2/tracking/acme/rl"

    mon_handle = StorageHandle(
        uri="gs://eyvan_monitoring_proj/tracking",
        bucket="eyvan_monitoring_proj",
        region="us-central1",
        role="monitoring",
        prefix="tracking",
    )
    assert mon_handle.tracking_uri() == "gs://eyvan_monitoring_proj/tracking/default/default"
    assert mon_handle.tracking_uri(org="acme", project="rl") == "gs://eyvan_monitoring_proj/tracking/acme/rl"

    for bad in ("../escape", "a/../b", ".", "/absolute", "a//b", "a\\b", "a?b=1", "a#frag", "bad\x00name"):
        with pytest.raises(ValueError):
            handle.path(bad)
        with pytest.raises(ValueError):
            _ = handle / bad


# ------------------------------------------------------------------ config store


def test_storage_config_store_atomic_owner_only_and_symlink_safe(tmp_path: Path) -> None:
    config_path = tmp_path / ".eyvan" / "storage.json"
    store = StorageConfigStore(config_path)
    assert store.load_if_exists() is None
    assert store.load() == StorageConfig()

    cfg = StorageConfig(
        project="example-project-a",
        monitoring=MonitoringBucketConfig(bucket="eyvan_monitoring_example-project-a", location="us-central1"),
        regional={"us-central2": RegionalBucketConfig(bucket="eyvan_us-central2", location="us-central2")},
        auto_create=True,
    )
    saved = store.save(cfg)
    assert saved == config_path
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config_path.parent.stat().st_mode) == 0o700
    assert store.load() == cfg

    # Refuse symlink on read and write.
    real_target = tmp_path / "elsewhere.json"
    real_target.write_text("{}")
    symlink_path = tmp_path / "link.json"
    symlink_path.symlink_to(real_target)
    link_store = StorageConfigStore(symlink_path)
    with pytest.raises(StorageConfigError, match="symlink"):
        link_store.load()
    with pytest.raises(StorageConfigError, match="symlink"):
        link_store.save(cfg)


# ------------------------------------------------------------------ GCP context detection


def test_detect_gcp_context_on_gce_metadata_and_spoof_rejection(tmp_path: Path) -> None:
    responses: dict[str, tuple[int, dict[str, str], bytes]] = {
        "http://169.254.169.254/computeMetadata/v1/project/project-id": (
            200,
            {"Metadata-Flavor": "Google"},
            b"example-project-a",
        ),
        "http://169.254.169.254/computeMetadata/v1/instance/zone": (
            200,
            {"Metadata-Flavor": "Google"},
            b"projects/123456/zones/us-central2-b",
        ),
        "http://169.254.169.254/computeMetadata/v1/instance/attributes/accelerator-type": (
            200,
            {"Metadata-Flavor": "Google"},
            b"v6e-8",
        ),
    }

    ctx = detect_gcp_context(
        config_path=tmp_path / "storage.json",
        metadata_fetcher=lambda url, _timeout: responses.get(url),
        allow_gcloud=False,
    )
    assert ctx == GcpContext(
        on_gcp=True,
        project="example-project-a",
        zone="us-central2-b",
        region="us-central2",
        accelerator_type="v6e-8",
        source="gce-metadata",
    )

    # Response missing Metadata-Flavor: Google header must not be trusted.
    spoof = detect_gcp_context(
        config_path=tmp_path / "storage.json",
        metadata_fetcher=lambda _url, _timeout: (200, {}, b"spoofed-project"),
        allow_gcloud=False,
    )
    assert spoof.on_gcp is False
    assert spoof.project is None


def test_detect_gcp_context_off_gcp_fallback_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    gcloud_calls: list[tuple[str, ...]] = []

    def fake_gcloud(args: tuple[str, ...], _timeout: float) -> tuple[int, str]:
        gcloud_calls.append(args)
        table = {
            ("config", "get-value", "project"): (0, "gcloud-proj-1"),
            ("config", "get-value", "compute/zone"): (0, "us-east5-a"),
            ("config", "get-value", "compute/region"): (0, "(unset)"),
        }
        return table.get(args, (1, ""))

    # 1. Only gcloud available (read-only get-value).
    ctx_gcloud = detect_gcp_context(
        metadata_fetcher=lambda _u, _t: None,
        gcloud_runner=fake_gcloud,
    )
    assert ctx_gcloud.on_gcp is False
    assert ctx_gcloud.project == "gcloud-proj-1"
    assert ctx_gcloud.zone == "us-east5-a"
    assert ctx_gcloud.region == "us-east5"
    assert ctx_gcloud.source == "gcloud"
    assert all(call[:2] == ("config", "get-value") for call in gcloud_calls)

    # 2. autoscale/defaults.yaml takes precedence over gcloud for project.
    autoscale_dir = tmp_path / ".eyvan" / "autoscale"
    autoscale_dir.mkdir(parents=True)
    (autoscale_dir / "defaults.yaml").write_text("version: 1\nproject: autoscale-proj-1\n")
    ctx_auto = detect_gcp_context(metadata_fetcher=lambda _u, _t: None, allow_gcloud=False)
    assert ctx_auto.project == "autoscale-proj-1"
    assert ctx_auto.source == "autoscale-defaults"

    # 3. gcp-native-runtime.json takes precedence over autoscale/defaults.yaml.
    (tmp_path / ".eyvan" / "gcp-native-runtime.json").write_text(json.dumps({"project_id": "runtime-proj-1"}))
    ctx_rt = detect_gcp_context(metadata_fetcher=lambda _u, _t: None, allow_gcloud=False)
    assert ctx_rt.project == "runtime-proj-1"
    assert ctx_rt.source == "gcp-native-runtime"

    # 4. storage.json takes highest precedence off-GCP.
    StorageConfigStore(tmp_path / ".eyvan" / "storage.json").save(
        StorageConfig(
            project="storage-proj-1",
            regional={"us-central2": RegionalBucketConfig(bucket="eyvan_us-central2", location="us-central2")},
        )
    )
    ctx_storage = detect_gcp_context(metadata_fetcher=lambda _u, _t: None, allow_gcloud=False)
    assert ctx_storage.project == "storage-proj-1"
    assert ctx_storage.region == "us-central2"
    assert ctx_storage.source == "storage-config"


# ------------------------------------------------------------------ GCS JSON backend


def test_gcs_bucket_backend_crud_race_and_security() -> None:
    server = FakeGcsHttpServer()
    backend = GcsBucketBackend(token_provider=lambda: "secret-token-1", transport=server)

    assert backend.get_bucket("eyvan_us-central2") is None
    assert backend.bucket_exists("eyvan_us-central2") is False

    created = backend.create_bucket("eyvan_us-central2", project="example-project-a", location="us-central2")
    assert created.name == "eyvan_us-central2"
    assert created.location == "us-central2"
    assert created.storage_class == "STANDARD"
    assert created.uniform_access is True
    assert created.created is True
    assert backend.bucket_exists("eyvan_us-central2") is True

    # Multi-host race: POST returns 409 because our own peer just created it in the same location.
    server.race_owned["eyvan_us-east5"] = {
        "name": "eyvan_us-east5",
        "location": "US-EAST5",
        "storageClass": "STANDARD",
        "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}},
    }
    raced = backend.create_bucket("eyvan_us-east5", project="example-project-a", location="us-east5")
    assert raced.name == "eyvan_us-east5"
    assert raced.created is False

    # Foreign bucket -> 403/409 raises BucketNameConflictError without leaking token.
    server.foreign.add("eyvan_taken")
    with pytest.raises(BucketNameConflictError) as get_err:
        backend.get_bucket("eyvan_taken")
    assert "secret-token" not in str(get_err.value)
    with pytest.raises(BucketNameConflictError) as post_err:
        backend.create_bucket("eyvan_taken", project="example-project-a", location="us-central2")
    assert "secret-token" not in str(post_err.value)

    # HTTP 500 error body and transport exception must never leak secrets.
    server.force_status = 500
    with pytest.raises(StorageBackendError) as err500:
        backend.get_bucket("eyvan_us-central2")
    assert "500" in str(err500.value)
    assert "secret-token" not in str(err500.value)

    def broken_transport(*_args: object) -> tuple[int, dict[str, str], bytes]:
        raise OSError("Authorization: Bearer secret-token-1")

    broken = GcsBucketBackend(token_provider=lambda: "secret-token-1", transport=broken_transport)
    with pytest.raises(StorageBackendError) as terr:
        broken.get_bucket("eyvan_us-central2")
    assert "secret-token" not in str(terr.value)
    assert terr.value.__suppress_context__ is True

    def broken_token() -> str:
        raise RuntimeError("leaked secret-token-1")

    auth_broken = GcsBucketBackend(token_provider=broken_token, transport=server)
    with pytest.raises(StorageAuthError) as aerr:
        auth_broken.get_bucket("eyvan_us-central2")
    assert "secret-token" not in str(aerr.value)


def test_default_token_provider_metadata_then_gcloud_fallback() -> None:
    gcloud_calls = 0

    def gcloud_runner(args: tuple[str, ...], _timeout: float) -> tuple[int, str]:
        nonlocal gcloud_calls
        gcloud_calls += 1
        assert args == ("auth", "print-access-token")
        return 0, "gcloud-access-token-abc"

    provider = default_gcp_token_provider(
        metadata_fetcher=lambda _u, _t: None,
        gcloud_runner=gcloud_runner,
    )
    assert provider() == "gcloud-access-token-abc"
    assert provider() == "gcloud-access-token-abc"
    assert gcloud_calls == 1


# ------------------------------------------------------------------ high-level resolution & consent


def test_get_storage_consent_semantics_and_fast_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "storage.json"
    backend = FakeBucketBackend()
    ctx = GcpContext(on_gcp=True, project="example-project-a", zone="us-central2-b", region="us-central2")

    # 1. Missing + create=False -> StorageNotConfiguredError.
    with pytest.raises(StorageNotConfiguredError):
        get_storage(create=False, config_path=cfg_path, backend=backend, gcp_context=ctx)
    assert backend.create_calls == []

    # 2. Missing + create=None + non-interactive + auto_create=False -> StorageConsentRequiredError.
    with pytest.raises(StorageConsentRequiredError, match="eyvan storage init"):
        get_storage(
            create=None,
            config_path=cfg_path,
            backend=backend,
            is_interactive=lambda: False,
            gcp_context=ctx,
        )
    assert backend.create_calls == []

    # 3. Interactive prompt declined -> StorageConsentRequiredError, nothing created.
    prompts: list[str] = []
    with pytest.raises(StorageConsentRequiredError):
        get_storage(
            create=None,
            config_path=cfg_path,
            backend=backend,
            confirm=lambda p: (prompts.append(p), False)[1],
            gcp_context=ctx,
        )
    assert len(prompts) == 1
    assert "eyvan_us-central2" in prompts[0]
    assert "us-central2" in prompts[0]
    assert "example-project-a" in prompts[0]
    assert "uniform_access=True" in prompts[0]
    assert backend.create_calls == []

    # 4. Interactive prompt approved -> creates eyvan_us-central2 and persists to storage.json.
    handle = get_storage(
        create=None,
        config_path=cfg_path,
        backend=backend,
        confirm=lambda _p: True,
        gcp_context=ctx,
    )
    assert handle.bucket == "eyvan_us-central2"
    assert handle.uri == "gs://eyvan_us-central2"
    assert handle.region == "us-central2"
    assert handle.zone == "us-central2-b"
    assert handle.project == "example-project-a"
    assert handle.role == "regional"
    assert handle.created is True
    assert len(backend.create_calls) == 1

    # 5. Fast path: second call returns immediately from storage.json with zero backend calls or prompts.
    get_count_before = len(backend.get_calls)
    cached = get_storage(
        config_path=cfg_path,
        backend=backend,
        confirm=lambda _p: pytest.fail("should not prompt when already configured"),
        gcp_context=ctx,
    )
    assert cached.uri == "gs://eyvan_us-central2"
    assert cached.created is False
    assert len(backend.get_calls) == get_count_before


def test_get_storage_auto_create_and_collision_safe_fallback(tmp_path: Path) -> None:
    cfg_path = tmp_path / "storage.json"
    configure_storage(project="example-project-a", auto_create=True, config_path=cfg_path)

    # eyvan_us-central2 is taken by another project (403 on GET) -> falls back to eyvan_us-central2_example-project-a.
    backend = FakeBucketBackend(foreign_get={"eyvan_us-central2"})
    uri = storage_path(
        "datasets",
        "gsm8k",
        zone="us-central2-b",
        config_path=cfg_path,
        backend=backend,
        is_interactive=lambda: False,
    )
    assert uri == "gs://eyvan_us-central2_example-project-a/datasets/gsm8k"
    saved = StorageConfigStore(cfg_path).load()
    assert saved.regional["us-central2"].bucket == "eyvan_us-central2_example-project-a"

    # Also test collision on POST (409 after GET returned 404).
    backend_post = FakeBucketBackend(foreign_create={"eyvan_europe-west4"})
    eu_handle = get_storage(
        region="europe-west4",
        create=True,
        config_path=cfg_path,
        backend=backend_post,
    )
    assert eu_handle.bucket == "eyvan_europe-west4_example-project-a"
    assert eu_handle.created is True


def test_monitoring_storage_and_resolve_default_tracking_uri(tmp_path: Path) -> None:
    cfg_path = tmp_path / "storage.json"
    off_gcp = GcpContext(on_gcp=False, project=None, region=None)

    # Unconfigured off-GCP -> None without prompting or network calls.
    assert resolve_default_tracking_uri(project="grpo-run", config_path=cfg_path, gcp_context=off_gcp) is None

    # Create monitoring bucket via get_monitoring_storage(create=True).
    backend = FakeBucketBackend()
    mon = get_monitoring_storage(
        project="example-project-a",
        region="us-central2",
        create=True,
        config_path=cfg_path,
        backend=backend,
    )
    assert mon.bucket == "eyvan_monitoring_example-project-a"
    assert mon.uri == "gs://eyvan_monitoring_example-project-a/tracking"
    assert mon.prefix == "tracking"
    assert mon.role == "monitoring"
    assert mon.created is True
    assert monitoring_path("default", "grpo", config_path=cfg_path) == (
        "gs://eyvan_monitoring_example-project-a/tracking/default/grpo"
    )

    # resolve_default_tracking_uri now resolves immediately from storage.json.
    assert resolve_default_tracking_uri(project="grpo", org="acme", config_path=cfg_path) == (
        "gs://eyvan_monitoring_example-project-a/tracking/acme/grpo"
    )

    # On a fresh TPU VM with no local storage.json, discovers an already-existing canonical monitoring bucket.
    fresh_cfg = tmp_path / "fresh-tpu-storage.json"
    on_gcp = GcpContext(on_gcp=True, project="example-project-a", zone="us-central2-b", region="us-central2")
    discovered = resolve_default_tracking_uri(
        project="rl-exp",
        config_path=fresh_cfg,
        backend=backend,
        gcp_context=on_gcp,
    )
    assert discovered == "gs://eyvan_monitoring_example-project-a/tracking/default/rl-exp"
    assert StorageConfigStore(fresh_cfg).load().monitoring is not None

    # Rejects path-traversal scope names.
    with pytest.raises(ValueError):
        resolve_default_tracking_uri(project="../evil", config_path=cfg_path)


def test_storage_handle_exists_and_ensure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "storage.json"
    backend = FakeBucketBackend()
    handle = StorageHandle(
        uri="gs://eyvan_us-central2",
        bucket="eyvan_us-central2",
        region="us-central2",
        project="example-project-a",
        _backend=backend,
        _config_path=cfg_path,
    )
    assert handle.exists() is False
    ensured = handle.ensure(create=True)
    assert ensured.created is True
    assert handle.exists() is True


# ------------------------------------------------------------------ lazy exports & lightness


def test_top_level_lazy_exports_stay_stdlib_light() -> None:
    expected = {
        "StorageHandle",
        "configure_storage",
        "get_monitoring_storage",
        "get_storage",
        "monitoring_path",
        "storage_path",
    }
    assert expected <= set(eyvan.__all__)
    probe = (
        "import sys, eyvan; "
        "fns = (eyvan.get_storage, eyvan.storage_path, eyvan.get_monitoring_storage, "
        "eyvan.monitoring_path, eyvan.configure_storage, eyvan.StorageHandle); "
        "assert all(fns); "
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'click','cloudpickle','fastapi','httpx','pydantic','sqlalchemy','starlette','uvicorn','yaml'} "
        "or m.startswith('eyvan.sdk') or m.startswith('eyvan.server')); "
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ------------------------------------------------------------------ CLI commands


def test_cli_storage_init_status_and_path_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    backend = FakeBucketBackend()
    fake_ctx = GcpContext(on_gcp=False, project="example-project-a", region=None, zone=None, source="gcloud")
    monkeypatch.setattr(storage_commands, "_backend_factory", lambda: backend)
    monkeypatch.setattr(storage_commands, "_context_detector", lambda _p: fake_ctx)

    runner = CliRunner()

    # Status before init (invoked through root CLI to verify plane-free bypass).
    before = runner.invoke(root_cli, ["storage", "status", "--json"])
    assert before.exit_code == 0, before.output
    before_data = json.loads(before.output)
    assert before_data["configured"] is False
    assert before_data["monitoring"] is None

    # Init with --yes creates monitoring + regional buckets and persists 0o600 config.
    init_res = runner.invoke(
        root_cli,
        [
            "storage",
            "init",
            "--project",
            "example-project-a",
            "--zone",
            "us-central2-b",
            "--auto-create",
            "--yes",
            "--json",
        ],
    )
    assert init_res.exit_code == 0, init_res.output
    init_data = json.loads(init_res.output)
    assert init_data["project"] == "example-project-a"
    assert init_data["auto_create"] is True
    assert init_data["monitoring"]["bucket"] == "eyvan_monitoring_example-project-a"
    assert init_data["monitoring"]["uri"] == "gs://eyvan_monitoring_example-project-a/tracking"
    assert init_data["regional"]["us-central2"]["bucket"] == "eyvan_us-central2"
    assert sorted(init_data["created"]) == ["eyvan_monitoring_example-project-a", "eyvan_us-central2"]
    assert stat.S_IMODE((tmp_path / ".eyvan" / "storage.json").stat().st_mode) == 0o600

    # `eyvan storage path` for regional and monitoring roles.
    reg_path = runner.invoke(root_cli, ["storage", "path", "checkpoints", "step-50", "--region", "us-central2"])
    assert reg_path.exit_code == 0, reg_path.output
    assert reg_path.output.strip() == "gs://eyvan_us-central2/checkpoints/step-50"

    mon_path = runner.invoke(root_cli, ["storage", "path", "default", "rl-exp", "--role", "monitoring"])
    assert mon_path.exit_code == 0, mon_path.output
    assert mon_path.output.strip() == "gs://eyvan_monitoring_example-project-a/tracking/default/rl-exp"

    # Unconfigured region fails without --create, succeeds with --create.
    missing_eu = runner.invoke(root_cli, ["storage", "path", "data", "--region", "europe-west4"])
    assert missing_eu.exit_code != 0
    created_eu = runner.invoke(root_cli, ["storage", "path", "data", "--region", "europe-west4", "--create"])
    assert created_eu.exit_code == 0, created_eu.output
    assert created_eu.output.strip() == "gs://eyvan_europe-west4/data"


def test_cli_storage_init_interactive_prompts_and_respects_decline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    backend = FakeBucketBackend()
    fake_ctx = GcpContext(on_gcp=False, project="example-project-a", region=None, zone=None, source="gcloud")
    monkeypatch.setattr(storage_commands, "_backend_factory", lambda: backend)
    monkeypatch.setattr(storage_commands, "_context_detector", lambda _p: fake_ctx)
    monkeypatch.setattr(storage_commands, "_is_interactive", lambda: True)

    # Blank region (monitoring only), auto_create=n, decline creating monitoring bucket (n).
    res = CliRunner().invoke(storage_commands.storage, ["init"], input="\nn\nn\n")
    assert res.exit_code == 0, res.output
    assert "Skipped bucket creation: monitoring" in res.output
    assert backend.create_calls == []
    saved = StorageConfigStore(tmp_path / ".eyvan" / "storage.json").load()
    assert saved.project == "example-project-a"
    assert saved.monitoring is None


def test_metadata_probe_timeout_negative_caching_storage_status_and_preferred_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eyvan.storage import _gcp

    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    assert _gcp._DEFAULT_METADATA_PROBE_TIMEOUT_S == 0.2

    probe_calls: list[tuple[str, float]] = []

    def fake_default_fetcher(url: str, timeout_s: float):
        probe_calls.append((url, timeout_s))
        return None

    monkeypatch.setattr(_gcp, "_default_metadata_fetcher", fake_default_fetcher)

    first = detect_gcp_context(allow_gcloud=False)
    second = detect_gcp_context(allow_gcloud=False)
    third = detect_gcp_context(config_path=tmp_path / "other.json", allow_gcloud=False)
    assert first.on_gcp is False and second.on_gcp is False and third.on_gcp is False
    # Negative GCE metadata probe is cached across different config paths and repeated calls.
    assert len(probe_calls) == 2  # ip + host once; zero extra calls on 2nd/3rd call
    assert all(timeout == 0.2 for _, timeout in probe_calls)

    # storage_status() returns the exact non-secret payload shape.
    cfg_path = tmp_path / ".eyvan" / "storage.json"
    StorageConfigStore(cfg_path).save(
        StorageConfig(
            project="proj-b",
            auto_create=True,
            monitoring=MonitoringBucketConfig(bucket="eyvan_monitoring_proj-b", location="us-central1"),
            regional={"us-central2": RegionalBucketConfig(bucket="eyvan_us-central2", location="us-central2")},
        )
    )
    snapshot = storage_status(config_path=cfg_path, allow_gcloud=False)
    assert snapshot["configured"] is True
    assert snapshot["auto_create"] is True
    assert snapshot["project"] == "proj-b"
    assert snapshot["config_path"] == str(cfg_path)
    assert isinstance(snapshot["monitoring"], dict) and snapshot["monitoring"]["bucket"] == "eyvan_monitoring_proj-b"
    assert isinstance(snapshot["regional"], dict) and "us-central2" in snapshot["regional"]
    assert isinstance(snapshot["context"], dict) and snapshot["context"]["project"] == "proj-b"

    # select_preferred_gcp_project prioritizes gcp-native-runtime.json > storage.json > fallback.
    assert select_preferred_gcp_project(["proj-a", "proj-b"], allow_gcloud=False) == "proj-b"
    (tmp_path / ".eyvan" / "gcp-native-runtime.json").write_text(json.dumps({"project_id": "proj-a"}))
    assert select_preferred_gcp_project(["proj-a", "proj-b"], allow_gcloud=False) == "proj-a"
    assert select_preferred_gcp_project(["proj-c", "proj-d"], allow_gcloud=False) == "proj-c"
    assert select_preferred_gcp_project([], allow_gcloud=False) is None

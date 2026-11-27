# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comprehensive unit and subprocess-kill tests for Eyvan's Layer 1 cross-region GCS guard."""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import typing as tp
import urllib.request
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.storage import (
    CROSS_REGION_ALLOW_MULTIREGION_ENV,
    CROSS_REGION_ALLOWED_BUCKETS_ENV,
    CROSS_REGION_KILL_ENV,
    CROSS_REGION_POLICY_ENV,
    CROSS_REGION_TRANSFER_BANNED_EXIT_CODE,
    CROSS_REGION_TRANSFER_BANNED_MARKER,
    CROSS_REGION_TRANSFER_BANNED_REASON,
    HOST_REGION_ENV,
    BucketLocation,
    CrossRegionTransferError,
    GcsBucketBackend,
    LocalityCheckResult,
    _region_guard,
    arm_cross_region_guard_from_environment,
    check_gcs_uri_locality,
    clear_bucket_location_cache,
    cross_region_guard,
    disable_cross_region_guard,
    enable_cross_region_guard,
    extract_bucket_from_gcs_http_url,
    extract_bucket_from_gcs_uri,
    extract_gcs_uris,
    is_location_local_to_region,
    resolve_bucket_location,
)
from eyvan.worker.distributed_environment import (
    is_distributed_topology_environment_name,
    is_runtime_owned_environment_name,
    remove_gce_tpu_topology_overrides,
    without_distributed_topology_overrides,
)
from eyvan.worker.runtime import _multihost_site, callable_protocol
from eyvan.worker.runtime.local_process import LocalProcessExecutor
from eyvan.worker.runtime.native_bootstrap import build_native_worker_startup_script

EYVAN_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_guard_state() -> tp.Iterator[None]:
    """Ensure every test begins and ends with the guard disarmed and caches cleared."""

    disable_cross_region_guard()
    clear_bucket_location_cache()
    try:
        yield
    finally:
        disable_cross_region_guard()
        clear_bucket_location_cache()


def _map_resolver(mapping: dict[str, BucketLocation]) -> tp.Callable[[str], BucketLocation | None]:
    def _resolve(bucket: str) -> BucketLocation | None:
        return mapping.get(bucket.strip().lower())

    return _resolve


_DEFAULT_BUCKETS: dict[str, BucketLocation] = {
    "example-us-bucket": BucketLocation("example-us-bucket", "us-central2", "region"),
    "example-eu-bucket": BucketLocation("example-eu-bucket", "europe-west4", "region"),
    "example-east-bucket": BucketLocation("example-east-bucket", "us-east1", "region"),
    "example-eur4-bucket": BucketLocation("example-eur4-bucket", "eur4", "dual-region"),
    "example-custom-dual": BucketLocation(
        "example-custom-dual",
        "us",
        "dual-region",
        ("europe-west4", "us-central1"),
    ),
    "example-eu-multi": BucketLocation("example-eu-multi", "eu", "multi-region"),
    "example-us-multi": BucketLocation("example-us-multi", "us", "multi-region"),
    "example-asia-multi": BucketLocation("example-asia-multi", "asia", "multi-region"),
}


# ------------------------------------------------------------------ stdlib & standalone invariants


def test_region_guard_is_pure_stdlib_and_execs_unregistered() -> None:
    """The staged file runs via ``spec_from_file_location`` without ``sys.modules`` or third-party deps."""

    module_path = Path(_region_guard.__file__)
    spec = spec_from_file_location("eyvan_region_guard_unregistered", module_path)
    assert spec is not None and spec.loader is not None
    standalone = module_from_spec(spec)
    sys.modules.pop("eyvan_region_guard_unregistered", None)
    spec.loader.exec_module(standalone)

    assert standalone.CROSS_REGION_TRANSFER_BANNED_EXIT_CODE == 95
    assert standalone.is_location_local_to_region(
        "europe-west4-a",
        standalone.BucketLocation("example-eu-bucket", "europe-west4", "region"),
    )
    assert not standalone.is_location_local_to_region(
        "europe-west4",
        standalone.BucketLocation("example-us-bucket", "us-central2", "region"),
    )

    probe = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('eyvan_region_guard', {str(module_path)!r}); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "non_stdlib = sorted("
        "m for m in sys.modules "
        "if m != 'eyvan_region_guard' "
        "and m.split('.')[0] not in sys.stdlib_module_names "
        "and not m.startswith('_')"
        "); "
        "print(non_stdlib)"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


# ------------------------------------------------------------------ region locality matrix


@pytest.mark.parametrize(
    ("host_region", "bucket_loc", "allow_multiregion", "expected"),
    [
        # Single-region exact match & mismatch
        ("europe-west4", BucketLocation("example-us-bucket", "us-central2", "region"), True, False),
        ("europe-west4", BucketLocation("example-eu-bucket", "europe-west4", "region"), True, True),
        ("EUROPE-WEST4-A", BucketLocation("example-eu-bucket", "EUROPE-WEST4", "REGION"), True, True),
        (
            "projects/123456/zones/europe-west4-b",
            BucketLocation("example-eu-bucket", "europe-west4", "region"),
            True,
            True,
        ),
        ("us-central2", BucketLocation("example-east-bucket", "us-east1", "region"), True, False),
        ("us-central1-ai1a", BucketLocation("example-c1", "us-central1", "region"), True, True),
        # Predefined GCP dual-regions
        ("europe-west4", BucketLocation("example-eur4", "eur4", "dual-region"), True, True),
        ("europe-north1", BucketLocation("example-eur4", "eur4", "dual-region"), True, True),
        ("us-central2", BucketLocation("example-eur4", "eur4", "dual-region"), True, False),
        ("us-central1", BucketLocation("example-nam4", "nam4", "dual-region"), True, True),
        ("us-east1", BucketLocation("example-nam4", "nam4", "dual-region"), True, True),
        ("us-central2", BucketLocation("example-nam4", "nam4", "dual-region"), True, False),
        ("europe-west1", BucketLocation("example-eur5", "eur5", "dual-region"), True, True),
        ("europe-west3", BucketLocation("example-eur7", "eur7", "dual-region"), True, True),
        ("europe-west6", BucketLocation("example-eur8", "eur8", "dual-region"), True, True),
        ("asia-northeast1", BucketLocation("example-asia1", "asia1", "dual-region"), True, True),
        # Custom dual-regions with explicit data_locations
        (
            "europe-west4",
            BucketLocation("example-custom", "us", "dual-region", ("EUROPE-WEST4", "US-CENTRAL1")),
            True,
            True,
        ),
        (
            "us-central1",
            BucketLocation("example-custom", "us", "dual-region", ("europe-west4", "us-central1")),
            False,
            True,
        ),
        (
            "us-central2",
            BucketLocation("example-custom", "us", "dual-region", ("europe-west4", "us-central1")),
            True,
            False,
        ),
        # Multi-regions with allow_multiregion=True vs False
        ("europe-west4", BucketLocation("example-eu-multi", "eu", "multi-region"), True, True),
        ("europe-west4", BucketLocation("example-eu-multi", "eu", "multi-region"), False, False),
        ("europe-west4", BucketLocation("example-us-multi", "us", "multi-region"), True, False),
        ("europe-west4", BucketLocation("example-us-multi", "us", "multi-region"), False, False),
        ("us-central2", BucketLocation("example-us-multi", "us", "multi-region"), True, True),
        ("us-central2", BucketLocation("example-us-multi", "us", "multi-region"), False, False),
        ("asia-northeast1", BucketLocation("example-asia-multi", "asia", "multi-region"), True, True),
        ("asia-northeast1", BucketLocation("example-asia-multi", "asia", "multi-region"), False, False),
        ("us-central1", BucketLocation("example-asia-multi", "asia", "multi-region"), True, False),
        # Multi-region inferred even if location_type is unknown
        ("europe-west4", BucketLocation("example-eu-multi", "eu", "unknown"), True, True),
        ("europe-west4", BucketLocation("example-us-multi", "us", "unknown"), True, False),
        # Unknown / unrecognized
        ("europe-west4", BucketLocation("example-unknown", "mars-north1", "unknown"), True, False),
        ("", BucketLocation("example-eu-bucket", "europe-west4", "region"), True, False),
    ],
)
def test_region_locality_matrix(
    host_region: str,
    bucket_loc: BucketLocation,
    allow_multiregion: bool,
    expected: bool,
) -> None:
    assert (
        is_location_local_to_region(
            host_region,
            bucket_loc,
            allow_multiregion=allow_multiregion,
        )
        is expected
    )


def test_check_gcs_uri_locality_allowlist_and_unknown_policy() -> None:
    resolver = _map_resolver(_DEFAULT_BUCKETS)

    denied = check_gcs_uri_locality(
        "gs://example-us-bucket/checkpoints/step-10",
        host_region="europe-west4",
        operation="read",
        location_resolver=resolver,
    )
    assert denied.allowed is False
    assert denied.host_region == "europe-west4"
    assert denied.bucket == "example-us-bucket"
    assert denied.bucket_location == "us-central2"
    assert denied.location_type == "region"

    allowlisted = check_gcs_uri_locality(
        "gs://example-us-bucket/checkpoints/step-10",
        host_region="europe-west4",
        operation="read",
        allowed_buckets=("example-us-bucket",),
        location_resolver=resolver,
    )
    assert allowlisted.allowed is True
    assert allowlisted.location_type == "allowlisted"

    unknown_allowed = check_gcs_uri_locality(
        "gs://example-mystery-bucket/data",
        host_region="europe-west4",
        location_resolver=resolver,
        on_unknown="allow",
    )
    assert unknown_allowed.allowed is True

    unknown_banned = check_gcs_uri_locality(
        "gs://example-mystery-bucket/data",
        host_region="europe-west4",
        location_resolver=resolver,
        on_unknown="ban",
    )
    assert unknown_banned.allowed is False


# ------------------------------------------------------------------ URI & HTTP URL parsing


def test_uri_and_http_url_extraction() -> None:
    text = (
        "gcloud storage cp gs://example-us-bucket/ckpt/step-1 gs://example-eu-bucket/out/ "
        '{"uri": "gs://example-us-bucket/ckpt/step-1", "extra": "gs://example-eur4-bucket"}'
    )
    assert extract_gcs_uris(text) == (
        "gs://example-us-bucket/ckpt/step-1",
        "gs://example-eu-bucket/out/",
        "gs://example-eur4-bucket",
    )
    assert extract_bucket_from_gcs_uri("gs://example-us-bucket/path/to/file") == "example-us-bucket"
    assert extract_bucket_from_gcs_uri("https://storage.googleapis.com/example-us-bucket/a") is None
    assert extract_bucket_from_gcs_uri("gs://user:pass@example-us-bucket/a") is None

    assert (
        extract_bucket_from_gcs_http_url(
            "https://storage.googleapis.com/storage/v1/b/example-us-bucket/o/ckpt%2Fstep?alt=media"
        )
        == "example-us-bucket"
    )
    assert (
        extract_bucket_from_gcs_http_url(
            "https://storage.googleapis.com/upload/storage/v1/b/example-us-bucket/o?uploadType=resumable"
        )
        == "example-us-bucket"
    )
    assert (
        extract_bucket_from_gcs_http_url("https://storage.googleapis.com/example-us-bucket/dir/blob.bin")
        == "example-us-bucket"
    )
    assert (
        extract_bucket_from_gcs_http_url("https://example-us-bucket.storage.googleapis.com/dir/blob.bin")
        == "example-us-bucket"
    )
    assert (
        extract_bucket_from_gcs_http_url("https://storage.cloud.google.com/example-us-bucket/dir/blob.bin")
        == "example-us-bucket"
    )
    assert (
        extract_bucket_from_gcs_http_url("/storage/v1/b/example-us-bucket/o/item", host="storage.googleapis.com")
        == "example-us-bucket"
    )
    assert extract_bucket_from_gcs_http_url("/dir/blob.bin", host="example-us-bucket.storage.googleapis.com") == (
        "example-us-bucket"
    )
    # Non-GCS hosts are ignored.
    assert extract_bucket_from_gcs_http_url("https://huggingface.co/example-us-bucket/model.bin") is None
    assert (
        extract_bucket_from_gcs_http_url(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        )
        is None
    )
    assert extract_bucket_from_gcs_http_url("/v1/items", host="api.example.com") is None

    # Metadata-only query is ignored when the re-entrancy bypass is set.
    meta_url = (
        "https://storage.googleapis.com/storage/v1/b/example-us-bucket"
        "?fields=location,locationType,customPlacementConfig"
    )
    assert extract_bucket_from_gcs_http_url(meta_url) == "example-us-bucket"
    with _region_guard._bypass_region_guard():
        assert extract_bucket_from_gcs_http_url(meta_url) is None


# ------------------------------------------------------------------ bucket location resolver & re-entrancy


def test_resolve_bucket_location_caches_and_bypasses_active_hooks() -> None:
    calls: list[tuple[str, str, dict[str, str]]] = []

    def fake_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, dict[str, str], bytes]:
        del body
        calls.append((method, url, headers))
        # Confirm re-entrancy bypass is active while resolving metadata.
        assert _region_guard._is_bypassed() is True
        return (
            200,
            {},
            json.dumps(
                {
                    "location": "US",
                    "locationType": "dual-region",
                    "customPlacementConfig": {"dataLocations": ["EUROPE-WEST4", "US-CENTRAL1"]},
                }
            ).encode("utf-8"),
        )

    with cross_region_guard(host_region="europe-west4", mode="ban"):
        loc1 = resolve_bucket_location(
            "example-custom-dual",
            transport=fake_transport,
            token_provider=lambda: "example-token-1",
        )
        loc2 = resolve_bucket_location(
            "example-custom-dual",
            transport=fake_transport,
            token_provider=lambda: "example-token-1",
        )

    assert loc1 == BucketLocation(
        bucket="example-custom-dual",
        location="us",
        location_type="dual-region",
        data_locations=("europe-west4", "us-central1"),
    )
    assert loc2 == loc1
    assert len(calls) == 1
    assert calls[0][0] == "GET"
    assert "fields=location,locationType,customPlacementConfig" in calls[0][1]
    assert calls[0][2]["Authorization"] == "Bearer example-token-1"


# ------------------------------------------------------------------ runtime hooks


def test_urllib_hook_blocks_cross_region_and_allows_local(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[str] = []

    def fake_orig_open(
        self: urllib.request.OpenerDirector,
        fullurl: str | urllib.request.Request,
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> str:
        del self, args, kwargs
        url = fullurl.full_url if isinstance(fullurl, urllib.request.Request) else str(fullurl)
        opened.append(url)
        return f"ok:{url}"

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_orig_open)
    violations: list[LocalityCheckResult] = []

    with cross_region_guard(
        host_region="europe-west4",
        mode="ban",
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
        on_violation=violations.append,
    ):
        # Cross-region bucket is blocked BEFORE opener runs.
        with pytest.raises(CrossRegionTransferError) as exc_info:
            urllib.request.urlopen("https://storage.googleapis.com/example-us-bucket/ckpt.bin")
        assert exc_info.value.host_region == "europe-west4"
        assert exc_info.value.bucket == "example-us-bucket"
        assert exc_info.value.bucket_location == "us-central2"
        assert opened == []
        assert len(violations) == 1
        err = capsys.readouterr().err
        assert CROSS_REGION_TRANSFER_BANNED_MARKER in err
        assert "host_region=europe-west4" in err
        assert "bucket=example-us-bucket" in err
        assert "bucket_location=us-central2" in err

        # Local bucket and non-GCS URL pass through cleanly.
        assert urllib.request.urlopen("https://storage.googleapis.com/example-eu-bucket/ckpt.bin") == (
            "ok:https://storage.googleapis.com/example-eu-bucket/ckpt.bin"
        )
        assert urllib.request.urlopen("https://example.org/data.json") == "ok:https://example.org/data.json"
        assert len(opened) == 2

    # Clean restoration.
    assert urllib.request.OpenerDirector.open is fake_orig_open


def test_http_client_hook_blocks_cross_region_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, str, str]] = []

    def fake_orig_request(
        self: http.client.HTTPConnection,
        method: str,
        url: str,
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> None:
        del args, kwargs
        sent.append((self.host, method, url))

    monkeypatch.setattr(http.client.HTTPConnection, "request", fake_orig_request)

    with cross_region_guard(
        host_region="europe-west4",
        mode="ban",
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
    ):
        conn = http.client.HTTPSConnection("storage.googleapis.com")
        with pytest.raises(CrossRegionTransferError) as err:
            conn.request("GET", "/storage/v1/b/example-us-bucket/o/weights.safetensors?alt=media")
        assert err.value.bucket == "example-us-bucket"
        assert err.value.bucket_location == "us-central2"
        assert sent == []

        vhost_conn = http.client.HTTPSConnection("example-us-bucket.storage.googleapis.com")
        with pytest.raises(CrossRegionTransferError):
            vhost_conn.request("PUT", "/weights.safetensors")
        assert sent == []

        # Local bucket succeeds.
        conn.request("GET", "/storage/v1/b/example-eu-bucket/o/weights.safetensors?alt=media")
        assert sent == [
            ("storage.googleapis.com", "GET", "/storage/v1/b/example-eu-bucket/o/weights.safetensors?alt=media")
        ]

    assert http.client.HTTPConnection.request is fake_orig_request


def test_subprocess_popen_hook_blocks_gcloud_storage_and_gsutil(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[tp.Any] = []

    def fake_orig_init(self: subprocess.Popen[tp.Any], args: tp.Any, *pargs: tp.Any, **kwargs: tp.Any) -> None:
        del pargs, kwargs
        spawned.append(args)
        self.returncode = 0

    monkeypatch.setattr(subprocess.Popen, "__init__", fake_orig_init)

    with cross_region_guard(
        host_region="europe-west4",
        mode="ban",
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
    ):
        with pytest.raises(CrossRegionTransferError) as err1:
            subprocess.Popen(["gcloud", "storage", "cp", "gs://example-us-bucket/a", "."])
        assert err1.value.bucket == "example-us-bucket"
        assert err1.value.operation == "subprocess.gcloud"

        with pytest.raises(CrossRegionTransferError) as err2:
            subprocess.Popen(["gsutil", "-m", "cp", "gs://example-us-bucket/a", "."])
        assert err2.value.bucket == "example-us-bucket"
        assert err2.value.operation == "subprocess.gsutil"

        with pytest.raises(CrossRegionTransferError) as err3:
            subprocess.Popen(["/bin/sh", "-c", "gcloud storage cp gs://example-us-bucket/a ."])
        assert err3.value.bucket == "example-us-bucket"

        with pytest.raises(CrossRegionTransferError) as err4:
            subprocess.Popen("gsutil cp gs://example-us-bucket/a .", shell=True)
        assert err4.value.bucket == "example-us-bucket"

        assert spawned == []

        # Local bucket & non-storage gcloud commands pass cleanly.
        subprocess.Popen(["gcloud", "storage", "cp", "gs://example-eu-bucket/a", "."])
        subprocess.Popen(["gcloud", "auth", "print-access-token"])
        assert len(spawned) == 2

    assert subprocess.Popen.__init__ is fake_orig_init


def test_tensorstore_hook_preimported_and_lazy_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Pre-imported fake tensorstore module.
    opened_specs: list[tp.Any] = []
    kv_specs: list[tp.Any] = []

    fake_ts = ModuleType("tensorstore")

    def ts_open(spec: tp.Any, **_kwargs: tp.Any) -> str:
        opened_specs.append(spec)
        return "opened"

    class FakeKvStore:
        @staticmethod
        def open(spec: tp.Any, **_kwargs: tp.Any) -> str:
            kv_specs.append(spec)
            return "kv-opened"

    class FakeSpec:
        def __init__(self, payload: dict[str, tp.Any]) -> None:
            self._payload = payload

        def to_json(self) -> dict[str, tp.Any]:
            return self._payload

    fake_ts.open = ts_open  # type: ignore[attr-defined]
    fake_ts.KvStore = FakeKvStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorstore", fake_ts)

    ocdbt_us_spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "ocdbt",
            "base": {
                "driver": "gcs",
                "bucket": "example-us-bucket",
                "path": "checkpoints/step-100/",
            },
        },
    }
    ocdbt_eu_spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "ocdbt",
            "base": {
                "driver": "gcs",
                "bucket": "example-eu-bucket",
                "path": "checkpoints/step-100/",
            },
        },
    }

    with cross_region_guard(
        host_region="europe-west4",
        mode="ban",
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
    ):
        with pytest.raises(CrossRegionTransferError) as err:
            fake_ts.open(ocdbt_us_spec)  # type: ignore[attr-defined]
        assert err.value.bucket == "example-us-bucket"
        assert err.value.operation == "tensorstore.open"

        with pytest.raises(CrossRegionTransferError):
            fake_ts.open(FakeSpec(ocdbt_us_spec))  # type: ignore[attr-defined]

        with pytest.raises(CrossRegionTransferError) as kv_err:
            fake_ts.KvStore.open({"driver": "gcs", "bucket": "example-us-bucket", "path": "x"})  # type: ignore[attr-defined]
        assert kv_err.value.bucket == "example-us-bucket"
        assert kv_err.value.operation == "tensorstore.KvStore.open"

        with pytest.raises(CrossRegionTransferError):
            fake_ts.KvStore.open("gs://example-us-bucket/ckpt")  # type: ignore[attr-defined]

        assert opened_specs == []
        assert kv_specs == []

        assert fake_ts.open(ocdbt_eu_spec) == "opened"  # type: ignore[attr-defined]
        assert fake_ts.KvStore.open("gs://example-eu-bucket/ckpt") == "kv-opened"  # type: ignore[attr-defined]

    # Restored cleanly on exit.
    assert fake_ts.open is ts_open  # type: ignore[attr-defined]
    assert fake_ts.KvStore.open is FakeKvStore.open  # type: ignore[attr-defined]

    # 2. Lazy import AFTER arming via sys.meta_path.
    monkeypatch.delitem(sys.modules, "tensorstore", raising=False)
    lazy_dir = tmp_path / "lazy_ts"
    lazy_dir.mkdir()
    (lazy_dir / "tensorstore.py").write_text(
        "calls = []\n"
        "def open(spec, **kwargs):\n"
        "    calls.append(('open', spec))\n"
        "    return 'lazy-ok'\n"
        "class KvStore:\n"
        "    @classmethod\n"
        "    def open(cls, spec, **kwargs):\n"
        "        calls.append(('kv', spec))\n"
        "        return 'lazy-kv-ok'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(lazy_dir))
    try:
        with cross_region_guard(
            host_region="europe-west4",
            mode="ban",
            location_resolver=_map_resolver(_DEFAULT_BUCKETS),
        ):
            import importlib

            lazy_ts = importlib.import_module("tensorstore")
            with pytest.raises(CrossRegionTransferError):
                lazy_ts.open(ocdbt_us_spec)
            with pytest.raises(CrossRegionTransferError):
                lazy_ts.KvStore.open({"driver": "gcs", "bucket": "example-us-bucket"})
            assert lazy_ts.calls == []
            assert lazy_ts.open(ocdbt_eu_spec) == "lazy-ok"
            assert lazy_ts.KvStore.open("gs://example-eu-bucket/x") == "lazy-kv-ok"
    finally:
        sys.modules.pop("tensorstore", None)


def test_gcsfs_hook_blocks_cross_region_sync_and_async(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tp.Any]] = []
    fake_gcsfs = ModuleType("gcsfs")

    class FakeBaseFS:
        def open(self, path: str, mode: str = "rb") -> str:
            calls.append(("open", (path, mode)))
            return "file-handle"

    class FakeGCSFileSystem(FakeBaseFS):
        def cat(self, path: tp.Any) -> bytes:
            calls.append(("cat", path))
            return b"data"

        def pipe(self, path: tp.Any, value: bytes | None = None) -> None:
            calls.append(("pipe", (path, value)))

        def get(self, rpath: tp.Any, lpath: str) -> None:
            calls.append(("get", (rpath, lpath)))

        def put(self, lpath: str, rpath: tp.Any) -> None:
            calls.append(("put", (lpath, rpath)))

        async def _call(self, method: str, path: str, *_args: tp.Any, **_kwargs: tp.Any) -> dict[str, str]:
            calls.append(("_call", (method, path)))
            return {"ok": "1"}

    fake_gcsfs.GCSFileSystem = FakeGCSFileSystem  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gcsfs", fake_gcsfs)

    with cross_region_guard(
        host_region="europe-west4",
        mode="ban",
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
    ):
        fs = FakeGCSFileSystem()
        with pytest.raises(CrossRegionTransferError):
            fs.open("example-us-bucket/train.parquet")
        with pytest.raises(CrossRegionTransferError):
            fs.cat(["gs://example-eu-bucket/ok.bin", "example-us-bucket/bad.bin"])
        with pytest.raises(CrossRegionTransferError):
            fs.pipe({"example-us-bucket/out.bin": b"123"})
        with pytest.raises(CrossRegionTransferError):
            fs.get("example-us-bucket/src.bin", "/tmp/dst.bin")
        with pytest.raises(CrossRegionTransferError):
            fs.put("/tmp/src.bin", "gs://example-us-bucket/dst.bin")
        with pytest.raises(CrossRegionTransferError):
            asyncio.run(fs._call("GET", "b/example-us-bucket/o/item"))
        assert calls == []

        # Local bucket operations succeed.
        assert fs.open("example-eu-bucket/train.parquet") == "file-handle"
        assert fs.cat("gs://example-eu-bucket/ok.bin") == b"data"
        fs.pipe("example-eu-bucket/out.bin", b"123")
        fs.put("/tmp/src.bin", "example-eu-bucket/dst.bin")
        assert asyncio.run(fs._call("GET", "b/example-eu-bucket/o/item")) == {"ok": "1"}
        assert len(calls) == 5

    # Inherited method was removed from subclass __dict__ cleanly on restore.
    assert "open" not in FakeGCSFileSystem.__dict__


def test_warn_mode_vs_ban_mode_and_disable_idempotence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[str] = []

    def fake_open(self: urllib.request.OpenerDirector, fullurl: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> str:
        del self, args, kwargs
        url = fullurl.full_url if isinstance(fullurl, urllib.request.Request) else str(fullurl)
        opened.append(url)
        return "allowed-in-warn"

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
    warn_events: list[LocalityCheckResult] = []

    with cross_region_guard(
        host_region="europe-west4",
        mode="warn",
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
        on_violation=warn_events.append,
    ):
        assert urllib.request.urlopen("https://storage.googleapis.com/example-us-bucket/a") == "allowed-in-warn"
        assert len(opened) == 1
        assert len(warn_events) == 1
        stderr = capsys.readouterr().err
        assert "eyvan-cross-region-transfer-warning" in stderr
        assert "bucket=example-us-bucket" in stderr

    # Ban mode with kill_on_violation=True invokes indirected terminate_on_violation(95).
    exits: list[int] = []

    class _Killed(BaseException):
        pass

    def fake_terminate(code: int) -> tp.NoReturn:
        exits.append(code)
        raise _Killed()

    monkeypatch.setattr(_region_guard, "terminate_on_violation", fake_terminate)
    with cross_region_guard(
        host_region="europe-west4",
        mode="ban",
        kill_on_violation=True,
        location_resolver=_map_resolver(_DEFAULT_BUCKETS),
    ):
        with pytest.raises(_Killed):
            urllib.request.urlopen("https://storage.googleapis.com/example-us-bucket/a")
        assert exits == [CROSS_REGION_TRANSFER_BANNED_EXIT_CODE]
        assert CROSS_REGION_TRANSFER_BANNED_MARKER in capsys.readouterr().err

    # Allow mode is a no-op.
    assert enable_cross_region_guard(host_region="europe-west4", mode="allow") is False
    # Calling disable multiple times is idempotent and leaves meta_path clean.
    meta_before = list(sys.meta_path)
    disable_cross_region_guard()
    disable_cross_region_guard()
    assert list(sys.meta_path) == meta_before
    assert urllib.request.OpenerDirector.open is fake_open


def test_gcs_bucket_backend_populates_location_type_and_enforces_active_guard() -> None:
    requests_seen: list[tuple[str, str]] = []

    def fake_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, dict[str, str], bytes]:
        del headers, body
        requests_seen.append((method, url))
        if method == "GET" and url.endswith("/b/example-custom-dual"):
            return (
                200,
                {},
                json.dumps(
                    {
                        "name": "example-custom-dual",
                        "location": "US",
                        "locationType": "dual-region",
                        "customPlacementConfig": {"dataLocations": ["EUROPE-WEST4", "US-CENTRAL1"]},
                        "storageClass": "STANDARD",
                    }
                ).encode("utf-8"),
            )
        if method == "GET" and url.endswith("/b/example-us-bucket"):
            return (
                200,
                {},
                json.dumps(
                    {
                        "name": "example-us-bucket",
                        "location": "US-CENTRAL2",
                        "locationType": "region",
                        "storageClass": "STANDARD",
                    }
                ).encode("utf-8"),
            )
        if method == "POST":
            return (
                200,
                {},
                json.dumps(
                    {
                        "name": "example-eu-bucket",
                        "location": "EUROPE-WEST4",
                        "locationType": "region",
                        "storageClass": "STANDARD",
                    }
                ).encode("utf-8"),
            )
        return 404, {}, b"{}"

    backend = GcsBucketBackend(token_provider=lambda: "example-token-1", transport=fake_transport)
    dual = backend.get_bucket("example-custom-dual")
    assert dual is not None
    assert dual.location_type == "dual-region"
    assert dual.data_locations == ("europe-west4", "us-central1")

    with cross_region_guard(host_region="europe-west4", mode="ban"):
        # Dual-region including europe-west4 is permitted.
        assert backend.get_bucket("example-custom-dual") is not None
        # Cross-region get_bucket is blocked.
        with pytest.raises(CrossRegionTransferError) as get_err:
            backend.get_bucket("example-us-bucket")
        assert get_err.value.bucket == "example-us-bucket"
        assert get_err.value.bucket_location == "us-central2"
        # Cross-region create_bucket is blocked BEFORE sending HTTP POST.
        post_count_before = sum(1 for m, _ in requests_seen if m == "POST")
        with pytest.raises(CrossRegionTransferError):
            backend.create_bucket(
                "example-us-new",
                project="example-project-a",
                location="us-central2",
            )
        assert sum(1 for m, _ in requests_seen if m == "POST") == post_count_before
        # Same-region create_bucket succeeds.
        created = backend.create_bucket(
            "example-eu-bucket",
            project="example-project-a",
            location="europe-west4",
        )
        assert created.location == "europe-west4"
        assert created.location_type == "region"


# ------------------------------------------------------------------ worker startup & env ownership


def test_distributed_environment_reserves_and_strips_cross_region_vars() -> None:
    protected = (
        HOST_REGION_ENV,
        CROSS_REGION_POLICY_ENV,
        CROSS_REGION_ALLOWED_BUCKETS_ENV,
        CROSS_REGION_ALLOW_MULTIREGION_ENV,
        CROSS_REGION_KILL_ENV,
    )
    for name in protected:
        assert is_distributed_topology_environment_name(name), name
        assert is_runtime_owned_environment_name(name), name

    user_env = {
        "USER_VAR": "kept",
        HOST_REGION_ENV: "us-central2",
        CROSS_REGION_POLICY_ENV: "allow",
        CROSS_REGION_KILL_ENV: "0",
        CROSS_REGION_ALLOWED_BUCKETS_ENV: "example-us-bucket",
        CROSS_REGION_ALLOW_MULTIREGION_ENV: "1",
    }
    stripped = without_distributed_topology_overrides(user_env)
    assert stripped == {"USER_VAR": "kept"}

    # GCE TPU topology scrubbing must NOT drop EYVAN_HOST_REGION.
    tpu_env = {HOST_REGION_ENV: "europe-west4", "GCE_METADATA_IP": "spoof"}
    remove_gce_tpu_topology_overrides(tpu_env)
    assert tpu_env == {HOST_REGION_ENV: "europe-west4"}


def test_multihost_site_native_bootstrap_and_callable_protocol_wiring(tmp_path: Path) -> None:
    # _multihost_site exports exact ban constants and arms guard on initialize_worker_runtime.
    assert _multihost_site.CROSS_REGION_TRANSFER_BANNED_EXIT_CODE == 95
    assert _multihost_site.CROSS_REGION_TRANSFER_BANNED_MARKER == CROSS_REGION_TRANSFER_BANNED_MARKER
    assert _multihost_site.CROSS_REGION_TRANSFER_BANNED_REASON == CROSS_REGION_TRANSFER_BANNED_REASON

    _multihost_site.initialize_worker_runtime(
        {
            CROSS_REGION_POLICY_ENV: "ban",
            HOST_REGION_ENV: "europe-west4",
            CROSS_REGION_KILL_ENV: "0",
        }
    )
    assert _region_guard.is_cross_region_guard_active() is True
    disable_cross_region_guard()

    # native_bootstrap stages eyvan_region_guard.py alongside eyvan_multihost.py.
    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/example-project-a/secrets/eyvan-native-cell/versions/latest",
    )
    assert "import eyvan.storage._region_guard as module" in script
    assert "/opt/eyvan-native/multihost/eyvan_region_guard.py" in script

    # callable_protocol arms guard during execute_callable when policy is set in environment.
    artifact_file = tmp_path / "callable.pkl"

    def _probe_callable() -> bool:
        return _region_guard.is_cross_region_guard_active()

    artifact = callable_protocol.dump_callable(_probe_callable)
    artifact_file.write_bytes(artifact.payload)

    class _UnusedGcs:
        def read_object(self, bucket_name: str, object_name: str) -> bytes:
            raise AssertionError(f"unexpected GCS read: {bucket_name}/{object_name}")

        def write_object(self, bucket_name: str, object_name: str, payload: bytes) -> None:
            raise AssertionError(f"unexpected GCS write: {bucket_name}/{object_name}")

    callable_env = {
        CROSS_REGION_POLICY_ENV: "ban",
        HOST_REGION_ENV: "europe-west4",
        CROSS_REGION_KILL_ENV: "0",
    }
    rc = callable_protocol.execute_callable(
        [
            "--uri",
            f"file://{artifact_file}",
            "--sha256",
            artifact.sha256,
            "--python",
            artifact.python,
        ],
        transfer=callable_protocol.UriArtifactTransfer(_UnusedGcs(), callable_env),
        environment=callable_env,
    )
    assert rc == 0
    assert _region_guard.is_cross_region_guard_active() is True
    disable_cross_region_guard()

    # arm_cross_region_guard_from_environment parses allowlist & multiregion flags.
    assert (
        arm_cross_region_guard_from_environment(
            {
                CROSS_REGION_POLICY_ENV: "ban",
                HOST_REGION_ENV: "europe-west4",
                CROSS_REGION_ALLOWED_BUCKETS_ENV: "example-us-bucket, example-other",
                CROSS_REGION_ALLOW_MULTIREGION_ENV: "0",
                CROSS_REGION_KILL_ENV: "0",
            },
            location_resolver=_map_resolver(_DEFAULT_BUCKETS),
        )
        is True
    )
    # Multi-region EU is now blocked because ALLOW_MULTIREGION=0, while allowlisted US bucket passes.
    with pytest.raises(CrossRegionTransferError):
        _region_guard.enforce_gcs_access("gs://example-eu-multi/ckpt", operation="read")
    allowed = _region_guard.enforce_gcs_access("gs://example-us-bucket/ckpt", operation="read")
    assert allowed is not None and allowed.allowed is True


# ------------------------------------------------------------------ REAL child Python subprocess kill test


def _stage_multihost_site(site_dir: Path) -> None:
    """Stage the exact files ``native_bootstrap`` writes into ``/opt/eyvan-native/multihost``."""

    site_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(_multihost_site.__file__), site_dir / "eyvan_multihost.py")
    from eyvan.worker import distributed_environment

    shutil.copyfile(Path(distributed_environment.__file__), site_dir / "eyvan_distributed_environment.py")
    shutil.copyfile(Path(_region_guard.__file__), site_dir / "eyvan_region_guard.py")
    (site_dir / "sitecustomize.py").write_text(
        "from eyvan_multihost import initialize_worker_runtime\n\ninitialize_worker_runtime()\n",
        encoding="utf-8",
    )


def test_real_subprocess_killed_with_exit_95_even_inside_try_except(tmp_path: Path) -> None:
    """A real Python subprocess armed via ``sitecustomize`` is killed with exit 95 on violation.

    User code wrapping the cross-region read in ``try: ... except Exception: pass``
    MUST NOT be able to swallow the ban.
    """

    site_dir = tmp_path / "multihost_site"
    _stage_multihost_site(site_dir)

    child_code = (
        "import urllib.request\n"
        "import eyvan_region_guard\n"
        "eyvan_region_guard.prime_bucket_location_cache(\n"
        "    eyvan_region_guard.BucketLocation('example-us-bucket', 'us-central2', 'region')\n"
        ")\n"
        "try:\n"
        "    urllib.request.urlopen('https://storage.googleapis.com/example-us-bucket/model/ckpt.bin')\n"
        "except Exception as exc:\n"
        "    print(f'SWALLOWED:{exc}', flush=True)\n"
        "print('SURVIVED', flush=True)\n"
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(site_dir),
        CROSS_REGION_POLICY_ENV: "ban",
        HOST_REGION_ENV: "europe-west4",
        CROSS_REGION_KILL_ENV: "1",
    }

    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == CROSS_REGION_TRANSFER_BANNED_EXIT_CODE, (
        f"expected exit 95, got {completed.returncode}; stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )
    assert "SWALLOWED" not in completed.stdout
    assert "SURVIVED" not in completed.stdout
    assert CROSS_REGION_TRANSFER_BANNED_MARKER in completed.stderr
    assert "host_region=europe-west4" in completed.stderr
    assert "bucket=example-us-bucket" in completed.stderr
    assert "bucket_location=us-central2" in completed.stderr


def test_local_process_executor_surfaces_cross_region_ban_from_exit_95(tmp_path: Path) -> None:
    """``LocalProcessExecutor`` recognizes exit 95 and surfaces ``cross-region-transfer-banned: <detail>``."""

    site_dir = tmp_path / "multihost_site"
    _stage_multihost_site(site_dir)

    child_code = (
        "import urllib.request, eyvan_region_guard; "
        "eyvan_region_guard.prime_bucket_location_cache("
        "eyvan_region_guard.BucketLocation('example-us-bucket', 'us-central2', 'region')); "
        "urllib.request.urlopen('https://storage.googleapis.com/example-us-bucket/ckpt.bin')"
    )
    executor = LocalProcessExecutor(tmp_path / "state")
    remote_id = executor.submit(
        {
            "entrypoint": shlex.join((sys.executable, "-c", child_code)),
            "submission_id": "banned-cross-region-job",
            "environment": {
                "PYTHONPATH": str(site_dir),
                CROSS_REGION_POLICY_ENV: "ban",
                HOST_REGION_ENV: "europe-west4",
                CROSS_REGION_KILL_ENV: "1",
            },
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
            "metadata": {"eyvan_workload_id": "banned-cross-region-job"},
        },
        MutationContext(idempotency_key="submit-ban-1", generation=1, fencing_token="fence-1"),
    )

    deadline = time.monotonic() + 10.0
    observation = executor.status(remote_id)
    while time.monotonic() < deadline and observation.status not in {
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.SUCCEEDED,
    }:
        time.sleep(0.05)
        observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.FAILED
    assert observation.message.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "host_region=europe-west4" in observation.message
    assert "bucket=example-us-bucket" in observation.message
    assert "bucket_location=us-central2" in observation.message

    # Also verify fallback re-derivation in LocalProcessExecutor.status() for a pre-existing exit_code=95 record.
    record_path, log_path, _owner_path = executor._paths("legacy-exit-95")
    log_path.write_text(
        f"training step 1\n{CROSS_REGION_TRANSFER_BANNED_MARKER} host_region=europe-west4 "
        "bucket=example-us-bucket bucket_location=us-central2 operation=tensorstore.open "
        "uri=gs://example-us-bucket/ckpt\n",
        encoding="utf-8",
    )
    record_path.write_text(
        json.dumps(
            {
                "remote_id": "legacy-exit-95",
                "status": ExecutionWorkloadStatus.FAILED.value,
                "message": "native-local command exited with code 95",
                "exit_code": CROSS_REGION_TRANSFER_BANNED_EXIT_CODE,
                "log_path": str(log_path),
                "group_quiescent": True,
            }
        ),
        encoding="utf-8",
    )
    recovered = executor.status("legacy-exit-95")
    assert recovered.status is ExecutionWorkloadStatus.FAILED
    assert recovered.message.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
    assert "bucket=example-us-bucket" in recovered.message
    assert "operation=tensorstore.open" in recovered.message

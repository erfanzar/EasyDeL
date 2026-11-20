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

"""Resolution of JAX's persistent compilation cache directory (explicit, existing, Google Cloud bucket, none)."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import pytest
from easydel.utils import compilation_cache as cc

pytestmark = pytest.mark.cpu_ok

HERE = cc.GcpLocation(project_id="my-proj", project_number="123", region="europe-west4")


class FakeStore:
    def __init__(self, buckets=None, fail=None):
        self.buckets = dict(buckets or {})
        self.created = []
        self.fail = fail

    def get(self, bucket):
        if self.fail:
            raise self.fail
        return self.buckets.get(bucket)

    def create(self, bucket, *, project_id, region):
        self.created.append((bucket, project_id, region))
        self.buckets[bucket] = {"projectNumber": "123", "location": region.upper()}


def _resolve(store, locate=lambda: HERE):
    return cc.resolve_gcs_cache_dir("GLM-5.3", locate=locate, store_factory=lambda: store, jax_version="0.11.2")


def test_detect_gcp_reads_project_and_region_from_metadata():
    values = {
        "instance/zone": "projects/123/zones/europe-west4-b",
        "project/project-id": "my-proj",
        "project/numeric-project-id": "123",
    }
    assert cc.detect_gcp(fetch=lambda path, timeout: values.get(path)) == HERE


def test_detect_gcp_is_none_off_google_cloud():
    assert cc.detect_gcp(fetch=lambda path, timeout: None) is None


def test_missing_bucket_is_created_in_this_project_and_region():
    store = FakeStore()
    assert _resolve(store) == "gs://easydel-caching-europe-west4-123/jax-0.11.2/GLM-5.3"
    assert store.created == [("easydel-caching-europe-west4-123", "my-proj", "europe-west4")]


def test_existing_owned_bucket_is_reused_without_creating():
    store = FakeStore({"easydel-caching-europe-west4-123": {"projectNumber": "123"}})
    assert _resolve(store) == "gs://easydel-caching-europe-west4-123/jax-0.11.2/GLM-5.3"
    assert store.created == []


def test_a_bucket_owned_by_another_project_is_refused():
    store = FakeStore({"easydel-caching-europe-west4-123": {"projectNumber": "999"}})
    assert _resolve(store) is None


def test_storage_errors_fall_back_to_no_cache():
    assert _resolve(FakeStore(fail=PermissionError("storage.buckets.create denied"))) is None


def test_off_google_cloud_resolves_nothing_and_never_touches_storage():
    def store_factory():
        raise AssertionError("no storage access off Google Cloud")

    assert cc.resolve_gcs_cache_dir("m", locate=lambda: None, store_factory=store_factory) is None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gs://bucket/GLM-5.3-Flash-easydel-bf16/", "GLM-5.3-Flash-easydel-bf16"),
        ("zai-org/GLM 5.3", "GLM-5.3"),
        (SimpleNamespace(config=SimpleNamespace(_name_or_path="org/Qwen3-8B")), "Qwen3-8B"),
        ("///", "default"),
    ],
)
def test_cache_namespace_is_directory_safe(model, expected):
    assert cc.cache_namespace(model) == expected


@pytest.fixture
def clean_jax_cache(monkeypatch):
    previous = jax.config.jax_compilation_cache_dir
    monkeypatch.setattr(cc, "_resolved", {})
    jax.config.update("jax_compilation_cache_dir", None)
    yield
    jax.config.update("jax_compilation_cache_dir", previous)


def test_explicit_directory_wins_over_everything(clean_jax_cache, monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "resolve_gcs_cache_dir", lambda ns: pytest.fail("explicit path must not probe GCP"))
    assert cc.configure_compilation_cache("m", explicit=str(tmp_path)) == str(tmp_path)
    assert jax.config.jax_compilation_cache_dir == str(tmp_path)


def test_a_directory_jax_already_has_is_left_alone(clean_jax_cache, monkeypatch, tmp_path):
    jax.config.update("jax_compilation_cache_dir", str(tmp_path))
    monkeypatch.setattr(cc, "resolve_gcs_cache_dir", lambda ns: pytest.fail("existing setting must win"))
    assert cc.configure_compilation_cache("m") == str(tmp_path)


def test_google_cloud_directory_is_applied_once_per_namespace(clean_jax_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "resolve_gcs_cache_dir", lambda ns: calls.append(ns) or f"gs://b/jax/{ns}")
    assert cc.configure_compilation_cache("m") == "gs://b/jax/m"
    assert jax.config.jax_compilation_cache_dir == "gs://b/jax/m"
    jax.config.update("jax_compilation_cache_dir", None)
    cc.configure_compilation_cache("m")
    assert calls == ["m"]


def test_no_google_cloud_keeps_jax_default(clean_jax_cache, monkeypatch):
    monkeypatch.setattr(cc, "resolve_gcs_cache_dir", lambda ns: None)
    assert cc.configure_compilation_cache("m") is None
    assert jax.config.jax_compilation_cache_dir is None

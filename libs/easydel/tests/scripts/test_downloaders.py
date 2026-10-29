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
"""Regression tests for the HuggingFace download scripts (no array computation)."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace
from typing import ClassVar

import pytest
from easydel.scripts import _hf_common as common
from easydel.scripts import download_hf_large_weights_to_gcs as large
from easydel.scripts import download_hf_repo_chunked_to_gcs as chunked

pytestmark = pytest.mark.cpu_ok


class _FakeLargeApi:
    """HfApi stand-in exposing one big gguf file per repo and recording the repos queried."""

    queried: ClassVar[list[str]] = []

    def __init__(self, token=None):
        del token

    def model_info(self, repo_id, **kwargs):
        del kwargs
        type(self).queried.append(repo_id)
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename="m.gguf", size=2 * 1024**3)])


def test_large_weights_download_uses_only_supported_hf_hub_kwargs(tmp_path, monkeypatch):
    calls = []
    installed_signature = inspect.signature(large.hf_hub_download)

    def strict_hf_hub_download(
        repo_id, filename, *, repo_type=None, revision=None, token=None, cache_dir=None, local_dir=None
    ):
        kwargs = dict(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        installed_signature.bind(**kwargs)
        calls.append(kwargs)

    _FakeLargeApi.queried = []
    monkeypatch.setattr(large, "HfApi", _FakeLargeApi)
    monkeypatch.setattr(large, "hf_hub_download", strict_hf_hub_download)

    code = large.main(
        [
            "--out-root",
            str(tmp_path),
            "--repo-id",
            "org/repo",
            "--min-size-mb",
            "1",
            "--revision",
            "test-revision",
            "--token",
            "test-token",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert code == 0
    assert calls == [
        {
            "repo_id": "org/repo",
            "filename": "m.gguf",
            "repo_type": "model",
            "revision": "test-revision",
            "token": "test-token",
            "cache_dir": str(tmp_path / "cache"),
            "local_dir": str(tmp_path / "org" / "repo"),
        }
    ]


def test_large_weights_match_repo_keeps_repos_matching_any_substring(tmp_path, monkeypatch):
    _FakeLargeApi.queried = []
    monkeypatch.setattr(large, "HfApi", _FakeLargeApi)

    code = large.main(
        [
            "--out-root",
            str(tmp_path),
            "--repo-id",
            "a/one",
            "--repo-id",
            "b/two",
            "--repo-id",
            "c/three",
            "--match-repo",
            "one",
            "--match-repo",
            "two",
            "--dry-run",
        ]
    )

    assert code == 0
    assert _FakeLargeApi.queried == ["a/one", "b/two"]


def test_chunked_continue_on_error_still_fails_exit_code_for_aborted_repo(tmp_path, monkeypatch):
    class _BrokenApi:
        def __init__(self, token=None):
            del token

        def list_repo_tree(self, **kwargs):
            raise RuntimeError("listing failed")

    monkeypatch.setattr(chunked, "HfApi", _BrokenApi)

    code = chunked.main(
        [
            "--out-root",
            str(tmp_path / "out"),
            "--repo-id",
            "org/repo",
            "--staging-dir",
            str(tmp_path / "stage"),
            "--continue-on-error",
        ]
    )

    assert code == 2


def test_shared_helpers_behave_identically_in_both_scripts(tmp_path):
    listing = tmp_path / "repos.txt"
    listing.write_text("# header\norg/a  # inline\n\norg/b\n", encoding="utf-8")

    assert large._read_models_file(listing) == ["org/a", "org/b"]
    assert chunked._read_repos_file(listing) == ["org/a", "org/b"]
    assert large._repo_out_dir(tmp_path, "o/n") == chunked._repo_out_dir_local(tmp_path, "o/n") == tmp_path / "o" / "n"
    assert large._matches_any_glob("x.gguf", ("*.gguf",)) and chunked._matches_any_glob("x.gguf", ("*.gguf",))
    assert not chunked._matches_any_glob("x.bin", ("*.gguf",))


def test_fast_transfer_enables_installed_xet_backend_after_hub_import(monkeypatch):
    pytest.importorskip("hf_xet")
    from huggingface_hub import constants
    from huggingface_hub.utils._runtime import is_xet_available

    monkeypatch.delattr(constants, "HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    env_before = dict(os.environ)
    assert not is_xet_available()

    warning = common.enable_hf_fast_transfer()

    assert warning is None
    assert is_xet_available()
    assert dict(os.environ) == env_before


def test_fast_transfer_missing_xet_warns_without_changing_backend_or_environment(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.delattr(constants, "HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    monkeypatch.setattr(common.importlib.util, "find_spec", lambda name: None)
    env_before = dict(os.environ)

    warning = common.enable_hf_fast_transfer()

    assert warning is not None
    assert constants.HF_HUB_DISABLE_XET is True
    assert dict(os.environ) == env_before


@pytest.mark.parametrize("installed", [True, False])
def test_fast_transfer_preserves_legacy_backend_and_handles_missing_package(monkeypatch, installed):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_ENABLE_HF_TRANSFER", False, raising=False)
    monkeypatch.setattr(common.importlib.util, "find_spec", lambda name: object() if installed else None)
    env_before = dict(os.environ)

    warning = common.enable_hf_fast_transfer()

    assert constants.HF_HUB_ENABLE_HF_TRANSFER is installed
    assert (warning is None) is installed
    assert dict(os.environ) == env_before


@pytest.mark.parametrize("failure", ["listing", "sync"])
def test_chunked_fatal_failure_is_counted_and_next_repository_is_processed(tmp_path, monkeypatch, failure):
    queried = []

    class _Api:
        def __init__(self, token=None):
            pass

    def files(api, repo_id, **kwargs):
        queried.append(repo_id)
        if failure == "listing" and repo_id == "org/bad":
            raise RuntimeError("listing failed")
        yield chunked.DownloadItem(path="file", size=1)

    def sync(payload, destination, **kwargs):
        if failure == "sync" and str(destination).endswith("bad"):
            raise RuntimeError("sync failed")

    monkeypatch.setattr(chunked, "HfApi", _Api)
    monkeypatch.setattr(chunked, "_iter_repo_files", files)
    monkeypatch.setattr(chunked, "_download_batch", lambda *a, **k: (1, 0))
    monkeypatch.setattr(chunked, "_sync_payload", sync)

    code = chunked.main(
        [
            "--out-root",
            str(tmp_path / "out"),
            "--staging-dir",
            str(tmp_path / "stage"),
            "--repo-id",
            "org/bad",
            "--repo-id",
            "org/good",
            "--continue-on-error",
        ]
    )

    assert code == 2
    assert queried == ["org/bad", "org/good"]

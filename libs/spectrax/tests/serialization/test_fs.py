# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Comprehensive tests for spectrax.serialization._fs."""

from __future__ import annotations

import uuid

import pytest

from spectrax.serialization._fs import (
    exists,
    is_dir,
    is_file,
    is_remote_path,
    iterdir,
    joinpath,
    mkdir,
    read_text,
    rm,
    write_text,
)

GCS_BASE = "gs://uscentral1stuff/spx-save-tmp"


class TestIsRemotePath:
    """Tests for is_remote_path."""

    def test_local_path_returns_false(self):
        """Local path returns false."""
        assert is_remote_path("/tmp/foo") is False
        assert is_remote_path("./foo") is False
        assert is_remote_path("foo/bar") is False

    def test_gs_path_returns_true(self):
        """Gs path returns true."""
        assert is_remote_path("gs://bucket/path") is True

    def test_s3_path_returns_true(self):
        """S3 path returns true."""
        assert is_remote_path("s3://bucket/path") is True

    def test_empty_path_returns_false(self):
        """Empty path returns false."""
        assert is_remote_path("") is False


class TestJoinpath:
    """Tests for joinpath."""

    def test_local_join(self):
        """Local join."""
        assert joinpath("/tmp", "foo", "bar") == "/tmp/foo/bar"

    def test_gs_join(self):
        """Gs join."""
        assert joinpath("gs://bucket", "path", "file.json") == "gs://bucket/path/file.json"

    def test_trailing_slash_handling(self):
        """Trailing slash handling."""
        assert joinpath("gs://bucket/", "/path/") == "gs://bucket/path/"

    def test_single_part(self):
        """Single part."""
        assert joinpath("gs://bucket") == "gs://bucket"


class TestMkdir:
    """Tests for mkdir."""

    def test_local_mkdir(self, tmp_path):
        """Local mkdir."""
        target = str(tmp_path / "nested" / "dir")
        mkdir(target)
        assert (tmp_path / "nested" / "dir").exists()

    def test_local_mkdir_exist_ok(self, tmp_path):
        """Local mkdir exist ok."""
        target = str(tmp_path / "existing")
        (tmp_path / "existing").mkdir()
        mkdir(target, exist_ok=True)

    def test_gs_mkdir(self, gcs_auth_ino):
        """Gs mkdir."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-mkdir-{run_id}"
        mkdir(path)
        write_text(joinpath(path, "marker.txt"), "ok")
        assert exists(joinpath(path, "marker.txt"))
        rm(path, recursive=True)


class TestExists:
    """Tests for exists."""

    def test_local_existing_file(self, tmp_path):
        """Local existing file."""
        f = tmp_path / "foo.txt"
        f.write_text("hello")
        assert exists(str(f)) is True

    def test_local_missing_file(self, tmp_path):
        """Local missing file."""
        assert exists(str(tmp_path / "missing.txt")) is False

    def test_gs_existing_file(self, gcs_auth_ino):
        """Gs existing file."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-exists-{run_id}/file.txt"
        write_text(path, "hello")
        assert exists(path) is True
        rm(joinpath(path, ".."), recursive=True)

    def test_gs_missing_file(self, gcs_auth_ino):
        """Gs missing file."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-exists-missing-{run_id}/file.txt"
        assert exists(path) is False


class TestIsDir:
    """Tests for is_dir."""

    def test_local_directory(self, tmp_path):
        """Local directory."""
        d = tmp_path / "subdir"
        d.mkdir()
        assert is_dir(str(d)) is True

    def test_local_file_returns_false(self, tmp_path):
        """Local file returns false."""
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert is_dir(str(f)) is False

    def test_gs_directory(self, gcs_auth_ino):
        """Gs directory."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-isdir-{run_id}"
        mkdir(path)
        write_text(joinpath(path, "marker.txt"), "ok")
        assert is_file(joinpath(path, "marker.txt")) is True
        rm(path, recursive=True)


class TestIsFile:
    """Tests for is_file."""

    def test_local_file(self, tmp_path):
        """Local file."""
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert is_file(str(f)) is True

    def test_local_directory_returns_false(self, tmp_path):
        """Local directory returns false."""
        d = tmp_path / "subdir"
        d.mkdir()
        assert is_file(str(d)) is False

    def test_gs_file(self, gcs_auth_ino):
        """Gs file."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-isfile-{run_id}/file.txt"
        write_text(path, "x")
        assert is_file(path) is True
        rm(joinpath(path, ".."), recursive=True)


class TestWriteText:
    """Tests for write_text."""

    def test_local_write_and_read(self, tmp_path):
        """Local write and read."""
        f = str(tmp_path / "test.txt")
        write_text(f, "hello world")
        assert read_text(f) == "hello world"

    def test_local_utf8(self, tmp_path):
        """Local utf8."""
        f = str(tmp_path / "utf8.txt")
        write_text(f, "日本語")
        assert read_text(f) == "日本語"

    def test_gs_write_and_read(self, gcs_auth_ino):
        """Gs write and read."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-writetext-{run_id}/file.txt"
        write_text(path, "gs content")
        assert read_text(path) == "gs content"
        rm(joinpath(path, ".."), recursive=True)


class TestReadText:
    """Tests for read_text."""

    def test_local_read_existing(self, tmp_path):
        """Local read existing."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        assert read_text(str(f)) == "content"

    def test_local_missing_raises(self, tmp_path):
        """Local missing raises."""
        with pytest.raises(FileNotFoundError):
            read_text(str(tmp_path / "missing.txt"))

    def test_gs_missing_raises(self, gcs_auth_ino):
        """Gs missing raises."""
        run_id = str(uuid.uuid4())[:8]
        path = f"{GCS_BASE}/test-readtext-missing-{run_id}/file.txt"
        with pytest.raises(FileNotFoundError):
            read_text(path)


class TestIterdir:
    """Tests for iterdir."""

    def test_local_iterdir(self, tmp_path):
        """Local iterdir."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "c.txt").write_text("x")
        names = [p for p in iterdir(str(tmp_path))]
        assert len(names) == 3
        basenames = [p.split("/")[-1] for p in names]
        assert set(basenames) == {"a", "b", "c.txt"}

    def test_gs_iterdir(self, gcs_auth_ino):
        """Gs iterdir."""
        run_id = str(uuid.uuid4())[:8]
        base = f"{GCS_BASE}/test-iterdir-{run_id}"
        write_text(joinpath(base, "a.txt"), "a")
        write_text(joinpath(base, "b.txt"), "b")
        names = [p for p in iterdir(base)]
        assert len(names) == 2
        basenames = [p.split("/")[-1] for p in names]
        assert set(basenames) == {"a.txt", "b.txt"}
        rm(base, recursive=True)


class TestRm:
    """Tests for rm."""

    def test_local_rm_file(self, tmp_path):
        """Local rm file."""
        f = tmp_path / "file.txt"
        f.write_text("x")
        rm(str(f))
        assert not f.exists()

    def test_local_rm_dir_recursive(self, tmp_path):
        """Local rm dir recursive."""
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "file.txt").write_text("x")
        rm(str(d), recursive=True)
        assert not d.exists()

    def test_local_rm_missing_noop(self, tmp_path):
        """Local rm missing noop."""
        rm(str(tmp_path / "missing"), recursive=True)

    def test_gs_rm_recursive(self, gcs_auth_ino):
        """Gs rm recursive."""
        run_id = str(uuid.uuid4())[:8]
        base = f"{GCS_BASE}/test-rm-{run_id}"
        write_text(joinpath(base, "file.txt"), "x")
        assert exists(base)
        rm(base, recursive=True)
        assert not exists(base)

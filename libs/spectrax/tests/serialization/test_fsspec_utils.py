# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for spectrax.serialization.fsspec_utils."""

from __future__ import annotations

import jax

from spectrax.serialization.fsspec_utils import should_write_shared_checkpoint_files


class TestShouldWriteSharedCheckpointFiles:
    """Tests for should_write_shared_checkpoint_files."""

    def test_local_path_returns_true(self):
        """Local path returns true."""
        assert should_write_shared_checkpoint_files("/tmp/foo") is True

    def test_relative_local_path_returns_true(self):
        """Relative local path returns true."""
        assert should_write_shared_checkpoint_files("./foo") is True

    def test_gs_path_process_0_returns_true(self):
        """Gs path process 0 returns true."""
        if jax.process_index() == 0:
            assert should_write_shared_checkpoint_files("gs://bucket/path") is True

    def test_gs_path_nonzero_process_returns_false(self):
        """Gs path nonzero process returns false."""
        if jax.process_index() != 0:
            assert should_write_shared_checkpoint_files("gs://bucket/path") is False

    def test_s3_path_process_0_returns_true(self):
        """S3 path process 0 returns true."""
        if jax.process_index() == 0:
            assert should_write_shared_checkpoint_files("s3://bucket/path") is True

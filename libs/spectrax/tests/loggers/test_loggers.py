# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for spectrax.loggers unified logging."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spectrax.loggers import ConsoleBackend, Logger, TensorBoardBackend, WandBBackend
from spectrax.loggers.base import _NullBackend


class TestNullBackend:
    """Fixture class for testing."""

    def test_null_backend_is_noop(self):
        """Null backend is noop."""
        b = _NullBackend()
        b.log_scalar("x", 1.0, step=0)
        b.log_histogram("x", np.array([1, 2, 3]), step=0)
        b.log_image("x", np.zeros((3, 3)), step=0)
        b.log_text("x", "hello", step=0)
        b.log_hparams({"lr": 0.01})
        b.flush()
        b.close()


class TestConsoleBackend:
    """Fixture class for testing."""

    def test_console_scalar(self, capsys):
        """Console scalar."""
        b = ConsoleBackend(prefix="[TEST]")
        b.log_scalar("loss", 0.5, step=10)
        captured = capsys.readouterr()
        assert "[TEST]" in captured.out
        assert "loss" in captured.out
        assert "0.5" in captured.out
        assert "step=" in captured.out and "10" in captured.out

    def test_console_histogram_is_silently_ignored(self, capsys):
        """Console histogram is silently ignored."""
        b = ConsoleBackend()
        b.log_histogram("weights", np.array([1.0, 2.0, 3.0]), step=5)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_console_image(self, capsys):
        """Console image."""
        b = ConsoleBackend()
        b.log_image("img", np.zeros((64, 64, 3)), step=1)
        captured = capsys.readouterr()
        assert "image" in captured.out
        assert "img" in captured.out
        assert "(64, 64, 3)" in captured.out

    def test_console_text(self, capsys):
        """Console text."""
        b = ConsoleBackend()
        b.log_text("sample", "hello world", step=2)
        captured = capsys.readouterr()
        assert "text" in captured.out
        assert "hello world" in captured.out

    def test_console_hparams(self, capsys):
        """Console hparams."""
        b = ConsoleBackend()
        b.log_hparams({"lr": 0.001, "batch_size": 32})
        captured = capsys.readouterr()
        assert "hparams" in captured.out
        import re

        clean = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)
        assert "lr = 0.001" in clean
        assert "batch_size = 32" in clean


class TestLogger:
    """Fixture class for testing."""

    def test_logger_with_no_backends_uses_null(self):
        """Logger with no backends uses null."""
        logger = Logger([])
        logger.log_scalar("x", 1.0, step=0)
        logger.close()

    def test_logger_multiplexes(self, capsys):
        """Logger multiplexes."""
        logger = Logger([ConsoleBackend(), ConsoleBackend()])
        logger.log_scalar("loss", 0.42, step=100)
        captured = capsys.readouterr()
        assert captured.out.count("loss") == 2

    def test_logger_context_manager(self):
        """Logger context manager."""
        with Logger([ConsoleBackend()]) as logger:
            logger.log_scalar("x", 1.0, step=0)

    def test_logger_auto_flush(self):
        """Logger auto flush."""
        backend = MagicMock()
        logger = Logger([backend], auto_flush=True)
        logger.log_scalar("x", 1.0, step=0)
        assert backend.flush.call_count >= 1
        logger.close()

    def test_logger_no_auto_flush(self):
        """Logger no auto flush."""
        backend = MagicMock()
        logger = Logger([backend], auto_flush=False)
        logger.log_scalar("x", 1.0, step=0)
        assert backend.flush.call_count == 0
        logger.close()

    def test_logger_closed_is_noop(self):
        """Logger closed is noop."""
        backend = MagicMock()
        logger = Logger([backend])
        logger.close()
        logger.log_scalar("x", 1.0, step=0)
        backend.log_scalar.assert_not_called()

    def test_logger_handles_backend_exception(self, capsys):
        """Logger handles backend exception."""

        class BrokenBackend(ConsoleBackend):
            """Fixture class for testing."""

            def log_scalar(self, tag, value, step):
                """Log a scalar value."""
                raise RuntimeError("boom")

        logger = Logger([BrokenBackend()])
        logger.log_scalar("x", 1.0, step=0)
        logger.close()

    def test_logger_flush_and_close_called(self):
        """Logger flush and close called."""
        backend = MagicMock()
        logger = Logger([backend], auto_flush=False)
        logger.flush()
        assert backend.flush.call_count == 1
        logger.close()
        assert backend.close.call_count == 1

    def test_logger_all_methods_dispatch(self):
        """Logger all methods dispatch."""
        backend = MagicMock()
        logger = Logger([backend], auto_flush=False)
        logger.log_scalar("s", 1.0, step=0)
        logger.log_histogram("h", np.array([1, 2]), step=0)
        logger.log_image("i", np.zeros((3, 3)), step=0)
        logger.log_text("t", "hi", step=0)
        logger.log_hparams({"lr": 0.01})
        logger.log_summary({"final_acc": 0.99})
        logger.log_table("tbl", ["a", "b"], [[1, 2], [3, 4]], step=0)
        assert backend.log_scalar.called
        assert backend.log_histogram.called
        assert backend.log_image.called
        assert backend.log_text.called
        assert backend.log_hparams.called
        assert backend.log_summary.called
        assert backend.log_table.called
        logger.close()

    def test_logger_easydel_aliases(self):
        """EasyDeL calls add_scalar / scalar on the writer directly."""
        backend = MagicMock()
        logger = Logger([backend], auto_flush=False)
        logger.add_scalar("s", 1.0, step=0)
        logger.add_histogram("h", np.array([1, 2]), step=0)
        logger.add_image("i", np.zeros((3, 3)), step=0)
        logger.add_text("t", "hi", step=0)
        logger.add_hparams({"lr": 0.01})
        logger.scalar("s2", 2.0, step=1)
        logger.histogram("h2", np.array([3, 4]), step=1)
        assert backend.log_scalar.call_count == 2
        assert backend.log_histogram.call_count == 2
        assert backend.log_image.call_count == 1
        assert backend.log_text.call_count == 1
        assert backend.log_hparams.call_count == 1
        logger.close()

    def test_logger_log_table_console_is_silently_ignored(self, capsys):
        """Logger log table console is silently ignored."""
        logger = Logger([ConsoleBackend()], auto_flush=False)
        logger.log_table("preview", ["step", "prompt"], [[1, "hello"], [2, "world"]], step=10)
        captured = capsys.readouterr()
        assert captured.out == ""
        logger.close()


class TestTensorBoardBackend:
    """Fixture class for testing."""

    def test_creates_log_dir(self, tmp_path):
        """Creates log dir."""
        log_dir = tmp_path / "nested" / "tb"
        assert not log_dir.exists()
        backend = TensorBoardBackend(log_dir)
        assert log_dir.exists()
        backend.close()

    def test_writes_event_file(self, tmp_path):
        """Writes event file."""
        backend = TensorBoardBackend(tmp_path)
        backend.log_scalar("loss", 0.5, step=10)
        backend.close()
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("events.out.tfevents")
        assert files[0].stat().st_size > 0

    def test_flush_and_close(self, tmp_path):
        """Flush and close."""
        backend = TensorBoardBackend(tmp_path)
        backend.log_scalar("x", 1.0, step=0)
        backend.flush()
        backend.close()
        assert backend._file is None or backend._file.closed

    def test_histogram(self, tmp_path):
        """Histogram."""
        backend = TensorBoardBackend(tmp_path)
        backend.log_histogram("weights", np.random.default_rng(0).standard_normal(100), step=5)
        backend.close()
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].stat().st_size > 0

    def test_text(self, tmp_path):
        """Text."""
        backend = TensorBoardBackend(tmp_path)
        backend.log_text("sample", "hello world", step=1)
        backend.close()
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].stat().st_size > 0


class TestWandBBackend:
    """Fixture class for testing."""

    def test_requires_wandb(self):
        """Requires wandb."""
        with patch("spectrax.loggers.wandb._WANDB_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="WandBBackend requires wandb"):
                WandBBackend()

    def test_logs_scalar(self):
        """Logs scalar."""
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.run = mock_run
        mock_wandb.init.return_value = mock_run
        mock_wandb.Histogram = MagicMock(side_effect=lambda x: f"hist({x.tolist()})")
        mock_wandb.Image = MagicMock(side_effect=lambda x: f"img({x.shape})")
        mock_wandb.Html = MagicMock(side_effect=lambda x: f"html({x})")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("spectrax.loggers.wandb.wandb", mock_wandb):
                with patch("spectrax.loggers.wandb._WANDB_AVAILABLE", True):
                    backend = WandBBackend(project="test-project")
                    backend.log_scalar("loss", 0.5, step=10)
                    mock_wandb.log.assert_called_with({"loss": 0.5}, step=10)
                    backend.close()

    def test_histogram(self):
        """Histogram."""
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.run = mock_run
        mock_wandb.init.return_value = mock_run
        mock_wandb.Histogram = MagicMock(side_effect=lambda x: f"hist({x.tolist()})")

        with patch("spectrax.loggers.wandb.wandb", mock_wandb):
            with patch("spectrax.loggers.wandb._WANDB_AVAILABLE", True):
                backend = WandBBackend(project="test-project")
                backend.log_histogram("w", np.array([1.0, 2.0]), step=0)
                mock_wandb.log.assert_called()
                backend.close()

    def test_log_hparams(self):
        """Log hparams."""
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.run = mock_run
        mock_wandb.init.return_value = mock_run

        with patch("spectrax.loggers.wandb.wandb", mock_wandb):
            with patch("spectrax.loggers.wandb._WANDB_AVAILABLE", True):
                backend = WandBBackend(project="test-project")
                backend.log_hparams({"lr": 0.01})
                mock_wandb.config.update.assert_called_with({"lr": 0.01})
                backend.close()


class TestLoggerIntegration:
    """Fixture class for testing."""

    def test_jax_array_histogram(self):
        """Histogram logging should accept JAX arrays."""
        backend = MagicMock()
        logger = Logger([backend], auto_flush=False)
        arr = np.array([1.0, 2.0, 3.0])
        logger.log_histogram("w", arr, step=0)
        backend.log_histogram.assert_called_once()
        logger.close()

    def test_process_index_zero_only(self):
        """Non-zero processes should silently drop writes."""
        with patch("jax.process_index", return_value=1):
            backend = MagicMock()
            logger = Logger([backend], auto_flush=False)
            logger.log_scalar("x", 1.0, step=0)
            backend.log_scalar.assert_not_called()
            logger.close()
            backend.close.assert_not_called()

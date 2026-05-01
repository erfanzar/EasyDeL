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

"""Lightweight logging utilities for EasyDeL worker processes.

This module provides a colorized log formatter, a lazy ``eLogger`` wrapper
that defers Python ``logging`` initialization until the first call, and a
``ProgressLogger`` for progress-bar style updates in worker scripts.

Exports:
    ColorFormatter: ANSI-colorized variant of ``logging.Formatter``.
    eLogger: Lazily initialized logger that exposes the standard logging
        method names via attribute access.
    get_logger: Factory for :class:`eLogger` instances.
    ProgressLogger: Terminal progress reporter with ETA support and a
        context-manager API.
    COLORS, LEVEL_COLORS: ANSI color tables consumed by the formatter.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
import typing as tp
from functools import wraps

COLORS: dict[str, str] = {
    "PURPLE": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "ORANGE": "\033[38;5;208m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "RESET": "\033[0m",
    "BLUE_PURPLE": "\033[38;5;99m",
}

LEVEL_COLORS: dict[str, str] = {
    "DEBUG": COLORS["ORANGE"],
    "INFO": COLORS["BLUE_PURPLE"],
    "WARNING": COLORS["YELLOW"],
    "ERROR": COLORS["RED"],
    "CRITICAL": COLORS["RED"] + COLORS["BOLD"],
    "FATAL": COLORS["RED"] + COLORS["BOLD"],
}

_LOGGING_LEVELS: dict[str, int] = {
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "WARN": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
    "critical": 50,
    "fatal": 50,
    "error": 40,
    "warning": 30,
    "warn": 30,
    "info": 20,
    "debug": 10,
    "notset": 0,
}


class ColorFormatter(logging.Formatter):
    """Logging formatter that adds ANSI color codes to log output.

    Colorizes the log level name and prepends a timestamp with the logger
    name to each line of the message. Multi-line messages are formatted so
    that every line receives the colored prefix.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a ``LogRecord`` with ANSI colors and a per-line prefix.

        Args:
            record: The log record produced by Python's ``logging`` module.

        Returns:
            str: The fully formatted, color-annotated log line(s). Multi-
            line messages have the colored ``(time loggername)`` prefix
            applied to every line.
        """
        orig_levelname = record.levelname
        color = LEVEL_COLORS.get(record.levelname, COLORS["RESET"])
        record.levelname = f"{color}{record.levelname:<8}{COLORS['RESET']}"
        current_time = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        formatted_name = f"{color}({current_time} {record.name}){COLORS['RESET']}"
        message = record.getMessage()
        lines = message.split("\n")
        formatted_lines = [f"{formatted_name} {line}" if line else formatted_name for line in lines]
        result = "\n".join(formatted_lines)

        record.levelname = orig_levelname
        return result


class eLogger:
    """Lazy-initializing logger that defers handler setup until first use.

    This avoids creating log handlers at import time, which can cause issues
    when modules are imported but never actually log. The underlying
    ``logging.Logger`` is created and configured on the first call to any
    logging method (``info``, ``debug``, ``warning``, etc.).

    Attributes:
        name: The logger name.
        level: The logging level (numeric).
    """

    def __init__(self, name: str, level: int | None = None):
        """Initialize the lazy logger.

        Args:
            name: The logger name (passed to ``logging.getLogger``).
            level: Optional explicit numeric or string log level. When
                ``None``, falls back to the ``LOGGING_LEVEL_ED`` environment
                variable, then to ``INFO``.
        """
        if level is None:
            env_level = os.getenv("LOGGING_LEVEL_ED", "INFO")
            level = _LOGGING_LEVELS.get(env_level, _LOGGING_LEVELS["INFO"])
        if isinstance(level, str):
            level = _LOGGING_LEVELS.get(level, _LOGGING_LEVELS["INFO"])

        self._name = name
        self._level = level
        self._logger: logging.Logger | None = None

    @property
    def level(self):
        """Return the configured numeric log level.

        Returns:
            int: The numeric log level set at construction time.
        """
        return self._level

    @property
    def name(self):
        """Return the logger name.

        Returns:
            str: The name passed to the constructor.
        """
        return self._name

    def _ensure_initialized(self) -> None:
        """Construct and configure the underlying ``logging.Logger`` once.

        Idempotent: subsequent calls are no-ops once the handler has been
        installed.
        """
        if self._logger is not None:
            return

        logger = logging.getLogger(self._name)
        logger.propagate = False

        logger.setLevel(self._level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)

        formatter = ColorFormatter()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._logger = logger

    def __getattr__(self, name: str) -> tp.Callable:
        """Forward log-level method names to the lazily-built ``Logger``.

        Args:
            name: Attribute name lookup. Recognized names are the standard
                logging level methods (``debug``, ``info``, ``warning``,
                ``error``, ``critical``, ``fatal``, ``warn``, ``notset``)
                and ``exception`` / ``log``.

        Returns:
            Callable: A wrapper that calls :meth:`_ensure_initialized`
            before delegating to the underlying ``logging.Logger`` method.

        Raises:
            AttributeError: When ``name`` does not correspond to a known
                logging method.
        """
        if name in _LOGGING_LEVELS or name.upper() in _LOGGING_LEVELS or name in ("exception", "log"):

            @wraps(getattr(logging.Logger, name))
            def wrapped_log_method(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                self._ensure_initialized()
                return getattr(self._logger, name)(*args, **kwargs)

            return wrapped_log_method
        raise AttributeError(f"'eLogger' object has no attribute '{name}'")


def get_logger(name: str, level: int | None = None) -> eLogger:
    """Build a lazily-initialised :class:`eLogger`.

    Wraps the :class:`eLogger` constructor so worker modules can grab a
    logger at import time without paying the cost of installing
    ``logging`` handlers until the first message is actually emitted.
    The returned object exposes the standard ``debug`` / ``info`` /
    ``warning`` / ``error`` / ``critical`` / ``exception`` methods.

    Args:
        name: Logger name passed to :func:`logging.getLogger`. Following
            the standard convention, dotted names produce hierarchical
            loggers (e.g. ``"AuthManager"``,
            ``"easydel.workers.response_store"``).
        level: Optional explicit numeric or string log level. When
            omitted, :class:`eLogger` reads the ``LOGGING_LEVEL_ED``
            environment variable and falls back to ``INFO`` if that is
            unset or invalid.

    Returns:
        eLogger: Lazy logger that resolves to a configured
        :class:`logging.Logger` with a :class:`ColorFormatter` console
        handler on first use.
    """
    return eLogger(name, level)


class ProgressLogger:
    """A progress logger that displays updating progress bars and messages.

    This class provides a clean way to show progress for long-running operations
    with support for progress bars, ETAs, and streaming updates that overwrite
    the same line in the terminal.

    Attributes:
        name: Logger name to use for fallback logging
        use_tty: Whether to use TTY features (auto-detected)
        start_time: Start time of the progress operation
        _logger: Underlying logger for fallback

    Example:
        >>> progress = ProgressLogger("Training")
        >>> for i in range(100):
        ...     progress.update(i, 100, f"Processing batch {i}")
        ...
        >>> progress.complete("Training finished!")
    """

    def __init__(self, name: str = "Progress", logger_instance: eLogger | None = None):
        """Initialize the progress logger.

        Args:
            name: Name to display in progress messages
            logger_instance: Optional logger instance to use for fallback
        """
        self.name = name
        self.use_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self.start_time = time.time()
        self._logger = logger_instance or get_logger(name)
        self._last_message_length = 0

    def update(
        self,
        current: int,
        total: int,
        message: str = "",
        bar_width: int = 20,
        show_eta: bool = True,
        extra_info: str = "",
    ) -> None:
        """Update the progress display.

        Args:
            current: Current progress value (0-based)
            total: Total number of items
            message: Message to display after the progress bar
            bar_width: Width of the progress bar in characters
            show_eta: Whether to show estimated time remaining
            extra_info: Additional info to append at the end
        """
        if total <= 0:
            return

        progress = min(current / total, 1.0)
        progress_pct = progress * 100

        filled = int(bar_width * progress)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        eta_str = ""
        if show_eta and current > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / current
            remaining = (total - current) * avg_time
            if remaining > 0:
                if remaining < 60:
                    eta_str = f" ETA: {remaining:.1f}s"
                elif remaining < 3600:
                    eta_str = f" ETA: {remaining / 60:.1f}m"
                else:
                    eta_str = f" ETA: {remaining / 3600:.1f}h"

        timestamp = time.strftime("%H:%M:%S")
        full_message = f"({timestamp} {self.name}) [{bar}] {progress_pct:5.1f}% {message}{eta_str}"
        if extra_info:
            full_message += f" {extra_info}"

        if self.use_tty:
            sys.stdout.write("\r" + " " * self._last_message_length + "\r")
            sys.stdout.write(full_message)
            sys.stdout.flush()
            self._last_message_length = len(full_message)
        else:
            self._logger.info(f"{progress_pct:.1f}% - {message}")

    def update_simple(self, message: str) -> None:
        """Update with a simple message without progress bar.

        Args:
            message: Message to display
        """
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"({timestamp} {self.name}) {message}"

        if self.use_tty:
            sys.stdout.write("\r" + " " * self._last_message_length + "\r")
            sys.stdout.write(full_message)
            sys.stdout.flush()
            self._last_message_length = len(full_message)
        else:
            self._logger.info(message)

    def complete(self, message: str | None = None, show_time: bool = True) -> None:
        """Complete the progress and show final message.

        Args:
            message: Optional completion message
            show_time: Whether to show total elapsed time
        """
        if message is None:
            message = "Completed"

        total_time = time.time() - self.start_time
        timestamp = time.strftime("%H:%M:%S")

        if show_time:
            time_str = ""
            if total_time < 60:
                time_str = f" in {total_time:.1f}s"
            elif total_time < 3600:
                time_str = f" in {total_time / 60:.1f}m"
            else:
                time_str = f" in {total_time / 3600:.1f}h"
            full_message = f"({timestamp} {self.name}) {message}{time_str}"
        else:
            full_message = f"({timestamp} {self.name}) {message}"

        if self.use_tty:
            sys.stdout.write("\r" + " " * self._last_message_length + "\r")
            sys.stdout.write(full_message + "\n")
            sys.stdout.flush()
        else:
            self._logger.info(full_message)

    def __enter__(self):
        """Enter the progress-logger context manager.

        Returns:
            ProgressLogger: ``self`` so callers can ``with progress as p:``.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, calling :meth:`complete` on success.

        Args:
            exc_type: Exception type (or ``None`` when exiting normally).
            exc_val: Exception value (unused).
            exc_tb: Exception traceback (unused).

        Returns:
            bool: ``False`` so any in-flight exception keeps propagating.
        """
        if exc_type is None:
            self.complete()
        return False

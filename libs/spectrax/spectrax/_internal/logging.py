# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logging infrastructure with colored output and once-only deduplication.

Provides :class:`LazyLogger`, :class:`ProgressLogger`, and :func:`get_logger`
for SpectraX's internal and user-facing logging needs. All formatters add
ANSI colours when ``stdout`` is a TTY; non-primary JAX processes
automatically have their log level raised to ``WARNING`` to reduce noise
in distributed training.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from functools import wraps

import jax
from jax._src import xla_bridge

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

_logged_once: set[tuple[object, ...]] = set()


class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds color to log output.

    Formats each record as ``(HH:MM:SS name) message`` with the level name
    padded to 8 characters and coloured according to :data:`LEVEL_COLORS`.
    Multi-line messages are indented so every line carries the header.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* with ANSI colours and a timestamp header.

        Args:
            record: The log record to format.

        Returns:
            The formatted log line (or lines) as a single string.
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


class LazyLogger:
    """A lazy-initialized logger with colored output and once-only logging support.

    Delays initialization until first use. On non-primary JAX processes,
    the log level is automatically raised to WARNING.
    """

    def __init__(self, name: str, level: int | None = None):
        """Create a lazy logger.

        Args:
            name: Logger name (forwarded to :func:`logging.getLogger`).
            level: Initial log level. ``None`` means read from the
                ``LOGGING_LEVEL_ED`` environment variable (default ``INFO``).
        """
        if level is None:
            level = _LOGGING_LEVELS[os.getenv("LOGGING_LEVEL_ED", "INFO")]
        if isinstance(level, str):
            level = _LOGGING_LEVELS[level]

        self._name = name
        self._level = level
        self._logger: logging.Logger | None = None
        self._logged_once_lock = threading.Lock()

    @property
    def level(self):
        """int: The effective log level for this logger."""
        return self._level

    @property
    def name(self):
        """str: The logger name."""
        return self._name

    def _ensure_initialized(self) -> None:
        """Lazy-initialize the underlying :class:`logging.Logger`."""
        if self._logger is not None:
            return

        try:
            if xla_bridge.backends_are_initialized():
                if jax.process_index() > 0:
                    self._level = logging.WARNING
        except RuntimeError:
            pass

        logger = logging.getLogger(self._name)
        logger.propagate = False
        logger.setLevel(self._level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)

        formatter = ColorFormatter()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._logger = logger

    @staticmethod
    def _make_hashable(value: object) -> object:
        """Return a hashable stand-in for *value*, falling back to ``repr``.

        Used by :meth:`_log_once` so non-hashable args (e.g. lists, dicts)
        can still participate in deduplication keys.

        Args:
            value: Any Python object.

        Returns:
            A hashable proxy — the original object if it is already
            hashable, otherwise ``(type(value), repr(value))`` or
            ``(type(value), id(value))`` as last resort.
        """
        try:
            hash(value)
        except Exception:
            try:
                return (type(value), repr(value))
            except Exception:
                return (type(value), id(value))
        return value

    def _log_once(self, level: int, message: str, *args: object, **kwargs: object) -> None:
        """Log *message* at *level* only if the exact call has not been seen before.

        The deduplication key covers the level name, message template, and
        every positional arg (made hashable via :meth:`_make_hashable`).

        Args:
            level: Python logging level constant (e.g. ``logging.INFO``).
            message: Format string.
            *args: Positional args forwarded to :meth:`logging.Logger.log`.
            **kwargs: Keyword args forwarded to :meth:`logging.Logger.log`.
        """
        safe_args = tuple(self._make_hashable(arg) for arg in args)
        message_key = (logging.getLevelName(level), message, *safe_args)
        with self._logged_once_lock:
            if message_key not in _logged_once:
                _logged_once.add(message_key)
                self._ensure_initialized()
                self._logger.log(level, message, *args, **kwargs)

    def debug_once(self, message: str, *args: object, **kwargs: object) -> None:
        """Log a ``DEBUG`` message once (deduplicated).

        Args:
            message: Message value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        self._log_once(logging.DEBUG, message, *args, **kwargs)

    def info_once(self, message: str, *args: object, **kwargs: object) -> None:
        """Log an ``INFO`` message once (deduplicated).

        Args:
            message: Message value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        self._log_once(logging.INFO, message, *args, **kwargs)

    def warn_once(self, message: str, *args: object, **kwargs: object) -> None:
        """Log a ``WARNING`` message once (deduplicated).

        Args:
            message: Message value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        self._log_once(logging.WARNING, message, *args, **kwargs)

    def warning_once(self, message: str, *args: object, **kwargs: object) -> None:
        """Alias for :meth:`warn_once`.

        Args:
            message: Message value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        self._log_once(logging.WARNING, message, *args, **kwargs)

    def error_once(self, message: str, *args: object, **kwargs: object) -> None:
        """Log an ``ERROR`` message once (deduplicated).

        Args:
            message: Message value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        self._log_once(logging.ERROR, message, *args, **kwargs)

    def clear_once_cache(self) -> None:
        """Clear the global once-only deduplication cache.

        After calling this, every subsequent ``*_once`` call will emit its
        message again (until the next deduplication).
        """
        with self._logged_once_lock:
            _logged_once.clear()

    def __getattr__(self, name: str) -> Callable[..., object]:
        """Dynamically resolve logging methods on first access.

        Args:
            name: Attribute name being accessed.

        Returns:
            A callable that delegates to the underlying ``logging.Logger``.

        Raises:
            AttributeError: If ``name`` is not a known logging method or level.
        """
        if name in ("exception", "log"):
            method_name = name

            @wraps(getattr(logging.Logger, method_name))
            def wrapped_log_method(*args: object, **kwargs: object) -> object:
                """Delegate to the underlying logger method.

                Args:
                    *args: Additional positional arguments forwarded to the wrapped callable or backend.
                    **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

                Returns:
                    Result described by this helper.
                """
                self._ensure_initialized()
                return getattr(self._logger, method_name)(*args, **kwargs)

            return wrapped_log_method

        if name in _LOGGING_LEVELS or name.upper() in _LOGGING_LEVELS:
            level = _LOGGING_LEVELS.get(name, _LOGGING_LEVELS.get(name.upper()))
            method_name = name.lower()
            if hasattr(logging.Logger, method_name):

                @wraps(getattr(logging.Logger, method_name))
                def wrapped_log_method(*args: object, **kwargs: object) -> object:
                    """Delegate to the named level method of the underlying logger.

                    Args:
                        *args: Additional positional arguments forwarded to the wrapped callable or backend.
                        **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

                    Returns:
                        Result described by this helper.
                    """
                    self._ensure_initialized()
                    return getattr(self._logger, method_name)(*args, **kwargs)

                return wrapped_log_method

            @wraps(logging.Logger.log)
            def wrapped_log_method(*args: object, **kwargs: object) -> object:
                """Log at the dynamically-resolved level.

                Args:
                    *args: Additional positional arguments forwarded to the wrapped callable or backend.
                    **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

                Returns:
                    Result described by this helper.
                """
                self._ensure_initialized()
                return self._logger.log(level, *args, **kwargs)

            return wrapped_log_method

        raise AttributeError(f"'LazyLogger' object has no attribute '{name}'")


def get_logger(name: str, level: int | None = None) -> LazyLogger:
    """Create a lazy logger that only initializes when first used.

    Args:
        name: Logger name.
        level: Optional initial level. ``None`` → ``LOGGING_LEVEL_ED`` env.

    Returns:
        A :class:`LazyLogger` instance.
    """
    return LazyLogger(name, level)


class ProgressLogger:
    """A progress logger that displays updating progress bars and messages.

    On a TTY the bar overwrites the same line; in a pipe or file the
    percentage is emitted as ordinary log lines via :func:`get_logger`.
    Use as a context manager for automatic completion logging.
    """

    def __init__(self, name: str = "Progress", logger_instance: LazyLogger | None = None):
        """Create a progress logger.

        Args:
            name: Display name printed next to the timestamp.
            logger_instance: Optional :class:`LazyLogger` for non-TTY output.
                When ``None``, a logger named ``name`` is created.
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
        """Render a progress bar for *current* out of *total*.

        Args:
            current: Completed items so far.
            total: Total items expected.
            message: Optional text displayed after the percentage.
            bar_width: Width of the Unicode bar in characters.
            show_eta: Whether to estimate and print remaining time.
            extra_info: Optional trailing text (appended after ETA).
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
        """Print a simple rotating message (no percentage bar).

        Args:
            message: The message to display.
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
        """Finalize the progress line and print elapsed time.

        Args:
            message: Completion text. ``None`` → ``"Completed"``.
            show_time: Whether to append the elapsed duration.
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
        """Context-manager entry — returns ``self``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context-manager exit — auto-completes if no exception was raised.

        Args:
            exc_type: Exc type value consumed by this operation.
            exc_val: Exc val value consumed by this operation.
            exc_tb: Exc tb value consumed by this operation.
        """
        if exc_type is None:
            self.complete()
        return False

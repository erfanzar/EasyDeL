# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Logging utilities for ejKernel with colored output and progress tracking.

This module provides enhanced logging capabilities built on top of the standard
``logging`` module:

- ANSI-colored console output with level-specific formatting and timestamps.
- Lazy logger initialization that defers creation until the first log call,
  avoiding issues with JAX runtime initialization order.
- Automatic suppression of log messages from non-primary JAX processes
  (process_index > 0 is silenced to WARNING level) in distributed settings.
- Terminal progress bars with ETA calculations via :class:`ProgressLogger`.
- JAX profiler integration helpers (``ignite_profiler``, ``extinguish_profiler``,
  ``create_step_profiler``) with optional Perfetto UI link generation.

Key Components:
    ColorFormatter: ``logging.Formatter`` subclass that adds ANSI colors and
        wall-clock timestamps to each log record.
    LazyLogger: Lazy-initialized, process-aware logger facade. Use
        :func:`get_logger` to obtain an instance.
    ProgressLogger: Terminal progress bar with ETA; supports context manager
        usage. Falls back to plain logging when stdout is not a TTY.
    get_logger: Factory function for creating :class:`LazyLogger` instances.
    ignite_profiler / extinguish_profiler: Start/stop the JAX trace profiler.
    create_step_profiler: Build a per-step callback that automatically starts
        and stops profiling within a configured training step window.

Module-level objects:
    COLORS (dict): Mapping of color names to ANSI escape codes.
    LEVEL_COLORS (dict): Mapping of log level names to their ANSI color codes.
    logger: Pre-built :class:`LazyLogger` named ``"ejKernelLoggings"`` for
        internal use by profiler utilities.

Example:
    >>> from ejkernel.loggings import get_logger, ProgressLogger
    >>>
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting training")
    >>>
    >>> with ProgressLogger("Training") as progress:
    ...     for i, batch in enumerate(batches):
    ...         progress.update(i, len(batches), f"Batch {i}")
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
import time
import typing as tp
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


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors and timestamps to log messages.

    This formatter applies ANSI color codes based on log level and formats
    multi-line messages with proper indentation and timestamps.

    Methods:
        format: Formats a log record with colors and timestamps.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors and timestamp.

        Args:
            record: The log record to format.

        Returns:
            Formatted string with ANSI color codes and timestamp.
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
    """Lazy-initialized logger that defers creation until first use.

    This logger automatically adjusts its level in distributed JAX environments,
    suppressing output from non-primary processes to avoid clutter. It provides
    colored output and lazy initialization to avoid JAX runtime issues.

    Attributes:
        name: Logger name.
        level: Current logging level.

    Example:
        >>> logger = LazyLogger("MyModule")
        >>> logger.info("This message only appears on process 0")
    """

    def __init__(self, name: str, level: int | None = None):
        """Initialize a lazy logger.

        Args:
            name: Name for the logger.
            level: Logging level (uses LOGGING_LEVEL_ED env var if None).
        """
        if level is None:
            level = _LOGGING_LEVELS[os.getenv("LOGGING_LEVEL_ED", "INFO")]
        if isinstance(level, str):
            level = _LOGGING_LEVELS[level]

        self._name = name
        self._level = level
        self._logger: logging.Logger | None = None

    @property
    def level(self) -> int:
        """Return the current logging level as an integer.

        Returns:
            Logging level integer (e.g., 10 for DEBUG, 20 for INFO).
        """
        return self._level

    @property
    def name(self) -> str:
        """Return the logger name.

        Returns:
            The name string provided at construction time.
        """
        return self._name

    def _ensure_initialized(self) -> None:
        """Initialize the underlying logger if not already done.

        Automatically adjusts log level for distributed processes,
        setting non-primary processes to WARNING level.
        """
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

    def __getattr__(self, name: str) -> tp.Callable:
        """Dynamically provide logging methods.

        Args:
            name: Method name to access (e.g., 'info', 'debug', 'error').

        Returns:
            Wrapped logging method that ensures initialization.

        Raises:
            AttributeError: If the requested attribute is not a logging method.
        """
        if name in _LOGGING_LEVELS or name.upper() in _LOGGING_LEVELS or name in ("exception", "log"):

            @wraps(getattr(logging.Logger, name))
            def wrapped_log_method(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                self._ensure_initialized()
                return getattr(self._logger, name)(*args, **kwargs)

            return wrapped_log_method
        raise AttributeError(f"'LazyLogger' object has no attribute '{name}'")


def get_logger(name: str, level: int | None = None) -> LazyLogger:
    """Create a lazy logger that only initializes when first used.

    This is the primary factory function for creating loggers in ejKernel.
    The logger defers initialization to avoid JAX runtime issues and automatically
    adjusts for distributed training scenarios.

    Args:
        name: The name of the logger, typically the module name.
        level: The logging level. Defaults to environment variable LOGGING_LEVEL_ED or "INFO".

    Returns:
        A lazy logger instance that initializes on first use.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return LazyLogger(name, level)


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

    def __init__(self, name: str = "Progress", logger_instance: LazyLogger | None = None):
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

    def __enter__(self) -> "ProgressLogger":
        """Enter the context manager, returning this instance.

        Returns:
            This ProgressLogger instance for use in ``with`` blocks.

        Example:
            >>> with ProgressLogger("Processing") as progress:
            ...     progress.update(0, 10, "Starting")
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the context manager and finalize progress display.

        Calls ``complete()`` if no exception occurred during the block.
        Does not suppress exceptions.

        Args:
            exc_type: Exception type, if any was raised.
            exc_val: Exception value, if any was raised.
            exc_tb: Exception traceback, if any was raised.

        Returns:
            False, indicating exceptions are never suppressed.
        """
        if exc_type is None:
            self.complete()
        return False


logger = get_logger("ejKernelLoggings")


def create_step_profiler(
    profile_path: str,
    start_step: int,
    duration_steps: int,
    enable_perfetto: bool,
) -> tp.Callable[[int], None]:
    """Create a step-aware profiler that activates during a specific training window.

    Creates a callback function that can be called at each training step to
    automatically start/stop JAX profiling at the specified step range.

    Args:
        profile_path: Directory to store profiling results (trace files).
        start_step: Step number to begin profiling (inclusive).
        duration_steps: Number of steps to profile.
        enable_perfetto: Whether to generate Perfetto UI links (primary process only).

    Returns:
        Callback function that takes step number and manages profiler lifecycle.
        Call this function at each training step with the current step number.

    Example:
        >>> profiler = create_step_profiler(
        ...     profile_path="./profiles",
        ...     start_step=100,
        ...     duration_steps=10,
        ...     enable_perfetto=True
        ... )
        >>> for step in range(1000):
        ...     profiler(step)
        ...     train_step(batch)
    """
    from ejkernel.utils import barrier_sync

    class ProfilerState:
        """Mutable state container for the step-profiler lifecycle.

        Tracks whether the JAX trace profiler is currently running and
        whether the profiling window has already completed, preventing
        the profiler from being started or stopped more than once.

        Attributes:
            active: ``True`` while the profiler is running (between
                ``ignite_profiler`` and ``extinguish_profiler`` calls).
            completed: ``True`` after the profiling window has finished.
                Once set, the ``profile_step`` callback becomes a no-op.
        """

        def __init__(self):
            """Initialize profiler state as inactive and not yet completed."""
            self.active = False
            self.completed = False

    state = ProfilerState()

    def profile_step(step: int) -> None:
        """Handle profiling lifecycle based on current step.

        Starts profiling one step before the target range and stops
        at the end of the duration. This function is idempotent after
        profiling completes.

        Args:
            step: Current training step number.
        """
        if state.completed:
            return

        if step == start_step - 1 and not state.active:
            logger.info(f"Activating profiler for steps {start_step}-{start_step + duration_steps - 1}")
            ignite_profiler(profile_path, enable_perfetto)
            state.active = True

        elif step == start_step + duration_steps - 1 and state.active:
            logger.info("Deactivating profiler")
            extinguish_profiler(enable_perfetto)
            barrier_sync()
            state.completed = True

    return profile_step


def ignite_profiler(profile_path: str, enable_perfetto: bool = False) -> None:
    """Start the JAX profiler with optional Perfetto integration.

    Begins tracing JAX operations for performance analysis. Trace files
    are written to the specified path and can be viewed in Perfetto UI.

    Args:
        profile_path: Directory to store profiling results (trace files).
        enable_perfetto: Whether to generate Perfetto UI links. Only enabled
            on primary process (process_index == 0) to avoid duplicate links.

    Note:
        Call extinguish_profiler() to stop tracing and finalize results.
    """
    should_enable_perfetto = enable_perfetto and jax.process_index() == 0
    jax.profiler.start_trace(profile_path, create_perfetto_link=should_enable_perfetto, create_perfetto_trace=True)


def extinguish_profiler(enable_perfetto: bool) -> None:
    """Stop the profiler and handle Perfetto link generation.

    Safely stops JAX tracing and finalizes trace files. When Perfetto
    is enabled, keeps output streams alive during the potentially
    long finalization process.

    Args:
        enable_perfetto: Whether Perfetto links were enabled during profiling.
            Used to determine if output pulsing is needed during shutdown.

    Note:
        This function should be called after ignite_profiler() to stop
        tracing and write results to disk.
    """
    completion_signal = threading.Event()
    if enable_perfetto and jax.process_index() == 0:
        _pulse_output_during_wait(completion_signal)

    jax.profiler.stop_trace()

    if enable_perfetto and jax.process_index() == 0:
        completion_signal.set()


def _pulse_output_during_wait(completion_signal: threading.Event) -> None:
    """Keep output streams alive during blocking profiler shutdown.

    Spawns a daemon thread that flushes stdout/stderr immediately and then
    prints ``"Profiler finalizing..."`` every 5 seconds until
    ``completion_signal`` is set. This prevents remote connection timeouts
    during long-running Perfetto trace finalization.

    Args:
        completion_signal: A ``threading.Event`` that is set by the caller
            (``extinguish_profiler``) after ``jax.profiler.stop_trace()``
            returns. The background thread exits once the event is set.

    Note:
        This is an internal helper used exclusively by
        :func:`extinguish_profiler`. The spawned thread is a daemon thread
        and will not prevent program exit.
    """

    def pulse_output() -> None:
        """Flush streams once, then print keep-alive messages every 5 seconds.

        Runs in a daemon thread. Flushes stdout/stderr immediately on entry,
        sleeps 5 seconds, then enters a loop that prints
        ``"Profiler finalizing..."`` to stdout and a blank line to stderr
        every 5 seconds until ``completion_signal`` is set.
        """
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)

        while not completion_signal.is_set():
            print("Profiler finalizing...", flush=True)
            print("\n", file=sys.stderr, flush=True)
            time.sleep(5)

    thread = threading.Thread(target=pulse_output, daemon=True)
    thread.start()

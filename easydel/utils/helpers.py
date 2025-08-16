# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Helper utilities for EasyDeL framework.

Provides logging, timing, caching, and general utility functions used
throughout the EasyDeL framework.

Classes:
    ColorFormatter: Colored console logging formatter
    LazyLogger: Deferred initialization logger
    Timer: Simple timing utility
    Timers: Multiple timer management with logging
    DummyStream: Null output stream for suppression

Functions:
    get_logger: Create a lazy logger instance
    set_loggers_level: Set logging level globally
    capture_time: Context manager for timing
    get_cache_dir: Get EasyDeL cache directory
    quiet: Context manager to suppress output
    check_bool_flag: Parse boolean environment variables

Constants:
    COLORS: Terminal color codes
    LEVEL_COLORS: Log level to color mapping
    _LOGGING_LEVELS: String to log level mapping

Example:
    >>> from easydel.utils.helpers import get_logger, Timer
    >>>
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting process...")
    >>>
    >>> with Timer("computation") as timer:
    ...     result = expensive_computation()
    >>> print(f"Took {timer.elapsed_time()} seconds")

"""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import sys
import time
import typing as tp
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import jax

if tp.TYPE_CHECKING:
    from flax.metrics.tensorboard import SummaryWriter
try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = None


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

# Mapping log levels to colors
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
    """Custom formatter that adds colors to log messages.

    Formats log messages with colored level names and timestamps.
    Colors are based on the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(ColorFormatter())
        >>> logger.addHandler(handler)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message with ANSI color codes.
        """
        orig_levelname = record.levelname
        color = LEVEL_COLORS.get(record.levelname, COLORS["RESET"])
        record.levelname = f"{color}{record.levelname:<8}{COLORS['RESET']}"
        current_time = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        formatted_name = f"{color}({current_time} {record.name}){COLORS['RESET']}"
        message = f"{formatted_name} {record.getMessage()}"
        record.levelname = orig_levelname
        return message


class LazyLogger:
    """Logger that initializes only when first used.

    Defers logger initialization until the first logging call,
    reducing startup overhead. Automatically adjusts log level
    for non-primary JAX processes.

    Attributes:
        _name: Logger name.
        _level: Logging level.
        _logger: Underlying logger (initialized on first use).

    Example:
        >>> logger = LazyLogger(__name__)
        >>> # Logger not initialized yet
        >>> logger.info("First message")  # Initializes here
    """

    def __init__(self, name: str, level: int | None = None):
        """Initialize LazyLogger.

        Args:
            name: Logger name.
            level: Optional logging level, defaults to LOGGING_LEVEL_ED env var.
        """
        self._name = name
        self._level = level or _LOGGING_LEVELS[os.getenv("LOGGING_LEVEL_ED", "INFO")]
        self._logger: logging.Logger | None = None

    def _ensure_initialized(self) -> None:
        if self._logger is not None:
            return

        try:
            if jax.process_index() > 0:
                self._level = logging.WARNING
        except RuntimeError:
            pass

        logger = logging.getLogger(self._name)
        logger.propagate = False

        # Set the logging level
        logger.setLevel(self._level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)

        # Use our custom color formatter
        formatter = ColorFormatter()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._logger = logger

    def __getattr__(self, name: str) -> tp.Callable:
        if name in _LOGGING_LEVELS or name.upper() in _LOGGING_LEVELS or name in ("exception", "log"):

            @wraps(getattr(logging.Logger, name))
            def wrapped_log_method(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                self._ensure_initialized()
                return getattr(self._logger, name)(*args, **kwargs)

            return wrapped_log_method
        raise AttributeError(f"'LazyLogger' object has no attribute '{name}'")


def get_logger(
    name: str,
    level: int | None = None,
) -> LazyLogger:
    """Create a lazy logger that only initializes when first used.

    Args:
        name: The name of the logger.
        level: The logging level. Defaults to environment
            variable LOGGING_LEVEL_ED or "INFO".

    Returns:
        A lazy logger instance that initializes on first use.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Process started")
        >>> logger.debug("Debug information")
    """
    return LazyLogger(name, level)


def set_loggers_level(level: int = logging.WARNING):
    """Set the logging level of all loggers globally.

    Args:
        level: The logging level to set. Defaults to logging.WARNING.

    Example:
        >>> import logging
        >>> set_loggers_level(logging.DEBUG)  # Enable debug logging
        >>> set_loggers_level(logging.ERROR)  # Only show errors
    """
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)


@contextlib.contextmanager
def capture_time():
    """Context manager that measures elapsed time.

    Yields a callable that returns the current elapsed time in seconds.
    The timer continues running until the context exits.

    Yields:
        Callable that returns elapsed time in seconds.

    Example:
        >>> with capture_time() as get_time:
        ...     # Do some work
        ...     print(f"After 1 second: {get_time()}")
        ...     # Do more work
        ...     print(f"After 2 seconds: {get_time()}")
        >>> print(f"Total time: {get_time()}")
    """
    start = time.perf_counter_ns()
    is_active = True

    def get_elapsed():
        """Get elapsed time in seconds."""
        current = time.perf_counter_ns() if is_active else end
        return (current - start) / 1e9

    try:
        yield get_elapsed
    finally:
        end = time.perf_counter_ns()
        is_active = False


logger = get_logger(__name__)


class Timer:
    """Simple timer for measuring execution time.

    Can be used as a context manager or manually with start/stop methods.
    Accumulates time across multiple start/stop cycles.

    Attributes:
        name: Timer name for identification.
        elapsed: Total elapsed time in seconds.
        started: Whether timer is currently running.
        start_time: Start time of current cycle.

    Example:
        >>> timer = Timer("training")
        >>> timer.start()
        >>> # Do work
        >>> timer.stop()
        >>> print(f"Elapsed: {timer.elapsed_time()} seconds")
        >>>
        >>> # Or as context manager
        >>> with Timer("inference") as t:
        ...     result = model(input)
    """

    def __init__(self, name):
        """Initialize Timer.

        Args:
            name: Name for this timer.
        """
        self.name = name
        self.elapsed = 0.0
        self.started = False
        self.start_time = 0.0

    def start(self):
        """Start the timer.

        Raises:
            RuntimeError: If timer is already running.
        """
        if self.started:
            raise RuntimeError(f"Timer '{self.name}' is already running")
        self.start_time = time.time()
        self.started = True

    def stop(self):
        """Stop the timer and accumulate elapsed time.

        Raises:
            RuntimeError: If timer is not running.
        """
        if not self.started:
            raise RuntimeError(f"Timer '{self.name}' is not running")
        self.elapsed += time.time() - self.start_time
        self.started = False

    def reset(self):
        self.elapsed = 0.0
        self.started = False
        self.start_time = 0.0

    def elapsed_time(self, reset=True):
        """Get total elapsed time.

        Args:
            reset: Whether to reset timer after reading.

        Returns:
            Total elapsed time in seconds.
        """
        if self.started:
            self.stop()
        total_time = self.elapsed
        if reset:
            self.reset()
        return total_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class Timers:
    """Manager for multiple named timers with logging support.

    Manages a collection of timers and integrates with logging backends
    like Weights & Biases and TensorBoard for metrics tracking.

    Attributes:
        timers: Dictionary of timer instances.
        use_wandb: Whether to log to Weights & Biases.
        tensorboard_writer: TensorBoard summary writer.

    Example:
        >>> timers = Timers(use_wandb=True, tensorboard_writer=writer)
        >>> with timers.timed("forward_pass"):
        ...     output = model(input)
        >>> timers.write(["forward_pass"], iteration=100)
    """

    def __init__(self, use_wandb, tensorboard_writer: SummaryWriter):
        """Initialize Timers.

        Args:
            use_wandb: Enable Weights & Biases logging.
            tensorboard_writer: TensorBoard writer instance.
        """
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        """Get or create a timer by name.

        Args:
            name: Timer name.

        Returns:
            Timer instance.
        """
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed_time(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                if wandb is None:
                    warnings.warn(
                        "`wandb` is not installed use `pip install wandb` (use_wandb=True will be ignored)",
                        stacklevel=1,
                    )
                    self.use_wandb = False
                else:
                    wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        assert normalizer > 0.0

        if isinstance(names, str):
            names = [names]
        for name in names:
            elapsed_time = self.timers[name].elapsed_time(reset=reset) * 1000.0 / normalizer
            self._print_log(name, elapsed_time)

    def _print_log(self, name, elapsed_time):
        if elapsed_time < 1000:
            time_str = f"{elapsed_time:.4f} ms"
            color = "\033[94m"  # Blue
        elif elapsed_time < 60000:
            time_str = f"{elapsed_time / 1000:.4f} sec"
            color = "\033[92m"  # Green
        elif elapsed_time < 3600000:
            time_str = f"{elapsed_time / 60000:.4f} min"
            color = "\033[93m"  # Yellow
        else:
            time_str = f"{elapsed_time / 3600000:.4f} hr"
            color = "\033[91m"  # Red

        logger.info(f"time took for {name} : {color}{time_str}\033[0m")

    @contextlib.contextmanager
    def timed(self, name, log=True, reset=True):
        timer = self(name)
        try:
            timer.start()
            yield timer
        finally:
            timer.stop()
            if log:
                elapsed_time = timer.elapsed_time(reset=reset) * 1000.0  # Convert to milliseconds
                self._print_log(name, elapsed_time)


def get_cache_dir() -> Path:
    """Get the EasyDeL cache directory.

    Returns the platform-specific cache directory for EasyDeL.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the cache directory.

    Example:
        >>> cache_dir = get_cache_dir()
        >>> print(cache_dir)
        /home/user/.cache/easydel
    """
    home_dir = Path.home()
    app_name = "easydel"
    if os.name == "nt":  # Windows
        cache_dir = Path(os.getenv("LOCALAPPDATA", home_dir / "AppData" / "Local")) / app_name
    elif os.name == "posix":  # Linux and macOS
        if "darwin" in os.sys.platform:  # macOS
            cache_dir = home_dir / "Library" / "Caches" / app_name
        else:  # Linux
            cache_dir = home_dir / ".cache" / app_name
    else:
        cache_dir = home_dir / ".cache" / app_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class DummyStream:
    """A null device-like stream that discards all writes.

    Used for suppressing output by replacing stdout/stderr.
    All write and flush operations are no-ops.
    """

    def write(self, *args, **kwargs):
        """Discard all write operations."""
        pass

    def flush(self, *args, **kwargs):
        """Discard all flush operations."""
        pass


@contextmanager
def quiet(suppress_stdout=True, suppress_stderr=True):
    """Context manager to temporarily suppress stdout and/or stderr output.

    Replaces stdout/stderr with null streams to discard all output.
    Restores original streams on exit.

    Args:
        suppress_stdout: Whether to suppress stdout.
        suppress_stderr: Whether to suppress stderr.

    Yields:
        None

    Example:
        >>> with quiet():
        ...     print("This won't be displayed")
        ...     noisy_function()
        >>> print("This will be displayed")

    Note:
        This will suppress ALL output to the specified streams within
        the context, including output from C extensions and system calls.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        if suppress_stdout:
            sys.stdout = DummyStream()
        if suppress_stderr:
            sys.stderr = DummyStream()
        yield

    finally:
        if suppress_stdout:
            sys.stdout = original_stdout
        if suppress_stderr:
            sys.stderr = original_stderr


def check_bool_flag(name: str, default: bool = True) -> bool:
    """Parse boolean environment variable.

    Interprets various string representations as boolean values.
    Accepts: 'true', 'yes', 'ok', '1', 'easy' (case-insensitive).

    Args:
        name: Environment variable name.
        default: Default value if variable not set.

    Returns:
        Boolean interpretation of the environment variable.

    Example:
        >>> os.environ['DEBUG'] = 'yes'
        >>> check_bool_flag('DEBUG')
        True
        >>> check_bool_flag('MISSING', default=False)
        False
    """
    default = "1" if default else "0"
    return str(os.getenv(name, default)).lower() in ["true", "yes", "ok", "1", "easy"]

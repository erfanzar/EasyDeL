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
    >>> from easydel.utils.helpers import Timer
    >>>
    >>> with Timer("computation") as timer:
    ...     result = expensive_computation()
    >>> print(f"Took {timer.elapsed_time()} seconds")

"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import typing as tp
import warnings
from contextlib import contextmanager
from pathlib import Path

from eformer.loggings import get_logger

if tp.TYPE_CHECKING:
    from flax.metrics.tensorboard import SummaryWriter
try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = None

logger = get_logger(__name__)


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

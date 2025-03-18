# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


import contextlib
import datetime
from functools import wraps
import logging
import os
import sys
import time
import typing as tp
import warnings
from contextlib import contextmanager
from pathlib import Path

import jax

if tp.TYPE_CHECKING:
	from flax.metrics.tensorboard import SummaryWriter
else:
	SummaryWriter = tp.Any
try:
	import wandb  # type: ignore
except ModuleNotFoundError:
	wandb = None


COLORS: tp.Dict[str, str] = {
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
LEVEL_COLORS: tp.Dict[str, str] = {
	"DEBUG": COLORS["ORANGE"],
	"INFO": COLORS["BLUE_PURPLE"],
	"WARNING": COLORS["YELLOW"],
	"ERROR": COLORS["RED"],
	"CRITICAL": COLORS["RED"] + COLORS["BOLD"],
	"FATAL": COLORS["RED"] + COLORS["BOLD"],
}

_LOGGING_LEVELS: tp.Dict[str, int] = {
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
	def format(self, record: logging.LogRecord) -> str:
		orig_levelname = record.levelname
		color = LEVEL_COLORS.get(record.levelname, COLORS["RESET"])
		record.levelname = f"{color}{record.levelname:<8}{COLORS['RESET']}"
		current_time = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
		formatted_name = f"{color}({current_time} {record.name}){COLORS['RESET']}"
		message = f"{formatted_name} {record.getMessage()}"
		record.levelname = orig_levelname
		return message


class LazyLogger:
	def __init__(self, name: str, level: tp.Optional[int] = None):
		self._name = name
		self._level = level or _LOGGING_LEVELS[os.getenv("LOGGING_LEVEL_ED", "INFO")]
		self._logger: tp.Optional[logging.Logger] = None

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
		if (
			name in _LOGGING_LEVELS
			or name.upper() in _LOGGING_LEVELS
			or name in ("exception", "log")
		):

			@wraps(getattr(logging.Logger, name))
			def wrapped_log_method(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
				self._ensure_initialized()
				return getattr(self._logger, name)(*args, **kwargs)

			return wrapped_log_method
		raise AttributeError(f"'LazyLogger' object has no attribute '{name}'")


def get_logger(
	name: str,
	level: tp.Optional[int] = None,
) -> LazyLogger:
	"""
	Function to create a lazy logger that only initializes when first used.

	Args:
	    name (str): The name of the logger.
	    level (Optional[int]): The logging level. Defaults to environment variable LOGGING_LEVEL_ED or "INFO".

	Returns:
	    LazyLogger: A lazy logger instance that initializes on first use.
	"""
	return LazyLogger(name, level)


def set_loggers_level(level: int = logging.WARNING):
	"""Function to set the logging level of all loggers to the specified level.

	Args:
	    level: int: The logging level to set. Defaults to
	        logging.WARNING.
	"""
	logging.root.setLevel(level)
	for handler in logging.root.handlers:
		handler.setLevel(level)


@contextlib.contextmanager
def capture_time():
	"""
	A context manager that measures elapsed time.

	Returns:
	    Callable that returns the current elapsed time while the context is active,
	    or the final elapsed time after the context exits.

	Example:
	    with capture_time() as get_time:
	        # Do some work
	        print(f"Current time: {get_time()}")
	    print(f"Final time: {get_time()}")
	"""
	start = time.perf_counter_ns()
	is_active = True

	def get_elapsed():
		current = time.perf_counter_ns() if is_active else end
		return (current - start) / 1e9

	try:
		yield get_elapsed
	finally:
		end = time.perf_counter_ns()
		is_active = False


logger = get_logger(__name__)


class Timer:
	def __init__(self, name):
		self.name = name
		self.elapsed = 0.0
		self.started = False
		self.start_time = 0.0

	def start(self):
		if self.started:
			raise RuntimeError(f"Timer '{self.name}' is already running")
		self.start_time = time.time()
		self.started = True

	def stop(self):
		if not self.started:
			raise RuntimeError(f"Timer '{self.name}' is not running")
		self.elapsed += time.time() - self.start_time
		self.started = False

	def reset(self):
		self.elapsed = 0.0
		self.started = False
		self.start_time = 0.0

	def elapsed_time(self, reset=True):
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
	def __init__(self, use_wandb, tensorboard_writer: SummaryWriter):
		self.timers = {}
		self.use_wandb = use_wandb
		self.tensorboard_writer = tensorboard_writer

	def __call__(self, name):
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
				elapsed_time = (
					timer.elapsed_time(reset=reset) * 1000.0
				)  # Convert to milliseconds
				self._print_log(name, elapsed_time)


def get_cache_dir() -> Path:
	home_dir = Path.home()
	app_name = "easydel"
	if os.name == "nt":  # Windows
		cache_dir = (
			Path(os.getenv("LOCALAPPDATA", home_dir / "AppData" / "Local")) / app_name
		)
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
	"""A null device-like stream that discards all writes."""

	def write(self, *args, **kwargs):
		pass

	def flush(self, *args, **kwargs):
		pass


@contextmanager
def quiet(suppress_stdout=True, suppress_stderr=True):
	"""
	Context manager to temporarily suppress stdout and/or stderr output.
	Args:
	  suppress_stdout (bool): Whether to suppress stdout
	  suppress_stderr (bool): Whether to suppress stderr
	Usage:
	  with suppress_output():
	    # Code that generates unwanted output
	    print("This won't be displayed")
	Note:
	  This will suppress ALL output to the specified streams within the context,
	  including output from C extensions and system calls.
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
	default = "1" if default else "0"
	return str(os.getenv(name, default)).lower() in ["true", "yes", "ok", "1", "easy"]

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
import os
import sys
import time
import typing as tp
import warnings
from contextlib import contextmanager
from pathlib import Path

from easydel.etils.etils import get_logger

if tp.TYPE_CHECKING:
	from flax.metrics.tensorboard import SummaryWriter
else:
	SummaryWriter = tp.Any
try:
	import wandb  # type: ignore
except ModuleNotFoundError:
	wandb = None

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
		logger.info(f"time took for {name} (ms) : {elapsed_time:.4f}")

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

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


import curses
import json
import os
import re
import subprocess
import sys
import threading
import time

import IPython.display
import jax


# Edited version of Jax-SMI from https://github.com/ayaka14732/jax-smi/
def run(note_book=None, interval: float = 1, dir_prefix: str = "/dev/shm", dpr=True):
	"""The run function is a simple wrapper around the go tool pprof command.
	It runs the command every interval seconds and prints out its output to stdout.
	If you are running this in a notebook, it will print to IPython's display instead of stdout.

	Args:
	    note_book: Determine whether the program is running in a
	        notebook or not
	    interval: float: Specify the time interval between each refresh
	    dir_prefix: str: Specify the directory where the memory
	    dpr: Control whether the output is displayed in a notebook or
	        not

	Returns:
	    The output of the pprof command
	"""
	if note_book is None:
		import os

		def is_notebook():
			"""Returns True if the code is being run in a notebook, False otherwise."""
			return os.environ.get("IPYTHON") is not None

		note_book = is_notebook()
	std = curses.initscr() if not note_book else None
	try:
		while True:
			if not note_book and dpr:
				std.clear()
			output = subprocess.run(
				args=["go", "tool", "pprof", "-tags", f"{dir_prefix}/memory.prof"],
				stdout=subprocess.PIPE,
				stderr=subprocess.DEVNULL,
			).stdout.decode("utf-8")
			if not note_book and dpr:
				std.addstr(output)
				std.refresh()
			if note_book and dpr:
				IPython.display.clear_output(True)
				print(output)

			with open(f"{dir_prefix}/memory.json", "w") as fin:
				json.dump({"log": output}, fin)
			time.sleep(interval)
	except KeyboardInterrupt:
		curses.endwin()


def get_mem(dir_prefix: str = "/dev/shm" if sys.platform != "win32" else "."):
	"""The get_mem function is a wrapper around the go tool pprof command.
	It takes in an optional argument, dir_prefix, which defaults to /dev/shm.
	The function then runs the go tool pprof command with arguments -tags and dir_prefix/memory.prof,
	and returns its stdout as a string.

	Args:
	    dir_prefix: str: Specify the directory where

	Returns:
	    A string of the memory profile
	"""
	return subprocess.run(
		args=["go", "tool", "pprof", "-tags", f"{dir_prefix}/memory.prof"],
		stdout=subprocess.PIPE,
		stderr=subprocess.DEVNULL,
	).stdout.decode("utf-8")


def initialise_tracking(
	interval: float = 0.5,
	dir_prefix: str = "/dev/shm" if sys.platform != "win32" else ".",
) -> None:
	"""The initialise_tracking function starts a daemon thread that periodically saves the current memory profile to disk.

	Args:
	    interval: float: Specify the time interval between each memory
	        profile
	    dir_prefix: str: Specify the directory where the memory profile
	        will be saved

	Returns:
	    Nothing, but it starts a thread that
	"""

	def inner():
		while True:
			jax.profiler.save_device_memory_profile(f"{dir_prefix}/memory.prof.new")
			os.rename(f"{dir_prefix}/memory.prof.new", f"{dir_prefix}/memory.prof")
			time.sleep(interval)
			time.sleep(1)

	thread = threading.Thread(target=inner, daemon=True)
	thread.start()


def get_capacity_matrix(dir_prefix: str = "/dev/shm"):
	pattern = r"(\d+\.\d+\wB) \((\d+\.\d+%)\): (\w+)(\(.*?\))?"

	def calculate_full_size(size, percent):
		size_in_gb = float(re.search(r"(\d+\.\d+)GB", size).group(1))
		percent_value = 100 / float(re.search(r"(\d+\.\d+)%", percent).group(1))
		full_size = size_in_gb * percent_value
		return full_size

	matches = re.findall(pattern, get_mem(dir_prefix=dir_prefix))
	information = {}
	try:
		for match in matches:
			information[match[2]] = {
				"Used": match[0],
				"Usage Percent": match[1],
				"Process": match[3][1:] if match[3] else "∞",
				"Full Capacity": calculate_full_size(match[0], match[1]),
			}
	except (ArithmeticError, AttributeError, KeyError, ValueError):
		...
	return information

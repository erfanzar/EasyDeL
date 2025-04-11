"""
Install EasyDel and dependencies on Ray TPU pods.

This script automates the installation of EasyDel and its dependencies across all hosts in a TPU pod.
It uses Ray for distributed execution and TPUExecutor for TPU-specific orchestration.

Usage:
    python -m easydel.scripts.install_on_hosts \
        --tpu-type <TPU_TYPE> \
        --source <pypi|github>

Options:
    --tpu-type     TPU pod slice type (e.g. v4-16, v3-8)
    --source       Installation source: 'pypi' for PyPI package or 'github' for latest from GitHub
    --num-tpu-hosts Override default host count for TPU type

Example:
    python -m easydel.scripts.install_on_hosts --tpu-type v4-16 --source github
"""

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
#!/usr/bin/env python

import argparse
import sys

try:
	import ray
	from eformer.escale import tpexec as tpx
except ImportError as e:
	print(f"Error: Failed to import Ray or TPExecutor: {e}")
	print(
		"Please ensure Ray and the required 'eformer' library are installed and configured."
	)
	sys.exit(1)


DEFAULT_KNOWN_TPU_TYPES = {
	# TPU v2 Pod Slices
	"v2-8": 1,
	"v2-32": 4,
	"v2-64": 8,
	"v2-128": 16,
	"v2-256": 32,
	"v2-512": 64,
	# TPU v3 Pod Slices
	"v3-8": 1,
	"v3-32": 4,
	"v3-64": 8,
	"v3-128": 16,
	"v3-256": 32,
	"v3-512": 64,
	"v3-1024": 128,
	"v3-2048": 256,
	# TPU v4 Pod Slices
	"v4-8": 1,
	"v4-16": 2,
	"v4-32": 4,
	"v4-64": 8,
	"v4-128": 16,
	"v4-256": 32,
	"v4-512": 64,
	"v4-1024": 128,
	"v4-2048": 256,
	"v4-4096": 512,
	# TPU v5e (Lite Efficiency) - Host counts can vary more here based on chips/host
	"v5e-4": 1,
	"v5e-8": 1,
	"v5e-16": 2,
	"v5e-32": 4,
	"v5e-64": 8,
	"v5e-128": 16,
	"v5e-256": 32,
	# TPU v5p (Performance)
	"v5p-8": 1,
	"v5p-16": 2,
	"v5p-32": 4,
	"v5p-64": 8,
	"v5p-128": 16,
	"v5p-256": 32,
}

# --- Ray Remote Functions ---


@ray.remote
def install_easydel_on_pods_pypi():
	"""Installs EasyDel[tf] from PyPI and other dependencies on Ray nodes."""
	import os  # Import within the function

	node_id = ray.get_runtime_context().get_node_id()
	print(f"Node {node_id}: Installing EasyDel from PyPI...")
	os.system("pip install --upgrade pip -q")
	os.system("pip install easydel[tf] -qU")
	os.system("pip install jax[tpu] -qU")
	os.system(
		"pip3 install torch torchvision torchaudio --index-url "
		"https://download.pytorch.org/whl/cpu -qU"
	)
	print(f"Node {node_id}: Installation from PyPI complete.")
	return True  # Indicate success


@ray.remote
def install_easydel_on_pods_github():
	"""Installs EasyDel[tf] from GitHub head and other dependencies on Ray nodes."""
	import os  # Import within the function

	node_id = ray.get_runtime_context().get_node_id()
	print(f"Node {node_id}: Installing EasyDel from GitHub head...")
	os.system("pip install --upgrade pip -q")
	os.system("pip uninstall easydel -y -q")
	os.system(
		"pip install 'easydel[tf] @ git+https://github.com/erfanzar/easydel.git' -qU"
	)
	os.system("pip install jax[tpu] -qU")
	os.system(
		"pip3 install torch torchvision torchaudio --index-url "
		"https://download.pytorch.org/whl/cpu -qU"
	)
	print(f"Node {node_id}: Installation from GitHub head complete.")
	return True  # Indicate success


def main():
	parser = argparse.ArgumentParser(
		description="Install EasyDel and dependencies on Ray TPU pods. Requires Ray and eformer.escale.tpexec.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument(
		"--source",
		choices=["pypi", "github"],
		default="pypi",
		help="Choose the source for EasyDel installation: PyPI package or GitHub head.",
	)

	parser.add_argument(
		"--tpu-type", type=str, default="v4-16", help="The type of TPU pod slice to use."
	)

	parser.add_argument(
		"--num-tpu-hosts",
		type=int,
		default=None,
		help=(
			"Optional host counts (e.g., 2,8,16). "
			"If provided, this overrides the internal default mapping."
		),
	)

	args = parser.parse_args()

	known_tpu_types = DEFAULT_KNOWN_TPU_TYPES

	if args.source == "github":
		install_func = install_easydel_on_pods_github
		print("Selected installation source: GitHub head ")
	else:
		install_func = install_easydel_on_pods_pypi
		print("Selected installation source: PyPI")

	tpu_type = args.tpu_type
	print(f"Selected TPU type: {tpu_type}")

	if args.num_tpu_hosts is None:
		num_hosts = known_tpu_types[tpu_type]
	print(f"Determined number of hosts for {tpu_type}: {num_hosts}")

	try:
		print("Initializing Ray...")
		ray.init("auto")
		print(f"Ray initialized successfully. Cluster resources: {ray.cluster_resources()}")
		print(
			f"\nExecuting installation function on "
			f"{num_hosts} host(s) of type '{tpu_type}' via TPUExecutor..."
		)

		results = ray.get(
			tpx.TPUExecutor.execute(
				install_func,
				tpu_type=tpu_type,
				num_hosts=num_hosts,
			)
		)
		print(
			"\nExecution command sent. Waiting for remote tasks "
			"(TPUExecutor might block or manage this)..."
		)
		if results:
			print(
				f"Received results from execution (structure depends on TPUExecutor): {results}"
			)

		print(
			"\nInstallation process initiated (or completed, depending on executor) on pods."
		)

	except Exception as e:
		print(f"\nAn error occurred during Ray initialization or execution: {e}")
		import traceback

		traceback.print_exc()
		if ray.is_initialized():
			print("Attempting to shutdown Ray...")
			ray.shutdown()
			print("Ray shutdown.")
		sys.exit(1)


if __name__ == "__main__":
	main()

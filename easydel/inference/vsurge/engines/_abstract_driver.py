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

import abc
import datetime
import typing as tp

import jax

if tp.TYPE_CHECKING:
	from easydel.infra.base_module import EasyDeLBaseModule
	from easydel.infra.utils import ProcessingClassType
else:
	ProcessingClassType = tp.Any
	EasyDeLBaseModule = tp.Any


class AbstractDriver(abc.ABC):
	"""
	Abstract base class for inference engine drivers.

	Defines the essential interface that all driver implementations must provide.
	"""

	@abc.abstractmethod
	def compile(self):
		"""
		Compiles or prepares the underlying inference engines.

		This method should handle any necessary setup, such as model compilation,
		weight loading, or resource allocation required before the driver can
		process requests.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def stop(self):
		"""
		Gracefully stops the driver and its associated background processes.

		This method should ensure that all ongoing operations are completed or
		cancelled safely, resources are released, and threads are joined.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def get_total_concurrent_requests(self) -> int:
		"""
		Returns the maximum number of concurrent requests the driver can handle.

		This indicates the capacity of the driver's decoding stage.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def submit_request(self, request: tp.Any):
		"""
		Submits a new inference request to the driver.

		Args:
		    request: The inference request object. The specific type may vary
		             depending on the implementation (e.g., ActiveRequest).
		"""
		raise NotImplementedError

	@property
	@abc.abstractmethod
	def processor(self) -> ProcessingClassType:
		"""
		Returns the processor (tokenizer) associated with the driver's engines.
		"""
		raise NotImplementedError

	def _get_model_name(self, model) -> str:
		"""
		Generate a standardized vsurge name combining model type, size, and timestamp.

		Format: {model_type}-{size_in_B}B-{timestamp}
		Example: llama-7.00B-20240311
		"""
		model_type = self._get_model_type(model)
		model_size = self._calculate_model_size(model.graphstate)
		timestamp = datetime.datetime.now().strftime("%Y%m%d")

		return f"{model_type}-{model_size}B-{timestamp}"

	def _get_model_type(self, model) -> str:
		"""Get the model type, with fallback to 'unknown' if not found."""
		return getattr(model.config, "model_type", "unknown").lower()

	def _calculate_model_size(self, graphstate) -> str:
		"""
		Calculate model size in billions of parameters.
		Returns formatted string with 2 decimal places.
		"""
		try:
			num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
			size_in_billions = num_params / 1e9
			return f"{size_in_billions:.2f}"
		except Exception:
			return "unknown"

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

"""Provides thread-safe ID generation for API requests."""

import threading


class ReqIDGenerator:
	"""
	A thread-safe generator for unique request IDs.

	This class ensures that each generated ID is unique even when accessed
	concurrently from multiple threads. It increments the ID by 8 each time
	to potentially group related sub-requests.
	"""

	def __init__(self):
		"""Initializes the ReqIDGenerator with the starting ID 0."""
		self.current_id = 0
		self.lock = threading.Lock()

	def generate_id(self) -> int:
		"""
		Generates a new, unique request ID in a thread-safe manner.

		Returns:
		    int: The newly generated unique ID.
		"""
		with self.lock:
			id = self.current_id
			self.current_id += 8
		return id


def convert_sub_id_to_group_id(sub_req_id: int) -> int:
	"""
	Converts a sub-request ID (generated by `ReqIDGenerator`) back to its
	associated group ID.

	Since the generator increments by 8, this effectively finds the base ID
	for a group of up to 8 related requests.

	Args:
	    sub_req_id (int): The ID generated for a specific sub-request.

	Returns:
	    int: The group ID associated with the sub-request ID.
	"""
	return (sub_req_id // 8) * 8

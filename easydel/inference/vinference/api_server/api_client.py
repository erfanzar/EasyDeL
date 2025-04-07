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

"""
A client for interacting with the vInference API server, mimicking OpenAI's API structure.
"""

import json
import typing as tp
import urllib.parse

import requests

from .api_models import (
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatCompletionStreamResponse,
	CompletionRequest,
	CompletionResponse,
	CompletionStreamResponse,
)


class vInferenceAPIError(Exception):
	"""Custom exception class for vInference API errors."""

	def __init__(
		self, status_code: int, message: str, response_content: tp.Optional[str] = None
	):
		"""
		Initializes the vInferenceAPIError.

		Args:
		    status_code (int): The HTTP status code of the error response.
		    message (str): The error message.
		    response_content (tp.Optional[str]): The raw response content, if available.
		"""
		self.status_code = status_code
		self.message = message
		self.response_content = response_content
		super().__init__(f"vInference API Error ({status_code}): {message}")


class vInferenceChatCompletionClient:
	"""
	Client for interacting with the vInference Chat Completion API endpoint.

	This client handles communication with the vInference server, including
	sending requests, handling responses (streaming or non-streaming), managing
	retries, and parsing errors.
	"""

	def __init__(self, base_url: str, max_retries: int = 5, timeout: float = 30.0):
		"""
		Initializes the vInferenceChatCompletionClient.

		Args:
		    base_url (str): The base URL of the vInference API server (e.g., "http://localhost:7860").
		    max_retries (int): Maximum number of retries for transient network errors. Defaults to 5.
		    timeout (float): Request timeout in seconds. Defaults to 30.0.
		"""
		url = urllib.parse.urlparse(base_url)
		self.base_url = f"{url.scheme}://{url.netloc}"
		self.max_retries = max_retries
		self.timeout = timeout

		self.session = requests.Session()
		retry_strategy = requests.adapters.Retry(
			total=max_retries, backoff_factor=1, status_forcelist=[502, 503, 504]
		)
		adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
		self.session.mount("http://", adapter)
		self.session.mount("https://", adapter)

	def _parse_error_response(self, response: requests.Response) -> str:
		"""
		Attempts to parse a detailed error message from the API response.

		Args:
		    response (requests.Response): The error response object.

		Returns:
		    str: The parsed error message or the raw response text if parsing fails.
		"""
		try:
			error_data = response.json()
			return error_data.get("error", {}).get("message", response.text)
		except (json.JSONDecodeError, AttributeError):
			return response.text

	def create_chat_completion(
		self,
		request: ChatCompletionRequest,
		extra_headers: tp.Optional[dict] = None,
	) -> tp.Generator[
		tp.Union[ChatCompletionStreamResponse, ChatCompletionResponse],
		None,
		None,
	]:
		"""
		Sends a chat completion request to the vInference API.

		Handles both streaming and non-streaming responses based on the `stream`
		attribute in the `request` object.

		Args:
		    request (ChatCompletionRequest): The chat completion request object.
		    extra_headers (tp.Optional[dict]): Optional dictionary of extra headers
		        to include in the request. Defaults to None.

		Yields:
		    tp.Union[ChatCompletionStreamResponse, ChatCompletionResponse]:
		        For streaming requests, yields `ChatCompletionStreamResponse` objects
		        for each chunk received. For non-streaming requests, yields a single
		        `ChatCompletionResponse` object.

		Raises:
		    vInferenceAPIError: If the API returns an error status code or if there's
		        an issue parsing the response.
		    requests.RequestException: For underlying network connection issues.
		"""
		url = f"{self.base_url}/v1/chat/completions"
		extra_headers = extra_headers or {}
		headers = {
			"bypass-tunnel-reminder": "true",
			"Content-Type": "application/json",
			"Accept": "application/json",
		}.update(extra_headers)
		out = ChatCompletionStreamResponse if request.stream else ChatCompletionResponse
		try:
			with self.session.post(
				url,
				data=request.model_dump_json(),
				headers=headers,
				stream=True,
				timeout=self.timeout,
			) as response:
				if response.status_code != 200:
					error_message = self._parse_error_response(response)
					raise vInferenceAPIError(
						status_code=response.status_code,
						message=error_message,
						response_content=response.text,
					)
				for line in response.iter_lines(decode_unicode=True):
					if line:
						if line.startswith("data: "):
							try:
								data = json.loads(line[6:])
								yield out(**data)
							except json.JSONDecodeError as e:
								raise vInferenceAPIError(
									status_code=response.status_code,
									message=f"Failed to parse response: {str(e)}",
									response_content=line,
								) from e
						else:
							try:
								data = json.loads(line)
								yield out(**data)
							except json.JSONDecodeError as e:
								raise vInferenceAPIError(
									status_code=response.status_code,
									message=f"Failed to parse response: {str(e)}",
									response_content=line,
								) from e

		except requests.RequestException as e:
			raise vInferenceAPIError(
				status_code=500, message=f"Network error occurred: {str(e)}"
			) from e

	def __enter__(self):
		"""Allows using the client with a 'with' statement."""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Closes the underlying requests session."""
		self.session.close()


class vInferenceCompletionClient:
	"""
	Client for interacting with the vInference Completion API endpoint.

	This client handles communication with the vInference server for text completions,
	supporting both streaming and non-streaming modes.
	"""

	def __init__(self, base_url: str, max_retries: int = 5, timeout: float = 30.0):
		"""
		Initializes the vInferenceCompletionClient.

		Args:
		    base_url (str): The base URL of the vInference API server (e.g., "http://localhost:7860").
		    max_retries (int): Maximum number of retries for transient network errors. Defaults to 5.
		    timeout (float): Request timeout in seconds. Defaults to 30.0.
		"""
		url = urllib.parse.urlparse(base_url)
		self.base_url = f"{url.scheme}://{url.netloc}"
		self.max_retries = max_retries
		self.timeout = timeout

		self.session = requests.Session()
		retry_strategy = requests.adapters.Retry(
			total=max_retries, backoff_factor=1, status_forcelist=[502, 503, 504]
		)
		adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
		self.session.mount("http://", adapter)
		self.session.mount("https://", adapter)

	def _parse_error_response(self, response: requests.Response) -> str:
		"""
		Attempts to parse a detailed error message from the API response.

		Args:
		    response (requests.Response): The error response object.

		Returns:
		    str: The parsed error message or the raw response text if parsing fails.
		"""
		try:
			error_data = response.json()
			return error_data.get("error", {}).get("message", response.text)
		except (json.JSONDecodeError, AttributeError):
			return response.text

	def create_completion(
		self,
		request: CompletionRequest,
		extra_headers: tp.Optional[dict] = None,
	) -> tp.Generator[
		tp.Union[CompletionStreamResponse, CompletionResponse],
		None,
		None,
	]:
		"""
		Sends a text completion request to the vInference API.

		Handles both streaming and non-streaming responses based on the `stream`
		attribute in the `request` object.

		Args:
		    request (CompletionRequest): The completion request object.
		    extra_headers (tp.Optional[dict]): Optional dictionary of extra headers
		        to include in the request. Defaults to None.

		Yields:
		    tp.Union[CompletionStreamResponse, CompletionResponse]:
		        For streaming requests, yields `CompletionStreamResponse` objects
		        for each chunk received. For non-streaming requests, yields a single
		        `CompletionResponse` object.

		Raises:
		    vInferenceAPIError: If the API returns an error status code or if there's
		        an issue parsing the response.
		    requests.RequestException: For underlying network connection issues.
		"""
		url = f"{self.base_url}/v1/completions"
		extra_headers = extra_headers or {}
		headers = {
			"bypass-tunnel-reminder": "true",
			"Content-Type": "application/json",
			"Accept": "application/json",
		}.update(extra_headers)
		out = CompletionStreamResponse if request.stream else CompletionResponse
		try:
			with self.session.post(
				url,
				data=request.model_dump_json(),
				headers=headers,
				stream=True,
				timeout=self.timeout,
			) as response:
				if response.status_code != 200:
					error_message = self._parse_error_response(response)
					raise vInferenceAPIError(
						status_code=response.status_code,
						message=error_message,
						response_content=response.text,
					)
				for line in response.iter_lines(decode_unicode=True):
					if line:
						if line.startswith("data: "):
							try:
								data = json.loads(line[6:])
								yield out(**data)
							except json.JSONDecodeError as e:
								raise vInferenceAPIError(
									status_code=response.status_code,
									message=f"Failed to parse response: {str(e)}",
									response_content=line,
								) from e
						else:
							try:
								data = json.loads(line)
								yield out(**data)
							except json.JSONDecodeError as e:
								raise vInferenceAPIError(
									status_code=response.status_code,
									message=f"Failed to parse response: {str(e)}",
									response_content=line,
								) from e

		except requests.RequestException as e:
			raise vInferenceAPIError(
				status_code=500, message=f"Network error occurred: {str(e)}"
			) from e

	def __enter__(self):
		"""Allows using the client with a 'with' statement."""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Closes the underlying requests session."""
		self.session.close()


class vInferenceClient:
	"""
	Unified client for interacting with all vInference API endpoints.

	This client provides access to both chat completions and text completions
	through a single interface.
	"""

	def __init__(self, base_url: str, max_retries: int = 5, timeout: float = 30.0):
		"""
		Initializes the vInferenceClient with connection parameters.

		Args:
			base_url (str): The base URL of the vInference API server
			max_retries (int): Maximum number of retries for network errors
			timeout (float): Request timeout in seconds
		"""
		self.chat = vInferenceChatCompletionClient(base_url, max_retries, timeout)
		self.completions = vInferenceCompletionClient(base_url, max_retries, timeout)

	def __enter__(self):
		"""Enters context for all client components."""
		self.chat.__enter__()
		self.completions.__enter__()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Exits context for all client components."""
		self.chat.__exit__(exc_type, exc_val, exc_tb)
		self.completions.__exit__(exc_type, exc_val, exc_tb)

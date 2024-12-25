import json
import typing as tp
import urllib.parse

import requests

from .api_models import (
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatCompletionStreamResponse,
)


class vInferenceAPIError(Exception):
	def __init__(
		self, status_code: int, message: str, response_content: tp.Optional[str] = None
	):
		self.status_code = status_code
		self.message = message
		self.response_content = response_content
		super().__init__(f"vInference API Error ({status_code}): {message}")


class vInferenceChatCompletionClient:
	def __init__(self, base_url: str, max_retries: int = 5, timeout: float = 30.0):
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
		Create a chat completion with streaming response.

		Args:
		    request: ChatCompletionRequest object containing the request parameters

		Yields:
		    dict: Parsed response chunks from the API

		Raises:
		    vInferenceAPIError: If the API returns an error response
		    requests.RequestException: For network-related errors
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
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.session.close()

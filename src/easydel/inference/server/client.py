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

"""Module for client-side interaction with EasyDeL API Engine."""

import json
from dataclasses import dataclass
from typing import Dict, Generator, List

import requests

from easydel.etils.etils import get_logger

logger = get_logger(__name__)
try:
	from websocket import create_connection  # noqa #type:ignore
except ModuleNotFoundError:
	create_connection = None
	logger.warn("couldn't import websocket, ServerEngine client side won't work.")


@dataclass
class GenerateGradioOutput:
	"""Dataclass to store Gradio generation output."""

	sequence: str
	sequence_stream: str
	token_count: int
	elapsed_time: float
	tokens_per_second: float


@dataclass
class GenerationProcessOutput:
	"""Dataclass to store generation process output."""

	step: int
	tokens_generated: int
	char_generated: int
	elapsed_time: float
	tokens_per_second: float


@dataclass
class SocketGenerationOutput:
	"""Dataclass to store socket generation output."""

	progress: GenerationProcessOutput
	response: str


@dataclass
class HttpGenerationProcessOutput:
	"""Dataclass for HTTP generation process output."""

	tokens_generated: int
	char_generated: int
	elapsed_time: float
	tokens_per_second: float


@dataclass
class HttpGenerationOutput:
	"""Dataclass for HTTP generation output."""

	progress: HttpGenerationProcessOutput
	response: str


def generate_https(
	hostname: str,
	conversation: List[Dict],
	path: str = "conversation",
	verify_ssl: bool = False,
) -> HttpGenerationOutput:
	"""Generate text using HTTPS.

	Args:
	    hostname (str): The hostname of the server.
	    conversation (List[Dict]): The conversation history.
	    path (str, optional): The path to the generation endpoint.
	        Defaults to "conversation".
	    verify_ssl (bool, optional): Whether to verify SSL certificate.
	        Defaults to False.

	Returns:
	    HttpGenerationOutput: The generated text and progress information.
	"""
	params = {"conversation": json.dumps(conversation)}
	try:
		response = requests.get(
			f"https://{hostname}/{path}",
			params=params,
			verify=verify_ssl,
		)
		response.raise_for_status()  # Raise an exception for bad status codes
	except requests.exceptions.RequestException as e:
		print(f"An error occurred: {e}")
		if e.response:
			print(f"Response status code: {e.response.status_code}")
			print(f"Response headers: {e.response.headers}")
			print(f"Response content: {e.response.text}")
		raise
	data = response.json()
	return HttpGenerationOutput(
		response=data["response"],
		progress=HttpGenerationProcessOutput(**data["progress"]),
	)


def generate_websocket(
	hostname: str,
	conversation: List[Dict],
	path: str = "conversation",
) -> Generator[SocketGenerationOutput, None, None]:
	"""Generate text using WebSockets.

	Args:
	    hostname (str): The hostname of the server.
	    conversation (List[Dict]): The conversation history.
	    path (str, optional): The path to the generation endpoint.
	        Defaults to "conversation".

	Yields:
	    SocketGenerationOutput: The generated text and progress information.
	"""
	hostname = (
		hostname.replace("https://", "")
		.replace("http://", "")
		.replace("ws://", "")
		.replace("/", "")
	)
	ws = create_connection(f"ws://{hostname}/{path}")
	data_to_send = {"conversation": conversation}

	ws.send(json.dumps(data_to_send))

	while True:
		response = ws.recv()
		response_data = json.loads(response)
		if response_data["done"]:
			break
		yield SocketGenerationOutput(
			response=response_data["response"],
			progress=GenerationProcessOutput(**response_data["progress"]),
		)


def generate_gradio(
	url: str,
	conversation: Dict,
):
	"""Generate responses using a Gradio client.

	Args:
	    url (str): The URL of the Gradio server.
	    conversation (Dict): The conversation history.

	Yields:
	    GenerateGradioOutput: The generated output information.
	"""
	try:
		from gradio_client import Client  # noqa # type:ignore
	except ModuleNotFoundError:
		raise ModuleNotFoundError(
			"`gradio_client` no found consider running " "`pip install gradio_client gradio`"
		) from None
	job_arguments = {
		"conversation": conversation,
	}

	client = Client(url, verbose=False)
	job = client.submit(**job_arguments, api_name="/generate_request")
	last_sequence = ""
	while not job.done():
		response = job.outputs()
		if len(response) > 0 and last_sequence != response[-1][0]:
			last_response = list(response[-1])
			last_response[1] = last_response[0][len(last_sequence) :]
			yield GenerateGradioOutput(*last_response)
			last_sequence = last_response[0]

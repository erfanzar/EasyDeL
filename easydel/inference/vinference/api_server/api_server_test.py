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
"""An example asynchronous client script for testing the vInference API server."""

import asyncio
import json
import typing as tp

import aiohttp


class ChatCompletionClient:
	"""An asynchronous client for interacting with the chat completion endpoint."""

	def __init__(self, base_url: str):
		"""
		Initializes the asynchronous client.

		Args:
		    base_url (str): The base URL of the vInference API server (e.g., "http://127.0.0.1:7680").
		"""
		self.base_url = base_url

	async def create_chat_completion(
		self,
		messages: tp.List[tp.Dict[str, str]],
		model: str,
		stream: bool = True,
		**kwargs,
	) -> tp.AsyncGenerator[tp.Dict[str, tp.Any], None]:
		"""
		Sends a chat completion request to the server and streams the response.

		Args:
		    messages (tp.List[tp.Dict[str, str]]): A list of message dictionaries, e.g.,
		        `[{"role": "user", "content": "Hello!"}]`.
		    model (str): The name of the model to use.
		    stream (bool): Whether to request a streaming response. Defaults to True.
		    **kwargs: Additional parameters to pass to the API (e.g., temperature, max_tokens).

		Yields:
		    tp.Dict[str, tp.Any]: Each chunk of the response as a dictionary.

		Raises:
		    Exception: If the server returns a non-200 status code.
		"""
		url = f"{self.base_url}/v1/chat/completions"

		payload = {"messages": messages, "model": model, "stream": stream, **kwargs}

		async with aiohttp.ClientSession() as session:
			async with session.post(url, json=payload) as response:
				if response.status != 200:
					raise Exception(f"Error: {response.status} - {await response.text()}")

				async for line in response.content:
					line = line.decode("utf-8").strip()
					if line.startswith("data: "):
						data = json.loads(line[6:])
						yield data


async def main():
	"""Main function to run the example chat completion interaction."""
	client = ChatCompletionClient("http://127.0.0.1:7680")  # Adjust URL if needed
	messages = [
		{"role": "system", "content": "You are a helpful assistant."},
		{
			"role": "user",
			"content": "write a neural network in c++ and rust and compare them",
		},
	]
	model_name = "llama-3-8b"  # Replace with your actual running model name

	print(f"Sending request to model: {model_name}")
	try:
		async for chunk in client.create_chat_completion(
			messages,
			model=model_name,
			max_tokens=512,  # Example: setting max_tokens
		):
			if (
				chunk["choices"]
				and chunk["choices"][0].get("delta")
				and chunk["choices"][0]["delta"].get("content")
			):
				print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

			# Check for finish reason in the last chunk
			if chunk["choices"] and chunk["choices"][0].get("finish_reason"):
				print("\n--- Finish Reason ---")
				print(chunk["choices"][0]["finish_reason"])
				if chunk.get("usage"):
					print("--- Usage --- ")
					print(chunk["usage"])
				break  # Stop after finish reason is received
	except Exception as e:
		print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
	asyncio.run(main())

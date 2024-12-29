import asyncio
import json
import typing as tp

import aiohttp


class ChatCompletionClient:
	def __init__(self, base_url: str):
		self.base_url = base_url

	async def create_chat_completion(
		self,
		messages: tp.List[tp.Dict[str, str]],
		model: str,
		stream: bool = True,
		**kwargs,
	) -> tp.AsyncGenerator[tp.Dict[str, tp.Any], None]:
		url = f"{self.base_url}/v1/chat/completions"

		payload = {"messages": messages, "model": model, "stream": stream, **kwargs}

		async with aiohttp.ClientSession() as session:
			async with session.post(url, json=payload) as response:
				if response.status != 200:
					raise Exception(f"Error: {response.status}")

				async for line in response.content:
					line = line.decode("utf-8").strip()
					if line.startswith("data: "):
						data = json.loads(line[6:])
						yield data


async def main():
	client = ChatCompletionClient("http://127.0.0.1:7680")
	messages = [
		{"role": "system", "content": "You are a helpful assistant."},
		{
			"role": "user",
			"content": "write a neural network in c++ and rust and compare them",
		},
	]

	async for chunk in client.create_chat_completion(
		messages, model="llama-1.53B-20241013"
	):
		if chunk["choices"][0]["finish_reason"] is None:
			print(chunk["choices"][0]["response"], end="", flush=True)
		else:
			print("\nFinish reason:", chunk["choices"][0]["finish_reason"])
			print("Usage:", chunk["usage"])


if __name__ == "__main__":
	asyncio.run(main())

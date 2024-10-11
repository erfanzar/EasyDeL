from http import HTTPStatus
from fastapi.responses import JSONResponse
from easydel.inference.vinference.api_models import (
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatCompletionResponseChoice,
	ChatCompletionStreamResponse,
	ChatCompletionStreamResponseChoice,
	ChatMessage,
)
from easydel.etils.etils import get_logger
from fastapi import FastAPI, Request
import uvicorn
from typing import Dict
from easydel.inference.vinference import vInference


app = FastAPI()
server = uvicorn.Server(uvicorn.Config(app=app))


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
	return JSONResponse({"message": message}, status_code=status_code.value)


class ApiServer:
	def __init__(self, inference_map: Dict[str, vInference] = None) -> None:
		assert inference_map is not None, "`inference_map` can not be None."
		for inference in inference_map.values():
			assert isinstance(
				inference, vInference
			), "values and inferences in inference_map must be `vInference`"

		self.inference_map = inference_map
		self.app = app
		self.server = server
		self.logger = get_logger(__name__)

	@app.post("/v1/chat/completions", response_model=ChatCompletionRequest)
	async def chat_completions(self, request: ChatCompletionRequest):
		print(await request)

	@app.get("/liveness")
	@app.post("/liveness")
	def liveness(self):
		return JSONResponse({"status": "ok"}, status_code=200)

	@app.get("/readiness")
	@app.post("/readiness")
	def readiness(self):
		return JSONResponse({"status": "ok"}, status_code=200)

	@app.get("/get_available_inference_names")
	@app.post("/get_available_inference_names")
	def get_available_inference_names(self):
		return JSONResponse({"inference_names": self.inference_map.keys()}, status_code=200)

	@app.get("/count_tokens")
	@app.post("/count_tokens")
	async def count_tokens(self, request: Request):
		try:
			request_dict = await request.json()
			prompt = request_dict.pop("text")
			model = request_dict.pop("model")
			return JSONResponse(
				{"ntokens": await self.inference_map[model].count_tokens(prompt)},
				status_code=200,
			)
		except Exception as e:
			return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


def main(): ...


if __name__ == "__main__":
	main()

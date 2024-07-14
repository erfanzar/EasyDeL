import json
from dataclasses import dataclass
from typing import Dict, Generator, List
import requests
from easydel.etils.etils import get_logger

logger = get_logger(__name__)
try:
    from websocket import create_connection
except ModuleNotFoundError:
    create_connection = None
    logger.warn("couldn't import websocket, ServerEngine client side won't work.")


@dataclass
class GenerateGradioOutput:
    sequence: str
    sequence_stream: str
    token_count: int
    elapsed_time: float
    tokens_per_second: float


@dataclass
class GenerationProcessOutput:
    step: int
    tokens_generated: int
    char_generated: int
    elapsed_time: float
    tokens_per_second: float


@dataclass
class SocketGenerationOutput:
    progress: GenerationProcessOutput
    response: str


@dataclass
class HttpGenerationProcessOutput:
    tokens_generated: int
    char_generated: int
    elapsed_time: float
    tokens_per_second: float


@dataclass
class HttpGenerationOutput:
    progress: HttpGenerationProcessOutput
    response: str


def generate_https(
    hostname: str,
    conversation: List[Dict],
    path: str = "conversation",
    verify_ssl: bool = False,
) -> HttpGenerationOutput:
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
    url,
    conversation: Dict,
):
    """
    Generate responses using a Gradio client.

    Args:
        url (str): The URL of the Gradio server.
        conversation (Dict): The conversation history.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Top-p sampling parameter.
        top_k (Optional[int]): Top-k sampling parameter.

    Yields:
        GenerateGradioOutput
    """
    try:
        from gradio_client import Client
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`gradio_client` no found consider running `pip install gradio_client gradio`"
        )
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

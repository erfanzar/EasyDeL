#!/usr/bin/env python3
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
Example of using the vWhisper FastAPI server.

This script demonstrates:
1. How to start the server
2. How to send requests to the server using the requests library
"""

import argparse

from jax import numpy as jnp

# Add the parent directory to the path so that we can import easydel
from easydel.inference.vwhisper.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Run the EasyDeL Whisper server example")

    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Name of the Whisper model to use (from HuggingFace)",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")

    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for the model",
    )

    args = parser.parse_args()

    # Convert dtype string to jnp type
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }

    dtype = dtype_map[args.dtype]

    print(f"Starting Whisper server with model {args.model}...")
    print(f"Server will be available at http://{args.host}:{args.port}")
    print("\nExample client usage (in another terminal):")
    print("------------------------------------------")
    print("import requests")
    print("import json")
    print("")
    print("# For transcription:")
    print("files = {'file': open('your_audio_file.mp3', 'rb')}")
    print("data = {'model': 'whisper-large-v3-turbo', 'response_format': 'json'}")
    print(f"response = requests.post('http://localhost:{args.port}/v1/audio/transcriptions', files=files, data=data)")
    print("print(json.dumps(response.json(), indent=2))")
    print("")
    print("# For translation:")
    print("files = {'file': open('your_audio_file.mp3', 'rb')}")
    print("data = {'model': 'whisper-large-v3-turbo', 'response_format': 'json'}")
    print(f"response = requests.post('http://localhost:{args.port}/v1/audio/translations', files=files, data=data)")
    print("print(json.dumps(response.json(), indent=2))")
    print("------------------------------------------")
    print("\nPress Ctrl+C to stop the server\n")

    # Run the server
    run_server(model_name=args.model, host=args.host, port=args.port, dtype=dtype)


if __name__ == "__main__":
    main()

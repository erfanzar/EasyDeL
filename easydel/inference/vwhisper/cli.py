# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Command-line interface for vWhisper server.

This module provides a CLI entry point for running the vWhisper FastAPI
server. It allows users to start a Whisper transcription/translation
service from the command line with configurable options.

Usage:
    Run the server from command line::

        $ python -m easydel.inference.vwhisper.cli \
            --model openai/whisper-large-v3-turbo \
            --host 0.0.0.0 \
            --port 8000 \
            --dtype bfloat16

    Or use the shorter form::

        $ python -m easydel.inference.vwhisper.cli --model openai/whisper-base

Arguments:
    --model: Name of the Whisper model from HuggingFace Hub.
        Default: "openai/whisper-large-v3-turbo"

    --host: Host address to bind the server to.
        Default: "0.0.0.0" (all interfaces)

    --port: Port number to listen on.
        Default: 8000

    --dtype: Data type for model weights. Options: float32, float16, bfloat16.
        Default: "bfloat16"

Example:
    Starting a server with a smaller model for testing::

        $ python -m easydel.inference.vwhisper.cli \
            --model openai/whisper-tiny \
            --port 9000 \
            --dtype float32

Note:
    This module requires FastAPI, uvicorn, and their dependencies to be
    installed. The server will be accessible at http://host:port/ once
    started, with API endpoints at /v1/audio/transcriptions and
    /v1/audio/translations.
"""

import argparse

from jax import numpy as jnp

from .server import run_server


def main():
    """Entry point for the vWhisper CLI server.

    Parses command-line arguments and starts the vWhisper FastAPI server
    with the specified configuration. The server provides OpenAI-compatible
    API endpoints for audio transcription and translation.

    The function performs the following steps:
        1. Parses command-line arguments for model, host, port, and dtype
        2. Converts the dtype string to the corresponding JAX dtype
        3. Launches the FastAPI server via uvicorn

    Command-line Arguments:
        --model (str): HuggingFace model identifier for the Whisper model.
            Defaults to "openai/whisper-large-v3-turbo".
        --host (str): Host address to bind the server.
            Defaults to "0.0.0.0" (all network interfaces).
        --port (int): Port number for the server.
            Defaults to 8000.
        --dtype (str): Data type for model computations.
            Choices are "float32", "float16", "bfloat16".
            Defaults to "bfloat16".

    Returns:
        None: This function does not return; it runs the server until
        interrupted.

    Raises:
        SystemExit: If argument parsing fails or invalid arguments
            are provided.

    Example:
        Running from command line::

            $ python -m easydel.inference.vwhisper.cli \
                --model openai/whisper-base \
                --host 127.0.0.1 \
                --port 8080

        Or calling directly from Python::

            >>> from easydel.inference.vwhisper.cli import main
            >>> import sys
            >>> sys.argv = ['cli', '--model', 'openai/whisper-tiny']
            >>> main()  # Starts the server

    Note:
        - The server blocks until interrupted (e.g., Ctrl+C)
        - Using bfloat16 provides good performance on TPU and modern GPUs
        - float32 may be needed for older hardware or debugging
    """
    parser = argparse.ArgumentParser(description="Run the EasyDeL Whisper FastAPI server")

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

    run_server(model_name=args.model, host=args.host, port=args.port, dtype=dtype)


if __name__ == "__main__":
    main()

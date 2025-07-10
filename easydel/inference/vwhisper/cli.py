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

import argparse

from jax import numpy as jnp

from .server import run_server


def main():
    """
    CLI entry point for running the vWhisper server.
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

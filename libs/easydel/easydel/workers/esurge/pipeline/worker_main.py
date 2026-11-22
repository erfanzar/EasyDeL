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

"""Subprocess entry point for the tokenizer ZMQ worker.

Detokenization is native and in-process in :mod:`surgecore`; only prompt tokenization retains a subprocess so its
HuggingFace call does not contend with the model process. The worker accepts ``tokenize``, ``drain`` and ``shutdown``
commands over ZMQ pyobj messages.

Invocation: ``python -m easydel.workers.esurge.pipeline.worker_main tokenizer --endpoint ENDPOINT
--tokenizer-path PATH --tokenizer-kwargs JSON_OBJECT``.
"""

from __future__ import annotations

import argparse
import json
import os


def _configure_worker_environment() -> None:
    """Isolate the tokenizer subprocess from the model process's JAX settings."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["ENABLE_DISTRIBUTED_INIT"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


if __name__ == "__main__":
    _configure_worker_environment()

# Third-party imports must follow subprocess environment isolation.
import zmq  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _tokenizer_worker(endpoint: str, tokenizer_path: str, tokenizer_kwargs: dict) -> None:
    """Serve prompt tokenization until ``shutdown`` arrives."""
    from easydel.utils import flags

    if "trust_remote_code" not in tokenizer_kwargs:
        tokenizer_kwargs["trust_remote_code"] = flags.get_bool(flags.ESURGE_WORKER_TRUST_REMOTE_CODE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)
    try:
        while True:
            message = socket.recv_pyobj()
            try:
                cmd = message.get("cmd")
                if cmd == "tokenize":
                    prompt = message["prompt"]
                    encoded = tokenizer(prompt, return_tensors=None)
                    token_ids = encoded["input_ids"]
                    if token_ids and isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                    socket.send_pyobj({"status": "ok", "tokens": list(token_ids)})
                elif cmd == "tokenize_batch":
                    # One call over the whole list: the fast tokenizer encodes the prompts in parallel, and each
                    # prompt's ids equal those of its own ``tokenize`` call.
                    encoded = tokenizer(list(message["prompts"]), return_tensors=None)
                    socket.send_pyobj({"status": "ok", "tokens": [list(ids) for ids in encoded["input_ids"]]})
                elif cmd == "drain":
                    socket.send_pyobj({"status": "ok"})
                elif cmd == "shutdown":
                    socket.send_pyobj({"status": "ok"})
                    break
                else:
                    socket.send_pyobj({"status": "error", "message": f"Unknown cmd {cmd}"})
            except Exception as error:
                socket.send_pyobj({"status": "error", "message": str(error), "exception_type": type(error).__name__})
    finally:
        socket.close(0)
        ctx.term()


def main() -> None:
    """Parse worker arguments and run the tokenizer loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tokenizer"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-kwargs", required=True)
    args = parser.parse_args()

    _configure_worker_environment()
    tokenizer_kwargs = json.loads(args.tokenizer_kwargs) if args.tokenizer_kwargs else {}
    _tokenizer_worker(args.endpoint, args.tokenizer_path, tokenizer_kwargs)


if __name__ == "__main__":
    main()

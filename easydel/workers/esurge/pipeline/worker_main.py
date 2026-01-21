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

"""Worker process entry point for tokenizer and detokenizer services.

This module implements the main worker processes that handle tokenization and
detokenization requests via ZeroMQ. It provides both tokenizer and detokenizer
worker implementations that can be spawned as separate processes.

Note:
    This module is for internal use only and is not part of EasyDeL's public API.
    It is only accessible to EasyDeL modules that require external worker processes
    to handle specific tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from collections.abc import Iterable

import zmq
from transformers import AutoTokenizer


class FastIncrementalDecoder:
    """
    Incrementally decode token streams while handling malformed UTF‑8
    (the “�” replacement character).

    Public API matches the original `SimpleDecoder.decode` method:

        delta, new_buffered_tokens, has_buffer = decoder.decode(
            tokens,
            previous_text="",
            buffered=decoder.buffered,   # mutable list that the decoder updates
            skip_special_tokens=True,
        )
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool) -> str:
        """Wrapper around `tokenizer.decode` that works with all HF versions."""
        try:
            return self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
            )

    def decode(
        self,
        new_tokens: Iterable[int],
        previous_text: str,
        buffered_tokens: list,
        *,
        skip_special_tokens: bool,
    ) -> tuple[str, list[int], bool]:
        """
        Incrementally decode `new_tokens`.

        * If the new tokens alone produce a clean string → return that delta.
          The buffer is cleared.

        * If they contain “�” we prepend the existing `self.buffer` and try
          again.  If that still contains “�” we keep the whole sequence in
          `self.buffer` and return an empty delta.

        * If ``finished=True`` (handled by the caller) we always decode
          everything in `self.buffer + new_tokens` and then drop it.

        Returns:
            delta_text, buffered (the same list that was passed in), has_buffer
        """
        decoded_new = self._decode(new_tokens, skip_special_tokens=skip_special_tokens)

        if "�" not in decoded_new:
            delta = (
                decoded_new[len(previous_text) :]
                if previous_text and decoded_new.startswith(previous_text)
                else decoded_new
            )
            return delta, [], False

        candidate = list(buffered_tokens) + list(new_tokens)
        decoded_candidate = self._decode(candidate, skip_special_tokens=skip_special_tokens)

        if "�" not in decoded_candidate:
            delta = (
                decoded_candidate[len(previous_text) :]
                if previous_text and decoded_candidate.startswith(previous_text)
                else decoded_candidate
            )
            return delta, [], False

        return "", candidate, True


def _tokenizer_worker(endpoint: str, tokenizer_path: str, tokenizer_kwargs: dict) -> None:
    """Run the tokenizer worker process.

    This function starts a ZeroMQ server that handles tokenization requests.

    Args:
        endpoint: ZeroMQ endpoint to bind to.
        tokenizer_path: Path or identifier for the tokenizer.
        tokenizer_kwargs: Additional kwargs for loading the tokenizer.
    """
    if "trust_remote_code" not in tokenizer_kwargs.keys():
        tokenizer_kwargs["trust_remote_code"] = os.getenv("ESURGE_WORKER_TRUST_REMOTE_CODE", "1") in ["1", "on", "yes"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)
    try:
        while True:
            message = socket.recv_pyobj()
            cmd = message.get("cmd")
            if cmd == "tokenize":
                prompt = message["prompt"]
                encoded = tokenizer(prompt, return_tensors=None)
                token_ids = encoded["input_ids"]
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                socket.send_pyobj({"status": "ok", "tokens": list(token_ids)})
            elif cmd == "drain":
                socket.send_pyobj({"status": "ok"})
            elif cmd == "shutdown":
                socket.send_pyobj({"status": "ok"})
                break
            else:
                socket.send_pyobj({"status": "error", "message": f"Unknown cmd {cmd}"})
    finally:
        socket.close(0)
        ctx.term()


def _detokenizer_worker(
    endpoint: str,
    tokenizer_path: str,
    tokenizer_kwargs: dict,
    max_states: int,
) -> None:
    """Run the detokenizer worker process.

    This function starts a ZeroMQ server that handles detokenization requests
    with support for incremental decoding and state management.

    Args:
        endpoint: ZeroMQ endpoint to bind to.
        tokenizer_path: Path or identifier for the tokenizer.
    tokenizer_kwargs: Additional kwargs for loading the tokenizer.
    max_states: Maximum number of decoding states to maintain.
    """
    if "trust_remote_code" not in tokenizer_kwargs.keys():
        tokenizer_kwargs["trust_remote_code"] = os.getenv("ESURGE_WORKER_TRUST_REMOTE_CODE", "1") in ["1", "on", "yes"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    decoder = FastIncrementalDecoder(tokenizer)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)
    states: OrderedDict[str, dict] = OrderedDict()

    def evict():
        while len(states) > max_states:
            states.popitem(last=False)

    try:
        while True:
            message = socket.recv_pyobj()
            cmd = message.get("cmd")
            if cmd == "decode":
                rid = message["request_id"]
                generated_tokens = message["tokens"]
                finished = message["finished"]
                skip_special = message["skip_special_tokens"]

                state = states.setdefault(rid, {"last_index": 0, "previous_text": "", "buffered": []})
                last_idx = min(state["last_index"], len(generated_tokens))
                new_tokens = generated_tokens[last_idx:]
                detokstart = time.time()
                if new_tokens or finished:
                    delta, new_buffered, _ = decoder.decode(
                        new_tokens,
                        state["previous_text"],
                        state["buffered"],
                        skip_special_tokens=skip_special,
                    )
                    state["buffered"] = new_buffered
                    accumulated = state["previous_text"] + delta
                    state["previous_text"] = accumulated
                    state["last_index"] = len(generated_tokens)

                    if finished:
                        try:
                            full_decoded = tokenizer.decode(
                                generated_tokens,
                                skip_special_tokens=skip_special,
                                clean_up_tokenization_spaces=False,
                            )
                        except TypeError:
                            full_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=skip_special)
                        if full_decoded.startswith(accumulated):
                            delta = full_decoded[len(accumulated) :]
                            accumulated = full_decoded
                        else:
                            delta = full_decoded
                            accumulated = full_decoded
                        states.pop(rid, None)
                    result = {
                        "accumulated_text": accumulated,
                        "delta_text": delta,
                        "last_decoded_index": state["last_index"],
                        "finished": finished,
                    }
                else:
                    result = {
                        "accumulated_text": state["previous_text"],
                        "delta_text": "",
                        "last_decoded_index": state["last_index"],
                        "finished": finished,
                    }
                detoktook = time.time() - detokstart
                result["detoktook"] = detoktook
                evict()
                socket.send_pyobj({"status": "ok", "result": result})
            elif cmd == "drain":
                states.clear()
                socket.send_pyobj({"status": "ok"})
            elif cmd == "reset":
                rid = message["request_id"]
                states.pop(rid, None)
                socket.send_pyobj({"status": "ok"})
            elif cmd == "shutdown":
                socket.send_pyobj({"status": "ok"})
                break
            else:
                socket.send_pyobj({"status": "error", "message": f"Unknown cmd {cmd}"})
    finally:
        socket.close(0)
        ctx.term()


def main():
    """Main entry point for worker processes.

    Parses command-line arguments and starts either a tokenizer or detokenizer
    worker process based on the specified mode.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tokenizer", "detokenizer"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-kwargs", required=True)
    parser.add_argument("--max-states", type=int, default=1 << 16)
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

    tokenizer_kwargs = json.loads(args.tokenizer_kwargs) if args.tokenizer_kwargs else {}
    if args.mode == "tokenizer":
        _tokenizer_worker(args.endpoint, args.tokenizer_path, tokenizer_kwargs)
    else:
        _detokenizer_worker(args.endpoint, args.tokenizer_path, tokenizer_kwargs, args.max_states)


if __name__ == "__main__":
    main()

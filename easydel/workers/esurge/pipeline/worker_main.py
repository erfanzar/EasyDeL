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
            skip_special_tokens=False,
        )
    """

    def __init__(self, tokenizer, *, context_window: int = 4):
        self.tokenizer = tokenizer
        self.context_window = max(0, int(context_window))

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
        context_tokens: Iterable[int] | None = None,
    ) -> tuple[str, list[int], bool]:
        """
        Incrementally decode `new_tokens` with optional token context.

        * We decode ``context_tokens + buffered_tokens + new_tokens`` and
          subtract the decoded context so tokenizers that need preceding
          context (e.g., WordPiece) still stream correctly.
        * If the resulting delta contains the UTF-8 replacement character
          (“�”), we buffer the tokens and emit nothing to avoid corrupt output.

        Returns:
            delta_text, buffered (the same list that was passed in), has_buffer
        """
        del previous_text
        context = list(context_tokens) if context_tokens else []
        buffered = list(buffered_tokens)
        fresh = list(new_tokens)
        candidate = context + buffered + fresh

        if not candidate:
            return "", buffered, bool(buffered)

        decoded_candidate = self._decode(candidate, skip_special_tokens=skip_special_tokens)
        decoded_context = self._decode(context, skip_special_tokens=skip_special_tokens) if context else ""

        if decoded_context and not decoded_candidate.startswith(decoded_context):
            # Fall back to no-context decode if prefix alignment fails.
            decoded_context = ""
            decoded_candidate = self._decode(buffered + fresh, skip_special_tokens=skip_special_tokens)

        delta = decoded_candidate[len(decoded_context) :]
        if "�" in delta:
            return "", buffered + fresh, True

        return delta, [], False


def _compute_suffix_delta(current_text: str, previous_text: str) -> str:
    """Return the append-only delta from ``previous_text`` to ``current_text``.

    Falls back to suffix-prefix overlap when strict prefix alignment fails to
    avoid replaying already streamed text.
    """

    if not current_text:
        return ""
    if not previous_text:
        return current_text
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]

    max_overlap = min(len(previous_text), len(current_text))
    for overlap in range(max_overlap, 0, -1):
        if previous_text.endswith(current_text[:overlap]):
            return current_text[overlap:]
    return ""


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
    try:
        context_window = int(os.getenv("ESURGE_DETOKENIZER_CONTEXT_WINDOW", "4"))
    except ValueError:
        context_window = 4
    decoder = FastIncrementalDecoder(tokenizer, context_window=context_window)

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
                    buffered = state["buffered"]
                    emitted_index = max(0, last_idx - len(buffered))
                    if decoder.context_window:
                        context_start = max(0, emitted_index - decoder.context_window)
                        context_tokens = generated_tokens[context_start:emitted_index]
                    else:
                        context_tokens = []
                    delta, new_buffered, _ = decoder.decode(
                        new_tokens,
                        state["previous_text"],
                        buffered,
                        skip_special_tokens=skip_special,
                        context_tokens=context_tokens,
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
                        delta = _compute_suffix_delta(full_decoded, accumulated)
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

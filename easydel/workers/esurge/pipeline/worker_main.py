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

import zmq
from transformers import AutoTokenizer


class SimpleDecoder:
    """Simple incremental decoder with UTF-8 error handling.

    This decoder handles incremental token decoding and buffers tokens that
    may result in malformed UTF-8 sequences until they can be properly decoded.

    Args:
        tokenizer: The tokenizer instance to use for decoding.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _decode_tokens(self, tokens, *, skip_special_tokens: bool) -> str:
        try:
            return self.tokenizer.decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        except Exception:
            return ""

    def decode(self, tokens, previous_text, buffered_tokens, *, skip_special_tokens: bool):
        """Decode tokens incrementally, handling UTF-8 boundary issues.

        Args:
            tokens: New tokens to decode.
            previous_text: Previously decoded text.
            buffered_tokens: Tokens buffered from previous decode attempts.

        Returns:
            A tuple of (delta_text, new_buffered_tokens, has_buffer).
        """
        merged = list(buffered_tokens) + list(tokens)
        if not merged:
            return "", [], False

        decoded = self._decode_tokens(merged, skip_special_tokens=skip_special_tokens)

        if "�" not in decoded:
            delta = decoded[len(previous_text) :] if previous_text and decoded.startswith(previous_text) else decoded
            return delta, [], False

        # Backtrack by trimming tokens until no malformed chars.
        remaining = list(merged)
        new_buffer = []
        decoded_text = previous_text

        while remaining:
            new_buffer.insert(0, remaining.pop())
            candidate = self._decode_tokens(remaining, skip_special_tokens=skip_special_tokens)
            if not candidate:
                continue
            if "�" not in candidate:
                decoded_text = candidate
                break

        delta = (
            decoded_text[len(previous_text) :]
            if previous_text and decoded_text.startswith(previous_text)
            else decoded_text
        )
        return delta, new_buffer, True


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
    decoder = SimpleDecoder(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
    eos_set = {int(tid) for tid in eos_ids if tid is not None}

    def _strip_eos(tokens: list[int]) -> list[int]:
        if not eos_set:
            return list(tokens)
        return [tok for tok in tokens if tok not in eos_set]

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)
    states: dict[str, dict] = {}

    def _evict():
        while len(states) > max_states:
            rid = next(iter(states))
            states.pop(rid, None)

    try:
        while True:
            message = socket.recv_pyobj()
            cmd = message.get("cmd")
            if cmd == "decode":
                rid = message["request_id"]
                generated_tokens = message["tokens"]
                finished = message["finished"]
                skip_special = message["skip_special_tokens"]
                generated_tokens = _strip_eos(generated_tokens)

                state = states.setdefault(rid, {"last_index": 0, "previous_text": "", "buffered": []})
                last_idx = min(state["last_index"], len(generated_tokens))
                new_tokens = generated_tokens[last_idx:]

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

                _evict()
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

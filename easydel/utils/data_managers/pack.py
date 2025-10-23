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

"""Token packing utilities for efficient training.

This module provides functions for packing tokenized sequences into
fixed-length chunks, optimizing GPU/TPU utilization during training.
"""

from __future__ import annotations

import random

import jax.numpy as jnp
import numpy as np


def pack_pre_tokenized(stream, seq_length: int, eos_token_id: int, batch_size: int, shuffle: bool, buffer_factor: int):
    """Pack pre-tokenized sequences into constant-length chunks.

    Takes a stream of pre-tokenized examples and packs them into fixed-length
    sequences for efficient training. Sequences are concatenated and split
    at the specified sequence length, with EOS tokens inserted as needed.

    Args:
        stream: Iterator of dictionaries containing 'tokens' field.
        seq_length: Target length for packed sequences.
        eos_token_id: Token ID to use for padding/separation.
        batch_size: Batch size (used for shuffle buffer calculation).
        shuffle: Whether to shuffle the packed sequences.
        buffer_factor: Multiplier for shuffle buffer size (batch_size * buffer_factor).

    Returns:
        Generator yielding dictionaries with 'input_ids' as JAX arrays.

    Example:
        >>> stream = iter([{"tokens": [1, 2, 3]}, {"tokens": [4, 5, 6, 7]}])
        >>> gen = pack_pre_tokenized(stream, seq_length=4, eos_token_id=0,
        ...                         batch_size=2, shuffle=False, buffer_factor=8)
        >>> for packed in gen():
        ...     print(packed["input_ids"])
    """

    def gen():
        buf = np.array([], dtype=np.int32)
        eos = np.array([eos_token_id], dtype=np.int32)
        shuffle_buf = []
        max_buf = batch_size * buffer_factor

        for sample in stream:
            toks = sample["tokens"]
            toks = np.array(toks, dtype=np.int32) if not isinstance(toks, np.ndarray) else toks.astype(np.int32)
            buf = np.concatenate([buf, toks], axis=0)
            if len(buf) % seq_length != 0:
                buf = np.concatenate([buf, eos], axis=0)
            while len(buf) >= seq_length:
                ex = {"input_ids": jnp.array(buf[:seq_length])}
                buf = buf[seq_length:]
                if shuffle:
                    if len(shuffle_buf) < max_buf:
                        shuffle_buf.append(ex)
                    else:
                        i = random.randrange(0, max_buf)
                        yield shuffle_buf[i]
                        shuffle_buf[i] = ex
                else:
                    yield ex
        random.shuffle(shuffle_buf)
        for ex in shuffle_buf:
            yield ex

    return gen


def pack_constant_length(
    stream,
    tokenize_fn,
    seq_length: int,
    eos_token_id: int,
    batch_size: int,
    shuffle: bool,
    buffer_factor: int,
):
    """Pack sequences with on-the-fly tokenization into constant-length chunks.

    Combines tokenization and packing in a single pipeline. Takes raw examples,
    tokenizes them using the provided function, and packs the results into
    fixed-length sequences.

    Args:
        stream: Iterator of raw examples to tokenize.
        tokenize_fn: Function that takes an example and returns token IDs.
        seq_length: Target length for packed sequences.
        eos_token_id: Token ID to use for padding/separation.
        batch_size: Batch size (used for shuffle buffer calculation).
        shuffle: Whether to shuffle the packed sequences.
        buffer_factor: Multiplier for shuffle buffer size (batch_size * buffer_factor).

    Returns:
        Generator yielding dictionaries with 'input_ids' as JAX arrays.

    Example:
        >>> def tokenize(ex):
        ...     return tokenizer.encode(ex["text"])
        >>> stream = iter([{"text": "Hello"}, {"text": "World"}])
        >>> gen = pack_constant_length(
        ...     stream, tokenize, seq_length=128,
        ...     eos_token_id=0, batch_size=4,
        ...     shuffle=True, buffer_factor=16
        ... )
        >>> for packed in gen():
        ...     print(packed["input_ids"].shape)
    """

    def token_iter():
        for ex in stream:
            toks = tokenize_fn(ex)
            yield {"tokens": toks}

    return pack_pre_tokenized(token_iter(), seq_length, eos_token_id, batch_size, shuffle, buffer_factor)

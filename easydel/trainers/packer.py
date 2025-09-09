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

"""Sequence packing utilities for efficient training.

This module provides functionality to pack multiple sequences into fixed-length
batches, maximizing GPU/TPU utilization by reducing padding waste. It handles
attention masks and position IDs correctly for packed sequences.

Packing is especially beneficial when training on datasets with varying
sequence lengths, as it reduces the amount of wasted computation on padding
tokens.
"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from datasets import Dataset


def pack_sequences(
    dataset: Dataset,
    max_length: int = 512,
    pad_token_id: int = 0,
    reset_position_ids: bool = False,
    num_proc: int | None = None,
):
    """Pack multiple sequences into fixed-length batches for efficient training.

    Combines multiple variable-length sequences into fixed-size packed sequences,
    reducing padding waste and improving training efficiency. Correctly handles
    attention masks and position IDs for packed sequences

    # With continuous position IDs
    packed_dataset = pack_sequences(
        dataset,
        max_length=512,
        pad_token_id=0,
        reset_position_ids=False
    )

    # With reset position IDs for each sequence
    packed_dataset = pack_sequences(
        dataset,
        max_length=512,
        pad_token_id=0,
        reset_position_ids=True
    )

    # Example output format for a packed sequence with two sequences:
    # reset_position_ids=False:
    {
        'input_ids': [seq1_tokens + [PAD] + seq2_tokens + [PAD] + padding],
        'attention_mask': [1,1,1,0,1,1,1,0,0,0],
        'position_ids': [0,1,2,3,4,5,6,7,0,0]
    }

    # reset_position_ids=True:
    {
        'input_ids': [seq1_tokens + [PAD] + seq2_tokens + [PAD] + padding],
        'attention_mask': [1,1,1,0,1,1,1,0,0,0],
        'position_ids': [0,1,2,0,0,1,2,0,0,0]
    }

    Args:
        dataset: HuggingFace Dataset containing 'input_ids' and 'attention_mask' columns.
                Each example should have variable-length sequences to pack.
        max_length: Maximum length of each packed sequence (default 512).
                   Sequences are packed until this limit is reached.
        pad_token_id: Token ID used for padding and as separator between
                     packed sequences (default 0).
        reset_position_ids: If True, position IDs reset to 0 for each sequence
                           within a pack. If False, position IDs are continuous
                           across packed sequences (default False).
        num_proc: Number of processes to use for parallel processing.
                 None uses single process (default None).

    Returns:
        Dataset: New dataset with packed sequences containing:
                - 'input_ids': Packed token sequences
                - 'attention_mask': Attention masks (0 for padding/separators)
                - 'position_ids': Position embeddings for each token

    Raises:
        KeyError: If dataset doesn't contain required columns.

    Note:
        - Sequences are separated by pad_token_id with attention_mask=0
        - Remaining space in the last pack is filled with padding
        - Position IDs handle both continuous and reset modes correctly
        - Efficient for training when sequences have varying lengths
    """

    def pack_examples(examples):
        """Pack a batch of examples into fixed-length sequences.

        Args:
            examples: Dictionary with 'input_ids' and 'attention_mask' lists.

        Returns:
            dict: Packed sequences with input_ids, attention_mask, and position_ids.
        """
        current_packed_input_ids = []
        current_packed_attention_mask = []
        current_packed_position_ids = []
        current_length = 0

        packed_input_ids = []
        packed_attention_mask = []
        packed_position_ids = []

        def get_position_ids(length, start_position=0):
            """Generate position IDs for a sequence.

            Args:
                length: Length of the sequence.
                start_position: Starting position for continuous mode.

            Returns:
                list: Position IDs for the sequence.
            """
            if reset_position_ids:
                return list(range(length))
            else:
                return list(range(start_position, start_position + length))

        # Iterate through all examples in the batch
        for input_ids, attention_mask in zip(examples["input_ids"], examples["attention_mask"], strict=False):
            seq_length = len(input_ids)

            # If adding this sequence would exceed max_length, start a new packed sequence
            if current_length + seq_length + 1 > max_length:
                # Pad the current packed sequence if needed
                if current_length < max_length:
                    padding_length = max_length - current_length
                    current_packed_input_ids.extend([pad_token_id] * padding_length)
                    current_packed_attention_mask.extend([0] * padding_length)
                    current_packed_position_ids.extend([0] * padding_length)

                # Add the completed packed sequence to results
                packed_input_ids.append(current_packed_input_ids)
                packed_attention_mask.append(current_packed_attention_mask)
                packed_position_ids.append(current_packed_position_ids)

                # Start new packed sequence
                current_packed_input_ids = []
                current_packed_attention_mask = []
                current_packed_position_ids = []
                current_length = 0

            # Generate position IDs for current sequence
            position_ids = get_position_ids(seq_length, start_position=current_length)

            # Add current sequence
            current_packed_input_ids.extend(input_ids)
            current_packed_attention_mask.extend(attention_mask)
            current_packed_position_ids.extend(position_ids)

            # Add separator token
            current_packed_input_ids.append(pad_token_id)
            current_packed_attention_mask.append(0)
            current_packed_position_ids.append(position_ids[-1] + 1 if not reset_position_ids else 0)

            current_length += seq_length + 1

        # Handle the last packed sequence
        if current_packed_input_ids:
            # Pad if needed
            if current_length < max_length:
                padding_length = max_length - current_length
                current_packed_input_ids.extend([pad_token_id] * padding_length)
                current_packed_attention_mask.extend([0] * padding_length)
                current_packed_position_ids.extend([0] * padding_length)

            packed_input_ids.append(current_packed_input_ids)
            packed_attention_mask.append(current_packed_attention_mask)
            packed_position_ids.append(current_packed_position_ids)

        return {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "position_ids": packed_position_ids,
        }

    # Process the dataset in batches using HuggingFace map function
    packed_dataset = dataset.map(
        pack_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Packing sequences",
        num_proc=num_proc,
    )

    return packed_dataset

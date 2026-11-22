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
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]


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

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")

    def pack_examples(examples):
        """Pack a batch of examples into fixed-length sequences.

        Args:
            examples: Dictionary with 'input_ids' and 'attention_mask' lists.

        Returns:
            dict: Packed sequences with input_ids, attention_mask, and position_ids.
        """
        has_labels = "labels" in examples
        has_completion_mask = "completion_mask" in examples
        has_assistant_masks = "assistant_masks" in examples

        current_packed_input_ids: list[int] = []
        current_packed_attention_mask: list[int] = []
        current_packed_position_ids: list[int] = []
        current_packed_segment_ids: list[int] = []
        current_packed_labels: list[int] = []
        current_packed_completion_mask: list[int] = []
        current_packed_assistant_masks: list[int] = []
        current_length = 0
        current_segment_id = 0

        packed_input_ids: list[list[int]] = []
        packed_attention_mask: list[list[int]] = []
        packed_position_ids: list[list[int]] = []
        packed_segment_ids: list[list[int]] = []
        packed_labels: list[list[int]] = []
        packed_completion_mask: list[list[int]] = []
        packed_assistant_masks: list[list[int]] = []

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

        def flush():
            """Emit a nonempty pack padded to exactly the requested capacity."""
            nonlocal current_length, current_segment_id
            if not current_packed_input_ids:
                return
            padding_length = max_length - current_length
            packed_input_ids.append(current_packed_input_ids[:] + [pad_token_id] * padding_length)
            packed_attention_mask.append(current_packed_attention_mask[:] + [0] * padding_length)
            packed_position_ids.append(current_packed_position_ids[:] + [0] * padding_length)
            packed_segment_ids.append(current_packed_segment_ids[:] + [-1] * padding_length)
            packed_labels.append(current_packed_labels[:] + [-100] * padding_length)
            if has_completion_mask:
                packed_completion_mask.append(current_packed_completion_mask[:] + [0] * padding_length)
            if has_assistant_masks:
                packed_assistant_masks.append(current_packed_assistant_masks[:] + [0] * padding_length)
            current_packed_input_ids.clear()
            current_packed_attention_mask.clear()
            current_packed_position_ids.clear()
            current_packed_segment_ids.clear()
            current_packed_labels.clear()
            current_packed_completion_mask.clear()
            current_packed_assistant_masks.clear()
            current_length = 0
            current_segment_id = 0

        labels_col = examples["labels"] if has_labels else [None] * len(examples["input_ids"])
        cm_col = examples["completion_mask"] if has_completion_mask else [None] * len(examples["input_ids"])
        am_col = examples["assistant_masks"] if has_assistant_masks else [None] * len(examples["input_ids"])

        for input_ids, attention_mask, labels, completion_mask, assistant_masks in zip(
            examples["input_ids"], examples["attention_mask"], labels_col, cm_col, am_col, strict=True
        ):
            if len(input_ids) != len(attention_mask):
                raise ValueError("input_ids and attention_mask must have the same length.")
            if labels is not None and len(labels) != len(input_ids):
                raise ValueError("input_ids and labels must have the same length.")
            if completion_mask is not None and len(completion_mask) != len(input_ids):
                raise ValueError("input_ids and completion_mask must have the same length.")
            if assistant_masks is not None and len(assistant_masks) != len(input_ids):
                raise ValueError("input_ids and assistant_masks must have the same length.")
            if not input_ids:
                continue
            # Keep short rows intact. Long rows stream through full-size chunks;
            # a separator is needed only when a row leaves room in its pack.
            if current_length and current_length + len(input_ids) > max_length:
                flush()
            for offset in range(0, len(input_ids), max_length):
                chunk_ids = list(input_ids[offset : offset + max_length])
                chunk_mask = list(attention_mask[offset : offset + max_length])
                if labels is not None:
                    chunk_labels = [
                        int(lbl) if int(active) else -100
                        for lbl, active in zip(labels[offset : offset + max_length], chunk_mask, strict=True)
                    ]
                else:
                    chunk_labels = [
                        int(tok) if int(active) else -100
                        for tok, active in zip(chunk_ids, chunk_mask, strict=True)
                    ]
                # Mask the first token of a non-leading packed document and any active token
                # preceded by PAD so PAD tokens never predict across document or padding boundaries
                # in shifted CE loss.
                after_separator = current_length > 0
                mask_after_pad = [
                    (i == 0 and after_separator) or (i > 0 and not int(chunk_mask[i - 1]))
                    for i in range(len(chunk_mask))
                ]
                for i, should_mask in enumerate(mask_after_pad):
                    if should_mask:
                        chunk_labels[i] = -100
                chunk_segments = [current_segment_id if int(active) else -1 for active in chunk_mask]
                position_ids = get_position_ids(len(chunk_ids), start_position=current_length)

                chunk_cm: list[int] | None = None
                if has_completion_mask:
                    chunk_cm = [
                        int(val) if int(active) else 0
                        for val, active in zip(completion_mask[offset : offset + max_length], chunk_mask, strict=True)
                    ]
                    for i, should_mask in enumerate(mask_after_pad):
                        if should_mask:
                            chunk_cm[i] = 0
                chunk_am: list[int] | None = None
                if has_assistant_masks:
                    chunk_am = [
                        int(val) if int(active) else 0
                        for val, active in zip(assistant_masks[offset : offset + max_length], chunk_mask, strict=True)
                    ]
                    for i, should_mask in enumerate(mask_after_pad):
                        if should_mask:
                            chunk_am[i] = 0
                if labels is None and (chunk_cm is not None or chunk_am is not None):
                    if chunk_cm is not None and (chunk_am is None or any(chunk_cm)):
                        effective_mask = chunk_cm
                    else:
                        effective_mask = chunk_am
                    assert effective_mask is not None
                    for i, keep in enumerate(effective_mask):
                        if not keep:
                            chunk_labels[i] = -100

                current_packed_input_ids.extend(chunk_ids)
                current_packed_attention_mask.extend(chunk_mask)
                current_packed_position_ids.extend(position_ids)
                current_packed_segment_ids.extend(chunk_segments)
                current_packed_labels.extend(chunk_labels)
                if chunk_cm is not None:
                    current_packed_completion_mask.extend(chunk_cm)
                if chunk_am is not None:
                    current_packed_assistant_masks.extend(chunk_am)

                current_length += len(chunk_ids)
                current_segment_id += 1
                if current_length == max_length:
                    flush()
                else:
                    current_packed_input_ids.append(pad_token_id)
                    current_packed_attention_mask.append(0)
                    current_packed_position_ids.append(position_ids[-1] + 1 if not reset_position_ids else 0)
                    current_packed_segment_ids.append(-1)
                    current_packed_labels.append(-100)
                    if has_completion_mask:
                        current_packed_completion_mask.append(0)
                    if has_assistant_masks:
                        current_packed_assistant_masks.append(0)
                    current_length += 1
                    if current_length == max_length:
                        flush()

        if current_packed_input_ids:
            flush()

        result = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "position_ids": packed_position_ids,
            "segment_ids": packed_segment_ids,
            "labels": packed_labels,
        }
        if has_completion_mask:
            result["completion_mask"] = packed_completion_mask
        if has_assistant_masks:
            result["assistant_masks"] = packed_assistant_masks
        return result

    packed_dataset = dataset.map(
        pack_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Packing sequences",
        num_proc=num_proc,
    )

    return packed_dataset

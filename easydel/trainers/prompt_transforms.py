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

"""Base transform classes for trainer preprocessing.

This module re-exports the base Transform class from data_managers
and provides trainer-specific utilities.
"""

from __future__ import annotations

import json
import typing as tp
from collections.abc import Iterator

# Re-export base classes from data_managers
from easydel.data.transforms.base import Example, ExpandTransform, Transform

from .prompt_utils import (
    maybe_apply_chat_template,
    maybe_convert_to_chatml,
    maybe_extract_prompt,
)

_TOKENIZED_FIELDS = {
    "input_ids",
    "attention_mask",
    "labels",
    "position_ids",
    "completion_mask",
    "assistant_masks",
    "prompt_input_ids",
    "prompt_attention_mask",
    "chosen_input_ids",
    "chosen_attention_mask",
    "rejected_input_ids",
    "rejected_attention_mask",
    "chosen_labels",
    "rejected_labels",
    "completion_input_ids",
    "completion_attention_mask",
    "completion_labels",
    "input_ids_chosen",
    "input_ids_rejected",
    "attention_mask_chosen",
    "attention_mask_rejected",
    "label",
    "embedding_input_ids",
    "embedding_attention_mask",
}


def purify_example(example: dict, keep_fields: set[str] | None = None) -> dict:
    """Remove non-tokenized fields from example, keeping only JAX-compatible arrays.

    This function filters out text fields (messages, prompt, chosen, rejected, etc.)
    and keeps only tokenized fields that can be passed to JAX functions.

    Args:
        example: Example dict that may contain both text and tokenized fields.
        keep_fields: Optional set of additional field names to keep.

    Returns:
        Purified example with only JAX-compatible fields.
    """
    fields_to_keep = _TOKENIZED_FIELDS.copy()
    if keep_fields:
        fields_to_keep.update(keep_fields)

    return {k: v for k, v in example.items() if k in fields_to_keep}


def is_conversational(example: dict) -> bool:
    """Check if example has conversational format (messages/conversations field).

    Args:
        example: Dataset example to check.

    Returns:
        True if example contains messages or prompt list format.
    """
    return ("messages" in example and isinstance(example["messages"], list)) or (
        "prompt" in example and isinstance(example["prompt"], list)
    )


def is_conversational_from_value(example: dict) -> bool:
    """Check if example has from/value conversation format.

    This is an alternative format where conversations use 'from' and 'value'
    keys instead of 'role' and 'content'.

    Args:
        example: Dataset example to check.

    Returns:
        True if example has conversations in from/value format.
    """
    conversations_key = None
    if "conversations" in example:
        conversations_key = "conversations"
    elif "conversation" in example:
        conversations_key = "conversation"

    if conversations_key is None:
        return False

    conversations = example[conversations_key]
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False

    first_turn = conversations[0]
    return isinstance(first_turn, dict) and "from" in first_turn and "value" in first_turn


def convert_to_chatml(example: dict) -> dict:
    """Convert from/value format to standard role/content ChatML format.

    Args:
        example: Example with 'conversations' or 'conversation' field
                using from/value format.

    Returns:
        Example with 'messages' field in role/content format.

    Raises:
        KeyError: If neither 'conversations' nor 'conversation' field exists.
    """
    if "conversations" in example:
        conversations_key = "conversations"
    elif "conversation" in example:
        conversations_key = "conversation"
    else:
        raise KeyError("Example must have 'conversations' or 'conversation' field")
    conversations = example[conversations_key]

    role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }

    messages = []
    for turn in conversations:
        role = turn.get("from", "user")
        content = turn.get("value", "")
        messages.append(
            {
                "role": role_mapping.get(role, role),
                "content": content,
            }
        )

    result = dict(example)
    result["messages"] = messages
    # Remove old format keys
    result.pop("conversations", None)
    result.pop("conversation", None)
    return result


def extract_prompt_from_preference(example: dict) -> dict:
    """Extract shared prompt from chosen/rejected if not present.

    For preference datasets where chosen and rejected share a common prefix,
    this extracts that prefix as the prompt. Uses the same logic as TRL's
    maybe_extract_prompt from prompt_utils.

    Args:
        example: Example with 'chosen' and 'rejected' fields.

    Returns:
        Example with 'prompt' field extracted if applicable.
    """
    result = maybe_extract_prompt(example)
    if "prompt" in result:
        return result
    # Some RL datasets store the prompt as a single-turn chat under `messages`.
    # GRPO/PPO expect a prompt field (string or list-of-messages), so normalize it.
    messages = result.get("messages")
    if isinstance(messages, list):
        out = dict(result)
        out["prompt"] = messages
        return out
    return result


def apply_chat_template_to_preference(
    example: dict,
    tokenizer: tp.Any,
    tools: list | None = None,
) -> dict:
    """Apply chat template to prompt/chosen/rejected for DPO.

    Args:
        example: Example with prompt, chosen, rejected fields.
        tokenizer: Tokenizer with apply_chat_template method.
        tools: Optional tools for function calling.

    Returns:
        Example with text formatted using chat template.

    Raises:
        KeyError: If required fields (prompt, chosen, rejected) are missing.
    """
    result = dict(example)

    # Strict field access - will raise KeyError if missing
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Handle message list format
    if isinstance(prompt, list):
        result["prompt"] = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )

        if isinstance(chosen, list):
            result["chosen"] = tokenizer.apply_chat_template(
                chosen,
                tokenize=False,
                add_generation_prompt=False,
            )
        if isinstance(rejected, list):
            result["rejected"] = tokenizer.apply_chat_template(
                rejected,
                tokenize=False,
                add_generation_prompt=False,
            )

    return result


class GRPOPreprocessTransform(Transform):
    """Preprocessing transform for Group Relative Policy Optimization.

    GRPO only needs tokenized prompts since completions are generated online.
    Uses left-padding for efficient batch generation.

    Produces:
        - input_ids: Left-padded prompt token IDs
        - attention_mask: Attention mask for prompt

    Args:
        tokenizer: Tokenizer for text encoding.
        max_prompt_length: Maximum tokens for prompt.
        tools: Optional tools for function calling.
        skip_apply_chat_template: If True, skip chat template application.

    Example:
        >>> transform = GRPOPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_prompt_length=512,
        ... )
        >>> result = transform({
        ...     "prompt": "Question: What is 2+2?",
        ... })
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_prompt_length: int | None = None,
        tools: list | None = None,
        skip_apply_chat_template: bool = False,
    ):
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
        self._tools = tools
        self._skip_apply_chat_template = skip_apply_chat_template
        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __call__(self, example: Example) -> Example:
        """Apply GRPO preprocessing to example.

        Args:
            example: Input example with prompt field or chosen/rejected fields.

        Returns:
            Preprocessed example with input_ids and attention_mask.
        """
        # Skip if already tokenized
        if "input_ids" in example:
            return example

        # Convert from/value format to role/content if needed (ShareGPT → ChatML)
        example = maybe_convert_to_chatml(example)

        # Extract prompt from preference data if needed (chosen/rejected → prompt)
        result = extract_prompt_from_preference(example)

        # Strict field access - will raise KeyError if prompt is missing
        prompt = result["prompt"]

        # Apply chat template if conversational format
        if isinstance(prompt, list) and not self._skip_apply_chat_template:
            prompt = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                tools=self._tools,
            )

        # Tokenize with left padding for generation
        tokenized = self._tokenizer(
            prompt,
            padding="max_length" if self._max_prompt_length else False,
            max_length=self._max_prompt_length,
            truncation=self._max_prompt_length is not None,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Apply left padding manually if tokenizer doesn't support it
        if self._max_prompt_length and len(input_ids) < self._max_prompt_length:
            pad_len = self._max_prompt_length - len(input_ids)
            input_ids = [self._pad_token_id] * pad_len + input_ids
            attention_mask = [0] * pad_len + attention_mask

        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        # Remove non-tokenized fields
        return purify_example(result)

    def __repr__(self) -> str:
        return f"GRPOPreprocessTransform(max_prompt={self._max_prompt_length})"


class PPOPreprocessTransform(GRPOPreprocessTransform):
    """Preprocessing transform for PPO training.

    PPO uses the same prompt-only preprocessing as GRPO: prompts are tokenized and
    left-padded; completions are generated online.
    """

    def __repr__(self) -> str:
        return f"PPOPreprocessTransform(max_prompt={self._max_prompt_length})"


class KTOPreprocessTransform(Transform):
    """Preprocessing transform for Kahneman-Tversky Optimization.

    KTO works with unpaired preference data (prompt + completion + label).

    Produces:
        - prompt_input_ids, prompt_attention_mask: Just the prompt
        - completion_input_ids, completion_attention_mask, completion_labels: Full sequence
        - label: Boolean indicating desirable (True) or undesirable (False)

    Labels have prompt tokens masked with label_pad_token_id (-100).

    Args:
        tokenizer: Tokenizer for text encoding.
        max_prompt_length: Maximum tokens for prompt.
        max_completion_length: Maximum tokens for completion.
        add_special_tokens: Whether to add BOS/EOS tokens.
        label_pad_token_id: Token ID for masking prompt tokens in labels.
        embedding_tokenizer: Optional tokenizer for BCO UDM embeddings.

    Example:
        >>> transform = KTOPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ... )
        >>> result = transform({
        ...     "prompt": "Question?",
        ...     "completion": "Answer",
        ...     "label": True,  # Good completion
        ... })
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = False,
        label_pad_token_id: int = -100,
        embedding_tokenizer: tp.Any = None,
    ):
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
        self._max_completion_length = max_completion_length
        self._add_special_tokens = add_special_tokens
        self._label_pad_token_id = label_pad_token_id
        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self._embedding_tokenizer = embedding_tokenizer

    def __call__(self, example: Example) -> Example:
        """Apply KTO preprocessing to example.

        Args:
            example: Input example with prompt, completion, label.

        Returns:
            Preprocessed example with all tokenized fields.

        Raises:
            KeyError: If required fields (prompt, completion, label) are missing.
        """
        # Skip if already tokenized
        if "prompt_input_ids" in example:
            return example

        # Convert from/value format to role/content if needed (ShareGPT → ChatML)
        example = maybe_convert_to_chatml(example)

        result = dict(example)

        # Strict field access - will raise KeyError if missing
        prompt = example["prompt"]
        completion = example["completion"]
        label = example["label"]

        # Handle conversational format
        if isinstance(prompt, list):
            prompt = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize prompt and completion separately
        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self._tokenizer(completion, add_special_tokens=False)["input_ids"]

        # Add BOS to prompt if requested
        bos_token_id = self._tokenizer.bos_token_id
        eos_token_id = self._tokenizer.eos_token_id

        if self._add_special_tokens and bos_token_id is not None:
            prompt_ids = [bos_token_id, *prompt_ids]

        # Add EOS to completion
        if eos_token_id is not None:
            completion_ids = [*completion_ids, eos_token_id]

        # Truncate prompt (from left, keep most recent context)
        if self._max_prompt_length is not None:
            prompt_ids = prompt_ids[-self._max_prompt_length :]

        # Truncate completion (from right)
        if self._max_completion_length is not None:
            completion_ids = completion_ids[: self._max_completion_length]

        # Build full sequence: prompt + completion
        full_ids = prompt_ids + completion_ids
        prompt_len = len(prompt_ids)

        # Create labels: mask prompt tokens with label_pad_token_id
        full_labels = [self._label_pad_token_id] * prompt_len + completion_ids

        # Create attention masks (all 1s for actual tokens)
        prompt_mask = [1] * len(prompt_ids)
        full_mask = [1] * len(full_ids)

        # Set outputs
        result["prompt_input_ids"] = prompt_ids
        result["prompt_attention_mask"] = prompt_mask
        result["completion_input_ids"] = full_ids
        result["completion_attention_mask"] = full_mask
        result["completion_labels"] = full_labels
        result["label"] = bool(label)

        # Add embedding tokens for BCO UDM if embedding tokenizer is provided
        if self._embedding_tokenizer is not None:
            emb_tokenized = self._embedding_tokenizer(
                prompt,
                truncation=True,
                add_special_tokens=False,
            )
            result["embedding_input_ids"] = emb_tokenized["input_ids"]
            result["embedding_attention_mask"] = emb_tokenized["attention_mask"]

        # Remove non-tokenized fields
        return purify_example(result)

    def __repr__(self) -> str:
        return (
            f"KTOPreprocessTransform(max_prompt={self._max_prompt_length}, max_completion={self._max_completion_length})"
        )


class BCOPreprocessTransform(ExpandTransform):
    """Preprocessing transform for Binary Classifier Optimization.

    BCO works with unpaired preference data (prompt + completion + label).
    This transform handles the full preprocessing pipeline per-example:
    1. Extract shared prompt from chosen/rejected conversations
    2. Apply chat template to convert conversations to text
    3. Unpair: yield TWO examples from one paired input (chosen + rejected)
    4. Tokenize prompt and completion with proper masking

    As an ExpandTransform, this yields multiple examples from each input:
    - For paired data (chosen/rejected): yields 2 examples
    - For unpaired data (prompt/completion/label): yields 1 example

    Produces per example:
        - prompt_input_ids, prompt_attention_mask: Just the prompt
        - completion_input_ids, completion_attention_mask, completion_labels: Full sequence
        - label: Boolean indicating desirable (True) or undesirable (False)

    Args:
        tokenizer: Tokenizer for text encoding.
        max_prompt_length: Maximum tokens for prompt.
        max_completion_length: Maximum tokens for completion.
        add_special_tokens: Whether to add BOS/EOS tokens.
        label_pad_token_id: Token ID for masking prompt tokens in labels.
        embedding_tokenizer: Optional tokenizer for BCO UDM embeddings.
        tools: Optional tools for function calling in chat template.

    Example:
        >>> transform = BCOPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ... )
        >>> # Paired input yields 2 examples
        >>> for result in transform({"chosen": [...], "rejected": [...]}):
        ...     print(result["label"])  # True, then False
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = False,
        label_pad_token_id: int = -100,
        embedding_tokenizer: tp.Any = None,
        tools: list | None = None,
    ):
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
        self._max_completion_length = max_completion_length
        self._add_special_tokens = add_special_tokens
        self._label_pad_token_id = label_pad_token_id
        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self._embedding_tokenizer = embedding_tokenizer
        self._tools = tools

    def __call__(self, example: Example) -> Iterator[Example]:
        """Process example and yield one or more tokenized examples.

        For paired data (chosen/rejected): yields 2 examples.
        For unpaired data (prompt/completion/label): yields 1 example.

        Args:
            example: Input example with either:
                - chosen/rejected fields (paired preference data)
                - prompt/completion/label fields (unpaired data)

        Yields:
            Tokenized examples with prompt_input_ids, completion_input_ids, label, etc.

        Raises:
            KeyError: If required fields are missing from the example.
        """
        # Skip if already tokenized
        if "prompt_input_ids" in example:
            yield example
            return

        # Check if already unpaired (has prompt/completion/label)
        if "completion" in example and "label" in example:
            # Already unpaired - just tokenize and yield single example
            yield self._tokenize_unpaired(example)
            return

        # Paired data: extract prompt, apply chat template, yield 2 examples

        # Step 1: Convert from/value format to role/content if needed (ShareGPT → ChatML)
        example = maybe_convert_to_chatml(example)

        # Step 2: Extract prompt from chosen/rejected if needed
        example = extract_prompt_from_preference(example)

        # Step 3: Apply chat template if conversational (uses maybe_apply_chat_template)
        example = maybe_apply_chat_template(example, self._tokenizer, self._tools)

        # Step 4: Get fields with strict access - missing fields will raise KeyError
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Step 5: Yield TWO tokenized examples (unpair operation)
        # Chosen example (label=True)
        yield self._tokenize_unpaired(
            {
                "prompt": prompt,
                "completion": chosen,
                "label": True,
            }
        )

        # Rejected example (label=False)
        yield self._tokenize_unpaired(
            {
                "prompt": prompt,
                "completion": rejected,
                "label": False,
            }
        )

    def _tokenize_unpaired(self, example: dict) -> dict:
        """Tokenize a single unpaired example.

        Args:
            example: Dict with prompt, completion, label fields.

        Returns:
            Tokenized example with prompt_input_ids, completion_input_ids, etc.

        Raises:
            KeyError: If required fields (prompt, completion, label) are missing.
        """
        result = dict(example)

        # Strict field access - will raise KeyError if missing
        prompt = example["prompt"]
        completion = example["completion"]
        label = example["label"]

        # Handle conversational format for prompt (if not already converted)
        if isinstance(prompt, list):
            prompt = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize prompt and completion separately
        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self._tokenizer(completion, add_special_tokens=False)["input_ids"]

        # Add BOS to prompt if requested
        bos_token_id = self._tokenizer.bos_token_id
        eos_token_id = self._tokenizer.eos_token_id

        if self._add_special_tokens and bos_token_id is not None:
            prompt_ids = [bos_token_id, *prompt_ids]

        # Add EOS to completion
        if eos_token_id is not None:
            completion_ids = [*completion_ids, eos_token_id]

        # Truncate prompt (from left, keep most recent context)
        if self._max_prompt_length is not None:
            prompt_ids = prompt_ids[-self._max_prompt_length :]

        # Truncate completion (from right)
        if self._max_completion_length is not None:
            completion_ids = completion_ids[: self._max_completion_length]

        # Build full sequence: prompt + completion
        full_ids = prompt_ids + completion_ids
        prompt_len = len(prompt_ids)

        # Create labels: mask prompt tokens with label_pad_token_id
        full_labels = [self._label_pad_token_id] * prompt_len + completion_ids

        # Create attention masks (all 1s for actual tokens)
        prompt_mask = [1] * len(prompt_ids)
        full_mask = [1] * len(full_ids)

        # Set outputs
        result["prompt_input_ids"] = prompt_ids
        result["prompt_attention_mask"] = prompt_mask
        result["completion_input_ids"] = full_ids
        result["completion_attention_mask"] = full_mask
        result["completion_labels"] = full_labels
        result["label"] = bool(label)

        # Add embedding tokens for BCO UDM if embedding tokenizer is provided
        if self._embedding_tokenizer is not None:
            emb_tokenized = self._embedding_tokenizer(
                prompt,
                truncation=True,
                add_special_tokens=False,
            )
            result["embedding_input_ids"] = emb_tokenized["input_ids"]
            result["embedding_attention_mask"] = emb_tokenized["attention_mask"]

        # Remove non-tokenized fields
        return purify_example(result)

    def __repr__(self) -> str:
        return (
            f"BCOPreprocessTransform(max_prompt={self._max_prompt_length}, max_completion={self._max_completion_length})"
        )


class DPOPreprocessTransform(Transform):
    """Preprocessing transform for Direct Preference Optimization.

    Handles the full DPO preprocessing pipeline:
    1. Extract shared prompt from chosen/rejected if needed
    2. Apply chat template to conversational data
    3. Tokenize prompt, chosen, and rejected sequences

    Supports pre-tokenized data - if prompt_input_ids exists, passes through.

    Args:
        tokenizer: Tokenizer for text encoding.
        max_prompt_length: Maximum tokens for prompt.
        max_completion_length: Maximum tokens for chosen/rejected.
        add_special_tokens: Whether to add BOS/EOS tokens.
        tools: Optional tools for function calling.

    Example:
        >>> transform = DPOPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ... )
        >>> result = transform({
        ...     "prompt": "What is 2+2?",
        ...     "chosen": "4",
        ...     "rejected": "5",
        ... })
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = False,
        tools: list | None = None,
        label_pad_token_id: int = -100,
    ):
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
        self._max_completion_length = max_completion_length
        self._add_special_tokens = add_special_tokens
        self._tools = tools
        self._label_pad_token_id = label_pad_token_id
        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __call__(self, example: Example) -> Example:
        """Apply DPO preprocessing to example.

        Args:
            example: Input example with prompt, chosen, rejected.

        Returns:
            Preprocessed example with prompt_input_ids, chosen_input_ids,
            rejected_input_ids.
        """
        # Skip if already tokenized
        if "prompt_input_ids" in example:
            return example

        # Step 1: Convert from/value format to role/content if needed (ShareGPT → ChatML)
        example = maybe_convert_to_chatml(example)

        # Step 2: Extract prompt if needed
        result = extract_prompt_from_preference(example)

        # Step 3: Apply chat template if conversational
        if isinstance(result["prompt"], list):
            result = apply_chat_template_to_preference(
                result,
                self._tokenizer,
                self._tools,
            )

        # Step 4: Tokenize
        return self._tokenize(result)

    def _tokenize(self, example: dict) -> dict:
        """Tokenize prompt, chosen, and rejected sequences.

        Produces:
            - prompt_input_ids, prompt_attention_mask: Just the prompt
            - chosen_input_ids, chosen_attention_mask, chosen_labels: Full chosen sequence
            - rejected_input_ids, rejected_attention_mask, rejected_labels: Full rejected sequence

        Labels have prompt tokens masked with label_pad_token_id (-100).

        Raises:
            KeyError: If required fields (prompt, chosen, rejected) are missing.
        """
        result = dict(example)

        # Strict field access - will raise KeyError if missing
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Tokenize each part separately
        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_completion_ids = self._tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_completion_ids = self._tokenizer(rejected, add_special_tokens=False)["input_ids"]

        # Add BOS to prompt if requested
        if self._add_special_tokens and self._tokenizer.bos_token_id is not None:
            prompt_ids = [self._tokenizer.bos_token_id, *prompt_ids]

        # Add EOS to completions
        if self._tokenizer.eos_token_id is not None:
            chosen_completion_ids = [*chosen_completion_ids, self._tokenizer.eos_token_id]
            rejected_completion_ids = [*rejected_completion_ids, self._tokenizer.eos_token_id]

        # Truncate prompt (from left, keep most recent context)
        if self._max_prompt_length is not None:
            prompt_ids = prompt_ids[-self._max_prompt_length :]

        # Truncate completions (from right)
        if self._max_completion_length is not None:
            chosen_completion_ids = chosen_completion_ids[: self._max_completion_length]
            rejected_completion_ids = rejected_completion_ids[: self._max_completion_length]

        # Build full sequences: prompt + completion
        chosen_full_ids = prompt_ids + chosen_completion_ids
        rejected_full_ids = prompt_ids + rejected_completion_ids

        # Create labels: mask prompt tokens with label_pad_token_id
        prompt_len = len(prompt_ids)
        chosen_labels = [self._label_pad_token_id] * prompt_len + chosen_completion_ids
        rejected_labels = [self._label_pad_token_id] * prompt_len + rejected_completion_ids

        # Pad to max lengths
        max_seq_length = (self._max_prompt_length or 0) + (self._max_completion_length or 0)

        def _pad_sequence(seq, max_len, pad_value):
            if max_len and len(seq) < max_len:
                return seq + [pad_value] * (max_len - len(seq))
            return seq

        # Pad prompt
        prompt_max = self._max_prompt_length
        if prompt_max:
            # Left-pad prompt
            if len(prompt_ids) < prompt_max:
                pad_len = prompt_max - len(prompt_ids)
                prompt_ids = [self._pad_token_id] * pad_len + prompt_ids
                prompt_attention_mask = [0] * pad_len + [1] * (prompt_max - pad_len)
            else:
                prompt_attention_mask = [1] * len(prompt_ids)
        else:
            prompt_attention_mask = [1] * len(prompt_ids)

        # Pad full sequences (right-pad)
        if max_seq_length:
            chosen_attention_mask = [1] * len(chosen_full_ids)
            rejected_attention_mask = [1] * len(rejected_full_ids)

            chosen_full_ids = _pad_sequence(chosen_full_ids, max_seq_length, self._pad_token_id)
            rejected_full_ids = _pad_sequence(rejected_full_ids, max_seq_length, self._pad_token_id)
            chosen_labels = _pad_sequence(chosen_labels, max_seq_length, self._label_pad_token_id)
            rejected_labels = _pad_sequence(rejected_labels, max_seq_length, self._label_pad_token_id)
            chosen_attention_mask = _pad_sequence(chosen_attention_mask, max_seq_length, 0)
            rejected_attention_mask = _pad_sequence(rejected_attention_mask, max_seq_length, 0)
        else:
            chosen_attention_mask = [1] * len(chosen_full_ids)
            rejected_attention_mask = [1] * len(rejected_full_ids)

        # Set all outputs
        result["prompt_input_ids"] = prompt_ids
        result["prompt_attention_mask"] = prompt_attention_mask
        result["chosen_input_ids"] = chosen_full_ids
        result["chosen_attention_mask"] = chosen_attention_mask
        result["chosen_labels"] = chosen_labels
        result["rejected_input_ids"] = rejected_full_ids
        result["rejected_attention_mask"] = rejected_attention_mask
        result["rejected_labels"] = rejected_labels

        # Remove non-tokenized fields
        return purify_example(result)

    def __repr__(self) -> str:
        return (
            f"DPOPreprocessTransform(max_prompt={self._max_prompt_length}, max_completion={self._max_completion_length})"
        )


class ORPOPreprocessTransform(DPOPreprocessTransform):
    """Preprocessing transform for Odds Ratio Preference Optimization.

    Uses the same preprocessing as DPO since the input format is identical.

    Example:
        >>> transform = ORPOPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ... )
    """

    def __repr__(self) -> str:
        return (
            f"ORPOPreprocessTransform(max_prompt={self._max_prompt_length}, "
            f"max_completion={self._max_completion_length})"
        )


# CPO uses the same preprocessing as DPO
CPOPreprocessTransform = DPOPreprocessTransform


class RewardPreprocessTransform(Transform):
    """Preprocessing transform for Reward Model training.

    Similar to DPO but focused on scoring completions.

    Args:
        tokenizer: Tokenizer for text encoding.
        max_length: Maximum sequence length.
        truncation: Whether to truncate sequences.

    Example:
        >>> transform = RewardPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_length=512,
        ... )
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_length: int,
        truncation: bool = True,
    ):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._truncation = truncation

    def __call__(self, example: Example) -> Example:
        """Apply reward model preprocessing.

        Raises:
            KeyError: If required fields (chosen, rejected) are missing.
        """
        # Skip if already tokenized
        if "input_ids_chosen" in example:
            return example

        # Convert from/value format to role/content if needed (ShareGPT → ChatML)
        example = maybe_convert_to_chatml(example)

        result = dict(example)

        # Strict field access - will raise KeyError if missing
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Handle conversational format
        if isinstance(chosen, list):
            chosen = self._tokenizer.apply_chat_template(chosen, tokenize=False)
        if isinstance(rejected, list):
            rejected = self._tokenizer.apply_chat_template(rejected, tokenize=False)

        # Tokenize both
        chosen_tokens = self._tokenizer(
            chosen,
            truncation=self._truncation,
            max_length=self._max_length,
            return_attention_mask=True,
        )
        rejected_tokens = self._tokenizer(
            rejected,
            truncation=self._truncation,
            max_length=self._max_length,
            return_attention_mask=True,
        )

        result["input_ids_chosen"] = chosen_tokens["input_ids"]
        result["attention_mask_chosen"] = chosen_tokens["attention_mask"]
        result["input_ids_rejected"] = rejected_tokens["input_ids"]
        result["attention_mask_rejected"] = rejected_tokens["attention_mask"]

        # Remove non-tokenized fields
        return purify_example(result)

    def __repr__(self) -> str:
        return f"RewardPreprocessTransform(max_length={self._max_length})"


class SFTPreprocessTransform(Transform):
    """Preprocessing transform for Supervised Fine-Tuning.

    Handles the full SFT preprocessing pipeline:
    1. Apply formatting function if provided
    2. Format detection (conversational vs text)
    3. Convert from/value format to ChatML if needed
    4. Apply chat template for conversational data
    5. Tokenization with optional completion masking

    Supports pre-tokenized data - if input_ids already exists, passes through.

    Args:
        tokenizer: Tokenizer for text encoding.
        max_length: Maximum sequence length.
        text_field: Field name for text data (default: "text").
        messages_field: Field name for messages (default: "messages").
        mask_prompt: Whether to create completion mask for completion-only loss.
        add_eos: Whether to add EOS token to text.
        truncation: Whether to truncate to max_length.
        formatting_func: Optional function to format examples before tokenization.
            Should take an example dict and return a string or dict with "text" field.

    Example:
        >>> transform = SFTPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_length=2048,
        ...     mask_prompt=True,
        ...     formatting_func=lambda x: x["instruction"] + x["response"],
        ... )
        >>> result = transform({"instruction": "Hi", "response": "Hello!"})
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_length: int,
        text_field: str = "text",
        messages_field: str = "messages",
        mask_prompt: bool = False,
        add_eos: bool = True,
        truncation: bool = True,
        formatting_func: tp.Callable | None = None,
    ):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._text_field = text_field
        self._messages_field = messages_field
        self._mask_prompt = mask_prompt
        self._add_eos = add_eos
        self._truncation = truncation
        self._formatting_func = formatting_func

    def __call__(self, example: Example) -> Example:
        """Apply SFT preprocessing to example.

        Args:
            example: Input example.

        Returns:
            Preprocessed example with input_ids, attention_mask, etc.
        """
        # Skip if already tokenized
        if "input_ids" in example:
            return example

        result = dict(example)

        # Step 0: Apply formatting function if provided
        if self._formatting_func is not None:
            formatted = self._formatting_func(example)
            if isinstance(formatted, str):
                result = dict(example)
                result[self._text_field] = formatted
            elif isinstance(formatted, dict):
                result = formatted
            else:
                result = dict(example)
                result[self._text_field] = str(formatted)

        # Step 1: Convert from/value format to ChatML
        if is_conversational_from_value(result):
            result = convert_to_chatml(result)

        # Step 2: Handle conversational format
        if is_conversational(result):
            messages = result.get(self._messages_field) or result.get("messages")
            if messages:
                return self._tokenize_conversational(result, messages)

        text_value = result.get(self._text_field)
        if isinstance(text_value, list) and text_value and all(isinstance(item, dict) for item in text_value):
            messages = self._normalize_message_list(text_value)
            return self._tokenize_conversational(result, messages)

        # Step 3: Handle prompt/completion format
        if "prompt" in result and "completion" in result:
            return self._tokenize_prompt_completion(result)

        # Step 4: Handle plain text format
        if self._text_field in result:
            return self._tokenize_text(result)

        # No recognized format, return as-is
        return result

    @staticmethod
    def _normalize_message_list(messages: list[dict]) -> list[dict]:
        if not messages:
            return messages
        first = messages[0]
        if "role" in first and "content" in first:
            return messages
        if "from" in first and "value" in first:
            role_mapping = {
                "human": "user",
                "gpt": "assistant",
                "system": "system",
                "user": "user",
                "assistant": "assistant",
            }
            normalized: list[dict] = []
            for turn in messages:
                if not isinstance(turn, dict):
                    continue
                role = role_mapping.get(turn.get("from", "user"), turn.get("from", "user"))
                normalized.append({"role": role, "content": turn.get("value", "")})
            return normalized
        return messages

    def _tokenize_conversational(self, example: dict, messages: list) -> dict:
        """Tokenize conversational data using chat template."""
        result = dict(example)

        # Handle tools if present
        tools = example.get("tools")
        if isinstance(tools, str):
            tools = json.loads(tools)

        try:
            processed = self._tokenizer.apply_chat_template(
                messages,
                return_dict=True,
                return_attention_mask=True,
                return_assistant_tokens_mask=self._mask_prompt,
                truncation=self._truncation,
                max_length=self._max_length,
                padding="max_length" if self._max_length else False,
                tools=tools,
            )
            result.update(processed)
        except Exception:
            # Fallback if chat template fails - try without tools first
            try:
                text = self._tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                # Final fallback: simple formatting without chat template
                text = self._simple_format_messages(messages)

            if self._add_eos and not text.endswith(self._tokenizer.eos_token):
                text = text + self._tokenizer.eos_token

            tokens = self._tokenizer(
                text,
                truncation=self._truncation,
                max_length=self._max_length,
                padding="max_length" if self._max_length else False,
                return_attention_mask=True,
            )
            result["input_ids"] = tokens["input_ids"]
            result["attention_mask"] = tokens["attention_mask"]

        # Remove non-tokenized fields
        return purify_example(result)

    def _simple_format_messages(self, messages: list) -> str:
        """Simple fallback formatting when chat template fails.

        This handles cases where the tokenizer's chat template has strict
        requirements (e.g., tool messages must follow assistant tool calls).
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle tool calls in assistant messages
            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tool_str = json.dumps(tool_calls, ensure_ascii=False)
                    content = (
                        f"{content}\n<tool_calls>{tool_str}</tool_calls>"
                        if content
                        else f"<tool_calls>{tool_str}</tool_calls>"
                    )

            # Handle tool response messages
            if role == "tool":
                tool_name = msg.get("name", "tool")
                content = f'<tool_response name="{tool_name}">{content}</tool_response>'

            parts.append(f"<|{role}|>\n{content}")

        return "\n".join(parts)

    def _tokenize_prompt_completion(self, example: dict) -> dict:
        """Tokenize prompt/completion format with optional masking."""
        result = dict(example)

        prompt = example["prompt"]
        completion = example["completion"]

        # Add EOS to completion if needed
        if self._add_eos and not completion.endswith(self._tokenizer.eos_token):
            completion = completion + self._tokenizer.eos_token

        # Check if conversational prompt
        if isinstance(prompt, list):
            prompt_ids = self._tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            full_text = self._tokenizer.apply_chat_template(
                [*prompt, {"role": "assistant", "content": completion}],
                tokenize=False,
            )
        else:
            prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_text = prompt + completion

        # Use tokenizer for truncation and padding
        tokens = self._tokenizer(
            full_text,
            truncation=self._truncation,
            max_length=self._max_length,
            padding="max_length" if self._max_length else False,
            return_attention_mask=True,
            add_special_tokens=False,
        )

        result["input_ids"] = tokens["input_ids"]
        result["attention_mask"] = tokens["attention_mask"]

        # Create completion mask if needed
        if self._mask_prompt:
            prompt_len = min(len(prompt_ids), len(tokens["input_ids"]))
            seq_len = len(tokens["input_ids"])
            # Mask: 0 for prompt tokens, 1 for completion tokens, 0 for padding
            completion_mask = [0] * prompt_len + [1] * (seq_len - prompt_len)
            # Apply attention mask to zero out padding positions
            completion_mask = [m * a for m, a in zip(completion_mask, tokens["attention_mask"], strict=True)]
            result["completion_mask"] = completion_mask

        # Remove non-tokenized fields
        return purify_example(result)

    def _tokenize_text(self, example: dict) -> dict:
        """Tokenize plain text format."""
        result = dict(example)

        text = example[self._text_field]

        # Add EOS if needed
        if self._add_eos and not text.endswith(self._tokenizer.eos_token):
            text = text + self._tokenizer.eos_token

        tokens = self._tokenizer(
            text,
            truncation=self._truncation,
            max_length=self._max_length,
            padding="max_length" if self._max_length else False,
            return_attention_mask=True,
        )

        result["input_ids"] = tokens["input_ids"]
        result["attention_mask"] = tokens["attention_mask"]

        # Remove non-tokenized fields
        return purify_example(result)

    def __repr__(self) -> str:
        return f"SFTPreprocessTransform(max_length={self._max_length}, mask_prompt={self._mask_prompt})"

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

"""Mistral3 / Pixtral tokenizer wrapper.

Wraps ``mistral_common.tokens.tokenizers.mistral.MistralTokenizer`` in a
HuggingFace-style API so the rest of EasyDeL can use it interchangeably with
``transformers`` tokenizers (``encode``, ``decode``, ``__call__``, batch
helpers, and ``apply_chat_template``).

Exports:
    - ``Mistral3Tokenizer``: the HF-shaped wrapper class.
"""

from typing import Any

import numpy as np

try:
    from mistral_common.tokens.tokenizers.mistral import ChatCompletionRequest, MistralTokenizer, SpecialTokenPolicy  # type:ignore #noqa
except ImportError:
    ChatCompletionRequest = MistralTokenizer = SpecialTokenPolicy = None


class Mistral3Tokenizer:
    """HuggingFace-style wrapper around ``mistral_common.MistralTokenizer``.

    Exposes the standard ``transformers`` tokenizer surface
    (``encode``/``decode``/``__call__``/``apply_chat_template``/batch
    helpers) on top of Mistral AI's official ``mistral-common`` tokenizer so
    that downstream EasyDeL code can treat Mistral-3 / Pixtral tokenization
    interchangeably with HuggingFace tokenizers. Supports ``"left"`` /
    ``"right"`` padding sides via :attr:`padding_side` and ``return_tensors="np"``
    for numpy output.

    Attributes:
        mistral_tokenizer (MistralTokenizer): The wrapped ``mistral-common``
            tokenizer (with chat / instruct templating capabilities).
        tokenizer: The inner SentencePiece-style tokenizer reached via
            ``mistral_tokenizer.instruct_tokenizer.tokenizer``.
        pad_token_id (int | None): Padding token id (``None`` if the
            underlying tokenizer has no pad token).
        eos_token_id (int): End-of-sequence token id.
        bos_token_id (int): Beginning-of-sequence token id.
        padding_side (str): Either ``"left"`` (default) or ``"right"``,
            controlling where pad tokens are inserted.
    """

    def __init__(self, mistral_tokenizer: MistralTokenizer):  # type: ignore[no-untyped-def]
        """Wrap a ``MistralTokenizer`` with a HuggingFace-style API.

        Args:
            mistral_tokenizer (MistralTokenizer): An already-loaded
                ``mistral_common`` tokenizer.

        Raises:
            ImportError: If ``mistral-common`` is not installed.
        """
        if MistralTokenizer is None:
            raise ImportError("mistral-common is not installed. Please install it with `pip install mistral-common`.")
        self.mistral_tokenizer = mistral_tokenizer
        tokenizer = self.mistral_tokenizer.instruct_tokenizer.tokenizer
        self.pad_token_id = tokenizer.pad_id
        self.eos_token_id = tokenizer.eos_id
        self.bos_token_id = tokenizer.bos_id
        self.tokenizer = tokenizer
        self.padding_side = "left"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a single string into a list of token ids.

        Maps the ``add_special_tokens`` flag to both the ``bos`` and ``eos``
        arguments of the underlying Mistral tokenizer (HuggingFace's
        single-flag semantics are not exposed by ``mistral-common``).

        Args:
            text: The input text to encode.
            add_special_tokens: Whether to prepend BOS and append EOS.

        Returns:
            list[int]: The encoded token ids.
        """
        return self.tokenizer.encode(text, bos=add_special_tokens, eos=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back into a string.

        Args:
            token_ids: Token ids to decode.
            skip_special_tokens: When ``True``, applies the IGNORE policy so
                special tokens are stripped from the output; when ``False``,
                the KEEP policy preserves them verbatim.

        Returns:
            str: The decoded text.
        """
        policy = SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP
        return self.mistral_tokenizer.decode(token_ids, policy)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> str | list[int] | dict[str, Any]:
        """Apply Mistral's chat template to a conversation history.

        Args:
            conversation: List of role/content message dicts describing the
                conversation (Mistral chat schema).
            tokenize: When ``False``, returns the formatted string;
                when ``True``, tokenizes the formatted text via
                :meth:`__call__`.
            add_special_tokens: Whether to add BOS/EOS during tokenization
                (only used when ``tokenize=True``).
            padding: Padding behaviour forwarded to :meth:`__call__`.
            truncation: Truncation behaviour forwarded to :meth:`__call__`.
            max_length: Maximum length forwarded to :meth:`__call__`.
            return_tensors: Tensor format for the tokenized output
                (e.g. ``"np"``); ignored when ``tokenize=False``.
            **kwargs: Extra keyword arguments forwarded to :meth:`__call__`.

        Returns:
            str | list[int] | dict[str, Any]: The templated string when
            ``tokenize=False``; otherwise the tokenizer output dictionary
            from :meth:`__call__`.
        """
        tokenized = self.mistral_tokenizer.encode_chat_completion(ChatCompletionRequest(messages=conversation))
        formatted_text = tokenized.text

        if not tokenize:
            return formatted_text

        return self.__call__(
            formatted_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Tokenize a single text or a batch of texts.

        Mimics the HuggingFace tokenizer call surface with padding and
        truncation control.

        Args:
            text: A single string, or a list of strings for batch tokenization.
            padding: Padding strategy. ``False`` / ``"do_not_pad"`` disables
                padding; ``True`` / ``"longest"`` pads to the longest sequence
                in the batch; ``"max_length"`` pads to ``max_length``.
            truncation: Truncation strategy. ``False`` / ``"do_not_truncate"``
                disables truncation; ``True`` / ``"longest_first"`` truncates
                to ``max_length``.
            max_length: Maximum sequence length. Required when ``truncation``
                is enabled or ``padding="max_length"``.
            return_tensors: When ``"np"``, returns numpy arrays; otherwise
                returns plain Python lists.
            add_special_tokens: Whether to prepend BOS / append EOS.
            **kwargs: Reserved for HF compatibility (currently unused).

        Returns:
            dict[str, Any]: A dictionary with ``"input_ids"`` and
            ``"attention_mask"``. For a single string input with
            ``return_tensors=None`` the leading batch dimension is squeezed
            out; otherwise outputs are batched.

        Raises:
            ValueError: If padding is enabled without a ``pad_token_id``, if
                ``padding="max_length"`` is requested without ``max_length``,
                or if :attr:`padding_side` is not ``"left"`` or ``"right"``.
        """
        is_single_input = isinstance(text, str)
        batch_texts = [text] if is_single_input else text

        if not batch_texts:
            return {"input_ids": [], "attention_mask": []}

        batch_token_ids = [self.encode(txt, add_special_tokens=add_special_tokens) for txt in batch_texts]

        if truncation and max_length:
            batch_token_ids = [tokens[:max_length] for tokens in batch_token_ids]

        if padding:
            if self.pad_token_id is None:
                raise ValueError(
                    "Padding is enabled, but the tokenizer does not have a `pad_token_id`. You can set one manually."
                )

            if padding == "longest" or padding is True:
                max_len = max(len(tokens) for tokens in batch_token_ids)
            elif padding == "max_length":
                if max_length is None:
                    raise ValueError("You must specify `max_length` when using `padding='max_length'`.")
                max_len = max_length
            else:
                max_len = 0
            if max_len > 0:
                if self.padding_side not in ["left", "right"]:
                    raise ValueError(f"padding_side must be 'left' or 'right', but got '{self.padding_side}'")

                padded_ids = []
                attention_masks = []
                for tokens in batch_token_ids:
                    num_to_pad = max_len - len(tokens)

                    if self.padding_side == "right":
                        padded_ids.append(tokens + [self.pad_token_id] * num_to_pad)
                        attention_masks.append([1] * len(tokens) + [0] * num_to_pad)
                    else:
                        padded_ids.append([self.pad_token_id] * num_to_pad + tokens)
                        attention_masks.append([0] * num_to_pad + [1] * len(tokens))

                batch_token_ids = padded_ids
            else:
                attention_masks = [[1] * len(tokens) for tokens in batch_token_ids]
        else:
            attention_masks = [[1] * len(tokens) for tokens in batch_token_ids]

        result = {
            "input_ids": batch_token_ids,
            "attention_mask": attention_masks,
        }

        if return_tensors == "np":
            result["input_ids"] = np.array(result["input_ids"], dtype=np.int64)
            result["attention_mask"] = np.array(result["attention_mask"], dtype=np.int64)

        if is_single_input and return_tensors is None:
            result["input_ids"] = result["input_ids"][0]
            result["attention_mask"] = result["attention_mask"][0]

        return result

    def batch_encode_plus(self, *args, **kwargs) -> dict[str, Any]:
        """HF-compatible alias for :meth:`__call__`.

        Args:
            *args: Positional arguments forwarded to :meth:`__call__`.
            **kwargs: Keyword arguments forwarded to :meth:`__call__`.

        Returns:
            dict[str, Any]: Same dictionary as :meth:`__call__`.
        """
        return self.__call__(*args, **kwargs)

    def encode_plus(self, *args, **kwargs) -> dict[str, Any]:
        """HF-compatible alias for :meth:`__call__`.

        Args:
            *args: Positional arguments forwarded to :meth:`__call__`.
            **kwargs: Keyword arguments forwarded to :meth:`__call__`.

        Returns:
            dict[str, Any]: Same dictionary as :meth:`__call__`.
        """
        return self.__call__(*args, **kwargs)

    @classmethod
    def from_hf_hub(cls, model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        """Construct a tokenizer from a HuggingFace Hub repo id.

        Args:
            model_name: Repository id of a Mistral model on the Hub
                (defaults to Mistral-Nemo-Instruct-2407).

        Returns:
            Mistral3Tokenizer: A configured wrapper around the downloaded
            ``mistral-common`` tokenizer.

        Raises:
            ImportError: If ``mistral-common`` is not installed.
        """
        if MistralTokenizer is None:
            raise ImportError("mistral-common is not installed. Please install it with `pip install mistral-common`.")
        mistral_tokenizer = MistralTokenizer.from_hf_hub(repo_id=model_name)
        return cls(mistral_tokenizer)

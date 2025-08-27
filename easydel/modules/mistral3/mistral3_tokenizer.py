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

from typing import Any

import numpy as np

try:
    from mistral_common.tokens.tokenizers.mistral import ChatCompletionRequest, MistralTokenizer, SpecialTokenPolicy
except ImportError:
    ChatCompletionRequest, MistralTokenizer, SpecialTokenPolicy = (
        type(None),
        type(None),
        type(None),
    )


class Mistral3Tokenizer:
    """
    A wrapper class to make the `mistral-common` tokenizer behave like a
    Hugging Face `transformers` tokenizer. This is useful for maintaining a
    consistent API in projects that might use various tokenizers.

    Attributes:
        mistral_tokenizer: The original MistralTokenizer instance.
        pad_token_id: The ID of the padding token.
        eos_token_id: The ID of the end-of-sequence token.
        bos_token_id: The ID of the beginning-of-sequence token.
    """

    def __init__(self, mistral_tokenizer: MistralTokenizer):  # type: ignore[no-untyped-def]
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
        """
        Encodes a single string into a list of token IDs.

        This method maps the `add_special_tokens` flag to the `bos` and `eos`
        arguments of the underlying Mistral tokenizer.

        Args:
            text: The input text to encode.
            add_special_tokens: Whether to add special tokens (BOS/EOS).

        Returns:
            A list of token IDs.
        """
        return self.tokenizer.encode(text, bos=add_special_tokens, eos=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            token_ids: The list of token IDs to decode.
            skip_special_tokens: Whether to remove special tokens from the
                                 decoded string.

        Returns:
            The decoded text string.
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
        """
        Applies a chat template to a conversation history.

        Args:
            conversation: A list of message dictionaries, each with 'role' and 'content'.
            tokenize: If False, returns the formatted string. If True, tokenizes it.
            add_special_tokens: Whether to add special tokens.
            padding: Whether to pad the sequences.
            truncation: Whether to truncate the sequences.
            max_length: The maximum length for truncation or padding.
            return_tensors: The tensor format for the output (e.g., 'np').

        Returns:
            The processed output, which can be a string, list of IDs, or a dict.
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
        """
        Tokenizes a single text or a batch of texts, with advanced options for
        padding and truncation, mimicking Hugging Face tokenizers.

        Args:
            text: A single string or a list of strings to tokenize.
            padding: Controls padding.
                - `False` or `'do_not_pad'`: No padding.
                - `True` or `'longest'`: Pad to the longest sequence in the batch.
                - `'max_length'`: Pad to `max_length`.
            truncation: Controls truncation.
                - `False` or `'do_not_truncate'`: No truncation.
                - `True` or `'longest_first'`: Truncate to `max_length`.
            max_length: The maximum sequence length. Required for truncation
                        and `padding='max_length'`.
            return_tensors: If 'np', returns numpy arrays. Otherwise, returns lists.
            add_special_tokens: Whether to add special tokens like BOS and EOS.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask'.
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
        """Alias for `__call__` for Hugging Face compatibility."""
        return self.__call__(*args, **kwargs)

    def encode_plus(self, *args, **kwargs) -> dict[str, Any]:
        """Alias for `__call__` for Hugging Face compatibility."""
        return self.__call__(*args, **kwargs)

    @classmethod
    def from_hf_hub(cls, model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        """
        Creates an instance from a model name on the Hugging Face Hub.

        Args:
            model_name: The name of the Mistral model on the Hub.

        Returns:
            An instance of Mistral3Tokenizer.
        """
        if MistralTokenizer is None:
            raise ImportError("mistral-common is not installed. Please install it with `pip install mistral-common`.")
        mistral_tokenizer = MistralTokenizer.from_hf_hub(repo_id=model_name)
        return cls(mistral_tokenizer)

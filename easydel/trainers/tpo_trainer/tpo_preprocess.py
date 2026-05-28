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
"""Triple preference optimization trainer and loss functions."""

from __future__ import annotations

import typing as tp

from jax import numpy as jnp

from ..prompt_transforms import (
    extract_prompt_from_preference,
    maybe_apply_chat_template,
    maybe_convert_to_chatml,
    purify_example,
    resolve_example_tools,
)
from ..utils import DataCollatorForPreferenceGrain, DataCollatorForPreferenceTFDS, pad


class TPOPreprocessTransform:
    """Tokenize triple-preference rows for EasyDeL TPO.

    Raw rows may contain explicit prompts or implicit prompts shared by chosen,
    rejected, and reference strings. The transform applies chat templates when
    needed, tokenizes all three completion branches, and emits the fields
    expected by the TPO collators.
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        tools: list | None = None,
        label_pad_token_id: int = -100,
    ) -> None:
        """Store tokenizer and truncation settings for triple preferences.

        Args:
            tokenizer: Tokenizer or processor tokenizer used for prompt and
                completion branch encoding.
            max_prompt_length: Optional left-truncation cap for shared prompt
                tokens.
            max_completion_length: Optional right-truncation cap for chosen,
                rejected, and reference branches.
            tools: Optional chat-template tool definitions resolved per
                example.
            label_pad_token_id: Label value used to mask prompt tokens in each
                completion branch.
        """
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
        self._max_completion_length = max_completion_length
        self._tools = tools
        self._label_pad_token_id = label_pad_token_id
        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __call__(self, example: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Convert one raw triple-preference row into tokenized TPO fields.

        Pretokenized rows containing prompt and reference ids pass through
        unchanged. Raw rows must include ``reference`` and either an explicit
        ``prompt`` or chosen/rejected strings with a shared implicit prompt; the
        transform applies chat templates before tokenizing all three branches.
        """
        if "prompt_input_ids" in example and "reference_input_ids" in example:
            return example
        if "reference" not in example:
            raise ValueError("TPO requires a `reference` (gold) completion column.")

        raw_example = maybe_convert_to_chatml(dict(example))
        result = extract_prompt_from_preference(raw_example)
        if "prompt" not in result:
            raise ValueError(
                "TPO requires an explicit `prompt` column or chosen/rejected completions with a shared prefix."
            )
        if "reference" not in result:
            result["reference"] = raw_example["reference"]
        if "prompt" not in raw_example:
            prompt = result["prompt"]
            reference = result["reference"]
            if isinstance(prompt, str) and isinstance(reference, str):
                if not reference.startswith(prompt):
                    raise ValueError(
                        "The `reference` completion does not start with the implicit prompt extracted from "
                        "`chosen`/`rejected`; provide an explicit `prompt` column."
                    )
                result["reference"] = reference[len(prompt) :]

        result = maybe_apply_chat_template(result, self._tokenizer, resolve_example_tools(result, self._tools))
        return self._tokenize(result)

    def _tokenize(self, example: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Tokenize one triple-preference example into prompt/completion fields.

        Prompt tokens are left-truncated by ``max_prompt_length`` and completion
        branches are right-truncated by ``max_completion_length``. EOS is added
        to each branch when the tokenizer exposes an EOS token id.
        """
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        reference = example["reference"]

        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_completion_ids = self._tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_completion_ids = self._tokenizer(rejected, add_special_tokens=False)["input_ids"]
        reference_completion_ids = self._tokenizer(reference, add_special_tokens=False)["input_ids"]

        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            chosen_completion_ids = [*chosen_completion_ids, eos_token_id]
            rejected_completion_ids = [*rejected_completion_ids, eos_token_id]
            reference_completion_ids = [*reference_completion_ids, eos_token_id]

        if self._max_prompt_length is not None:
            prompt_ids = prompt_ids[-self._max_prompt_length :]
        if self._max_completion_length is not None:
            chosen_completion_ids = chosen_completion_ids[: self._max_completion_length]
            rejected_completion_ids = rejected_completion_ids[: self._max_completion_length]
            reference_completion_ids = reference_completion_ids[: self._max_completion_length]

        prompt_len = len(prompt_ids)
        result = dict(example)
        result["prompt_input_ids"] = prompt_ids
        result["prompt_attention_mask"] = [1] * prompt_len
        for prefix, completion_ids in (
            ("chosen", chosen_completion_ids),
            ("rejected", rejected_completion_ids),
            ("reference", reference_completion_ids),
        ):
            full_ids = prompt_ids + completion_ids
            result[f"{prefix}_input_ids"] = full_ids
            result[f"{prefix}_attention_mask"] = [1] * len(full_ids)
            result[f"{prefix}_labels"] = [self._label_pad_token_id] * prompt_len + completion_ids
        return purify_example(result)


class DataCollatorForTriplePreferenceTFDS(DataCollatorForPreferenceTFDS):
    """TFDS collator for TPO chosen/rejected/reference triples.

    The collator extends the normal preference collator with reference branch
    fields when ``include_reference`` is enabled. It pads prompt and completion
    tensors to the configured caps for TFDS-backed dataloaders.
    """

    include_reference: bool = True

    def __init__(
        self,
        max_prompt_length: int,
        max_completion_length: int,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool | None = False,
        pad_to_multiple_of: int | None = None,
        include_reference: bool = True,
    ) -> None:
        """Create the TFDS collator used for batched TPO examples.

        The base preference collator handles prompt, chosen, and rejected
        tensors. This initializer records whether the reference completion
        branch should also be padded and emitted, which is only required when
        the TPO alpha term is active.
        """
        super().__init__(
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            pad_token_id=pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.include_reference = include_reference

    def __call__(self, features: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
        """Pad a list of TFDS examples into a TPO model batch.

        The parent collator produces normal preference tensors. When reference
        output is enabled, this method extracts each reference completion,
        resolves a shared completion pad length across all three branches, and
        appends ``reference_input_ids`` plus ``reference_attention_mask``.
        """
        output = super().__call__(features)
        if not self.include_reference:
            return output

        chosen_arrays = [self._extract_completion_arrays(feature, "chosen") for feature in features]
        rejected_arrays = [self._extract_completion_arrays(feature, "rejected") for feature in features]
        reference_arrays = [self._extract_completion_arrays(feature, "reference") for feature in features]
        chosen_input_ids = [input_ids for input_ids, _ in chosen_arrays]
        rejected_input_ids = [input_ids for input_ids, _ in rejected_arrays]
        reference_input_ids = [input_ids for input_ids, _ in reference_arrays]
        reference_attention_mask = [attention_mask for _, attention_mask in reference_arrays]
        completion_pad_length = self._resolve_pad_length(
            [*chosen_input_ids, *rejected_input_ids, *reference_input_ids],
            self.max_completion_length,
        )
        output["reference_input_ids"] = pad(
            reference_input_ids,
            completion_pad_length,
            padding_value=self.pad_token_id,
        )
        output["reference_attention_mask"] = pad(reference_attention_mask, completion_pad_length, padding_value=0)
        return output


class DataCollatorForTriplePreferenceGrain(DataCollatorForPreferenceGrain):
    """Grain collator for TPO chosen/rejected/reference triples.

    This variant mirrors the TFDS collator behavior for Grain dataloaders,
    including optional reference completion tensors used by the TPO alpha term.
    """

    include_reference: bool = True

    def __init__(
        self,
        max_prompt_length: int,
        max_completion_length: int,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool | None = False,
        pad_to_multiple_of: int | None = None,
        include_reference: bool = True,
    ) -> None:
        """Create the Grain collator used for per-example TPO batching.

        Grain supplies one feature mapping at a time, but the same reference
        branch toggle is needed as in the TFDS collator. The stored flag keeps
        TPO alpha-zero runs from materializing unused reference tensors.
        """
        super().__init__(
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            pad_token_id=pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.include_reference = include_reference

    def __call__(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Pad one Grain feature mapping into TPO tensor fields.

        The parent collator handles prompt, chosen, and rejected branches. This
        method adds a padded reference branch for TPO runs that need the
        dataset-provided reference completion in the loss.
        """
        output = super().__call__(features)
        if not self.include_reference:
            return output

        chosen_input_ids, _ = self._extract_completion_arrays(features, "chosen")
        rejected_input_ids, _ = self._extract_completion_arrays(features, "rejected")
        reference_input_ids, reference_attention_mask = self._extract_completion_arrays(features, "reference")
        completion_pad_length = self._resolve_pad_length(
            [chosen_input_ids, rejected_input_ids, reference_input_ids],
            self.max_completion_length,
        )
        output["reference_input_ids"] = pad(
            [jnp.asarray(reference_input_ids)],
            completion_pad_length,
            padding_value=self.pad_token_id,
        )[0]
        output["reference_attention_mask"] = pad(
            [jnp.asarray(reference_attention_mask)],
            completion_pad_length,
            padding_value=0,
        )[0]
        return output

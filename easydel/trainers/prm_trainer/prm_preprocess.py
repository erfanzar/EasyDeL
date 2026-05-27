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
"""Process reward model preprocessing and trainer."""

from __future__ import annotations

import typing as tp


class PRMPreprocessTransform:
    """Tokenize process-reward examples into token-classification labels.

    Raw rows contain a prompt, ordered completion steps, and one label per step.
    The transform creates one token sequence where prompt tokens and non-final
    step tokens are masked with ``-100`` and each step label is placed on the
    final emitted token for that step.
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        *,
        step_separator: str = "\n",
        max_length: int | None = 1024,
        max_completion_length: int | None = None,
        train_on_last_step_only: bool = False,
        is_eval: bool = False,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        """Store tokenizer and labeling options for PRM row conversion.

        Args:
            tokenizer: Tokenizer or processor tokenizer used to encode prompts,
                completion steps, separators, and optional BOS tokens.
            step_separator: Text appended after each completion step before the
                sparse step label is placed on the final emitted token.
            max_length: Optional total sequence cap after prompt and completion
                concatenation.
            max_completion_length: Optional cap applied to concatenated
                completion-step tokens before prompt truncation is resolved.
            train_on_last_step_only: If true, non-final step labels are masked
                during training.
            is_eval: Keeps all labels available for evaluation even when
                last-step-only training is requested.
            pad_to_multiple_of: Optional padding multiple used after
                ``max_length`` is selected.
        """
        self._tokenizer = tokenizer
        self._step_separator = step_separator
        self._max_length = max_length
        self._max_completion_length = max_completion_length
        self._train_on_last_step_only = train_on_last_step_only
        self._is_eval = is_eval
        self._pad_to_multiple_of = pad_to_multiple_of
        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __call__(self, example: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Tokenize a raw PRM example unless it already contains model fields.

        Pretokenized examples with both ``input_ids`` and ``labels`` are
        returned unchanged. Raw examples are expected to contain ``prompt``,
        ``completions``, and ``labels`` and are converted with the constructor's
        separator, length, padding, and last-step supervision options.
        """
        if "input_ids" in example and "labels" in example:
            return example
        return self.tokenize_row(
            example,
            tokenizer=self._tokenizer,
            step_separator=self._step_separator,
            max_length=self._max_length,
            max_completion_length=self._max_completion_length,
            train_on_last_step_only=self._train_on_last_step_only,
            is_eval=self._is_eval,
            pad_token_id=self._pad_token_id,
            pad_to_multiple_of=self._pad_to_multiple_of,
        )

    @staticmethod
    def tokenize_row(
        features: dict[str, tp.Any],
        *,
        tokenizer: tp.Any,
        step_separator: str,
        max_length: int | None,
        max_completion_length: int | None,
        train_on_last_step_only: bool,
        is_eval: bool,
        pad_token_id: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> dict[str, list[int]]:
        """Tokenize one PRM row into input ids, attention mask, and step labels.

        Prompt tokens are masked with ``-100``. Each completion step contributes
        its label only on the separator-final token for that step, optionally
        limiting supervision to the final step.
        """
        prompt_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        completions = list(features["completions"])
        raw_labels = list(features["labels"])
        if len(completions) != len(raw_labels):
            raise ValueError("PRM examples must have the same number of `completions` and `labels`.")
        if not completions:
            raise ValueError("PRM examples require at least one completion step.")

        step_labels = [int(label) for label in raw_labels]
        if train_on_last_step_only and not is_eval:
            step_labels = [-100] * (len(step_labels) - 1) + [step_labels[-1]]

        separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
        completion_pieces = [
            tokenizer(completion, add_special_tokens=False)["input_ids"] + separator_ids for completion in completions
        ]

        def build_completion_labels(token_budget: int | None = None) -> tuple[list[int], list[int]]:
            """Build completion tokens and sparse PRM labels within a token budget.

            The budget is applied across the concatenated completion steps. Each
            emitted step contributes separator-terminated tokens and a single
            supervised label on its final token.
            """
            remaining = token_budget
            out_completion_ids: list[int] = []
            out_labels: list[int] = []
            for completion_ids, step_label in zip(completion_pieces, step_labels, strict=True):
                if remaining is not None:
                    if remaining <= 0:
                        break
                    completion_ids = completion_ids[:remaining]
                    remaining -= len(completion_ids)
                if not completion_ids:
                    continue
                out_completion_ids.extend(completion_ids)
                out_labels.extend([-100] * (len(completion_ids) - 1) + [step_label])
            return out_completion_ids, out_labels

        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            prompt_ids = [bos_token_id, *prompt_ids]

        completion_budget = max_completion_length
        if max_length is not None:
            completion_budget = max_length if completion_budget is None else min(completion_budget, max_length)
        completion_ids, labels = build_completion_labels(completion_budget)

        if max_length is not None:
            prompt_budget = max(max_length - len(completion_ids), 0)
            if prompt_budget == 0 and len(completion_ids) > max_length:
                completion_ids, labels = build_completion_labels(max_length)
                prompt_ids = []
            elif len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + labels

        attention_mask = [1] * len(input_ids)
        target_length = max_length
        if target_length is not None and pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
            target_length = ((target_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        if target_length is not None and len(input_ids) < target_length:
            pad_id = 0 if pad_token_id is None else int(pad_token_id)
            pad_len = target_length - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

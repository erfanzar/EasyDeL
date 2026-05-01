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

"""Verifiable reward functions for RLVR training.

This module provides rule-based reward verifiers that score model
completions against ground-truth answers without a learned reward
model. Each verifier implements a domain-specific correctness check
(math answer extraction, code test execution, format validation, etc.).

Verifiers follow the standard GRPO reward function signature::

    def verifier(*, prompts, completions, **kwargs) -> list[float]

They can be passed directly as ``reward_funcs`` to the RLVR trainer.

Available verifiers:
    - ``MathVerifier``: Extract ``\\boxed{}`` or ``####`` answers and
      compare against gold answers numerically.
    - ``CodeVerifier``: Execute generated code against test cases
      and return pass/fail rewards.
    - ``FormatVerifier``: Check that completions match a required
      format (regex pattern).
    - ``LengthPenaltyVerifier``: Penalise overly long or short
      completions relative to a target length.
"""

from __future__ import annotations

import re
import typing as tp

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_HASH_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _normalize_number(text: str) -> str | None:
    """Strip formatting and return a canonical number string."""
    text = text.strip().replace(",", "").replace(" ", "")
    try:
        return str(float(text))
    except ValueError:
        return None


def _extract_answer(text: str) -> str | None:
    """Extract a numeric answer from model output.

    Checks for ``\\boxed{...}`` first, then ``#### ...``, then the
    last number in the text.
    """
    match = _BOXED_RE.search(text)
    if match:
        return _normalize_number(match.group(1))
    match = _HASH_ANSWER_RE.search(text)
    if match:
        return _normalize_number(match.group(1))
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return _normalize_number(numbers[-1])
    return None


class MathVerifier:
    """Rule-based math answer verifier.

    Extracts the predicted answer from model output (via ``\\boxed{}``
    or ``####`` format) and compares it against the gold answer.
    Returns binary reward: 1.0 if correct, 0.0 otherwise.

    The gold answer is read from ``batch["answer"]`` which should
    contain the GSM8K-style answer string (ending with ``#### <num>``).

    Args:
        answer_key: Key in the batch dict containing gold answers.
        tolerance: Absolute tolerance for numeric comparison.
    """

    def __init__(self, answer_key: str = "answer", tolerance: float = 1e-6):
        """Initialize a math answer verifier.

        Args:
            answer_key (str): Column name in the batch dict carrying gold
                answer strings.
            tolerance (float): Absolute tolerance used for numeric
                comparison between predicted and gold answers.
        """
        self._answer_key = answer_key
        self._tolerance = tolerance

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions against gold math answers.

        Args:
            prompts: List of prompt strings or message lists.
            completions: List of completion strings or message lists.
            batch: Batch dict containing the gold answer column.
            **kwargs: Ignored.

        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect).
        """
        if completions is None:
            return [0.0]

        completions_list = list(completions) if not isinstance(completions, list) else completions
        rewards = []

        gold_answers = self._get_gold_answers(batch, len(completions_list))

        for completion, gold in zip(completions_list, gold_answers, strict=False):
            text = completion if isinstance(completion, str) else str(completion)
            predicted = _extract_answer(text)

            if predicted is None or gold is None:
                rewards.append(0.0)
                continue

            try:
                correct = abs(float(predicted) - float(gold)) <= self._tolerance
            except (ValueError, TypeError):
                correct = predicted == gold

            rewards.append(1.0 if correct else 0.0)

        return rewards

    def _get_gold_answers(
        self,
        batch: dict[str, tp.Any] | None,
        n: int,
    ) -> list[str | None]:
        """Extract gold answers from the batch dict.

        Args:
            batch (dict[str, tp.Any] | None): Source batch.
            n (int): Number of completions; the result list is padded with
                ``None`` to this length.

        Returns:
            list[str | None]: Normalized gold-answer strings of length
            ``n``; entries are ``None`` when the answer is missing or
            cannot be parsed.
        """
        if batch is None:
            return [None] * n

        raw_answers = batch.get(self._answer_key)
        if raw_answers is None:
            return [None] * n

        if isinstance(raw_answers, str):
            raw_answers = [raw_answers] * n

        result = []
        for ans in raw_answers:
            if isinstance(ans, str):
                match = _HASH_ANSWER_RE.search(ans)
                if match:
                    result.append(_normalize_number(match.group(1)))
                else:
                    result.append(_normalize_number(ans))
            else:
                result.append(str(ans))

        while len(result) < n:
            result.append(None)
        return result


class CodeVerifier:
    """Code execution verifier.

    Extracts code from the completion, runs it with provided test
    cases, and returns binary reward based on pass/fail.

    The verifier looks for code inside markdown code blocks
    (`` ```python ... ``` ``) or ``<code>...</code>`` tags.

    Args:
        timeout: Maximum execution time per test case in seconds.
        test_key: Key in the batch dict containing test case strings.
    """

    def __init__(self, timeout: float = 10.0, test_key: str = "tests"):
        """Initialize a code-execution verifier.

        Args:
            timeout (float): Hard time budget in seconds for each test
                case execution.
            test_key (str): Column name in the batch dict carrying test
                source code.
        """
        self._timeout = timeout
        self._test_key = test_key

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by executing code against test cases.

        Args:
            prompts: List of prompts.
            completions: List of code completions.
            batch: Batch dict with test cases under ``test_key``.
            **kwargs: Ignored.

        Returns:
            List of rewards (1.0 if all tests pass, 0.0 otherwise).
        """
        if completions is None:
            return [0.0]

        completions_list = list(completions) if not isinstance(completions, list) else completions
        tests = self._get_tests(batch, len(completions_list))
        rewards = []

        for completion, test_code in zip(completions_list, tests, strict=False):
            text = completion if isinstance(completion, str) else str(completion)
            code = self._extract_code(text)
            if not code or not test_code:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if self._run_tests(code, test_code) else 0.0)

        return rewards

    @staticmethod
    def _extract_code(text: str) -> str | None:
        """Extract code from markdown or XML code blocks."""
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _run_tests(self, code: str, test_code: str) -> bool:
        """Execute code + test assertions in a sandboxed environment.

        Args:
            code (str): The candidate code to execute.
            test_code (str): The test/assertion source appended to the
                candidate code.

        Returns:
            bool: True if the combined script ran to completion without
            raising or timing out, False otherwise.
        """
        import contextlib
        import io
        import signal

        combined = f"{code}\n\n{test_code}"
        try:

            def _handler(signum, frame):
                """Convert ``SIGALRM`` into a ``TimeoutError`` for ``exec``.

                Args:
                    signum: Signal number (unused).
                    frame: Current stack frame (unused).

                Raises:
                    TimeoutError: Always, to abort the running ``exec``.
                """
                raise TimeoutError

            old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(self._timeout))
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(combined, {"__builtins__": __builtins__})
                return True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        except Exception:
            return False

    def _get_tests(self, batch: dict[str, tp.Any] | None, n: int) -> list[str | None]:
        """Extract per-completion test sources from the batch dict.

        Args:
            batch (dict[str, tp.Any] | None): Source batch.
            n (int): Number of completions; the returned list is padded
                with ``None`` to length ``n``.

        Returns:
            list[str | None]: Test-code strings of length ``n``; entries
            are ``None`` when no test source is provided for that index.
        """
        if batch is None:
            return [None] * n
        tests = batch.get(self._test_key)
        if tests is None:
            return [None] * n
        if isinstance(tests, str):
            return [tests] * n
        result = list(tests)
        while len(result) < n:
            result.append(None)
        return result


class FormatVerifier:
    """Format compliance verifier.

    Checks that completions match a required regex pattern.
    Returns 1.0 if the pattern is found, 0.0 otherwise.

    Useful for enforcing output formats like ``\\boxed{}``,
    JSON output, or specific answer templates.

    Args:
        pattern: Regex pattern the completion must match.
        require_full_match: If True, the entire completion must
            match. If False (default), a partial match suffices.
    """

    def __init__(self, pattern: str, require_full_match: bool = False):
        """Initialize a format-compliance verifier.

        Args:
            pattern (str): Regex pattern compiled with ``re.DOTALL``.
            require_full_match (bool): When True, require ``fullmatch``
                instead of ``search``.
        """
        self._pattern = re.compile(pattern, re.DOTALL)
        self._full = require_full_match

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by format compliance.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            **kwargs: Ignored.

        Returns:
            List of rewards (1.0 if format matches, 0.0 otherwise).
        """
        if completions is None:
            return [0.0]
        completions_list = list(completions) if not isinstance(completions, list) else completions
        rewards = []
        for c in completions_list:
            text = c if isinstance(c, str) else str(c)
            if self._full:
                rewards.append(1.0 if self._pattern.fullmatch(text) else 0.0)
            else:
                rewards.append(1.0 if self._pattern.search(text) else 0.0)
        return rewards


class LengthPenaltyVerifier:
    """Length-based reward modifier.

    Applies a penalty to completions that are too long or too short.
    The reward is ``max(0, 1 - |len - target| / target)``, giving
    1.0 at the target length and decaying linearly to 0.

    Args:
        target_length: Ideal completion length in characters.
        min_length: Minimum length; completions shorter get 0.0.
        max_length: Maximum length; completions longer get 0.0.
    """

    def __init__(
        self,
        target_length: int = 500,
        min_length: int = 10,
        max_length: int = 5000,
    ):
        """Initialize a length-penalty verifier.

        Args:
            target_length (int): Desired completion length (in
                characters); produces reward 1.0 at the target.
            min_length (int): Lower bound; completions shorter than this
                receive 0.0.
            max_length (int): Upper bound; completions longer than this
                receive 0.0.
        """
        self._target = target_length
        self._min = min_length
        self._max = max_length

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by length proximity to target.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            **kwargs: Ignored.

        Returns:
            List of rewards in ``[0, 1]``.
        """
        if completions is None:
            return [0.0]
        completions_list = list(completions) if not isinstance(completions, list) else completions
        rewards = []
        for c in completions_list:
            text = c if isinstance(c, str) else str(c)
            length = len(text)
            if length < self._min or length > self._max:
                rewards.append(0.0)
            else:
                penalty = abs(length - self._target) / max(self._target, 1)
                rewards.append(max(0.0, 1.0 - penalty))
        return rewards

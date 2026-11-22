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
    - ``ExactMatchVerifier``: Normalised exact match against a gold
      string, with optional regex answer extraction.
    - ``MultipleChoiceVerifier``: Extract the chosen option letter and
      compare against the gold choice.
    - ``ContainsVerifier``: Require/forbid keywords (all/any, graded).
    - ``JSONVerifier``: Reward valid JSON and required top-level keys.
    - ``ReasoningFormatVerifier``: Reward the R1-style
      ``<think>...</think>`` + final-answer structure.
    - ``RepetitionPenaltyVerifier``: Reward lexical diversity via the
      distinct-n-gram ratio.
    - ``CompositeVerifier``: Weighted combination of several verifiers.
"""

from __future__ import annotations

import math
import re
import typing as tp

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_HASH_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_FRAC_RE = re.compile(r"^([+-]?)\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}$")
_SIMPLE_FRAC_RE = re.compile(r"^([+-]?\d[\d,]*(?:\.\d+)?)/([+-]?\d[\d,]*(?:\.\d+)?)$")
_TEXT_WRAPPER_RE = re.compile(r"^\\(?:text|mathrm|mathbf|operatorname)\{([^{}]+)\}$")


def _extract_boxed_contents(text: str) -> list[str]:
    """Extract all balanced-brace ``\\boxed{...}`` payloads in appearance order."""
    results: list[str] = []
    marker = r"\boxed{"
    idx = 0
    while True:
        start = text.find(marker, idx)
        if start == -1:
            break
        content_start = start + len(marker)
        depth = 1
        pos = content_start
        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[content_start : pos - 1])
            idx = pos
        else:
            idx = content_start
    return results


def _normalize_number(text: str) -> str | None:
    """Strip formatting and return a canonical number string.

    Removes whitespace, optional ``$`` / ``\\text{...}`` wrappers, and
    thousands-separator commas; evaluates simple LaTeX/slash fractions such as
    ``\\frac{1}{2}`` or ``3/4``; and serialises the finite ``float`` back to
    ``str`` so that lexically distinct but numerically equal answers
    (``"1,000"`` vs ``"1000.0"``, ``"\\frac{1}{2}"`` vs ``"0.5"``) compare
    equal.

    Args:
        text: Free-form input that should represent a number.

    Returns:
        A canonical ``str(float(...))`` representation, or ``None``
        when the input does not parse as a finite number.
    """
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == "$" and cleaned[-1] == "$":
        cleaned = cleaned[1:-1].strip()
    while True:
        wrapper = _TEXT_WRAPPER_RE.match(cleaned)
        if not wrapper:
            break
        cleaned = wrapper.group(1).strip()

    compact = cleaned.replace(" ", "")
    frac_match = _FRAC_RE.match(compact)
    if frac_match:
        sign_str, num_str, den_str = frac_match.groups()
        num_norm = _normalize_number(num_str)
        den_norm = _normalize_number(den_str)
        if num_norm is None or den_norm is None:
            return None
        den_val = float(den_norm)
        if den_val == 0.0:
            return None
        val = float(num_norm) / den_val
        if sign_str == "-":
            val = -val
        return str(val) if math.isfinite(val) else None

    simple_frac = _SIMPLE_FRAC_RE.match(compact)
    if simple_frac:
        num_str, den_str = simple_frac.groups()
        try:
            num_val = float(num_str.replace(",", ""))
            den_val = float(den_str.replace(",", ""))
        except ValueError:
            return None
        if den_val == 0.0:
            return None
        val = num_val / den_val
        return str(val) if math.isfinite(val) else None

    numeric = compact.replace(",", "")
    if not numeric:
        return None
    try:
        val = float(numeric)
    except ValueError:
        return None
    return str(val) if math.isfinite(val) else None


def _extract_answer(text: str) -> str | None:
    """Extract a numeric answer from model output.

    Tries three strategies in order: the *last* parseable balanced-brace
    ``\\boxed{...}`` (LaTeX), the *last* GSM8K-style ``#### <number>`` marker,
    and finally the *last* digit-leading number anywhere in the text. Falls
    through if an earlier marker is empty or unparseable.

    Args:
        text: Raw model completion to scan.

    Returns:
        The extracted canonical number string, or ``None`` if no
        candidate matched.
    """
    for boxed in reversed(_extract_boxed_contents(text)):
        normalized = _normalize_number(boxed)
        if normalized is not None:
            return normalized
    for hash_num in reversed(_HASH_ANSWER_RE.findall(text)):
        normalized = _normalize_number(hash_num)
        if normalized is not None:
            return normalized
    for num in reversed(_NUMBER_RE.findall(text)):
        normalized = _normalize_number(num)
        if normalized is not None:
            return normalized
    return None


class MathVerifier:
    """Rule-based math answer verifier.

    Extracts the predicted answer from model output (via ``\\boxed{}``
    or ``####`` format) and compares it against the gold answer.
    Returns binary reward: 1.0 if correct, 0.0 otherwise.

    The gold answer is read from ``batch["answer"]`` which should
    contain the GSM8K-style answer string (ending with ``#### <num>``).

    Attributes:
        _answer_key (str): Key in the batch dict carrying the gold
            answer column.
        _tolerance (float): Absolute tolerance used when comparing the
            predicted and gold numeric values.
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

        completions_list = _to_text_list(completions)
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

        result: list[str | None] = []
        for ans in _get_column(batch, self._answer_key, n):
            if ans is None:
                result.append(None)
            elif isinstance(ans, str):
                normalized: str | None = None
                for boxed in reversed(_extract_boxed_contents(ans)):
                    normalized = _normalize_number(boxed)
                    if normalized is not None:
                        break
                if normalized is None:
                    for hash_num in reversed(_HASH_ANSWER_RE.findall(ans)):
                        normalized = _normalize_number(hash_num)
                        if normalized is not None:
                            break
                if normalized is None:
                    normalized = _normalize_number(ans)
                result.append(normalized)
            else:
                result.append(_normalize_number(str(ans)))

        while len(result) < n:
            result.append(None)
        return result


class CodeVerifier:
    """Code execution verifier.

    Extracts code from the completion, runs it with provided test
    cases, and returns binary reward based on pass/fail.

    The verifier looks for code inside markdown code blocks
    (`` ```python ... ``` ``) or ``<code>...</code>`` tags.

    Attributes:
        _timeout (float): Hard time budget (seconds) per test case.
        _test_key (str): Key in the batch dict carrying the test source
            column.
    """

    def __init__(self, timeout: float = 10.0, test_key: str = "tests", use_jail: str = "auto"):
        """Initialize a code-execution verifier.

        Args:
            timeout (float): Hard time budget in seconds for each test
                case execution.
            test_key (str): Column name in the batch dict carrying test
                source code.
            use_jail (str): Sandbox jail policy (``"auto"``, ``"required"``,
                ``"bwrap"``, ``"nsjail"``, or ``"off"``).
        """
        from .._shared.sandbox import Sandbox

        self._timeout = timeout
        self._test_key = test_key
        self._use_jail = use_jail
        self._sandbox = Sandbox(timeout=self._timeout, use_jail=self._use_jail)

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

        completions_list = _to_text_list(completions)
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
        """Extract code from markdown or XML code blocks.

        Searches for a fenced markdown block (``\\`\\`\\`python`` or
        plain ``\\`\\`\\```) first, then a ``<code>...</code>`` tag.
        The first match wins and the inner content is returned with
        leading / trailing whitespace stripped.

        Args:
            text: Raw completion text to scan.

        Returns:
            Extracted code string, or ``None`` if neither a markdown
            block nor a ``<code>`` tag is found.
        """
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _run_tests(self, code: str, test_code: str) -> bool:
        """Execute code + test assertions in a hardened out-of-process sandbox.

        Both candidate code and tests are staged in temporary files (avoiding
        Linux ``-c`` ``E2BIG`` argv limits) and run with a per-run cryptographic
        sentinel written to a private file *only after* ``test_code`` completes
        normally, defeating ``sys.exit(0)`` / ``os._exit(0)`` reward hacking.

        Args:
            code (str): The candidate code to execute.
            test_code (str): The test/assertion source run after the
                candidate code.

        Returns:
            bool: True iff the sandbox exited cleanly (exit code 0), did not
            time out, and the post-test sentinel matched exactly.
        """
        _, passed = self._sandbox.run_python_with_sentinel(code, test_code)
        return passed

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
        return _get_column(batch, self._test_key, n)


class FormatVerifier:
    """Format compliance verifier.

    Checks that completions match a required regex pattern.
    Returns 1.0 if the pattern is found, 0.0 otherwise.

    Useful for enforcing output formats like ``\\boxed{}``,
    JSON output, or specific answer templates.

    Attributes:
        _pattern (re.Pattern[str]): Compiled regex pattern (with
            ``re.DOTALL``).
        _full (bool): When True, require ``fullmatch`` instead of
            ``search`` when scoring.
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
        completions_list = _to_text_list(completions)
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

    Attributes:
        _target (int): Ideal completion length (characters); produces
            reward 1.0 when ``len(text) == target``.
        _min (int): Lower bound; completions shorter than this receive
            ``0.0``.
        _max (int): Upper bound; completions longer than this receive
            ``0.0``.
    """

    def __init__(
        self,
        target_length: int = 500,
        min_length: int = 10,
        max_length: int | None = None,
    ):
        """Initialize a length-penalty verifier.

        Args:
            target_length (int): Desired completion length (in
                characters); produces reward 1.0 at the target.
            min_length (int): Lower bound; completions shorter than this
                receive 0.0.
            max_length (int | None): Optional upper bound; completions
                longer than this receive 0.0. When ``None`` (the default),
                no arbitrary upper cap is applied beyond the natural linear
                decay to 0.0 at ``2 * target_length``.
        """
        self._target = int(target_length)
        self._min = int(min_length)
        self._max: int | None = None if max_length is None else int(max_length)

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
        completions_list = _to_text_list(completions)
        rewards = []
        for c in completions_list:
            text = c if isinstance(c, str) else str(c)
            length = len(text)
            if length < self._min or (self._max is not None and length > self._max):
                rewards.append(0.0)
            else:
                penalty = abs(length - self._target) / max(self._target, 1)
                rewards.append(max(0.0, 1.0 - penalty))
        return rewards


def _to_text_list(completions: tp.Any) -> list[str]:
    """Coerce a completions payload into a list of plain strings.

    Args:
        completions: List of completion strings or message-list objects
            (or ``None``).

    Returns:
        List of strings; empty when ``completions`` is ``None``.
    """
    if completions is None:
        return []

    def content_text(value: tp.Any) -> str:
        """Extract chat message/text-block content without Python repr escaping."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "content" in value:
                return content_text(value["content"])
            if value.get("type") in {"text", "output_text"}:
                return content_text(value.get("text", ""))
            return ""
        if isinstance(value, (list, tuple)):
            return "".join(content_text(part) for part in value)
        return "" if value is None else str(value)

    items = [completions] if isinstance(completions, (str, dict)) else list(completions)
    return [content_text(c) for c in items]


def _get_column(batch: dict[str, tp.Any] | None, key: str, n: int) -> list[tp.Any]:
    """Read a per-example column from the batch, padded/broadcast to length ``n``.

    A scalar column is broadcast to every row; a column whose length divides
    ``n`` is repeated prompt-major (each prompt's value covers all of its
    generations). Mirrors the gold-answer extraction used by the built-in
    verifiers so custom verifiers stay aligned with the completion order.

    Args:
        batch: Source batch dict (or ``None``).
        key: Column name to read.
        n: Number of completions the result must cover.

    Returns:
        A list of length ``>= n``; ``[None] * n`` when the batch or column is
        absent or empty.

    Raises:
        ValueError: If the column is non-empty, shorter than ``n`` and its
            length does not divide ``n``. Padding with ``None`` there would
            silently misalign metadata with completions (the tail completions
            would be scored against no gold answer).
    """
    if batch is None:
        return [None] * n
    col = batch.get(key)
    if col is None:
        return [None] * n
    if isinstance(col, str):
        return [col] * n
    try:
        result = list(col)
    except TypeError:
        # Includes Python numeric values and rank-zero NumPy/JAX arrays.
        return [col] * n
    if 0 < len(result) < n and n % len(result) == 0:
        # GRPO emits prompt-major completions: each prompt's reward metadata
        # must accompany all of its generations, not only its first row.
        return [value for value in result for _ in range(n // len(result))]
    if 0 < len(result) < n:
        raise ValueError(
            f"Reward metadata column {key!r} has {len(result)} entries, which does not divide the {n} completions; "
            "cannot align it prompt-major with the rollout layout."
        )
    while len(result) < n:
        result.append(None)
    return result


def _normalize_text(text: str, *, lower: bool = True, strip_punctuation: bool = False) -> str:
    """Canonicalise free text for string comparison.

    Collapses runs of whitespace to a single space and trims; optionally
    lowercases and removes punctuation so lexically-noisy but semantically
    equal answers compare equal.

    Args:
        text: Input string.
        lower: Lowercase the text when True.
        strip_punctuation: Drop non-word, non-space characters when True.

    Returns:
        The normalised string.
    """
    t = text.strip()
    if lower:
        t = t.lower()
    if strip_punctuation:
        t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


class ExactMatchVerifier:
    """Normalised exact-match verifier.

    Optionally extracts an answer span from the completion via a regex
    capture group, then compares it against the gold answer in
    ``batch[answer_key]`` after whitespace/case/punctuation normalisation.
    Returns 1.0 on match, 0.0 otherwise. Generalises answer-string tasks
    (closed-book QA, entity extraction, templated answers).

    Attributes:
        _answer_key (str): Batch column carrying the gold answer.
        _extract (re.Pattern[str] | None): Optional capture-group pattern;
            the last match's group 1 is compared instead of the full text.
        _lower (bool): Case-insensitive comparison when True.
        _strip_punct (bool): Ignore punctuation when True.
    """

    def __init__(
        self,
        answer_key: str = "answer",
        extract_pattern: str | None = None,
        case_sensitive: bool = False,
        strip_punctuation: bool = False,
    ):
        """Initialize an exact-match verifier.

        Args:
            answer_key: Batch column with the gold answer string.
            extract_pattern: Optional regex with one capture group; when
                given, the last match's group 1 is compared instead of the
                whole completion.
            case_sensitive: Compare case-sensitively when True.
            strip_punctuation: Ignore punctuation when True.
        """
        self._answer_key = answer_key
        self._extract = re.compile(extract_pattern, re.DOTALL) if extract_pattern else None
        self._lower = not case_sensitive
        self._strip_punct = strip_punctuation

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by normalised exact match against gold answers.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            batch: Batch dict carrying the gold-answer column.
            **kwargs: Ignored.

        Returns:
            List of rewards (1.0 for match, 0.0 otherwise).
        """
        texts = _to_text_list(completions)
        golds = _get_column(batch, self._answer_key, len(texts))
        rewards: list[float] = []
        for text, gold in zip(texts, golds, strict=False):
            if gold is None:
                rewards.append(0.0)
                continue
            candidate = text
            if self._extract is not None:
                matches = list(self._extract.finditer(text))
                if not matches:
                    candidate = ""
                else:
                    last_match = matches[-1]
                    if self._extract.groups == 0:
                        candidate = last_match.group(0)
                    else:
                        non_empty = [g for g in last_match.groups() if g is not None and g != ""]
                        candidate = non_empty[-1] if non_empty else ""
            pred = _normalize_text(candidate, lower=self._lower, strip_punctuation=self._strip_punct)
            target = _normalize_text(str(gold), lower=self._lower, strip_punctuation=self._strip_punct)
            rewards.append(1.0 if pred == target and pred != "" else 0.0)
        return rewards


class MultipleChoiceVerifier:
    """Multiple-choice answer verifier.

    Extracts the selected option letter (``A``-``E`` by default) from the
    completion and compares it against the gold letter in
    ``batch[answer_key]``. Extraction tries, in order: ``\\boxed{X}``, an
    ``answer is/: (X)`` phrase, then the last standalone option letter.

    Attributes:
        _answer_key (str): Batch column carrying the gold option letter.
        _choices (str): Allowed option letters (upper-case).
    """

    def __init__(self, answer_key: str = "answer", choices: str = "ABCDE"):
        """Initialize a multiple-choice verifier.

        Args:
            answer_key: Batch column with the gold option letter.
            choices: Allowed option letters (case-insensitive).
        """
        self._answer_key = answer_key
        self._choices = choices.upper()
        cls = f"[{re.escape(self._choices)}]"
        self._boxed = re.compile(rf"\\boxed\{{\s*({cls})\s*\}}", re.IGNORECASE)
        # Require an explicit "is" or ":" connector plus a trailing non-letter
        # boundary. Lowercase bare letters (such as the English article "a" in
        # "Answer: a simple equation gives B") are accepted only when parenthesized
        # or terminated by punctuation / end-of-line.
        self._phrase = re.compile(
            rf"(?i:\banswer\s*(?:is\b\s*:?|:))\s*"
            rf"(?:\(\s*(?i:({cls}))\s*\)"
            rf"|({cls})(?![A-Za-z])"
            rf"|(?i:({cls}))(?=[.,;:!?)\]\n]|$))"
        )
        self._lone_upper = re.compile(rf"(?<![A-Za-z])({cls})(?![A-Za-z])")
        self._lone = re.compile(rf"(?<![A-Za-z])({cls})(?![A-Za-z])", re.IGNORECASE)

    def _extract_choice(self, text: str) -> str | None:
        """Extract the chosen option letter from a completion.

        Args:
            text: Raw completion text.

        Returns:
            The upper-cased option letter, or ``None`` if none was found.
        """
        boxed_matches = self._boxed.findall(text)
        if boxed_matches:
            return boxed_matches[-1].upper()
        phrase_matches = list(self._phrase.finditer(text))
        if phrase_matches:
            last = phrase_matches[-1]
            for group in last.groups():
                if group is not None:
                    return group.upper()
        lone_upper = self._lone_upper.findall(text)
        if lone_upper:
            return lone_upper[-1].upper()
        lone = self._lone.findall(text)
        return lone[-1].upper() if lone else None

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by matching the extracted option to gold.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            batch: Batch dict carrying the gold option column.
            **kwargs: Ignored.

        Returns:
            List of rewards (1.0 for the correct option, 0.0 otherwise).
        """
        texts = _to_text_list(completions)
        golds = _get_column(batch, self._answer_key, len(texts))
        rewards: list[float] = []
        for text, gold in zip(texts, golds, strict=False):
            if gold is None:
                rewards.append(0.0)
                continue
            pred = self._extract_choice(text)
            target = str(gold).strip().upper()[:1]
            rewards.append(1.0 if pred is not None and pred == target else 0.0)
        return rewards


class ContainsVerifier:
    """Keyword presence/absence verifier.

    Rewards completions that contain required keywords and avoid
    forbidden ones. With ``graded=True`` the reward is the fraction of
    required keywords present (still zeroed if any forbidden keyword
    appears); otherwise it is binary under the ``mode`` condition.

    Attributes:
        _required (list[str]): Keywords that should appear.
        _forbidden (list[str]): Keywords that must not appear.
        _mode (str): ``"all"`` or ``"any"`` for the binary condition.
        _ci (bool): Case-insensitive matching when True.
        _graded (bool): Return the presence fraction instead of binary.
    """

    def __init__(
        self,
        required: tp.Sequence[str] | None = None,
        forbidden: tp.Sequence[str] | None = None,
        mode: str = "all",
        case_insensitive: bool = True,
        graded: bool = False,
    ):
        """Initialize a keyword verifier.

        Args:
            required: Keywords that should appear in the completion.
            forbidden: Keywords that must not appear.
            mode: ``"all"`` (every required keyword) or ``"any"`` (at least
                one) for the binary reward.
            case_insensitive: Lowercase both sides before matching.
            graded: When True, return the fraction of required keywords
                present rather than a binary reward.

        Raises:
            ValueError: If ``mode`` is not ``"all"`` or ``"any"``.
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any'; got {mode!r}.")
        self._required = list(required or [])
        self._forbidden = list(forbidden or [])
        self._mode = mode
        self._ci = case_insensitive
        self._graded = graded

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by keyword presence/absence.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            **kwargs: Ignored.

        Returns:
            List of rewards in ``[0, 1]``.
        """
        rewards: list[float] = []
        for text in _to_text_list(completions):
            hay = text.lower() if self._ci else text
            req = [k.lower() if self._ci else k for k in self._required]
            forb = [k.lower() if self._ci else k for k in self._forbidden]
            if any(k in hay for k in forb):
                rewards.append(0.0)
                continue
            if not req:
                rewards.append(1.0)
                continue
            present = [k in hay for k in req]
            if self._graded:
                rewards.append(sum(present) / len(present))
            elif self._mode == "all":
                rewards.append(1.0 if all(present) else 0.0)
            else:
                rewards.append(1.0 if any(present) else 0.0)
        return rewards


class JSONVerifier:
    """Valid-JSON verifier.

    Rewards completions that contain parseable JSON (optionally inside a
    ```` ```json ```` fence), and — when ``required_keys`` is set — that the
    parsed object contains those top-level keys. With ``graded=True`` and
    required keys, returns the fraction of required keys present (0.0 when
    the JSON does not parse).

    Attributes:
        _required (list[str]): Top-level keys the JSON object must contain.
        _graded (bool): Return the key-coverage fraction instead of binary.
    """

    _JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
    _FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", re.DOTALL)
    _PARSE_FAILURE = object()

    def __init__(self, required_keys: tp.Sequence[str] | None = None, graded: bool = False):
        """Initialize a JSON verifier.

        Args:
            required_keys: Top-level keys the parsed object must contain.
            graded: When True (with ``required_keys``), reward the fraction
                of required keys present rather than a binary check.
        """
        self._required = list(required_keys or [])
        self._graded = graded

    def _parse(self, text: str) -> tp.Any:
        """Parse JSON from a completion, preferring explicit ```` ```json ```` blocks.

        Args:
            text: Raw completion text.

        Returns:
            The parsed object (including ``None`` for JSON null), or the
            private parse-failure sentinel if nothing parsed.
        """
        import json

        candidates: list[str] = []
        # Prefer explicit ```json fences (last to first) over earlier non-JSON code fences.
        for block in reversed(self._JSON_FENCE.findall(text)):
            candidates.append(block.strip())
        for block in reversed(self._FENCE.findall(text)):
            stripped_block = block.strip()
            if stripped_block not in candidates:
                candidates.append(stripped_block)
        stripped = text.strip()
        candidates.append(stripped)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1])
        arr_start = stripped.find("[")
        arr_end = stripped.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(stripped[arr_start : arr_end + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                continue
        return self._PARSE_FAILURE

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by JSON validity and required-key coverage.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            **kwargs: Ignored.

        Returns:
            List of rewards in ``[0, 1]``.
        """
        rewards: list[float] = []
        for text in _to_text_list(completions):
            obj = self._parse(text)
            if obj is self._PARSE_FAILURE:
                rewards.append(0.0)
                continue
            if not self._required:
                rewards.append(1.0)
                continue
            keys = obj.keys() if isinstance(obj, dict) else []
            present = [k in keys for k in self._required]
            if self._graded:
                rewards.append(sum(present) / len(present))
            else:
                rewards.append(1.0 if all(present) else 0.0)
        return rewards


class ReasoningFormatVerifier:
    """Reasoning-format verifier (R1-style ``<think>`` + answer).

    Rewards completions that wrap their chain-of-thought in a single
    ``<think>...</think>`` block and place a final answer after it. Grants
    half credit for a well-formed, non-empty think block and half for an
    answer marker (``\\boxed{}``, ``<answer>...</answer>``, or the
    GSM8K-style ``#### <num>``) appearing after ``</think>``.

    Attributes:
        _think_open / _think_close (str): The reasoning tag pair.
    """

    _ANSWER_MARKERS = (
        re.compile(r"\\boxed\{.+?\}", re.DOTALL),
        re.compile(r"<answer>.+?</answer>", re.DOTALL),
        re.compile(r"####\s*\S+"),
    )

    def __init__(self, think_open: str = "<think>", think_close: str = "</think>"):
        """Initialize a reasoning-format verifier.

        Args:
            think_open: Opening reasoning tag.
            think_close: Closing reasoning tag.
        """
        self._think_open = think_open
        self._think_close = think_close

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by reasoning-format compliance.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            **kwargs: Ignored.

        Returns:
            List of rewards in ``{0.0, 0.5, 1.0}``.
        """
        rewards: list[float] = []
        for text in _to_text_list(completions):
            reward = 0.0
            o_count = text.count(self._think_open)
            c_count = text.count(self._think_close)
            o_idx = text.find(self._think_open)
            c_idx = text.find(self._think_close)
            think_ok = (
                o_count == 1
                and c_count == 1
                and o_idx != -1
                and c_idx > o_idx
                and text[o_idx + len(self._think_open) : c_idx].strip() != ""
            )
            if think_ok:
                reward += 0.5
                tail = text[c_idx + len(self._think_close) :]
            else:
                tail = text
            if any(marker.search(tail) for marker in self._ANSWER_MARKERS):
                reward += 0.5
            rewards.append(reward)
        return rewards


class RepetitionPenaltyVerifier:
    """N-gram repetition penalty.

    Rewards lexical diversity: the fraction of *distinct* word n-grams in
    the completion (``distinct-n``). A value of 1.0 means no repeated
    n-grams; degenerate loops trend toward 0. Texts shorter than one
    n-gram receive 1.0.

    Attributes:
        _n (int): N-gram size.
    """

    def __init__(self, ngram: int = 3):
        """Initialize a repetition-penalty verifier.

        Args:
            ngram: N-gram size used for the distinct-n ratio (>= 1).
        """
        self._n = max(1, int(ngram))

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by distinct-n-gram ratio.

        Args:
            prompts: Ignored.
            completions: List of completion strings.
            **kwargs: Ignored.

        Returns:
            List of rewards in ``[0, 1]`` (higher = less repetition).
        """
        rewards: list[float] = []
        for text in _to_text_list(completions):
            tokens = text.split()
            if len(tokens) < self._n:
                rewards.append(1.0)
                continue
            ngrams = [tuple(tokens[i : i + self._n]) for i in range(len(tokens) - self._n + 1)]
            rewards.append(len(set(ngrams)) / len(ngrams))
        return rewards


class CompositeVerifier:
    """Weighted combination of several verifiers.

    Calls each sub-verifier with the full reward payload and returns the
    per-completion weighted sum, so a single ``reward_funcs`` entry can
    blend, e.g., correctness + format + anti-repetition. Sub-verifier
    outputs are aligned to the shortest returned length.

    Attributes:
        _verifiers (list): The wrapped verifier callables.
        _weights (list[float]): Per-verifier weights (default all 1.0).
    """

    def __init__(
        self,
        verifiers: tp.Sequence[tp.Callable[..., list[float]]],
        weights: tp.Sequence[float] | None = None,
    ):
        """Initialize a composite verifier.

        Args:
            verifiers: Sub-verifiers to combine.
            weights: Per-verifier weights; defaults to ``1.0`` each.

        Raises:
            ValueError: If ``weights`` is given with a mismatched length.
        """
        self._verifiers = list(verifiers)
        if weights is None:
            weights = [1.0] * len(self._verifiers)
        if len(weights) != len(self._verifiers):
            raise ValueError(f"weights length {len(weights)} != verifiers length {len(self._verifiers)}.")
        self._weights = [float(w) for w in weights]

    def __call__(
        self,
        *,
        prompts: tp.Any = None,
        completions: tp.Any = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float]:
        """Score completions by the weighted sum of sub-verifier rewards.

        Args:
            prompts: Forwarded to each sub-verifier.
            completions: Forwarded to each sub-verifier.
            batch: Forwarded to each sub-verifier.
            **kwargs: Forwarded to each sub-verifier.

        Returns:
            List of weighted-sum rewards.
        """
        n = len(_to_text_list(completions))
        per_verifier: list[list[float]] = []
        for verifier in self._verifiers:
            scores = verifier(prompts=prompts, completions=completions, batch=batch, **kwargs)
            if scores is None:
                per_verifier.append([0.0] * n)
            else:
                per_verifier.append([0.0 if s is None else float(s) for s in scores])
        if not per_verifier:
            return [0.0] * n
        length = min(len(scores) for scores in per_verifier)
        rewards: list[float] = []
        for i in range(length):
            rewards.append(sum(w * scores[i] for w, scores in zip(self._weights, per_verifier, strict=True)))
        return rewards

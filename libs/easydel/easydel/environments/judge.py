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

"""LLM-as-judge rewards: an OpenAI-compatible judge client, rubric and pairwise judges, and an environment.

The pieces, bottom up:

* :class:`JudgeClient` posts to an OpenAI-compatible ``/chat/completions``
  endpoint through stdlib ``urllib`` (or an injected transport callable). It
  bounds in-flight requests with a semaphore, spaces requests with a
  :class:`RequestRateLimiter` shared by every thread, and retries transient
  failures with exponential backoff and jitter, honouring ``Retry-After`` /
  ``x-ratelimit-reset`` hints. All waiting happens *outside* the concurrency
  semaphore, so a sleeping request never blocks a sibling.
* :class:`RubricJudge` asks for a JSON rubric (``correctness`` 0-10, secondary
  criteria 0-10, ``fatal_issue``) and blends it with
  :func:`~easydel.environments.rewards.lexicographic_blend`, either one
  candidate per call or a whole group per call.
* :class:`PairwiseJudge` compares responses to the same prompt pairwise and
  turns win rates into scores in ``[0, 1]``.
* :class:`LLMJudgeEnvironment` (registered as ``"llm_judge"``) and
  :func:`make_judge_reward_fn` plug the judges into environment rollouts and
  GRPO ``reward_funcs``.

A sample the judge could not score is **unscorable**: reward functions return
``None`` for it (never a fabricated ``0``), and the environment marks the step
with ``info["env_error"]`` so collectors drop it.
"""

from __future__ import annotations

import collections.abc
import concurrent.futures
import email.utils
import itertools
import json
import math
import os
import random
import re
import threading
import time
import typing as tp
import urllib.error
import urllib.request

from eformer.loggings import get_logger

from easydel.trainers._shared.environment import AgenticEnvironment, ResetResult, StepResult

from ._base import register_environment
from .rewards import lexicographic_blend

logger = get_logger(__name__)

__all__ = (
    "JudgeClient",
    "JudgeError",
    "LLMJudgeEnvironment",
    "PairwiseJudge",
    "RequestRateLimiter",
    "RubricJudge",
    "make_judge_reward_fn",
    "parse_judge_json",
    "urllib_transport",
)

TransportResponse = tuple[int, collections.abc.Mapping[str, str], bytes]
"""``(status, headers, body)`` returned by a transport."""

Transport = tp.Callable[[str, collections.abc.Mapping[str, str], bytes, float], TransportResponse]
"""``transport(url, headers, body, timeout) -> (status, headers, body)``; raise on connection errors."""

_RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_DURATION_UNITS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


class JudgeError(RuntimeError):
    """A judge call failed for good (retries exhausted, non-retryable status, or unparsable output).

    Attributes:
        status: Last HTTP status, or ``None`` for transport/parse errors.
        rate_limited: ``True`` when every attempt was rejected with HTTP 429.
    """

    def __init__(self, message: str, *, status: int | None = None, rate_limited: bool = False):
        super().__init__(message)
        self.status = status
        self.rate_limited = rate_limited


def urllib_transport(
    url: str, headers: collections.abc.Mapping[str, str], body: bytes, timeout: float
) -> TransportResponse:
    """POST ``body`` with stdlib ``urllib`` and return ``(status, headers, body)``.

    HTTP error statuses are returned, not raised; connection errors and timeouts raise.
    """
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as err:
        err_headers = dict(err.headers.items()) if err.headers is not None else {}
        return int(err.code), err_headers, err.read() or b""


def _parse_duration(value: str, *, now_wall: float) -> float | None:
    """Parse a rate-limit hint: seconds, an epoch timestamp, ``"1m30s"``/``"250ms"``, or an HTTP date."""
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        parts = _DURATION_PART_RE.findall(text)
        if parts and "".join(f"{num}{unit}" for num, unit in parts) == text:
            return sum(float(num) * _DURATION_UNITS[unit] for num, unit in parts)
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        return max(0.0, when.timestamp() - now_wall)
    if not math.isfinite(seconds):
        return None
    if seconds > 1e9:  # an absolute epoch timestamp
        return max(0.0, seconds - now_wall)
    return max(0.0, seconds)


class RequestRateLimiter:
    """Smooth requests-per-minute limiter based on slot reservation.

    Each call to :meth:`reserve` books the next free slot ``60 / rpm`` seconds
    after the previous one and returns how long the caller must wait for it.
    Reservation is a constant-time critical section; the wait itself happens in
    the caller, outside any lock, so concurrent threads queue up evenly instead
    of bursting. With ``world_size > 1`` the budget is split across hosts that
    share one endpoint quota (each host gets ``rpm / world_size``).

    Args:
        requests_per_minute: Global request budget. Must be positive.
        world_size: Number of hosts sharing the budget.
        clock: Monotonic clock in seconds (injectable for tests).
    """

    def __init__(
        self,
        requests_per_minute: float,
        *,
        world_size: int = 1,
        clock: tp.Callable[[], float] = time.monotonic,
    ):
        if not requests_per_minute > 0:
            raise ValueError(f"`requests_per_minute` must be positive; got {requests_per_minute}.")
        if int(world_size) < 1:
            raise ValueError(f"`world_size` must be >= 1; got {world_size}.")
        self.interval = 60.0 * int(world_size) / float(requests_per_minute)
        self._clock = clock
        self._lock = threading.Lock()
        self._next_slot: float | None = None

    def reserve(self) -> float:
        """Book the next slot and return the delay (seconds, ``>= 0``) until it."""
        with self._lock:
            now = self._clock()
            slot = now if self._next_slot is None else max(now, self._next_slot)
            self._next_slot = slot + self.interval
            return slot - now


class JudgeClient:
    """Thread-safe client for an OpenAI-compatible ``/chat/completions`` judge endpoint.

    Args:
        base_url: Endpoint base, e.g. ``"http://host:8000/v1"``; ``/chat/completions`` is appended.
        model: Judge model name sent in each request.
        api_key: API key. Takes precedence over ``api_key_env``. Never logged.
        api_key_env: Name of an environment variable holding the API key.
        transport: Callable performing the HTTP POST (defaults to :func:`urllib_transport`).
        max_concurrency: Maximum in-flight requests across all threads.
        requests_per_minute: Optional global request budget (see :class:`RequestRateLimiter`).
        world_size: Hosts sharing ``requests_per_minute``.
        max_retries: Retries after the first attempt for transient failures.
        backoff_base: Base delay (seconds) of the exponential backoff.
        backoff_max: Cap on one backoff delay.
        retry_after_cap: Cap on a server-provided ``Retry-After`` / reset hint.
        timeout: Per-request timeout in seconds.
        temperature: Sampling temperature of the judge.
        max_tokens: Optional completion-token cap of the judge.
        extra_body: Extra JSON fields merged into each request body.
        clock: Monotonic clock (injectable for tests).
        sleep: Sleep function (injectable for tests).
        seed: Seed of the jitter RNG.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
        transport: Transport | None = None,
        max_concurrency: int = 8,
        requests_per_minute: float | None = None,
        world_size: int = 1,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        retry_after_cap: float = 60.0,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: collections.abc.Mapping[str, tp.Any] | None = None,
        clock: tp.Callable[[], float] = time.monotonic,
        sleep: tp.Callable[[float], None] = time.sleep,
        seed: int | None = None,
    ):
        if not base_url or not model:
            raise ValueError("`base_url` and `model` must be given explicitly.")
        if int(max_concurrency) < 1:
            raise ValueError(f"`max_concurrency` must be >= 1; got {max_concurrency}.")
        if int(max_retries) < 0:
            raise ValueError(f"`max_retries` must be >= 0; got {max_retries}.")
        if api_key is None and api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"Environment variable {api_key_env!r} holding the judge API key is not set.")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._api_key = api_key
        self._transport = transport or urllib_transport
        self.max_concurrency = int(max_concurrency)
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._limiter = (
            RequestRateLimiter(requests_per_minute, world_size=world_size, clock=clock)
            if requests_per_minute is not None
            else None
        )
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.retry_after_cap = float(retry_after_cap)
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self._sleep = sleep
        self._rng = random.Random(seed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={self.url!r}, model={self.model!r}, max_concurrency={self.max_concurrency})"

    def _retry_delay(self, attempt: int, headers: collections.abc.Mapping[str, str]) -> float:
        """Delay before retry ``attempt + 1``: the server hint (capped) or jittered exponential backoff."""
        lowered = {str(k).lower(): str(v) for k, v in headers.items()}
        for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
            if key in lowered:
                hint = _parse_duration(lowered[key], now_wall=time.time())
                if hint is not None:
                    return min(hint, self.retry_after_cap)
        backoff = self.backoff_base * (2.0**attempt) * self._rng.uniform(0.5, 1.5)
        return min(backoff, self.backoff_max)

    def _request_body(self, messages: collections.abc.Sequence[collections.abc.Mapping[str, tp.Any]]) -> bytes:
        body: dict[str, tp.Any] = {"model": self.model, "messages": list(messages), "temperature": self.temperature}
        if self.max_tokens is not None:
            body["max_tokens"] = int(self.max_tokens)
        body.update(self.extra_body)
        return json.dumps(body).encode("utf-8")

    @staticmethod
    def _extract_content(body: bytes) -> str:
        payload = json.loads(body.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not a string")
        return content

    def chat(self, messages: collections.abc.Sequence[collections.abc.Mapping[str, tp.Any]]) -> str:
        """Send one chat request and return the assistant message content.

        Transient failures (connection errors, HTTP 408/409/425/429/5xx, malformed
        2xx bodies) are retried up to ``max_retries`` times. Rate-limit and backoff
        sleeps happen outside the concurrency semaphore.

        Args:
            messages: OpenAI-style chat messages.

        Returns:
            The judge's reply text.

        Raises:
            JudgeError: When the request fails for good. ``rate_limited`` is set
                when every attempt was answered with HTTP 429.
        """
        body = self._request_body(messages)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        attempts = self.max_retries + 1
        n_rate_limited = 0
        status: int | None = None
        reason = ""
        for attempt in range(attempts):
            if self._limiter is not None:
                delay = self._limiter.reserve()
                if delay > 0:
                    self._sleep(delay)
            response_headers: collections.abc.Mapping[str, str] = {}
            with self._semaphore:
                try:
                    status, response_headers, response_body = self._transport(self.url, headers, body, self.timeout)
                except Exception as exc:  # connection error / timeout: transient
                    status, response_body, reason = None, b"", f"{type(exc).__name__}: {exc}"
            if status is not None and 200 <= status < 300:
                try:
                    return self._extract_content(response_body)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    reason = f"malformed response body: {type(exc).__name__}"
            elif status is not None:
                reason = f"HTTP {status}"
                if status == 429:
                    n_rate_limited += 1
                elif status not in _RETRYABLE_STATUSES and status < 500:
                    raise JudgeError(f"Judge request rejected: {reason}", status=status)
            if attempt + 1 < attempts:
                self._sleep(self._retry_delay(attempt, response_headers))
        raise JudgeError(
            f"Judge request failed after {attempts} attempt(s): {reason}",
            status=status,
            rate_limited=n_rate_limited == attempts,
        )


def parse_judge_json(text: str) -> dict[str, tp.Any]:
    """Strictly parse one JSON object from a judge reply.

    Accepts the bare object or a single fenced ```` ```json ```` block, and also
    tolerates prose around exactly one top-level ``{...}`` span. Anything else
    (no object, a non-object JSON value, invalid JSON) raises.

    Raises:
        ValueError: If no JSON object can be parsed.
    """
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge reply contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge reply is not a JSON object")
    return value


def _score_0_10(value: tp.Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"`{field}` must be a number")
    if not 0 <= value <= 10:
        raise ValueError(f"`{field}` must be in [0, 10]")
    return float(value)


def _join_text(value: tp.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [m.get("content", "") if isinstance(m, dict) else m for m in value]
        return "\n".join(p if isinstance(p, str) else str(p) for p in parts if p)
    return str(value)


RUBRIC_SYSTEM_PROMPT = "You are a strict, impartial grader. Reply with a single JSON object and nothing else."

RUBRIC_TEMPLATE = """Grade the response to the task below.

[Task]
{prompt}
{reference_block}
[Response]
{response}

Score each criterion from 0 (worst) to 10 (best):
- correctness: is the response correct and does it solve the task?
{criteria_block}
Set "fatal_issue" to true if the response is unsafe, refuses a benign task, is in the wrong language, or is otherwise \
unusable regardless of its other qualities.

Reply with exactly this JSON object:
{{"correctness": <0-10>, {criteria_json}"fatal_issue": <true|false>}}"""

RUBRIC_BATCH_TEMPLATE = """Grade each candidate response to the task below independently.

[Task]
{prompt}
{reference_block}
{candidates_block}
Score each criterion from 0 (worst) to 10 (best):
- correctness: is the response correct and does it solve the task?
{criteria_block}
Set "fatal_issue" to true if a response is unsafe, refuses a benign task, is in the wrong language, or is otherwise \
unusable regardless of its other qualities.

Reply with exactly one JSON object containing one entry per candidate, using the candidate indices shown above:
{{"scores": [{{"index": <candidate index>, "correctness": <0-10>, {criteria_json}"fatal_issue": <true|false>}}, ...]}}"""


class RubricJudge:
    """Pointwise rubric judge returning scores in ``[0, 1]``.

    The judge replies with ``correctness`` and each secondary criterion on
    ``0-10`` plus a boolean ``fatal_issue``. The score is::

        lexicographic_blend(correctness, secondaries, weights, primary_scale=1, secondary_max=10, band=band)
        / (10 + band)

    so secondaries break ties but never outweigh one correctness point, and
    ``fatal_issue`` forces ``0``. Malformed or failed judgments yield ``None``.

    Args:
        client: The judge client.
        criteria: Mapping or sequence of secondary criteria. A mapping maps the
            JSON field name to its description; a sequence uses the names as
            descriptions.
        weights: Optional weights of the secondary criteria.
        band: Secondary band width as a fraction of one correctness point, in ``[0, 1)``.
        template: Pointwise prompt template.
        batch_template: Batch prompt template.
        system_prompt: System message sent with each request.
        max_parse_retries: Re-asks after a reply that fails strict parsing.
        max_batch_retries: Re-asks after a batch reply whose indices are not a bijection.
    """

    def __init__(
        self,
        client: JudgeClient,
        *,
        criteria: collections.abc.Mapping[str, str] | collections.abc.Sequence[str] = (
            ("clarity", "is the response clear and well organized?"),
            ("completeness", "does the response address every part of the task?"),
        ),
        weights: collections.abc.Sequence[float] | None = None,
        band: float = 0.5,
        template: str = RUBRIC_TEMPLATE,
        batch_template: str = RUBRIC_BATCH_TEMPLATE,
        system_prompt: str = RUBRIC_SYSTEM_PROMPT,
        max_parse_retries: int = 1,
        max_batch_retries: int = 1,
    ):
        if isinstance(criteria, collections.abc.Mapping):
            items = list(criteria.items())
        else:
            items = [(c, c) if isinstance(c, str) else (str(c[0]), str(c[1])) for c in criteria]
        names = [name for name, _ in items]
        if "correctness" in names or "fatal_issue" in names or "index" in names or len(set(names)) != len(names):
            raise ValueError(f"Invalid secondary criteria names: {names}.")
        if weights is not None and len(weights) != len(names):
            raise ValueError(f"Got {len(weights)} weights for {len(names)} criteria.")
        lexicographic_blend(0.0, [0.0] * len(names), weights, secondary_max=10.0, band=band)  # validate
        self.client = client
        self.criteria = tuple(names)
        self._criteria_block = "".join(f"- {name}: {desc}\n" for name, desc in items)
        self._criteria_json = "".join(f'"{name}": <0-10>, ' for name in names)
        self.weights = tuple(float(w) for w in weights) if weights is not None else None
        self.band = float(band)
        self.template = template
        self.batch_template = batch_template
        self.system_prompt = system_prompt
        self.max_parse_retries = int(max_parse_retries)
        self.max_batch_retries = int(max_batch_retries)

    def score_from_judgment(self, judgment: collections.abc.Mapping[str, tp.Any]) -> float:
        """Turn one parsed rubric object into a score in ``[0, 1]``.

        Raises:
            ValueError: If a field is missing or out of range.
        """
        fatal = judgment.get("fatal_issue")
        if not isinstance(fatal, bool):
            raise ValueError("`fatal_issue` must be a boolean")
        correctness = _score_0_10(judgment.get("correctness"), "correctness")
        secondaries = [_score_0_10(judgment.get(name), name) for name in self.criteria]
        if fatal:
            return 0.0
        blended = lexicographic_blend(
            correctness, secondaries, self.weights, primary_scale=1.0, secondary_max=10.0, band=self.band
        )
        return blended / (10.0 + self.band)

    @staticmethod
    def _reference_block(reference: str | None) -> str:
        return f"\n[Reference answer]\n{reference}\n" if reference else ""

    def _messages(self, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def score(self, prompt: tp.Any, response: tp.Any, reference: str | None = None) -> float | None:
        """Score one response; ``None`` when the judge fails or never returns a valid rubric."""
        user = self.template.format(
            prompt=_join_text(prompt),
            response=_join_text(response),
            reference_block=self._reference_block(reference),
            criteria_block=self._criteria_block,
            criteria_json=self._criteria_json,
        )
        for _ in range(self.max_parse_retries + 1):
            try:
                reply = self.client.chat(self._messages(user))
            except JudgeError as exc:
                logger.debug(f"Rubric judge call failed: {exc}")
                return None
            try:
                return self.score_from_judgment(parse_judge_json(reply))
            except ValueError:
                continue
        return None

    def _parse_batch(self, reply: str, n: int) -> list[float]:
        """Parse a batch reply; raises unless its indices are an exact bijection with ``range(n)``."""
        entries = parse_judge_json(reply).get("scores")
        if not isinstance(entries, list) or len(entries) != n:
            raise ValueError("batch reply must list exactly one entry per candidate")
        scores: dict[int, float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("batch entry is not an object")
            index = entry.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n or index in scores:
                raise ValueError("batch indices are not a bijection with the candidates")
            scores[index] = self.score_from_judgment(entry)
        return [scores[i] for i in range(n)]

    def score_many(
        self, prompt: tp.Any, responses: collections.abc.Sequence[tp.Any], reference: str | None = None
    ) -> list[float | None]:
        """Score responses one call each, concurrently; one failure never cancels the others."""
        if not responses:
            return []
        workers = min(len(responses), self.client.max_concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edjudge") as pool:
            futures = [pool.submit(self.score, prompt, r, reference) for r in responses]
            results: list[float | None] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.debug(f"Rubric judge worker failed: {type(exc).__name__}")
                    results.append(None)
            return results

    def score_batch(
        self, prompt: tp.Any, responses: collections.abc.Sequence[tp.Any], reference: str | None = None
    ) -> list[float | None]:
        """Score several responses to one prompt in a single judge call.

        The reply must carry one entry per candidate whose indices form an exact
        bijection with the inputs. Otherwise the call is retried up to
        ``max_batch_retries`` times and then falls back to :meth:`score_many`,
        except when the judge is persistently rate limited (every attempt got
        HTTP 429): fanning out then would only amplify the rate limiting, so all
        candidates come back unscorable instead.
        """
        n = len(responses)
        if n == 0:
            return []
        if n == 1:
            return [self.score(prompt, responses[0], reference)]
        candidates = "".join(f"[Candidate {i}]\n{_join_text(r)}\n\n" for i, r in enumerate(responses))
        user = self.batch_template.format(
            prompt=_join_text(prompt),
            reference_block=self._reference_block(reference),
            candidates_block=candidates,
            criteria_block=self._criteria_block,
            criteria_json=self._criteria_json,
        )
        for _ in range(self.max_batch_retries + 1):
            try:
                reply = self.client.chat(self._messages(user))
            except JudgeError as exc:
                if exc.rate_limited:
                    logger.debug("Batch judge persistently rate limited; not fanning out.")
                    return [None] * n
                break
            try:
                return list(self._parse_batch(reply, n))
            except ValueError:
                continue
        return self.score_many(prompt, responses, reference)


PAIRWISE_TEMPLATE = """Compare two responses to the same task and decide which is better overall \
(correctness first, then clarity and completeness).

[Task]
{prompt}
{reference_block}
[Response A]
{response_a}

[Response B]
{response_b}

Reply with exactly this JSON object: {{"winner": "A" | "B" | "tie"}}"""


class PairwiseJudge:
    """Group scorer from pairwise judge comparisons.

    For a group of responses to one prompt it compares pairs, either all
    ``n * (n - 1) / 2`` of them or, when ``max_pairs`` is smaller, a balanced
    schedule of ``max_pairs`` distinct pairs in which every member appears in at
    least one comparison and per-member counts differ by at most one. A
    ``max_pairs`` below ``ceil(n / 2)`` cannot cover every member and raises
    ``ValueError`` when a group of ``n`` is scheduled. A member's score is its win rate
    ``(wins + 0.5 * ties) / completed comparisons`` in ``[0, 1]``. Failed
    comparisons are excluded; a member with no completed comparison, and every
    member of a single-member group, is unscorable (``None``).

    Args:
        client: The judge client.
        max_pairs: Optional cap on comparisons per group.
        swap_positions: Randomly swap A/B presentation per comparison to cancel position bias.
        template: Pairwise prompt template.
        system_prompt: System message sent with each request.
        seed: Seed of the presentation-order RNG.
    """

    def __init__(
        self,
        client: JudgeClient,
        *,
        max_pairs: int | None = None,
        swap_positions: bool = True,
        template: str = PAIRWISE_TEMPLATE,
        system_prompt: str = RUBRIC_SYSTEM_PROMPT,
        seed: int = 0,
    ):
        if max_pairs is not None and int(max_pairs) < 1:
            raise ValueError(f"`max_pairs` must be >= 1; got {max_pairs}.")
        self.client = client
        self.max_pairs = None if max_pairs is None else int(max_pairs)
        self.swap_positions = bool(swap_positions)
        self.template = template
        self.system_prompt = system_prompt
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()

    def pairs(self, n: int) -> list[tuple[int, int]]:
        """Return the comparison schedule ``[(i, j), ...]`` with ``i < j`` for a group of ``n``."""
        total = n * (n - 1) // 2
        if self.max_pairs is None or self.max_pairs >= total:
            return list(itertools.combinations(range(n), 2))
        if self.max_pairs < math.ceil(n / 2):
            raise ValueError(
                f"`max_pairs={self.max_pairs}` cannot give each of {n} group members a comparison; "
                f"it must be at least ceil(n / 2) = {math.ceil(n / 2)}."
            )
        # Near-regular target degrees (max - min <= 1, every member >= 1) realised as a simple graph with the
        # Havel-Hakimi construction; such a degree sequence is always graphical, so the loop never runs short.
        q, r = divmod(2 * self.max_pairs, n)
        remaining = [q + 1 if i < r else q for i in range(n)]
        schedule: list[tuple[int, int]] = []
        while True:
            v = max(range(n), key=lambda i: (remaining[i], -i))
            if remaining[v] == 0:
                break
            partners = sorted((u for u in range(n) if u != v and remaining[u] > 0), key=lambda u: (-remaining[u], u))
            partners = partners[: remaining[v]]
            for u in partners:
                remaining[u] -= 1
                schedule.append((min(u, v), max(u, v)))
            remaining[v] = 0
        return sorted(schedule)

    def compare(self, prompt: tp.Any, first: tp.Any, second: tp.Any, reference: str | None = None) -> float | None:
        """Return ``first``'s outcome against ``second``: ``1`` win, ``0.5`` tie, ``0`` loss, ``None`` on failure."""
        with self._rng_lock:
            swapped = self.swap_positions and self._rng.random() < 0.5
        a, b = (second, first) if swapped else (first, second)
        user = self.template.format(
            prompt=_join_text(prompt),
            reference_block=f"\n[Reference answer]\n{reference}\n" if reference else "",
            response_a=_join_text(a),
            response_b=_join_text(b),
        )
        try:
            reply = self.client.chat(
                [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
            )
            winner = parse_judge_json(reply).get("winner")
        except (JudgeError, ValueError):
            return None
        if not isinstance(winner, str) or winner.strip().lower() not in {"a", "b", "tie"}:
            return None
        winner = winner.strip().lower()
        if winner == "tie":
            return 0.5
        a_wins = winner == "a"
        return float(a_wins != swapped)

    def score_group(
        self, prompt: tp.Any, responses: collections.abc.Sequence[tp.Any], reference: str | None = None
    ) -> list[float | None]:
        """Score a group by pairwise win rate; single-member groups are unscorable."""
        n = len(responses)
        if n < 2:
            return [None] * n
        schedule = self.pairs(n)
        workers = min(len(schedule), self.client.max_concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edjudge-pair") as pool:
            futures = [pool.submit(self.compare, prompt, responses[i], responses[j], reference) for i, j in schedule]
            outcomes: list[float | None] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except Exception:
                    outcomes.append(None)
        points = [0.0] * n
        games = [0] * n
        for (i, j), outcome in zip(schedule, outcomes, strict=True):
            if outcome is None:
                continue
            points[i] += outcome
            points[j] += 1.0 - outcome
            games[i] += 1
            games[j] += 1
        return [points[k] / games[k] if games[k] else None for k in range(n)]


@register_environment("llm_judge")
class LLMJudgeEnvironment(AgenticEnvironment):
    """Single-turn environment whose reward is an LLM rubric judgment.

    ``reset`` returns the prompt; ``step`` scores the action with a
    :class:`RubricJudge` and terminates. When the judge cannot score the
    action the step carries ``info["env_error"] = True`` and
    ``info["unscorable"] = True`` (the reward field is a placeholder ``0.0``
    that collectors must not train on).

    Args:
        prompt: Task prompt shown to the policy.
        reference: Optional reference answer shown to the judge.
        judge: A ready :class:`RubricJudge`. When omitted, one is built from
            ``base_url``/``model`` and ``client_kwargs``.
        base_url: Judge endpoint base URL (used when ``judge`` is omitted).
        model: Judge model name (used when ``judge`` is omitted).
        criteria: Secondary criteria for a built judge.
        client_kwargs: Extra :class:`JudgeClient` arguments for a built judge.
        problem_statement: Prompt alias filled in by :meth:`AutoEnvironment.from_task`;
            used only when ``prompt`` is empty.
        task_prompt: Prompt alias filled in by :meth:`AutoEnvironment.from_task`;
            used only when ``prompt`` and ``problem_statement`` are empty.

    The constructor has no ``**kwargs`` on purpose: :class:`AutoEnvironment` and
    :func:`make_environment_reward_fn` filter dataset-row keys against this
    signature, so unrelated columns are dropped instead of reaching the client.
    """

    def __init__(
        self,
        prompt: str = "",
        *,
        reference: str | None = None,
        judge: RubricJudge | None = None,
        base_url: str | None = None,
        model: str | None = None,
        criteria: collections.abc.Mapping[str, str] | collections.abc.Sequence[str] | None = None,
        client_kwargs: collections.abc.Mapping[str, tp.Any] | None = None,
        problem_statement: str | None = None,
        task_prompt: str | None = None,
    ):
        if judge is None:
            if not base_url or not model:
                raise ValueError("LLMJudgeEnvironment needs `judge` or both `base_url` and `model`.")
            client = JudgeClient(base_url, model, **dict(client_kwargs or {}))
            judge = RubricJudge(client) if criteria is None else RubricJudge(client, criteria=criteria)
        elif client_kwargs or base_url or model:
            raise ValueError("Pass either a ready `judge` or client settings, not both.")
        if not prompt:
            prompt = problem_statement or task_prompt or ""
        self._prompt = prompt
        self._reference = reference
        self.judge = judge

    @property
    def max_steps(self) -> int | None:
        return 1

    def reset(self, seed: int | None = None) -> ResetResult:
        return ResetResult(observation=self._prompt, info={"env_error": False})

    def step(self, action: str) -> StepResult:
        score = self.judge.score(self._prompt, action, self._reference)
        if score is None:
            return StepResult(
                observation="",
                reward=0.0,
                terminated=True,
                info={"env_error": True, "unscorable": True, "env_error_message": "LLM judge could not score."},
            )
        return StepResult(observation="", reward=score, terminated=True, info={"env_error": False, "judge_score": score})


def _group_slices(prompt_keys: list[str], num_generations: int | None) -> list[range]:
    n = len(prompt_keys)
    if num_generations is not None:
        if num_generations < 1 or n % num_generations:
            raise ValueError(f"{n} completions do not form groups of num_generations={num_generations}.")
        return [range(s, s + num_generations) for s in range(0, n, num_generations)]
    groups: list[range] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or prompt_keys[i] != prompt_keys[start]:
            groups.append(range(start, i))
            start = i
    return groups


def _find_references(
    kwargs: collections.abc.Mapping[str, tp.Any], reference_key: str | None, n: int
) -> list[tp.Any] | None:
    """Return the ``n``-long reference column from the top-level kwargs or the trainer's ``batch`` dict."""
    if not reference_key:
        return None
    candidates = [kwargs.get(reference_key)]
    batch_payload = kwargs.get("batch")
    if isinstance(batch_payload, collections.abc.Mapping):
        candidates.append(batch_payload.get(reference_key))
    for value in candidates:
        if value is None or isinstance(value, (str, bytes)):
            continue
        if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()  # numpy column
        if isinstance(value, (list, tuple)) and len(value) == n:
            return list(value)
    return None


def make_judge_reward_fn(
    judge: RubricJudge | PairwiseJudge,
    *,
    num_generations: int | None = None,
    batch: bool = True,
    reference_key: str | None = "reference",
    max_workers: int = 8,
    name: str = "llm_judge",
) -> tp.Callable[..., list[float | None]]:
    """Build a GRPO ``reward_funcs`` callable backed by an LLM judge.

    Completions arrive prompt-major; groups are consecutive blocks of
    ``num_generations`` (or, when ``None``, runs of identical prompts). Each
    group is judged concurrently: a :class:`RubricJudge` uses
    :meth:`RubricJudge.score_batch` (``batch=True``) or one call per completion;
    a :class:`PairwiseJudge` uses :meth:`PairwiseJudge.score_group`. Unscorable
    samples are ``None``, never ``0``.

    Args:
        judge: The rubric or pairwise judge.
        num_generations: Group size; ``None`` infers groups from prompt runs.
        batch: Rubric mode only: judge a whole group per call.
        reference_key: Field holding per-completion reference answers. It is read
            from the top-level kwargs first, then from the trainer's ``batch``
            dict (where GRPO/PPO put dataset columns). When no ``n``-long column
            is found, judging proceeds without references and a warning is logged
            once. ``None`` disables references.
        max_workers: Groups judged in parallel.
        name: Metric name of the reward function.

    Returns:
        A reward callable ``(prompts, completions, **kwargs) -> list[float | None]``.
        It carries ``discard_unscorable = True`` so trainers that honour the
        marker drop groups containing ``None`` instead of training on them.
    """
    warned = [False]

    def _reward_fn(
        *,
        prompts: collections.abc.Sequence[tp.Any] | None = None,
        completions: collections.abc.Sequence[tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float | None]:
        if not completions:
            return []
        n = len(completions)
        prompt_list = list(prompts) if prompts is not None else [None] * n
        if len(prompt_list) != n:
            raise ValueError(f"Got {len(prompt_list)} prompts for {n} completions.")
        references = _find_references(kwargs, reference_key, n)
        if references is None:
            if reference_key and not warned[0]:
                warned[0] = True
                logger.warning(
                    f"Judge reward {name!r}: no {n}-long `{reference_key}` column found at top level or in "
                    "`batch`; judging without reference answers."
                )
            references = [None] * n
        groups = _group_slices([_join_text(p) for p in prompt_list], num_generations)
        if isinstance(judge, PairwiseJudge):
            for rows in groups:
                judge.pairs(len(rows))  # a too-small ``max_pairs`` is a config error, not an unscorable group

        def _judge_group(rows: range) -> list[float | None]:
            prompt = prompt_list[rows.start]
            reference = references[rows.start]
            reference = None if reference is None else str(reference)
            responses = [completions[i] for i in rows]
            if isinstance(judge, PairwiseJudge):
                return judge.score_group(prompt, responses, reference)
            if batch:
                return judge.score_batch(prompt, responses, reference)
            return judge.score_many(prompt, responses, reference)

        scores: list[float | None] = [None] * n
        workers = max(1, min(int(max_workers), len(groups)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edjudge-group") as pool:
            futures = [(rows, pool.submit(_judge_group, rows)) for rows in groups]
            for rows, future in futures:
                try:
                    group_scores = future.result()
                except Exception as exc:
                    logger.debug(f"Judge group failed: {type(exc).__name__}")
                    continue
                for i, value in zip(rows, group_scores, strict=True):
                    scores[i] = value
        return scores

    _reward_fn.__name__ = name
    setattr(_reward_fn, "discard_unscorable", True)  # noqa: B010 - opt-in marker read by GRPO
    return _reward_fn

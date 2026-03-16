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

"""Self-play environment and question generator for agentic MoshPit.

This module provides a self-play training loop where one agent (or API)
generates questions on a given topic, the solver model tries to answer
them, and optionally the same or a different agent verifies the answer.

Three question-generation backends are supported:

- **Local model**: Uses the trainer's own model (or a separate EasyDeL
  model) with a questioner system prompt. The model is called via
  ``generate_fn`` which is injected by the trainer at rollout time.
- **OpenAI-compatible API**: Calls any OpenAI-compatible endpoint
  (OpenAI, Anthropic via proxy, vLLM, etc.) to generate questions
  and optionally verify answers.
- **Static callable**: Any ``(topic, seed) -> question`` function.

The ``SelfPlayEnvironment`` is a standard ``AgenticEnvironment`` that
plugs directly into ``AgenticMoshPitTrainer`` — no new trainer needed.

Example:
    >>> gen = LocalQuestionGenerator(
    ...     questioner_system_prompt="Generate a hard calculus problem.",
    ...     verifier_system_prompt="Check if the answer is correct.",
    ... )
    >>> env = SelfPlayEnvironment(topic="calculus", generator=gen)
    >>> trainer = AgenticMoshPitTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     env_factory=lambda: SelfPlayEnvironment(topic="calculus", generator=gen),
    ...     processing_class=tokenizer,
    ... )
"""

from __future__ import annotations

import hashlib
import re
import typing as tp
from dataclasses import dataclass, field

from .environment import AgenticEnvironment, ResetResult, StepResult


def _deterministic_variety(seed: int | None, index: int = 0) -> int:
    """Generate a deterministic variety integer from seed and index.

    Uses SHA-256 hashing so the result is identical across all JAX
    devices/hosts for the same ``(seed, index)`` pair — no Python
    ``random`` module involved.
    """
    raw = f"{seed}:{index}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1_000_000


@dataclass
class GeneratedQuestion:
    """A question produced by a QuestionGenerator.

    Attributes:
        question: The question text presented to the solver.
        metadata: Arbitrary metadata (ground truth, difficulty, topic, etc.).
    """

    question: str
    metadata: dict[str, tp.Any] = field(default_factory=dict)


class QuestionGenerator:
    """Abstract interface for question generation backends.

    Subclass this to create custom question generators. The two
    core methods are ``generate`` (produce a question) and
    ``verify`` (score a solver's answer). Both have batched
    variants (``generate_batch`` / ``verify_batch``) that default
    to looping over the single-item methods but can be overridden
    for backends that support true batching (e.g. local LLM).

    If ``verify`` is not overridden it always returns 0.0.
    """

    def generate(self, topic: str, seed: int | None = None) -> GeneratedQuestion:
        """Generate a single question on the given topic.

        Args:
            topic: The subject area or instructions for question generation.
            seed: Optional seed for deterministic generation.

        Returns:
            A ``GeneratedQuestion`` with the question text and metadata.
        """
        raise NotImplementedError

    def generate_batch(
        self,
        topics: list[str],
        seeds: list[int | None],
    ) -> list[GeneratedQuestion]:
        """Generate questions for multiple topics/seeds in one batched call.

        The default implementation loops over ``generate()`` sequentially.
        Override this for backends that can batch (e.g. local LLM).

        Args:
            topics: List of topic strings, one per environment.
            seeds: List of seeds, one per environment.

        Returns:
            List of ``GeneratedQuestion``, same length as inputs.
        """
        return [self.generate(t, s) for t, s in zip(topics, seeds, strict=True)]

    def verify(
        self,
        question: str,
        answer: str,
        metadata: dict[str, tp.Any] | None = None,
    ) -> float:
        """Verify a solver's answer and return a reward.

        Args:
            question: The original question.
            answer: The solver's final answer text.
            metadata: Metadata from ``GeneratedQuestion.metadata``.

        Returns:
            Reward float, typically in ``[0, 1]``.
        """
        return 0.0

    def verify_batch(
        self,
        questions: list[str],
        answers: list[str],
        metadatas: list[dict[str, tp.Any] | None],
    ) -> list[float]:
        """Verify multiple answers in one batched call.

        The default implementation loops over ``verify()`` sequentially.
        Override this for backends that can batch.

        Args:
            questions: List of original questions.
            answers: List of solver answers.
            metadatas: List of metadata dicts.

        Returns:
            List of reward floats, same length as inputs.
        """
        return [self.verify(q, a, m) for q, a, m in zip(questions, answers, metadatas, strict=True)]

    def set_generate_fn(self, generate_fn: tp.Callable[[list[str]], list[str]]) -> None:
        """Inject the batched generation function from the trainer.

        Called by ``SelfPlayEnvironment`` before rollouts start.
        Only used by ``LocalQuestionGenerator``; other backends
        ignore this.
        """


class LocalQuestionGenerator(QuestionGenerator):
    """Generate questions and verify answers using a local LLM.

    Uses the trainer's own ``generate_fn`` (injected at rollout time)
    to call the model with a questioner system prompt. Supports
    separate sampling parameters for question generation and
    verification (e.g. high temperature for creative questions,
    low temperature for deterministic grading).

    Args:
        questioner_system_prompt: System prompt for question generation.
        verifier_system_prompt: System prompt for answer verification.
            If ``None``, the questioner prompt is reused.
        tokenizer: Tokenizer for formatting chat templates.
        tool_schemas: Optional tool schemas passed to chat template.
        questioner_temperature: Sampling temperature for question
            generation. Higher values produce more diverse questions.
        questioner_top_p: Top-p for question generation.
        questioner_top_k: Top-k for question generation.
        verifier_temperature: Sampling temperature for verification.
            Lower values produce more deterministic scores.
        verifier_top_p: Top-p for verification.
        verifier_top_k: Top-k for verification.
    """

    def __init__(
        self,
        questioner_system_prompt: str,
        verifier_system_prompt: str | None = None,
        tokenizer: tp.Any | None = None,
        tool_schemas: list[dict[str, tp.Any]] | None = None,
        questioner_temperature: float = 0.9,
        questioner_top_p: float = 0.95,
        questioner_top_k: int | None = None,
        verifier_temperature: float = 0.1,
        verifier_top_p: float = 0.95,
        verifier_top_k: int | None = None,
        strip_reasoning_fn: tp.Callable[[str], str] | None = None,
    ):
        self._questioner_prompt = questioner_system_prompt
        self._verifier_prompt = verifier_system_prompt
        self._tokenizer = tokenizer
        self._tool_schemas = tool_schemas
        self._q_temperature = questioner_temperature
        self._q_top_p = questioner_top_p
        self._q_top_k = questioner_top_k
        self._v_temperature = verifier_temperature
        self._v_top_p = verifier_top_p
        self._v_top_k = verifier_top_k
        self._generate_fn: tp.Callable[..., list[str]] | None = None
        if strip_reasoning_fn is None:
            _closed = re.compile(r"<think>.*?</think>", re.DOTALL)
            _unclosed = re.compile(r"<think>.*", re.DOTALL)

            def strip_reasoning_fn(text: str) -> str:
                return _unclosed.sub("", _closed.sub("", text)).strip()

        self._strip_reasoning = strip_reasoning_fn

    def set_generate_fn(self, generate_fn: tp.Callable[[list[str]], list[str]]) -> None:
        self._generate_fn = generate_fn

    def generate(self, topic: str, seed: int | None = None) -> GeneratedQuestion:
        return self.generate_batch([topic], [seed])[0]

    def generate_batch(
        self,
        topics: list[str],
        seeds: list[int | None],
    ) -> list[GeneratedQuestion]:
        """Batch-generate questions with a single ``generate_fn`` call."""
        if self._generate_fn is None:
            raise RuntimeError(
                "generate_fn not set. LocalQuestionGenerator must be used "
                "with SelfPlayEnvironment inside AgenticMoshPitTrainer."
            )

        prompts = []
        varieties = []
        for idx, (topic, seed) in enumerate(zip(topics, seeds, strict=True)):
            variety = _deterministic_variety(seed, idx)
            varieties.append(variety)
            user_msg = (
                f"Topic: {topic}\n"
                f"Variation seed: {variety}\n\n"
                "Generate a single challenging question on this topic. "
                "Output ONLY the question, nothing else."
            )
            prompts.append(self._format_prompt(self._questioner_prompt, user_msg, enable_thinking=False))

        responses = self._generate_fn(
            prompts,
            temperature=self._q_temperature,
            top_p=self._q_top_p,
            top_k=self._q_top_k,
            num_return_sequences=1,
            strip_thinking=True,
        )

        return [
            GeneratedQuestion(
                question=resp.strip(),
                metadata={"topic": t, "seed": s, "variety": v},
            )
            for resp, t, s, v in zip(responses, topics, seeds, varieties, strict=True)
        ]

    def verify(
        self,
        question: str,
        answer: str,
        metadata: dict[str, tp.Any] | None = None,
    ) -> float:
        return self.verify_batch([question], [answer], [metadata])[0]

    def verify_batch(
        self,
        questions: list[str],
        answers: list[str],
        metadatas: list[dict[str, tp.Any] | None],
    ) -> list[float]:
        """Batch-verify answers with a single ``generate_fn`` call."""
        if self._generate_fn is None:
            return [0.0] * len(questions)

        sys_prompt = self._verifier_prompt or self._questioner_prompt
        max_answer_chars = 2000
        prompts = []
        for question, answer in zip(questions, answers, strict=True):
            clean_answer = self._strip_reasoning(answer)
            if len(clean_answer) > max_answer_chars:
                clean_answer = clean_answer[:max_answer_chars] + "\n... [truncated]"
            user_msg = (
                f"Question: {question}\n\n"
                f"Student's answer:\n{clean_answer}\n\n"
                "Grade the student's answer. Think through whether the reasoning and "
                "final result are correct. Then output your score as an integer from "
                "0 to 10 inside <reward></reward> tags.\n"
                "Example: <reward>7</reward>"
            )
            prompts.append(self._format_prompt(sys_prompt, user_msg, enable_thinking=False))

        responses = self._generate_fn(
            prompts,
            temperature=self._v_temperature,
            top_p=self._v_top_p,
            top_k=self._v_top_k,
            num_return_sequences=1,
            strip_thinking=False,
        )
        self._last_verify_responses = responses
        return [self._parse_score(r) for r in responses]

    def _format_prompt(
        self,
        system: str,
        user: str,
        enable_thinking: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self._tokenizer is None:
            raise RuntimeError(
                "LocalQuestionGenerator requires a tokenizer for chat template formatting. "
                "Either pass one explicitly or let the trainer inject it automatically."
            )
        kwargs: dict[str, tp.Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": self._tool_schemas,
        }
        if not enable_thinking:
            try:
                return self._tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                pass
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _parse_score(text: str) -> float:
        import re

        tag_match = re.search(r"<reward>\s*(\d+(?:\.\d+)?)\s*</reward>", text)
        if tag_match:
            return min(max(float(tag_match.group(1)) / 10.0, 0.0), 1.0)
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
        if numbers:
            return min(max(float(numbers[-1]) / 10.0, 0.0), 1.0)
        return 0.0


class OpenAIQuestionGenerator(QuestionGenerator):
    """Generate questions and verify answers via an OpenAI-compatible API.

    Works with OpenAI, Azure OpenAI, vLLM, Ollama, or any endpoint
    that implements the ``/v1/chat/completions`` API.

    Args:
        base_url: API base URL (e.g. ``"https://api.openai.com/v1"``).
        api_key: API key for authentication.
        model: Model name to use (e.g. ``"gpt-4o"``).
        questioner_system_prompt: System prompt for question generation.
        verifier_system_prompt: System prompt for verification.
            If ``None``, the questioner acts as verifier.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens for generated responses.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        questioner_system_prompt: str,
        verifier_system_prompt: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._questioner_prompt = questioner_system_prompt
        self._verifier_prompt = verifier_system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def generate(self, topic: str, seed: int | None = None) -> GeneratedQuestion:
        variety = _deterministic_variety(seed)
        user_msg = (
            f"Topic: {topic}\n"
            f"Variation seed: {variety}\n\n"
            "Generate a single challenging question on this topic. "
            "Output ONLY the question, nothing else."
        )
        response = self._chat(self._questioner_prompt, user_msg)
        return GeneratedQuestion(
            question=response.strip(),
            metadata={"topic": topic, "seed": seed, "variety": variety},
        )

    def verify(
        self,
        question: str,
        answer: str,
        metadata: dict[str, tp.Any] | None = None,
    ) -> float:
        sys_prompt = self._verifier_prompt or self._questioner_prompt
        user_msg = (
            f"Question: {question}\n\n"
            f"Student's answer: {answer}\n\n"
            "Is this answer correct? Reply with ONLY a score from 0 to 10, "
            "where 0 is completely wrong and 10 is perfectly correct."
        )
        response = self._chat(sys_prompt, user_msg)
        return LocalQuestionGenerator._parse_score(response)

    def _chat(self, system: str, user: str) -> str:
        import json
        import urllib.request

        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


class CallableQuestionGenerator(QuestionGenerator):
    """Wrap plain functions as a QuestionGenerator.

    Args:
        generate_fn: ``(topic, seed) -> question_text`` callable.
        verify_fn: Optional ``(question, answer, metadata) -> reward``
            callable. If ``None``, verification always returns 0.0.
    """

    def __init__(
        self,
        generate_fn: tp.Callable[[str, int | None], str],
        verify_fn: tp.Callable[[str, str, dict[str, tp.Any] | None], float] | None = None,
    ):
        self._gen_fn = generate_fn
        self._ver_fn = verify_fn

    def generate(self, topic: str, seed: int | None = None) -> GeneratedQuestion:
        return GeneratedQuestion(question=self._gen_fn(topic, seed))

    def verify(
        self,
        question: str,
        answer: str,
        metadata: dict[str, tp.Any] | None = None,
    ) -> float:
        if self._ver_fn is not None:
            return self._ver_fn(question, answer, metadata)
        return 0.0


class SelfPlayEnvironment(AgenticEnvironment):
    """Self-play environment where questions are generated by an LLM.

    On ``reset``, the questioner generates a new question on the
    configured topic. The solver interacts over multiple turns
    (optionally using tools). On the final ``step`` (when
    terminated), the verifier scores the solver's accumulated
    response and returns a reward.

    If ``verify=False``, the environment never terminates on its
    own — it runs for ``max_steps`` turns and returns 0.0 reward,
    making it a pure generation/exploration run.

    Args:
        topic: Topic string or detailed instructions passed to the
            questioner for generating questions.
        generator: A ``QuestionGenerator`` instance (local model,
            OpenAI API, or callable).
        verify: Whether to verify answers and return rewards.
            If ``False``, reward is always 0.0 and the environment
            never self-terminates.
        answer_pattern: Optional regex pattern to detect when the
            solver has provided a final answer. If ``None``, every
            non-tool response is treated as a potential final answer
            submitted for verification after ``max_steps``.
        max_steps_override: Override the environment's max_steps.
            If ``None``, defaults to 5.
    """

    def __init__(
        self,
        topic: str,
        generator: QuestionGenerator,
        verify: bool = True,
        answer_pattern: str | None = None,
        max_steps_override: int | None = None,
    ):
        self._topic = topic
        self._generator = generator
        self._verify = verify
        self._answer_pattern = answer_pattern
        self._max_steps_override = max_steps_override or 5
        self._question: str = ""
        self._metadata: dict[str, tp.Any] = {}
        self._solver_turns: list[str] = []
        self._step_count = 0
        self._defer_verify: bool = False
        self._final_answer: str | None = None

    def reset(self, seed: int | None = None) -> ResetResult:
        """Reset the environment.

        If ``reset_with_question`` was called beforehand (batched path),
        the pre-generated question is used. Otherwise falls back to
        calling ``generator.generate()`` directly (single-env path).
        """
        if self._question and self._step_count == 0:
            return ResetResult(
                observation=self._question,
                info={"question": self._question, "topic": self._topic, **self._metadata},
            )

        result = self._generator.generate(self._topic, seed)
        self._question = result.question
        self._metadata = result.metadata
        self._solver_turns = []
        self._step_count = 0
        return ResetResult(
            observation=self._question,
            info={"question": self._question, "topic": self._topic, **self._metadata},
        )

    def reset_with_question(self, question: GeneratedQuestion) -> ResetResult:
        """Reset with a pre-generated question (used by batched rollout)."""
        self._question = question.question
        self._metadata = question.metadata
        self._solver_turns = []
        self._step_count = 0
        return ResetResult(
            observation=self._question,
            info={"question": self._question, "topic": self._topic, **self._metadata},
        )

    def step(self, action: str) -> StepResult:
        self._solver_turns.append(action)
        self._step_count += 1

        if self._answer_pattern:
            import re

            if re.search(self._answer_pattern, action):
                self._final_answer = action
                if self._defer_verify:
                    return StepResult(observation="", reward=float("nan"), terminated=True, info={})
                reward = self._do_verify(action)
                return StepResult(
                    observation="",
                    reward=reward,
                    terminated=True,
                    info=self._make_info(reward, action),
                )

        if self._step_count >= self._max_steps_override:
            full_answer = "\n".join(self._solver_turns)
            self._final_answer = full_answer
            if self._defer_verify:
                return StepResult(observation="", reward=float("nan"), terminated=True, info={})
            reward = self._do_verify(full_answer)
            return StepResult(
                observation="",
                reward=reward,
                terminated=True,
                info=self._make_info(reward, full_answer),
            )

        return StepResult(
            observation="Continue working on the problem.",
            reward=0.0,
            terminated=False,
            info={},
        )

    def set_generate_fn(self, generate_fn: tp.Callable[[list[str]], list[str]]) -> None:
        """Propagate the trainer's generate_fn to the question generator."""
        self._generator.set_generate_fn(generate_fn)

    def _do_verify(self, answer: str) -> float:
        if not self._verify:
            return 0.0
        return self._generator.verify(self._question, answer, self._metadata)

    def _make_info(self, reward: float, answer: str) -> dict[str, tp.Any]:
        return {
            "question": self._question,
            "answer": answer,
            "reward": reward,
            "topic": self._topic,
            "num_turns": len(self._solver_turns),
            "all_turns": list(self._solver_turns),
            **self._metadata,
        }

    @property
    def max_steps(self) -> int:
        return self._max_steps_override

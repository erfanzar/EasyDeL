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

"""Agentic MoshPit trainer smoke test with GSM8K math reasoning.

Uses the GSM8K dataset to train a multi-turn coding agent that can
write and execute Python code to solve grade-school math problems.
The agent gets multiple turns (up to ``max_steps``) where it can
optionally invoke a Python code execution tool, then provide its
final answer.

Environment:
    - Each episode presents a GSM8K question as the initial observation.
    - The agent may write a Python snippet via the ``python_code`` tool
      (hermes-style ``<tool_call>`` format) to compute intermediate
      results. Tool output is returned as an observation.
    - On any turn the agent may also provide a final answer by writing
      ``\\boxed{<number>}`` somewhere in its response.
    - The episode terminates when a ``\\boxed`` answer is detected or
      ``max_steps`` is reached.
    - Binary reward: 1.0 if the extracted number matches the gold
      answer, 0.0 otherwise.

This mirrors Alibaba ROLL's math environment pattern (GEMMathEnv)
with answer extraction via ``\\boxed`` and optional tool-use turns.
"""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

import easydel as ed
from easydel.trainers.agentic_moshpit import (
    AgenticEnvironment,
    PythonCodeTool,
    ResetResult,
    StepResult,
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )
else:
    from ._common import (
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_HASH_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _normalize_number(text: str) -> str | None:
    """Strip commas / whitespace and return a canonical number string."""
    text = text.strip().replace(",", "").replace(" ", "")
    try:
        return str(float(text))
    except ValueError:
        return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the number inside ``\\boxed{...}`` from model output."""
    match = _BOXED_RE.search(text)
    if match:
        return _normalize_number(match.group(1))
    return None


def extract_gold_answer(answer_text: str) -> str | None:
    """Extract the gold numeric answer from a GSM8K answer string.

    GSM8K answers end with ``#### <number>`` after the chain-of-thought.
    """
    match = _HASH_ANSWER_RE.search(answer_text)
    if match:
        return _normalize_number(match.group(1))
    return None


class GSM8KCodeEnvironment(AgenticEnvironment):
    """Multi-turn math reasoning environment backed by GSM8K.

    Each episode:
    1. Presents a GSM8K word problem.
    2. The agent can write Python code via the ``python_code`` tool
       to compute intermediate results (multi-turn).
    3. The agent provides its final answer inside ``\\boxed{}``.
    4. Receives binary reward: 1.0 if correct, 0.0 otherwise.

    If the agent does not produce a ``\\boxed{}`` answer within
    ``max_steps``, it receives 0.0 reward.

    Args:
        questions: List of question strings.
        answers: List of gold answer strings (containing ``#### <number>``).
    """

    def __init__(
        self,
        questions: list[str],
        answers: list[str],
    ):
        self._questions = questions
        self._answers = answers
        self._rng = random.Random()
        self._gold: str | None = None
        self._question: str | None = None
        self._idx: int = 0

    def reset(self, seed: int | None = None) -> ResetResult:
        """Pick a question deterministically from seed."""
        if seed is not None:
            self._rng = random.Random(seed)

        self._idx = self._rng.randint(0, len(self._questions) - 1)
        self._question = self._questions[self._idx]
        self._gold = extract_gold_answer(self._answers[self._idx])

        prompt = (
            "Solve the following math problem step by step.\n"
            "You may use the python_code tool to run computations.\n"
            "When you have the final answer, write it as \\boxed{<number>}.\n"
            "\n"
            f"{self._question}"
        )
        return ResetResult(
            observation=prompt,
            info={
                "question": self._question,
                "gold_answer": self._gold,
                "dataset_idx": self._idx,
            },
        )

    def step(self, action: str) -> StepResult:
        """Check if the agent provided a boxed answer.

        If a ``\\boxed{...}`` is found, compare against gold and
        terminate. If not found, return an empty observation so the
        agent can continue reasoning (tool calls are handled by the
        ToolEnvWrapper, not here).
        """
        predicted = extract_boxed_answer(action)

        if predicted is not None:
            correct = predicted == self._gold
            return StepResult(
                observation="",
                reward=1.0 if correct else 0.0,
                terminated=True,
                info={
                    "correct": correct,
                    "predicted": predicted,
                    "expected": self._gold,
                },
            )

        return StepResult(
            observation=("Please provide your final answer inside \\boxed{<number>}."),
            reward=0.0,
            terminated=False,
            info={},
        )

    @property
    def max_steps(self) -> int:
        return 5


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()

    logger.info("Loading GSM8K dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    questions = gsm8k["question"]
    answers = gsm8k["answer"]
    logger.info("Loaded %d GSM8K problems.", len(questions))

    def env_factory() -> GSM8KCodeEnvironment:
        return GSM8KCodeEnvironment(questions=questions, answers=answers)

    python_tool = PythonCodeTool(timeout=5.0, max_output_length=2048)
    max_length = int(os.environ.get("EASYDEL_MPMD_AGENTIC_MAX_LENGTH", "128"))
    trainer_args = make_config(
        ed.AgenticMoshPitConfig,
        "agentic-moshpit-gsm8k",
        overrides={
            "max_prompt_length": max_length // 2,
            "max_completion_length": max_length // 2,
            "max_length": max_length,
            "max_steps": 1,
            "group_size": 1,
            "num_env_groups": 1,
            "num_return_sequences": 1,
            "generation_num_return_sequences": 1,
            "esurge_max_num_batched_tokens": max_length // 2,
            "reward_mode": "episode",
            "advantage_estimator": "grpo",
            "system_prompt": (
                "You are a helpful math assistant. "
                "Solve the problem step by step. "
                "When you have the final answer, write it as \\boxed{<answer>}."
            ),
            "tool_caller": "hermes",
            "loss_type": "dapo",
            "beta": 0.04,
            "scale_rewards": "group",
        },
    )

    scaffold_dataset = gsm8k.rename_column("question", "prompt").select(range(min(500, len(gsm8k))))

    logger.info("Launching Agentic MoshPit trainer on GSM8K (multi-turn code agent).")
    trainer = ed.AgenticMoshPitTrainer(
        arguments=trainer_args,
        model=model,
        env_factory=env_factory,
        tools=[python_tool],
        train_dataset=scaffold_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("Agentic MoshPit GSM8K run finished.")


if __name__ == "__main__":
    main()

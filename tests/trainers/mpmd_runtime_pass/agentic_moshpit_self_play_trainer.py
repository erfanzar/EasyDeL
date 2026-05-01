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

"""Self-play agentic MoshPit trainer smoke test.

The model plays all three roles:
1. **Questioner**: generates math word problems on a given topic.
2. **Solver**: reasons through the problem (optionally using tools).
3. **Verifier**: scores the solver's answer on a 0-10 scale.

All three use the same underlying model via ``LocalQuestionGenerator``,
which injects the trainer's batched ``generate_fn`` at rollout time.
No dataset is needed — questions come from the model itself.

Trajectory conversations (all turns) are logged to Weights & Biases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import easydel as ed
from easydel.trainers.agentic_moshpit import (
    LocalQuestionGenerator,
    PythonCodeTool,
    SelfPlayEnvironment,
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


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()

    generator = LocalQuestionGenerator(
        questioner_system_prompt=(
            "You are a math teacher who creates challenging word problems. "
            "Given a topic, create a single original math word problem that "
            "requires multi-step reasoning. The problem should have a clear "
            "numeric answer. Output ONLY the problem statement, nothing else."
        ),
        verifier_system_prompt=(
            "You are a math grader. Given a question and a student's answer, "
            "determine if the answer is correct. Consider the reasoning and "
            "the final numeric result. Reply with ONLY a score from 0 to 10."
        ),
        tokenizer=tokenizer,
    )

    topic = "arithmetic and algebra word problems involving money, distances, or time"

    def env_factory() -> SelfPlayEnvironment:
        return SelfPlayEnvironment(
            topic=topic,
            generator=generator,
            verify=True,
            answer_pattern=r"\\boxed\{.+\}",
            max_steps_override=1,
        )

    python_tool = PythonCodeTool(timeout=5.0, max_output_length=2048)

    max_length = int(os.environ.get("EASYDEL_MPMD_AGENTIC_MAX_LENGTH", "128"))
    trainer_args = make_config(
        ed.AgenticMoshPitConfig,
        "agentic-moshpit-self-play",
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
                "You are a math problem solver. "
                "Think step by step and use the python_code tool if needed. "
                "When you have the final answer, write it as \\boxed{<answer>}."
            ),
            "tool_caller": "hermes",
            "loss_type": "dapo",
            "beta": 0.04,
            "scale_rewards": "group",
        },
    )

    logger.info("Launching self-play Agentic MoshPit trainer.")
    logger.info("Topic: %s", topic)
    trainer = ed.AgenticMoshPitTrainer(
        arguments=trainer_args,
        model=model,
        env_factory=env_factory,
        tools=[python_tool],
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("Self-play Agentic MoshPit run finished.")


if __name__ == "__main__":
    main()

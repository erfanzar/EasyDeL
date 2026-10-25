"""Pure-Python coverage of the intentional public rollout adapters."""

from types import SimpleNamespace

import pytest
from easydel.trainers.esurge_rollout import (
    eSurgeRolloutConfig,
    eSurgeRolloutGenerator,
    generate_rollout_completions,
)

pytestmark = pytest.mark.cpu_ok


class Trainer:
    arguments = SimpleNamespace(use_esurge_generation=False)

    def rollout(self, *, prompts, use_esurge, apply_chat_template, config_overrides, release_runtime_after_generation):
        prefix = "chat:" if apply_chat_template else "text:"
        suffix = str(config_overrides["max_new_tokens"])
        return SimpleNamespace(
            generation_results=[prefix + prompt + suffix for prompt in prompts],
            prompt_ids=[[index] for index in range(len(prompts))],
            sequences=[[index, 42] for index in range(len(prompts))],
        )


def test_generator_defaults_and_per_call_overrides_preserve_rows():
    config = eSurgeRolloutConfig(generation_overrides={"max_new_tokens": 3}, as_chat=True)
    generator = eSurgeRolloutGenerator(Trainer(), config)
    assert generator(["a", "b"]) == [
        {"text": "chat:a3", "prompt_ids": [0], "sequences": [0, 42]},
        {"text": "chat:b3", "prompt_ids": [1], "sequences": [1, 42]},
    ]
    assert generator.generate(["c"], generation_overrides={"max_new_tokens": 7}, as_chat=False)[0]["text"] == "text:c7"
    assert generator(["d"])[0]["text"] == "chat:d3"
    assert config.generation_overrides == {"max_new_tokens": 3}


def test_empty_and_invalid_trainer():
    assert generate_rollout_completions(object(), []) == []
    with pytest.raises(TypeError):
        generate_rollout_completions(object(), ["hello"])

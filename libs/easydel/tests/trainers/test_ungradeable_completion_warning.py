"""Detecting rollouts that can never produce a reward signal.

Reward functions score the *parsed* completion text. A reasoning chat template
that prefills ``<think>`` and never closes it within the completion budget
leaves that parsed text empty, so every reward is 0 forever while generation
itself is perfectly healthy. Observed on Qwen3.5-4B SAO: 2,639 characters of
coherent reasoning per rollout, parsed text empty, ``ExecutionReward=0`` at both
1024- and 2048-token budgets, with nothing in the logs explaining why.

The detector must fire on that shape, distinguish it from "the policy emitted
nothing at all", and stay quiet when completions are merely short or partially
empty so it does not become noise that gets ignored.

The warning is asserted through a stub logger rather than ``caplog``: EasyDeL's
``LazyLogger`` does not propagate to the root logger, and its ``warn_once``
dedupes globally, which would make assertions order-dependent across tests.
"""

from __future__ import annotations

import pytest
from easydel.trainers import base_trainer as base_trainer_module
from easydel.trainers.base_trainer import BaseTrainer


class _RecordingLogger:
    """Captures ``warn_once`` calls with their interpolated message."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warn_once(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)

    def __getattr__(self, _name: str):  # any other log level is a no-op here
        return lambda *args, **kwargs: None


@pytest.fixture
def spy(monkeypatch) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(base_trainer_module, "logger", recorder)
    return recorder


def test_all_empty_after_parsing_with_raw_output_is_reported(spy):
    """The observed failure: plenty generated, nothing survived parsing."""
    BaseTrainer._warn_on_ungradeable_completions(
        clean_texts=["", "", "", ""],
        raw_texts=["long reasoning trace"] * 4,
        truncated=[True, True, True, True],
    )
    assert len(spy.messages) == 1
    message = spy.messages[0]
    assert "ungradeable" in message
    assert "4/4" in message
    assert "100%" in message


def test_fully_graded_batch_is_silent(spy):
    BaseTrainer._warn_on_ungradeable_completions(
        clean_texts=["def solve(): pass", "print(1)"],
        raw_texts=["def solve(): pass", "print(1)"],
        truncated=[False, False],
    )
    assert spy.messages == []


def test_minority_of_empty_completions_is_silent(spy):
    """One dud in a healthy batch is normal RL variance, not a config fault."""
    BaseTrainer._warn_on_ungradeable_completions(
        clean_texts=["print(1)", "print(2)", "print(3)", ""],
        raw_texts=["print(1)", "print(2)", "print(3)", ""],
        truncated=[False, False, False, True],
    )
    assert spy.messages == []


def test_whitespace_only_counts_as_empty(spy):
    """A parser that strips everything but a newline is still ungradeable."""
    BaseTrainer._warn_on_ungradeable_completions(
        clean_texts=["   ", "\n", "\t", ""],
        raw_texts=["reasoning"] * 4,
        truncated=[True] * 4,
    )
    assert len(spy.messages) == 1
    assert "ungradeable" in spy.messages[0]


def test_empty_raw_output_reports_the_simpler_cause(spy):
    """Nothing generated at all is a different fault than nothing parsed."""
    BaseTrainer._warn_on_ungradeable_completions(
        clean_texts=["", ""],
        raw_texts=["", ""],
        truncated=[True, True],
    )
    assert len(spy.messages) == 1
    assert "no learning signal" in spy.messages[0]
    assert "ungradeable" not in spy.messages[0]


def test_reported_counts_reflect_the_batch(spy):
    """Operators act on the numbers, so they must match the batch exactly."""
    BaseTrainer._warn_on_ungradeable_completions(
        clean_texts=["", "", "", "ok"],
        raw_texts=["reasoning", "reasoning", "", "ok"],
        truncated=[True, False, True, False],
    )
    message = spy.messages[0]
    assert "3/4" in message  # three empty of four
    assert "75%" in message


def test_no_completions_is_silent(spy):
    BaseTrainer._warn_on_ungradeable_completions(clean_texts=[], raw_texts=[], truncated=[])
    BaseTrainer._warn_on_ungradeable_completions(clean_texts=None, raw_texts=None, truncated=None)
    assert spy.messages == []


def test_missing_metadata_does_not_raise(spy):
    """Generation metadata is best-effort; a detector must never break a step."""
    BaseTrainer._warn_on_ungradeable_completions(clean_texts=["", ""], raw_texts=None, truncated=None)
    assert len(spy.messages) == 1

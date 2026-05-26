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
"""eSurge rollout and reward-spec helpers."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from importlib import import_module


def _load_openreward() -> object:
    """Import the optional OpenReward SDK or raise a targeted dependency error.

    OpenReward is not required for normal EasyDeL training imports. Delaying the
    import until this helper runs keeps the trainer package usable without the
    optional integration installed.
    """
    try:
        return import_module("openreward")
    except ImportError as exc:
        raise ImportError("OpenReward integration requires the optional `openreward` package.") from exc


def _call_first_available(module: object, names: tuple[str, ...], **kwargs: object) -> object:
    """Call the first available OpenReward SDK entrypoint from ``names``.

    OpenReward versions may expose dataset, environment, and reward factories
    under different function names. This helper centralizes that compatibility
    lookup while still failing loudly when none of the expected callables exist.
    """
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn(**kwargs)
    raise AttributeError(f"OpenReward SDK does not expose any of: {', '.join(names)}")


def generate_rollout_completions(
    trainer: object,
    prompts: list[str],
    *,
    generation_overrides: dict[str, object] | None = None,
    as_chat: bool | None = None,
) -> list[dict[str, object]]:
    """Generate completions from an EasyDeL trainer through ``generate_unified``.

    Args:
        trainer: Trainer object exposing EasyDeL's ``generate_unified`` method.
        prompts: Raw prompt strings to complete.
        generation_overrides: Optional generation config overrides passed to
            eSurge/local generation.
        as_chat: Whether the trainer should apply a chat template to prompts.

    Returns:
        One dictionary per generated completion, carrying text plus available
        token/sequences side data returned by ``generate_unified``.
    """
    if not prompts:
        return []
    generate_unified = getattr(trainer, "generate_unified", None)
    if generate_unified is None:
        raise TypeError("`trainer` must provide EasyDeL `generate_unified` for rollout generation.")
    use_esurge = bool(getattr(getattr(trainer, "arguments", None), "use_esurge_generation", True))
    results = generate_unified(
        prompts=prompts,
        use_esurge=use_esurge,
        apply_chat_template=bool(as_chat),
        config_overrides=generation_overrides,
        release_runtime_after_generation=False,
    )
    texts = results.generation_results
    if isinstance(texts, str):
        texts = [texts]
    prompt_ids = getattr(results, "prompt_ids", None)
    sequences = getattr(results, "sequences", None)
    rollout_rows: list[dict[str, object]] = []
    for idx, text in enumerate(texts):
        row: dict[str, object] = {"text": text}
        if prompt_ids is not None:
            row["prompt_ids"] = prompt_ids[idx] if len(prompt_ids) == len(texts) else prompt_ids
        if sequences is not None:
            row["sequences"] = sequences[idx] if len(sequences) == len(texts) else sequences
        rollout_rows.append(row)
    return rollout_rows


@dataclass(frozen=True)
class eSurgeRolloutConfig:
    """Configuration for trainer-backed eSurge rollout generation.

    The config stores default generation overrides and chat-template policy for
    :class:`eSurgeRolloutGenerator`. Per-call overrides can replace these
    values without mutating the shared config object.
    """

    generation_overrides: dict[str, object] | None = None
    as_chat: bool | None = None


class eSurgeRolloutGenerator:
    """Callable wrapper around :func:`generate_rollout_completions`.

    The wrapper binds a trainer instance and default rollout config so the same
    object can be passed into APIs that expect a simple callable generator. It
    still delegates all actual generation to the trainer's ``generate_unified``
    method.
    """

    def __init__(self, trainer: object, config: eSurgeRolloutConfig | None = None) -> None:
        """Bind a trainer and default rollout options for repeated generation.

        Args:
            trainer: Trainer object exposing EasyDeL ``generate_unified``. The
                generator does not own or modify the trainer state.
            config: Optional immutable rollout defaults for generation
                overrides and chat-template handling. A default empty config is
                created when omitted.
        """
        self.trainer = trainer
        self.config = config or eSurgeRolloutConfig()

    def generate(
        self,
        prompts: list[str],
        *,
        generation_overrides: dict[str, object] | None = None,
        as_chat: bool | None = None,
    ) -> list[dict[str, object]]:
        """Generate rollout rows using wrapper defaults and call overrides.

        Explicit ``generation_overrides`` or ``as_chat`` arguments take
        precedence over the defaults stored in ``self.config`` for this call
        only. The returned rows preserve the text and token side data exposed by
        EasyDeL generation.
        """
        return generate_rollout_completions(
            self.trainer,
            prompts,
            generation_overrides=(
                self.config.generation_overrides if generation_overrides is None else generation_overrides
            ),
            as_chat=self.config.as_chat if as_chat is None else as_chat,
        )

    __call__ = generate


@dataclass(frozen=True)
class OpenRewardSpec:
    """Lazy OpenReward integration spec for GRPO-style trainer construction.

    The spec resolves optional OpenReward dataset, environment, and reward
    function factories only when the corresponding property is accessed. This
    keeps the base EasyDeL import path independent from the optional SDK.
    """

    name: str | None = None
    num_tasks: int | None = None
    split: str = "train"
    base_url: str | None = None
    env_name: str | None = None
    include_metadata: bool = True

    @property
    def train_dataset(self) -> object:
        """Load the OpenReward training dataset using configured task fields.

        The property tries the known OpenReward dataset factory names and passes
        through the task name, split, task limit, base URL, and metadata flag.
        The return type is intentionally SDK-defined.
        """
        openreward = _load_openreward()
        return _call_first_available(
            openreward,
            ("load_dataset", "get_dataset", "dataset"),
            name=self.name,
            split=self.split,
            num_tasks=self.num_tasks,
            base_url=self.base_url,
            include_metadata=self.include_metadata,
        )

    @property
    def environment_factory(self) -> tp.Callable[[], object]:
        """Return a zero-argument factory for configured OpenReward environments.

        Trainers call this factory per rollout interaction, so it captures the
        spec fields and creates a fresh SDK environment each time rather than
        sharing state across prompts.
        """
        openreward = _load_openreward()

        def _factory() -> object:
            """Create one OpenReward environment instance for a rollout step.

            The nested factory keeps the OpenReward module and spec values
            closed over while presenting the zero-argument interface expected by
            EasyDeL environment-backed GRPO trainers.
            """
            return _call_first_available(
                openreward,
                ("make_env", "create_environment", "environment"),
                name=self.name,
                env_name=self.env_name,
                base_url=self.base_url,
            )

        return _factory

    @property
    def reward_funcs(self) -> list[tp.Callable[..., float]]:
        """Load OpenReward reward functions and normalize them to a list.

        Some SDK versions return a single callable while others return an
        iterable of callables. This property normalizes both shapes to the list
        form consumed by GRPO-style trainers.
        """
        openreward = _load_openreward()
        funcs = _call_first_available(
            openreward,
            ("load_reward_functions", "get_reward_functions", "reward_functions"),
            name=self.name,
            num_tasks=self.num_tasks,
            base_url=self.base_url,
        )
        if callable(funcs):
            return [funcs]
        return list(funcs)

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
"""Async GRPO trainer backed by local eSurge async execution."""

from __future__ import annotations

import typing as tp

import jax  # noqa: F401 - kept for test monkeypatching compatibility

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from ..async_rollout_mixin import AsyncRolloutResult as _AsyncRolloutResult  # noqa: F401
from ..group_relative_policy_optimization import GRPOTrainer
from .async_grpo_config import AsyncGRPOConfig


def environment_reward_placeholder(
    *,
    prompts: tp.Sequence[tp.Any] | None = None,
    completions: tp.Sequence[tp.Any] | None = None,
    **_kwargs: tp.Any,
) -> list[float]:
    """Zero-weight ``reward_funcs`` entry for environment-only GRPO runs.

    When a GRPO-family trainer has an ``environment_factory``, it steps the
    environments itself and appends their rewards as an extra reward column
    with weight ``1.0``. GRPO still requires at least one reward function, and
    a function that echoed ``environment_rewards`` would count the environment
    reward a second time. This placeholder returns ``0.0`` per completion and
    carries ``weight = 0.0``, so the aggregated reward equals the environment
    reward exactly once. An explicit ``reward_weights`` in the trainer config
    overrides the attribute weight; its first entry should then be ``0.0``.

    Args:
        prompts: Prompt records (used only for the output length fallback).
        completions: Completion records; one zero is returned per entry.

    Returns:
        A list of zeros, one per completion.
    """
    count = len(completions) if completions is not None else len(prompts or ())
    return [0.0] * count


environment_reward_placeholder.weight = 0.0


@Registry.register("trainer", "async_grpo")
class AsyncGRPOTrainer(GRPOTrainer):
    """AsyncGRPO trainer using GRPO updates with eSurge async rollouts.

    Thin convenience wrapper over :class:`GRPOTrainer` with
    ``arguments.async_mode=True`` and plug-and-play ``environment`` support.
    With an environment and no ``reward_funcs``, the trainer installs
    :func:`environment_reward_placeholder` so the environment reward column is
    the only reward source.
    """

    arguments: AsyncGRPOConfig
    emit_legacy_async_grpo_metrics: tp.ClassVar[bool] = True

    def __init__(
        self,
        arguments: AsyncGRPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: object | list[object] | None = None,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | list[object] | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
        tools: list[dict | str | tp.Callable[..., object]] | None = None,
        environment_factory: tp.Callable[[], object] | None = None,
        environment: str | tp.Mapping[str, tp.Any] | tp.Any | None = None,
    ) -> None:
        if not isinstance(arguments, AsyncGRPOConfig):
            raise TypeError(f"arguments must be AsyncGRPOConfig, got {type(arguments)}")

        env_spec = environment if environment is not None else arguments.environment
        resolved_runtime = None
        if env_spec is not None and environment_factory is None:
            from easydel.environments import AutoEnvironment, KiriRuntime, SandboxPool

            if arguments.environment_endpoints:
                resolved_runtime = SandboxPool(
                    endpoints=arguments.environment_endpoints,
                    backend=arguments.environment_backend,
                )
            else:
                resolved_runtime = KiriRuntime(backend=arguments.environment_backend)
            if callable(env_spec) and not isinstance(env_spec, type):
                environment_factory = env_spec
            else:
                environment_factory = AutoEnvironment.as_factory(env_spec, runtime=resolved_runtime)

        if reward_funcs is None:
            if environment_factory is not None:
                # GRPO appends the stepped environment rewards as their own column; a reward fn that
                # echoed them would double-count, so install a zero-weight placeholder instead.
                reward_funcs = environment_reward_placeholder
            else:
                raise ValueError("AsyncGRPOTrainer requires `reward_funcs` unless `environment` is provided.")

        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
            tools=tools,
            environment_factory=environment_factory,
        )

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


@Registry.register("trainer", "async_grpo")
class AsyncGRPOTrainer(GRPOTrainer):
    """AsyncGRPO trainer using GRPO updates with eSurge async rollouts.

    Thin convenience wrapper over :class:`GRPOTrainer` with
    ``arguments.async_mode=True`` and plug-and-play ``environment`` support.
    """

    arguments: AsyncGRPOConfig

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
            if environment_factory is not None or env_spec is not None:
                from easydel.environments import make_environment_reward_fn

                reward_funcs = make_environment_reward_fn(
                    env_spec if (env_spec is not None and not callable(env_spec)) else "auto",
                    runtime=resolved_runtime,
                )
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

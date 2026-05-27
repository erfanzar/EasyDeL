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
"""NeMo Gym style GRPO trainer backed by EasyDeL/eSurge generation."""

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from ..training_utils import filter_kwargs_for_callable
from ._fn import _as_mapping, _environment_reward_func, _normalize_step_result
from .nemo_gym_config import NeMoGymConfig


@Registry.register("trainer", "nemo_gym")
class NeMoGymTrainer(GRPOTrainer):
    """GRPO trainer for NeMo Gym style environments using local eSurge rollouts.

    The trainer carries task metadata and agent references from each prompt
    batch into environment construction, generates actions with EasyDeL/eSurge,
    and converts environment step results into GRPO reward side channels.
    """

    arguments: NeMoGymConfig

    def __init__(
        self,
        arguments: NeMoGymConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: object | list[object] | None = None,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | list[object] | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
        tools: list[dict | str | tp.Callable[..., object]] | None = None,
        environment_factory: tp.Callable[..., object] | None = None,
    ) -> None:
        """Create a GRPO trainer that routes generated actions through environments.

        Args:
            arguments: NeMo Gym config controlling eSurge generation, metadata
                column names, timeout handling, and the environment reward
                scale.
            model: Initialized EasyDeL policy module or state used by the
                inherited GRPO rollout and optimization path.
            reward_funcs: Optional extra reward functions. When omitted, the
                trainer installs the environment reward collector and zeroes the
                normal GRPO reward weights so only environment feedback affects
                the batch.
            train_dataset: Prompt dataset. Rows may include metadata and agent
                reference columns named by the config.
            eval_dataset: Optional evaluation dataset or named evaluation
                dataset mapping.
            processing_class: Tokenizer or processor used for prompt encoding
                and completion decoding.
            reward_processing_classes: Optional processors for non-environment
                reward functions.
            data_tokenize_fn: Optional dataset tokenization override accepted
                by the GRPO base trainer.
            tools: Tool definitions exposed to the generation path.
            environment_factory: Callable that builds one environment per
                generated action; supported signatures are filtered at runtime.

        Raises:
            TypeError: If ``arguments`` is not a ``NeMoGymConfig``.
        """
        if not isinstance(arguments, NeMoGymConfig):
            raise TypeError(f"arguments must be NeMoGymConfig, got {type(arguments)}")
        self._nemo_active_metadata: list[dict[str, object]] = []
        self._nemo_active_agent_refs: list[object] = []
        using_default_reward = reward_funcs is None
        if reward_funcs is None:
            reward_funcs = _environment_reward_func
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
        if using_default_reward:
            self.reward_weights = self.reward_weights * 0.0

    def _extract_nemo_sidechannels(self, batch: dict[str, object]) -> tuple[list[dict[str, object]], list[object]]:
        """Extract per-prompt NeMo metadata and agent references from a batch.

        Metadata entries are normalized to dictionaries. Agent references are
        read from a dedicated batch column when present, otherwise from each
        metadata row using the configured key.
        """
        metadata_values = batch.get(self.arguments.metadata_key)
        agent_refs = batch.get(self.arguments.agent_ref_key)

        if metadata_values is None:
            return [], []
        if not isinstance(metadata_values, list | tuple):
            metadata_values = [metadata_values]
        metadata = [_as_mapping(value) for value in metadata_values]

        if agent_refs is None:
            agent_refs = [item.get(self.arguments.agent_ref_key) for item in metadata]
        elif not isinstance(agent_refs, list | tuple):
            agent_refs = [agent_refs]
        return metadata, list(agent_refs)

    @staticmethod
    def _expand_sidechannel(values: list[object], target_len: int) -> list[object | None]:
        """Repeat side-channel values to match generated completion count.

        A batch may have one metadata row per prompt while generation produces
        multiple completions per prompt. This expands or pads side-channel lists
        so each generated action can receive one matching context value.
        """
        if not values:
            return [None] * target_len
        if len(values) == target_len:
            return list(values)
        repeat = max(1, target_len // len(values))
        expanded = [value for value in values for _ in range(repeat)]
        if len(expanded) < target_len:
            expanded.extend([values[-1]] * (target_len - len(expanded)))
        return expanded[:target_len]

    def _make_nemo_environment(self, metadata: object, agent_ref: object) -> object:
        """Construct one NeMo environment using supported factory arguments.

        The factory is called with filtered keyword arguments first, then with
        metadata only, then with no arguments. That keeps EasyDeL compatible
        with common NeMo/Gym factory signatures without an adapter class.
        """
        if self.environment_factory is None:
            raise ValueError("NeMoGymTrainer requires `environment_factory` for environment reward feedback.")
        kwargs = {"metadata": metadata, "agent_ref": agent_ref, "request_timeout": self.arguments.request_timeout}
        filtered = filter_kwargs_for_callable(self.environment_factory, kwargs)
        try:
            return self.environment_factory(**filtered)
        except TypeError:
            try:
                return self.environment_factory(metadata)
            except TypeError:
                return self.environment_factory()

    def _run_environment_feedback(
        self,
        *,
        action_texts: list[str],
        tool_calls: list[object | None],
    ) -> dict[str, list[object]] | None:
        """Run NeMo environment feedback for generated actions.

        Each generated action receives a fresh environment. The method resets
        when possible, uses ``step_with_tool_calls``/``run``/``step`` in that
        order, normalizes the result, and closes the environment if it exposes
        ``close``.
        """
        if self.environment_factory is None:
            return None

        metadata_rows = self._expand_sidechannel(self._nemo_active_metadata, len(action_texts))
        agent_refs = self._expand_sidechannel(self._nemo_active_agent_refs, len(action_texts))
        observations: list[object] = []
        rewards: list[object] = []
        terminated: list[object] = []
        truncated: list[object] = []
        infos: list[object] = []

        for idx, action_text in enumerate(action_texts):
            env = self._make_nemo_environment(metadata_rows[idx], agent_refs[idx])
            try:
                reset = getattr(env, "reset", None)
                if callable(reset):
                    reset_payload = filter_kwargs_for_callable(
                        reset,
                        {"metadata": metadata_rows[idx], "agent_ref": agent_refs[idx]},
                    )
                    reset(**reset_payload)

                step_with_tool_calls = getattr(env, "step_with_tool_calls", None)
                if callable(step_with_tool_calls):
                    step_result = step_with_tool_calls(
                        action_text,
                        tool_calls=tool_calls[idx] if idx < len(tool_calls) else None,
                    )
                else:
                    run = getattr(env, "run", None)
                    step = getattr(env, "step", None)
                    runner = run if callable(run) else step
                    if not callable(runner):
                        raise TypeError("NeMo Gym environments must provide run(action) or step(action).")
                    step_result = runner(action_text)

                observation, reward, done, is_truncated, info = _normalize_step_result(step_result)
                observations.append(observation)
                rewards.append(float(reward) * float(self.arguments.environment_reward_weight))
                terminated.append(done)
                truncated.append(is_truncated)
                infos.append(info)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

        return {
            "environment_observations": observations,
            "environment_rewards": rewards,
            "environment_terminated": terminated,
            "environment_truncated": truncated,
            "environment_infos": infos,
        }

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, object],
        is_train: bool,
    ) -> tuple[dict[str, object], dict[str, float | int | str]]:
        """Expose NeMo sidechannels during GRPO generation and scoring.

        The active metadata is stored only around the parent GRPO preprocessing
        call and cleared in ``finally``. This prevents stale side-channel data
        from leaking into later batches if generation or scoring raises.
        """
        metadata, agent_refs = self._extract_nemo_sidechannels(batch)
        self._nemo_active_metadata = metadata
        self._nemo_active_agent_refs = agent_refs
        try:
            model_batch, metrics = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)
        finally:
            self._nemo_active_metadata = []
            self._nemo_active_agent_refs = []
        metrics = {
            **metrics,
            "nemo_gym/environment_enabled": int(self.environment_factory is not None),
            "nemo_gym/metadata_rows": len(metadata),
        }
        return model_batch, metrics

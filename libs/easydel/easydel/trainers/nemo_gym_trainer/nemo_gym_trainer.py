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

import inspect
import pickle
import typing as tp
from collections.abc import Mapping

import jax
import numpy as np
from jax.experimental import multihost_utils

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from .._shared.environment import ToolEnvWrapper
from ..group_relative_policy_optimization import GRPOTrainer
from ..training_utils import filter_kwargs_for_callable
from ._fn import (
    _as_mapping,
    _decode_json_container_if_needed,
    _environment_reward_func,
    _NeMoGymGrainCollator,
    _NeMoGymPreprocessTransform,
    _NeMoGymTFDSCollator,
    _normalize_step_result,
)
from .nemo_gym_config import NeMoGymConfig


def _is_array_like(value: object) -> bool:
    """Return True for NumPy or JAX array containers (excluding strings/bytes/mappings)."""
    if isinstance(value, str | bytes | bytearray | Mapping | list | tuple | np.generic):
        return False
    return isinstance(value, np.ndarray | jax.Array)


def _to_python_scalar_or_value(value: object) -> object:
    """Convert 0-d arrays and NumPy scalars to native Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray | jax.Array):
        arr = np.asarray(jax.device_get(value)) if isinstance(value, jax.Array) else value
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    return value


def _collect_pytree_batch_info(tree: Mapping[tp.Any, tp.Any]) -> tuple[bool, int | None, bool]:
    """Inspect a nested mapping to check if all leaves share a single batch length.

    Returns ``(all_batch_like, batch_len, has_array_leaf)``.
    """
    batch_len: int | None = None
    has_leaf = False
    has_array_leaf = False

    def _visit(node: object) -> bool:
        nonlocal batch_len, has_leaf, has_array_leaf
        if isinstance(node, Mapping):
            if not node:
                return False
            return all(_visit(v) for v in node.values())
        has_leaf = True
        if _is_array_like(node):
            arr = np.asarray(jax.device_get(node)) if isinstance(node, jax.Array) else np.asarray(node)
            if arr.ndim == 0:
                return False
            has_array_leaf = True
            length = int(arr.shape[0])
        elif isinstance(node, list | tuple):
            length = len(node)
        else:
            return False
        if batch_len is None:
            batch_len = length
        elif batch_len != length:
            return False
        return True

    ok = _visit(tree)
    return (ok and has_leaf), batch_len, has_array_leaf


def _slice_pytree_mapping(tree: Mapping[tp.Any, tp.Any], index: int) -> dict[tp.Any, tp.Any]:
    """Extract row ``index`` from a PyTree-transposed mapping."""
    out: dict[tp.Any, tp.Any] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            out[key] = _slice_pytree_mapping(value, index)
        elif _is_array_like(value):
            arr = np.asarray(jax.device_get(value)) if isinstance(value, jax.Array) else np.asarray(value)
            out[key] = _to_python_scalar_or_value(arr[index])
        elif isinstance(value, list | tuple):
            out[key] = _to_python_scalar_or_value(value[index])
        else:
            out[key] = _to_python_scalar_or_value(value)
    return out


def _unbatch_sidechannel_column(value: object, *, expected_len: int | None = None) -> list[object]:
    """Convert a batched side-channel column into a per-example Python list.

    Handles Python lists/tuples, 1D/0D NumPy & JAX arrays, and PyTree-transposed
    nested mappings emitted by ``grain.Batch``.
    """
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [_to_python_scalar_or_value(item) for item in value]
    if _is_array_like(value):
        arr = np.asarray(jax.device_get(value)) if isinstance(value, jax.Array) else np.asarray(value)
        if arr.ndim == 0:
            return [arr.item()]
        return [_to_python_scalar_or_value(item) for item in arr]
    if isinstance(value, Mapping):
        all_batch_like, batch_len, has_array_leaf = _collect_pytree_batch_info(value)
        if all_batch_like and batch_len is not None:
            # When leaves are pure Python lists (not arrays) and expected_len
            # disagrees, treat as a single dict with a list field rather than a batch.
            if has_array_leaf or expected_len is None or batch_len == expected_len:
                return [_slice_pytree_mapping(value, i) for i in range(batch_len)]
        return [dict(value)]
    return [_to_python_scalar_or_value(value)]


def _broadcast_bytes_from_host0(data: bytes | None) -> bytes:
    """Broadcast a variable-length byte buffer from host 0 to all JAX hosts."""
    if jax.process_count() <= 1:
        return data if data is not None else b""
    is_root = jax.process_index() == 0
    raw = data if is_root and data is not None else b""
    length_arr = np.array([len(raw)], dtype=np.int64)
    length = int(np.asarray(multihost_utils.broadcast_one_to_all(length_arr))[0])
    if length == 0:
        return b""
    buf = np.frombuffer(raw, dtype=np.uint8) if is_root else np.zeros(length, dtype=np.uint8)
    synced = np.asarray(multihost_utils.broadcast_one_to_all(buf), dtype=np.uint8)
    return synced.tobytes()


def _broadcast_object_from_host0(obj: object) -> object:
    """Broadcast a picklable Python object from host 0 to all JAX hosts."""
    if jax.process_count() <= 1:
        return obj
    raw = pickle.dumps(obj) if jax.process_index() == 0 else None
    synced = _broadcast_bytes_from_host0(raw)
    return pickle.loads(synced)


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

    def _get_preprocess_transform(self):
        """Wrap GRPO preprocessing so ``metadata_key`` and ``agent_ref_key`` are never dropped."""
        base_transform = super()._get_preprocess_transform()
        if base_transform is None:
            return None
        return _NeMoGymPreprocessTransform(
            base_transform=base_transform,
            metadata_key=self.arguments.metadata_key,
            agent_ref_key=self.arguments.agent_ref_key,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Wrap the GRPO Grain collator so NeMo sidechannels bypass numeric coercion."""
        base_collator = super().create_grain_collect_function(
            max_sequence_length=max_sequence_length,
            truncation_mode=truncation_mode,
        )
        return _NeMoGymGrainCollator(
            base_collator=base_collator,
            metadata_key=self.arguments.metadata_key,
            agent_ref_key=self.arguments.agent_ref_key,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Wrap the GRPO TFDS collator so NeMo sidechannels stay Python lists."""
        base_collator = super().create_tfds_collect_function(
            max_sequence_length=max_sequence_length,
            truncation_mode=truncation_mode,
        )
        return _NeMoGymTFDSCollator(
            base_collator=base_collator,
            metadata_key=self.arguments.metadata_key,
            agent_ref_key=self.arguments.agent_ref_key,
        )

    def _extract_nemo_sidechannels(
        self,
        batch: Mapping[str, object] | tp.Sequence[Mapping[str, object]],
    ) -> tuple[list[dict[str, object]], list[object]]:
        """Extract per-prompt NeMo metadata and agent references from a batch.

        Supports collated dicts (including ``np.ndarray`` / ``jax.Array``
        columns and PyTree-transposed ``dict[str, np.ndarray]`` structures
        from ``grain.Batch``) as well as uncollated ``list[dict]`` batches.
        Agent reference extraction is independent of whether metadata is present.
        """
        metadata_key = self.arguments.metadata_key
        agent_ref_key = self.arguments.agent_ref_key

        if isinstance(batch, list | tuple):
            has_metadata = any(isinstance(row, Mapping) and row.get(metadata_key) is not None for row in batch)
            metadata = (
                [
                    _as_mapping(row.get(metadata_key)) if isinstance(row, Mapping) else {}
                    for row in batch
                ]
                if has_metadata
                else []
            )
            has_agent_refs = any(isinstance(row, Mapping) and row.get(agent_ref_key) is not None for row in batch)
            if has_agent_refs:
                agent_refs = [
                    _decode_json_container_if_needed(row.get(agent_ref_key)) if isinstance(row, Mapping) else None
                    for row in batch
                ]
                if metadata and len(agent_refs) == len(metadata):
                    agent_refs = [
                        ref if ref is not None else metadata[i].get(agent_ref_key)
                        for i, ref in enumerate(agent_refs)
                    ]
            elif metadata:
                agent_refs = [item.get(agent_ref_key) for item in metadata]
            else:
                agent_refs = []
            return metadata, agent_refs

        if not isinstance(batch, Mapping):
            return [], []

        expected_len: int | None = None
        input_ids = batch.get("input_ids")
        if _is_array_like(input_ids):
            arr = np.asarray(jax.device_get(input_ids)) if isinstance(input_ids, jax.Array) else np.asarray(input_ids)
            if arr.ndim >= 1:
                expected_len = int(arr.shape[0])
        elif isinstance(input_ids, list | tuple):
            expected_len = len(input_ids)

        metadata_values = batch.get(metadata_key)
        agent_ref_values = batch.get(agent_ref_key)

        if metadata_values is not None:
            unbatched_metadata = _unbatch_sidechannel_column(metadata_values, expected_len=expected_len)
            metadata = [_as_mapping(value) for value in unbatched_metadata]
        else:
            metadata = []

        if agent_ref_values is not None:
            unbatched_refs = _unbatch_sidechannel_column(agent_ref_values, expected_len=expected_len)
            agent_refs = [_decode_json_container_if_needed(value) for value in unbatched_refs]
            if metadata and len(agent_refs) == len(metadata):
                agent_refs = [
                    ref if ref is not None else metadata[i].get(agent_ref_key)
                    for i, ref in enumerate(agent_refs)
                ]
        elif metadata:
            agent_refs = [item.get(agent_ref_key) for item in metadata]
        else:
            agent_refs = []

        return metadata, list(agent_refs)

    @staticmethod
    def _expand_sidechannel(values: list[object], target_len: int) -> list[object | None]:
        """Repeat side-channel values to match generated completion count.

        A batch may have one metadata row per prompt while generation produces
        multiple completions per prompt. Requires exact divisibility
        (``target_len % len(values) == 0``) instead of silently truncating or
        repeating ``values[-1]``.
        """
        if target_len < 0:
            raise ValueError(f"target_len must be non-negative, got {target_len}.")
        if not values:
            return [None] * target_len
        if target_len == 0:
            return []
        if len(values) == target_len:
            return list(values)
        if target_len % len(values) != 0:
            raise ValueError(
                f"Cannot align side-channel values of length {len(values)} to {target_len} completions: "
                f"{target_len} is not divisible by {len(values)}."
            )
        repeat = target_len // len(values)
        return [value for value in values for _ in range(repeat)]

    def _make_nemo_environment(self, metadata: object, agent_ref: object) -> object:
        """Construct one NeMo environment using supported factory arguments.

        Resolve filtered keyword, metadata-positional, and no-argument
        signatures before calling the factory exactly once. Exceptions from
        inside the factory propagate unchanged without repeating side effects.
        """
        if self.environment_factory is None:
            raise ValueError("NeMoGymTrainer requires `environment_factory` for environment reward feedback.")
        kwargs = {"metadata": metadata, "agent_ref": agent_ref, "request_timeout": self.arguments.request_timeout}
        filtered = filter_kwargs_for_callable(self.environment_factory, kwargs)
        try:
            signature = inspect.signature(self.environment_factory)
        except (TypeError, ValueError):
            return self.environment_factory(**filtered)
        for args, call_kwargs in (((), filtered), ((metadata,), {}), ((), {})):
            try:
                signature.bind(*args, **call_kwargs)
            except TypeError:
                continue
            return self.environment_factory(*args, **call_kwargs)
        raise TypeError(f"Unsupported NeMo environment factory signature: {signature}")

    def _execute_environment_feedback_local(
        self,
        *,
        action_texts: list[str],
        tool_calls: list[object | None],
    ) -> dict[str, list[object]]:
        """Execute NeMo environment feedback locally for each generated action."""
        metadata_rows = self._expand_sidechannel(self._nemo_active_metadata, len(action_texts))
        agent_refs = self._expand_sidechannel(self._nemo_active_agent_refs, len(action_texts))
        env_tools = getattr(self, "_environment_tools", None)
        tool_parser = getattr(self, "_environment_tool_parser", None)

        observations: list[object] = []
        rewards: list[object] = []
        terminated: list[object] = []
        truncated: list[object] = []
        infos: list[object] = []

        for idx, action_text in enumerate(action_texts):
            env = self._make_nemo_environment(metadata_rows[idx], agent_refs[idx])
            if env_tools and not isinstance(env, ToolEnvWrapper):
                cloned_tools = [
                    tool.clone_for_episode() if hasattr(tool, "clone_for_episode") else tool for tool in env_tools
                ]
                env = ToolEnvWrapper(
                    env=env,
                    tools=cloned_tools,
                    tool_call_parser=tool_parser,
                    max_tool_calls_per_step=self.arguments.max_tool_calls_per_step,
                )
            try:
                reset = getattr(env, "reset", None)
                if callable(reset):
                    if isinstance(env, ToolEnvWrapper):
                        inner_reset = getattr(env.env, "reset", None)
                        if callable(inner_reset):
                            reset_payload = filter_kwargs_for_callable(
                                inner_reset,
                                {"metadata": metadata_rows[idx], "agent_ref": agent_refs[idx]},
                            )
                            env.reset(**reset_payload)
                        else:
                            env._tool_calls_this_step = 0
                            env._reset_tools()
                    else:
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

    def _run_environment_feedback(
        self,
        *,
        action_texts: list[str],
        tool_calls: list[object | None],
    ) -> dict[str, list[object]] | None:
        """Run NeMo environment feedback for generated actions.

        Each generated action receives a fresh environment (wrapped in
        :class:`ToolEnvWrapper` when ``self._environment_tools`` is configured).
        On multi-host runs (``jax.process_count() > 1``), process 0 executes
        all environments once and broadcasts results to follower hosts so
        external environments never suffer redundant per-host side effects.
        """
        if self.environment_factory is None:
            return None

        if jax.process_count() <= 1:
            return self._execute_environment_feedback_local(action_texts=action_texts, tool_calls=tool_calls)

        if jax.process_index() == 0:
            try:
                feedback = self._execute_environment_feedback_local(action_texts=action_texts, tool_calls=tool_calls)
            except BaseException as exc:
                _broadcast_object_from_host0({"ok": False, "type": type(exc).__name__, "message": str(exc)})
                raise
            _broadcast_object_from_host0({"ok": True, "payload": feedback})
            return feedback

        envelope = _broadcast_object_from_host0(None)
        if not isinstance(envelope, dict) or not envelope.get("ok"):
            err_type = envelope.get("type", "RuntimeError") if isinstance(envelope, dict) else "RuntimeError"
            err_msg = envelope.get("message", "unknown error") if isinstance(envelope, dict) else str(envelope)
            raise RuntimeError(f"Host 0 NeMo environment feedback failed ({err_type}): {err_msg}")
        return envelope["payload"]

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, object] | list[dict[str, object]],
        is_train: bool,
    ) -> tuple[dict[str, object], dict[str, float | int | str]]:
        """Expose NeMo sidechannels during GRPO generation and scoring.

        The active metadata is stored only around the parent GRPO preprocessing
        call and cleared in ``finally``. This prevents stale side-channel data
        from leaking into later batches if generation or scoring raises.
        """
        batch = self._apply_user_data_collator(batch)
        metadata, agent_refs = self._extract_nemo_sidechannels(batch)
        if isinstance(batch, Mapping):
            normalized_batch: dict[str, object] = dict(batch)
            if self.arguments.metadata_key in normalized_batch and metadata:
                normalized_batch[self.arguments.metadata_key] = metadata
            if self.arguments.agent_ref_key in normalized_batch and agent_refs:
                normalized_batch[self.arguments.agent_ref_key] = agent_refs
            parent_batch: object = normalized_batch
        else:
            parent_batch = batch
        self._nemo_active_metadata = metadata
        self._nemo_active_agent_refs = agent_refs
        try:
            model_batch, metrics = super()._preprocess_batch_input(state=state, batch=parent_batch, is_train=is_train)
        finally:
            self._nemo_active_metadata = []
            self._nemo_active_agent_refs = []
        metrics = {
            **metrics,
            "nemo_gym/environment_enabled": int(self.environment_factory is not None),
            "nemo_gym/metadata_rows": len(metadata),
        }
        return model_batch, metrics

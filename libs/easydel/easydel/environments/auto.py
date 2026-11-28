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

"""Hugging Face-style ``AutoEnvironment`` discovery, instantiation, and reward adapters."""

from __future__ import annotations

import collections.abc
import concurrent.futures
import inspect
import math
import typing as tp

from easydel.trainers._shared.environment import AgenticEnvironment, StepResult
from easydel.utils import Registry

from ._base import build_environment, list_environments, register_environment
from .harnesses import HarnessPreset
from .runtime import ContainerConfig, KiriRuntime, SandboxPool


def _extract_text(value: tp.Any) -> str:
    """Extract plain text from a string, chat message mapping, or list of messages/content blocks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, collections.abc.Mapping):
        if "content" in value:
            return _extract_text(value["content"])
        if "text" in value:
            return _extract_text(value["text"])
        return str(value)
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            part = _extract_text(item)
            if part:
                parts.append(part)
        return "\n".join(parts)
    return str(value)


def _as_column(val: tp.Any, n: int) -> list[tp.Any] | None:
    """Return ``val`` as an ``n``-element list if it represents a per-sample column (including numpy arrays)."""
    if val is None or isinstance(val, (str, bytes, collections.abc.Mapping)):
        return None
    if isinstance(val, (list, tuple)):
        return list(val) if len(val) == n else None
    if hasattr(val, "tolist") and hasattr(val, "shape"):
        shape = getattr(val, "shape", ())
        if isinstance(shape, tuple) and len(shape) >= 1 and shape[0] == n:
            return list(val.tolist())
        return None
    if hasattr(val, "tolist"):
        converted = val.tolist()
        if isinstance(converted, list) and len(converted) == n:
            return converted
    return None


def _row_prompt(row: collections.abc.Mapping[str, tp.Any]) -> str:
    for key in ("prompt", "problem_statement", "task_prompt", "question", "instruction", "query"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list) and val:
            last = val[-1]
            if isinstance(last, collections.abc.Mapping) and "content" in last:
                return _extract_text(last["content"])
    return ""


def _filter_ctor_kwargs(cls: type, kwargs: collections.abc.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    allowed = set(params.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


class AutoEnvironment:
    """Hugging Face-style factory for EasyDeL RL environments.

    Provides plug-and-play environment resolution from a registered name, a
    configuration mapping, a :class:`HarnessPreset`, or a raw dataset row::

        env = AutoEnvironment.from_task(row, backend="kiri", runtime=pool)
        env = AutoEnvironment.from_name("code_judge", stdin_tests=tests)
        factory = AutoEnvironment.as_factory("auto", runtime=pool)
    """

    @staticmethod
    def list() -> list[str]:
        """Return all registered environment names."""
        return list_environments()

    @staticmethod
    def register(name: str | collections.abc.Sequence[str]):
        """Decorator registering a custom :class:`AgenticEnvironment` subclass."""
        return register_environment(name)

    @classmethod
    def infer_name(cls, task: collections.abc.Mapping[str, tp.Any]) -> str:
        """Infer the most suitable registered environment name from a task/dataset row."""
        explicit = task.get("environment") or task.get("env_name") or task.get("env_type")
        if isinstance(explicit, str) and explicit:
            return explicit
        if "harness" in task:
            return "harness"
        if any(k in task for k in ("test_patch", "fail_to_pass", "pass_to_pass", "repo_dir")):
            return "swe_repo"
        if any(k in task for k in ("verify_script", "setup_script")):
            return "terminal"
        if any(k in task for k in ("stdin_tests", "test_code")):
            return "code_judge"
        if "initial_state" in task and "tools" in task:
            return "synthetic_tool"
        if "answer" in task or "verifier" in task:
            return "python_tir"
        raise ValueError(
            "Cannot infer environment from task keys "
            f"{sorted(task.keys())}. Pass `environment=<name>` or include task-specific fields "
            "(e.g. `test_patch`/`fail_to_pass`, `stdin_tests`/`test_code`, `verify_script`, `answer`, or `harness`)."
        )

    @classmethod
    def from_name(
        cls,
        name: str | HarnessPreset | collections.abc.Mapping[str, tp.Any],
        **kwargs: tp.Any,
    ) -> AgenticEnvironment:
        """Instantiate an environment from a registered name, dict spec, or :class:`HarnessPreset`."""
        if isinstance(name, HarnessPreset):
            return build_environment("harness", harness=name, **kwargs)
        if isinstance(name, collections.abc.Mapping):
            return build_environment(name, **kwargs)
        env_cls = Registry.get_or_raise("environment", name)
        return env_cls(**_filter_ctor_kwargs(env_cls, kwargs))

    @classmethod
    def from_task(
        cls,
        task: collections.abc.Mapping[str, tp.Any],
        *,
        environment: str | HarnessPreset | collections.abc.Mapping[str, tp.Any] | None = None,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str | None = None,
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        **overrides: tp.Any,
    ) -> AgenticEnvironment:
        """Build an environment directly from a dataset row or task dictionary.

        Automatically maps standard prompt columns (``prompt``,
        ``problem_statement``, ``task_prompt``, ``question``) and filters row
        keys against the target environment's constructor signature.
        """
        merged: dict[str, tp.Any] = dict(task)
        prompt = _row_prompt(task)
        if prompt:
            if not isinstance(merged.get("prompt"), str):
                merged["prompt"] = prompt
            if not isinstance(merged.get("problem_statement"), str):
                merged["problem_statement"] = prompt
            if not isinstance(merged.get("task_prompt"), str):
                merged["task_prompt"] = prompt

        if runtime is not None:
            merged["runtime"] = runtime
        if backend is not None:
            merged["backend"] = backend
        if endpoint is not None:
            merged["endpoint"] = endpoint
        if container_config is not None:
            merged["container_config"] = container_config
        merged.update(overrides)

        if isinstance(environment, HarnessPreset):
            env_name = "harness"
            merged.setdefault("harness", environment)
        elif isinstance(environment, collections.abc.Mapping):
            spec_copy = dict(environment)
            env_name = str(spec_copy.pop("name", None) or spec_copy.pop("type", None) or cls.infer_name(merged))
            merged = {**spec_copy, **merged}
        elif environment is not None and environment != "auto":
            env_name = str(environment)
        else:
            env_name = cls.infer_name(merged)

        env_cls = Registry.get_or_raise("environment", env_name)
        filtered = _filter_ctor_kwargs(env_cls, merged)
        return env_cls(**filtered)

    @classmethod
    def from_dataset(
        cls,
        dataset: collections.abc.Iterable[collections.abc.Mapping[str, tp.Any]],
        *,
        environment: str | HarnessPreset | collections.abc.Mapping[str, tp.Any] | None = None,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str | None = None,
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        **overrides: tp.Any,
    ) -> collections.abc.Iterator[AgenticEnvironment]:
        """Lazily yield one configured :class:`AgenticEnvironment` per dataset row."""
        for row in dataset:
            yield cls.from_task(
                row,
                environment=environment,
                runtime=runtime,
                backend=backend,
                endpoint=endpoint,
                container_config=container_config,
                **overrides,
            )

    @classmethod
    def as_factory(
        cls,
        environment: str | HarnessPreset | collections.abc.Mapping[str, tp.Any] = "auto",
        *,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str | None = None,
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        **defaults: tp.Any,
    ) -> tp.Callable[..., AgenticEnvironment]:
        """Create a trainer-compatible ``environment_factory(**task_or_empty)``.

        Factory ``defaults`` apply only when not overridden by per-call ``task`` keys or ``kwargs``
        (precedence: ``defaults < task < kwargs``).
        """
        base_defaults: dict[str, tp.Any] = {
            **{
                k: v
                for k, v in {
                    "runtime": runtime,
                    "backend": backend,
                    "endpoint": endpoint,
                    "container_config": container_config,
                }.items()
                if v is not None
            },
            **defaults,
        }

        def _factory(task: collections.abc.Mapping[str, tp.Any] | None = None, **kwargs: tp.Any) -> AgenticEnvironment:
            combined_task = {**base_defaults, **(task or {}), **kwargs}
            if environment == "auto" or task or kwargs:
                return cls.from_task(
                    combined_task,
                    environment=None if environment == "auto" else environment,
                )
            return cls.from_name(environment, **base_defaults)

        if isinstance(environment, str):
            env_label = environment.strip()
            if env_label == "harness" and isinstance(defaults.get("harness"), str) and defaults["harness"].strip():
                env_label = f"harness:{defaults['harness'].strip()}"
        elif isinstance(environment, collections.abc.Mapping):
            raw_label = environment.get("name") or environment.get("type")
            env_label = str(raw_label).strip() if raw_label else "auto"
        else:
            env_label = getattr(environment, "name", None) or type(environment).__name__
        _factory.environment_name = env_label  # type: ignore[attr-defined]
        return _factory


def is_unscorable(result: StepResult) -> bool:
    """Whether ``result`` carries no usable reward.

    Environments flag a sample unscorable with ``info["discard"] = True`` when the grader itself failed
    (sandbox/infra error, judge failure). Results with ``info["env_error"] = True`` are treated the same way.
    """
    info = result.info or {}
    return bool(info.get("discard") or info.get("env_error"))


def _grade_single_turn(env: AgenticEnvironment, completion: str) -> StepResult | None:
    """Grade one completion terminally, isolating per-sample exceptions into ``None`` (unscorable)."""
    try:
        grader = getattr(env, "grade_single_turn", None)
        if callable(grader):
            return grader(completion)
        return env.step(completion)
    except Exception:
        return None


def _grade_single_turn_batch(
    envs: collections.abc.Sequence[AgenticEnvironment],
    completions: collections.abc.Sequence[str],
    *,
    max_workers: int,
) -> list[StepResult | None]:
    if len(envs) <= 1:
        return [_grade_single_turn(e, c) for e, c in zip(envs, completions, strict=True)]
    workers = min(max(1, max_workers), len(envs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edenv-grade") as pool:
        futs = [pool.submit(_grade_single_turn, e, c) for e, c in zip(envs, completions, strict=True)]
        results: list[StepResult | None] = []
        for f in futs:
            try:
                results.append(f.result())
            except Exception:
                results.append(None)
        return results


def make_environment_reward_fn(
    environment: str | HarnessPreset | collections.abc.Mapping[str, tp.Any] = "auto",
    *,
    runtime: KiriRuntime | SandboxPool | None = None,
    backend: str | None = None,
    container_config: ContainerConfig | None = None,
    max_workers: int = 16,
    **defaults: tp.Any,
) -> tp.Callable[..., list[float | None]]:
    """Build a GRPO/AsyncGRPO-compatible ``reward_funcs`` callable for environments.

    If the trainer already stepped environments during rollout and attached
    ``environment_rewards`` in ``kwargs``, those rewards are returned directly
    with zero duplicate execution. Otherwise (single-turn RLVR / code / math),
    completions are graded concurrently via :class:`AutoEnvironment` in ``max_workers``
    threads. Every completion is graded terminally: environments exposing
    ``grade_single_turn`` (e.g. ``python_tir``) grade the final answer without
    executing code blocks.

    A ``None`` entry means the sample is unscorable (its environment failed to
    reset, or the grader reported ``info["discard"]``/``info["env_error"]``);
    trainers must drop such samples rather than treat them as reward 0.

    The returned callable carries ``discard_unscorable = True``. Group-relative trainers use that marker to
    opt this reward function into dropping a group when one of its samples scores ``None``; reward functions
    without the marker keep NaN-aware handling of missing rewards.
    """

    def _reward_fn(
        *,
        prompts: collections.abc.Sequence[tp.Any] | None = None,
        completions: collections.abc.Sequence[tp.Any] | None = None,
        environment_rewards: collections.abc.Sequence[float | None] | None = None,
        **kwargs: tp.Any,
    ) -> list[float | None]:
        if environment_rewards is not None:
            out_rewards: list[float | None] = []
            for r in environment_rewards:
                if r is None:
                    out_rewards.append(None)
                else:
                    val = float(r)
                    out_rewards.append(val if math.isfinite(val) else None)
            return out_rewards
        if completions is None:
            return []

        comp_texts = [_extract_text(c) for c in completions]
        n = len(comp_texts)
        if n == 0:
            return []

        columns: dict[str, list[tp.Any]] = {}
        batch_payload = kwargs.get("batch")
        if isinstance(batch_payload, collections.abc.Mapping):
            for key, val in batch_payload.items():
                col = _as_column(val, n)
                if col is not None:
                    columns[key] = col
        for key, val in kwargs.items():
            if key == "batch":
                continue
            col = _as_column(val, n)
            if col is not None:
                columns[key] = col

        envs: list[AgenticEnvironment | None] = []
        for idx in range(n):
            row: dict[str, tp.Any] = dict(defaults)
            for key, col in columns.items():
                row[key] = col[idx]
            if prompts is not None and idx < len(prompts):
                row["prompt"] = _extract_text(prompts[idx])
            try:
                env = AutoEnvironment.from_task(
                    row,
                    environment=None if environment == "auto" else environment,
                    runtime=runtime,
                    backend=backend,
                    container_config=container_config,
                )
            except Exception:
                envs.append(None)
                continue
            envs.append(env)

        try:
            active_envs: list[AgenticEnvironment] = []
            active_comps: list[str] = []
            active_indices: list[int] = []
            for idx, (env, comp) in enumerate(zip(envs, comp_texts, strict=True)):
                if env is None:
                    continue
                try:
                    reset_res = env.reset()
                except Exception:
                    continue
                if not reset_res.info.get("env_error"):
                    active_envs.append(env)
                    active_comps.append(comp)
                    active_indices.append(idx)
            # A sample whose environment could not be constructed, reset, or graded is unscorable: ``None``, never 0.0.
            scores: list[float | None] = [None] * n
            if active_envs:
                results = _grade_single_turn_batch(active_envs, active_comps, max_workers=max_workers)
                for idx, res in zip(active_indices, results, strict=True):
                    if res is None or is_unscorable(res):
                        scores[idx] = None
                    else:
                        val = float(res.reward)
                        scores[idx] = val if math.isfinite(val) else None
            return scores
        finally:
            for env in envs:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    _reward_fn.__dict__["discard_unscorable"] = True
    return _reward_fn


__all__ = (
    "AutoEnvironment",
    "is_unscorable",
    "make_environment_reward_fn",
)

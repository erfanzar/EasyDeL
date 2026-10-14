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

"""Benchmark orchestration for ELM models via lm-evaluation-harness.

Provides utilities for normalizing benchmark configurations, resolving
task variants (e.g., instruct siblings), running evaluations against an
eSurge inference engine, and flattening result metrics for logging.
"""

from __future__ import annotations

import collections.abc
import contextlib
import os
import time
from typing import Any

from eformer.loggings import get_logger

from .types import BenchmarkConfig, BenchmarkTask, BenchmarkTasks, ResolvedBenchmarkConfig

logger = get_logger("eLargeModelBenchmarking")

_cached_task_manager: Any = None
_cached_task_manager_key: tuple | None = None


def _get_or_create_task_manager(
    *,
    verbosity: str | None = None,
    include_path: str | list | None = None,
    include_defaults: bool = True,
    metadata: dict | None = None,
    summary_logger: Any = None,
) -> Any:
    """Return a cached ``lm_eval.tasks.TaskManager``, building one on first call.

    The TaskManager eagerly indexes every registered lm-eval task (14k+
    YAML files) which can be slow under CPU contention.  Caching by
    ``(include_path, include_defaults)`` avoids repeating that work when
    the same evaluation harness is invoked multiple times in one process.
    """
    global _cached_task_manager, _cached_task_manager_key
    resolved_logger = summary_logger if summary_logger is not None else logger

    ip = tuple(include_path) if isinstance(include_path, list) else include_path
    key = (ip, include_defaults)

    if _cached_task_manager is not None and _cached_task_manager_key == key:
        resolved_logger.info(
            "Reusing cached lm-eval TaskManager (%s indexed tasks).",
            len(getattr(_cached_task_manager, "all_tasks", []) or []),
        )
        return _cached_task_manager

    from lm_eval.tasks import TaskManager  # type:ignore

    task_manager_start = time.perf_counter()
    resolved_logger.info("Creating lm-eval TaskManager (this is a one-time cost per process).")
    tm = TaskManager(
        verbosity=verbosity,
        include_path=include_path,
        include_defaults=include_defaults,
        metadata=metadata,
    )
    elapsed = time.perf_counter() - task_manager_start
    resolved_logger.info(
        "lm-eval TaskManager ready in %.2fs with %s indexed tasks.",
        elapsed,
        len(getattr(tm, "all_tasks", []) or []),
    )
    _cached_task_manager = tm
    _cached_task_manager_key = key
    return tm


def override_lm_eval_code_exec(*, num_workers: int | None = None, timeout: float | None = None):
    """Return a context manager that patches lm-eval's code-execution scorer.

    Delegates to :func:`~easydel.infra.elarge.processing.override_lm_eval_code_exec`.

    Args:
        num_workers: Number of parallel worker processes for code evaluation.
            ``None`` leaves the lm-eval default unchanged.
        timeout: Per-sample execution timeout in seconds.  ``None`` leaves the
            lm-eval default unchanged.

    Returns:
        A context manager; enter it to apply the overrides, exit to restore
        the original scorer.
    """
    from .processing import override_lm_eval_code_exec as _override_lm_eval_code_exec

    return _override_lm_eval_code_exec(num_workers=num_workers, timeout=timeout)


def _task_name_fragment(task: BenchmarkTask) -> str | None:
    """Extract a short human-readable label from a task specification.

    Tries, in order: the task string itself, well-known mapping keys
    (``task``, ``alias``, ``group``, ``dataset_path``, ``dataset_name``),
    and finally object attributes with the same names.

    Args:
        task: A task string, mapping, or lm-eval task object.

    Returns:
        A non-empty string fragment, or ``None`` if nothing useful was found.
    """
    if isinstance(task, str):
        return task
    if isinstance(task, collections.abc.Mapping):
        for key in ("task", "alias", "group", "dataset_path", "dataset_name"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
    for attr in ("task", "alias", "name"):
        value = getattr(task, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _default_benchmark_name(tasks: collections.abc.Sequence[BenchmarkTask], index: int) -> str:
    """Build an auto-generated benchmark name from its task list.

    Joins up to three task name fragments with ``+``, appending a
    ``+Nmore`` suffix when there are more.  Falls back to
    ``benchmark_<index+1>`` when no fragment can be extracted.

    Args:
        tasks: Sequence of task specifications in the benchmark.
        index: Zero-based position of the benchmark in the parent list,
            used only for the fallback name.

    Returns:
        A non-empty string suitable for use as a dict key.
    """
    fragments = [frag for task in tasks if (frag := _task_name_fragment(task))]
    if not fragments:
        return f"benchmark_{index + 1}"
    head = "+".join(fragment.replace("/", "_") for fragment in fragments[:3])
    if len(fragments) > 3:
        head += f"+{len(fragments) - 3}more"
    return head


def _normalize_tasks(tasks: BenchmarkTasks) -> list[BenchmarkTask]:
    """Coerce a flexible task specification into a flat, non-empty list.

    Args:
        tasks: A single task string/object or a sequence of them.

    Returns:
        A list with at least one element.

    Raises:
        ValueError: If ``tasks`` is an empty sequence.
    """
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, collections.abc.Sequence):
        normalized = list(tasks)
        if not normalized:
            raise ValueError("benchmark `tasks` must not be empty.")
        return normalized
    return [tasks]


def is_benchmark_config_like(value: Any) -> bool:
    """Return ``True`` when *value* looks like a :class:`BenchmarkConfig` mapping.

    A value is considered benchmark-config-like if it is a ``Mapping`` that
    contains the ``"tasks"`` key.

    Args:
        value: Arbitrary object to inspect.

    Returns:
        ``True`` if *value* is a mapping with benchmark task specifications.
    """
    return isinstance(value, collections.abc.Mapping) and "tasks" in value


def normalize_benchmark_configs(
    benchmarks: BenchmarkConfig | collections.abc.Sequence[BenchmarkConfig] | None,
    *,
    default_eval_config: collections.abc.Mapping[str, Any] | None = None,
) -> list[ResolvedBenchmarkConfig]:
    """Validate and resolve a raw benchmark specification into a list of :class:`ResolvedBenchmarkConfig`.

    Each benchmark entry is merged on top of *default_eval_config* (benchmark-
    level settings win).  The ``tasks`` and ``name`` keys are extracted
    and the remaining keys become the ``eval_kwargs`` passed to lm-eval.

    Args:
        benchmarks: A single :class:`BenchmarkConfig` mapping, a sequence of
            them, or ``None`` (returns an empty list).
        default_eval_config: Optional base eval settings shared across all
            benchmarks.  Individual benchmark entries can override any key.

    Returns:
        Ordered list of :class:`ResolvedBenchmarkConfig`, one per input entry.

    Raises:
        TypeError: If *benchmarks* is not a mapping, a sequence of mappings,
            or ``None``; or if any entry lacks a ``tasks`` key.
        ValueError: If a benchmark entry's task list is empty, or if the tasks
            value is ``None``.
    """
    if benchmarks is None:
        return []

    if isinstance(benchmarks, collections.abc.Mapping) and "task" in benchmarks and "tasks" not in benchmarks:
        raise TypeError("benchmark configs must use `tasks`; the `task` alias is no longer supported.")

    if is_benchmark_config_like(benchmarks):
        raw_configs = [benchmarks]
    elif isinstance(benchmarks, collections.abc.Sequence) and not isinstance(benchmarks, (str, bytes)):
        raw_configs = list(benchmarks)
    else:
        raise TypeError("benchmarks must be a BenchmarkConfig or a sequence of BenchmarkConfig values.")

    resolved: list[ResolvedBenchmarkConfig] = []
    for index, benchmark in enumerate(raw_configs):
        if isinstance(benchmark, collections.abc.Mapping) and "task" in benchmark and "tasks" not in benchmark:
            raise TypeError("benchmark configs must use `tasks`; the `task` alias is no longer supported.")
        if not is_benchmark_config_like(benchmark):
            raise TypeError("each benchmark entry must be a mapping containing `tasks`.")

        merged = dict(default_eval_config or {})
        merged.update(dict(benchmark))

        tasks_value = merged.pop("tasks", None)
        if tasks_value is None:
            raise ValueError("benchmark config must define `tasks`.")

        tasks = _normalize_tasks(tasks_value)
        explicit_name = str(merged.pop("name", "") or "").strip()
        name = explicit_name or _default_benchmark_name(tasks, index)
        resolved.append(ResolvedBenchmarkConfig(name=name, tasks=tasks, eval_kwargs=merged))

    return resolved


def task_uses_code_eval(task: Any) -> bool:
    """Return True when a task specification looks like Humaneval/MBPP style code eval."""
    code_eval_hints = ("humaneval", "mbpp")

    def _matches(value: Any) -> bool:
        """Check if a value contains a code-eval task hint.

        Args:
            value: An arbitrary value to test. Only strings are matched.

        Returns:
            True if *value* is a string whose lowercase form contains
            any of the ``code_eval_hints`` (e.g. ``"humaneval"``,
            ``"mbpp"``), False otherwise.
        """
        return isinstance(value, str) and any(hint in value.lower() for hint in code_eval_hints)

    if _matches(task):
        return True
    if isinstance(task, collections.abc.Mapping):
        return any(_matches(task.get(key)) for key in ("task", "alias", "group", "dataset_path", "dataset_name"))
    return any(_matches(getattr(task, attr, None)) for attr in ("task", "alias", "name"))


def auto_code_eval_num_workers() -> int:
    """Return a conservative default worker count for Hugging Face ``code_eval``.

    ``code_eval`` uses a thread pool where each worker also launches a
    subprocess to execute generated code. Large TPU hosts often report very
    large CPU counts, and using that raw value leads to heavy oversubscription
    and extremely long post-generation stalls. Cap the automatic default at 16;
    callers can still override the value explicitly.
    """
    return max(1, min(int(os.cpu_count() or 1), 16))


def maybe_resolve_instruct_task_variants(
    tasks: collections.abc.Sequence[BenchmarkTask],
    *,
    task_manager: Any | None,
    apply_chat_template: bool | str,
    summary_logger: Any | None = None,
) -> list[BenchmarkTask]:
    """Swap raw task names for available ``*_instruct`` variants when chat mode is enabled.

    This keeps benchmark selection dynamic: if lm-eval registers an instruct
    sibling for a task and the caller enables chat templating, prefer the
    instruct variant automatically. Non-string task objects and already-resolved
    instruct task names are preserved unchanged.
    """
    if not apply_chat_template:
        return list(tasks)

    candidates: list[tuple[int, str, str]] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, str) or task.endswith("_instruct"):
            continue
        candidates.append((idx, task, f"{task}_instruct"))

    if not candidates:
        return list(tasks)

    if task_manager is not None:
        available_tasks = set(getattr(task_manager, "all_tasks", []) or [])
    else:
        available_tasks = _probe_task_names([c[2] for c in candidates])

    resolved_tasks = list(tasks)
    resolved_logger = summary_logger or logger
    for idx, orig, instruct in candidates:
        if instruct in available_tasks:
            resolved_logger.info(
                "apply_chat_template enabled; using instruct task variant %s instead of %s.",
                instruct,
                orig,
            )
            resolved_tasks[idx] = instruct
    return resolved_tasks


def _probe_task_names(names: list[str]) -> set[str]:
    """Check which task names are registered in lm-eval without building a full index.

    Falls back to a full TaskManager if targeted lookup isn't possible.
    """
    try:
        global _cached_task_manager
        if _cached_task_manager is not None:
            all_tasks = set(getattr(_cached_task_manager, "all_tasks", []) or [])
            return {n for n in names if n in all_tasks}

        tm = _get_or_create_task_manager()
        all_tasks = set(getattr(tm, "all_tasks", []) or [])
        return {n for n in names if n in all_tasks}
    except Exception as exc:
        logger.warning("Failed to probe lm-eval task names: %s", exc)
        return set()


def _task_declares_generation_prefix(task: BenchmarkTask, task_manager: Any | None) -> bool:
    """Return True when an lm-eval task config already defines a generation prefill."""
    if not isinstance(task, str) or task_manager is None:
        return False
    get_config = getattr(task_manager, "_get_config", None)
    if not callable(get_config):
        return False
    try:
        config = get_config(task)
    except Exception:
        return False
    gen_prefix = config.get("gen_prefix") if isinstance(config, collections.abc.Mapping) else None
    return isinstance(gen_prefix, str) and bool(gen_prefix.strip())


def maybe_disable_chat_template_for_prefilled_tasks(
    tasks: collections.abc.Sequence[BenchmarkTask],
    *,
    task_manager: Any | None,
    apply_chat_template: bool | str,
    summary_logger: Any | None = None,
) -> bool | str:
    """Disable lm-eval chat templating when tasks already define a generation prefix.

    Some lm-eval tasks, especially code tasks such as ``humaneval_instruct``,
    already provide a concrete assistant-side prefix via ``gen_prefix``. Wrapping
    those prompts in an additional tokenizer chat template can turn code
    continuation tasks back into prose-answer tasks. When that prefill is
    present, keep the task-selected prompt but skip the extra chat wrapper.
    """
    if not apply_chat_template:
        return apply_chat_template
    prefixed_tasks = [task for task in tasks if _task_declares_generation_prefix(task, task_manager)]
    if not prefixed_tasks:
        return apply_chat_template
    resolved_logger = summary_logger or logger
    resolved_logger.info(
        "Disabling apply_chat_template for tasks with lm-eval gen_prefix prefill: %s",
        prefixed_tasks,
    )
    return False


def flatten_benchmark_metrics(
    benchmark_name: str,
    results: collections.abc.Mapping[str, Any],
) -> dict[str, float]:
    """Flatten lm-eval result metrics into a single dict with namespaced keys.

    Produces keys of the form
    ``benchmark/<benchmark_name>/<task_name>/<metric_name>`` for every
    numeric metric value found in ``results["results"]``.

    Args:
        benchmark_name: Label for the benchmark suite (used as the second path
            component in each key).
        results: The raw dict returned by ``lm_eval.evaluator.simple_evaluate``.
            Only the ``"results"`` sub-mapping is inspected.

    Returns:
        A flat ``{key: float}`` dict suitable for logging to WandB, TensorBoard,
        or similar.  Boolean metric values are cast to ``float`` (0.0 / 1.0).
        Non-numeric values are silently omitted.
    """
    flattened: dict[str, float] = {}
    result_metrics = results.get("results", {})
    if not isinstance(result_metrics, collections.abc.Mapping):
        return flattened

    for task_name, metrics in result_metrics.items():
        if not isinstance(metrics, collections.abc.Mapping):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, bool):
                flattened[f"benchmark/{benchmark_name}/{task_name}/{metric_name}"] = float(value)
            elif isinstance(value, int | float):
                flattened[f"benchmark/{benchmark_name}/{task_name}/{metric_name}"] = float(value)
    return flattened


def run_lm_eval_with_esurge(
    *,
    surge: Any,
    processor: Any,
    tasks: str | collections.abc.Sequence[BenchmarkTask],
    max_length: int,
    fallback_batch_size: int | str | None,
    num_fewshot: int | None = None,
    eval_config: collections.abc.Mapping[str, Any] | None = None,
    stop_engine: bool = True,
    summary_logger: Any | None = None,
) -> dict[str, Any]:
    """Run lm-evaluation-harness tasks against an :class:`~easydel.inference.eSurge` engine.

    Constructs an :class:`~easydel.inference.evaluations.eSurgeLMEvalAdapter`,
    optionally patches lm-eval's code-execution scorer, and calls
    ``lm_eval.evaluator.simple_evaluate``.  After evaluation the adapter is
    stopped if *stop_engine* is ``True``.

    ELM-specific keys (``max_new_tokens``, ``temperature``, ``sampling_params``,
    ``batch_size``, ``code_eval_num_workers``, etc.) are popped from
    *eval_config* before the remainder is forwarded to lm-eval so that no
    unknown keyword errors occur.

    Args:
        surge: A running :class:`~easydel.inference.eSurge` instance.
        processor: Tokenizer / processor compatible with the model.
        tasks: A single task name string or a sequence of task specifications
            accepted by lm-eval.
        max_length: Maximum sequence length (context + generation) supported by
            the model, forwarded to the adapter.
        fallback_batch_size: Batch size used when *eval_config* does not
            explicitly set ``"batch_size"``.
        num_fewshot: Default few-shot count; overridden by ``"num_fewshot"``
            in *eval_config* if present.
        eval_config: Optional dict of lm-eval and ELM eval settings.  ELM-
            specific keys are consumed here; the rest reach lm-eval unchanged.
        stop_engine: If ``True``, call ``eval_adapter.stop()`` in the
            ``finally`` block.  Pass ``False`` when reusing the engine across
            multiple benchmarks.
        summary_logger: Logger instance for progress messages.  Falls back to
            the module-level logger when ``None``.

    Returns:
        The raw result dict produced by ``lm_eval.evaluator.simple_evaluate``,
        typically ``{"results": {...}, "config": {...}, ...}``.

    Raises:
        ImportError: If ``lm-eval`` is not installed.
    """
    try:
        from lm_eval import evaluator  # type:ignore
    except ImportError as e:
        raise ImportError("lm-eval is required for evaluation. Install with: pip install easydel[torch,lm-eval]") from e

    from easydel.inference.evaluations import eSurgeLMEvalAdapter

    resolved_logger = summary_logger or logger
    task_list = [tasks] if isinstance(tasks, str) else list(tasks)
    config = dict(eval_config or {})
    config.setdefault("apply_chat_template", False)
    apply_chat_template = config.get("apply_chat_template", False)

    batch_size = config.pop("batch_size", fallback_batch_size)
    max_new_tokens = config.pop("max_new_tokens", 8192)
    hard_max_new_tokens = bool(config.pop("hard_max_new_tokens", False))
    enable_thinking = bool(config.pop("enable_thinking", False))
    chat_template_args = config.pop("chat_template_args", None)
    think_start_token = config.pop("think_start_token", None)
    think_end_token = config.pop("think_end_token", None)
    ignore_benchmark_eos_flags = bool(config.pop("ignore_benchmark_eos_flags", False))
    temperature = config.pop("temperature", 0.0)
    top_p = config.pop("top_p", 0.95)
    sampling_params = config.pop("sampling_params", None)
    device = config.pop("device", "cpu")
    num_fewshot = config.pop("num_fewshot", num_fewshot)
    normalize_math_answers = bool(config.pop("normalize_math_answers", True))
    math_answer_task_hints = config.pop("math_answer_task_hints", None)
    code_eval_num_workers = config.pop("code_eval_num_workers", None)
    code_eval_timeout = config.pop("code_eval_timeout", None)
    if code_eval_num_workers is None and any(task_uses_code_eval(task) for task in task_list):
        code_eval_num_workers = auto_code_eval_num_workers()
        resolved_logger.info(
            "Auto-selected code_eval_num_workers=%s (os.cpu_count()=%s).",
            code_eval_num_workers,
            int(os.cpu_count() or 1),
        )
    include_path = config.pop("include_path", None)
    include_defaults = bool(config.pop("include_defaults", True))
    task_manager = config.pop("task_manager", None)

    adapter_start = time.perf_counter()
    resolved_logger.info("Preparing eSurge lm-eval adapter.")
    eval_adapter = eSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        hard_max_new_tokens=hard_max_new_tokens,
        enable_thinking=enable_thinking,
        chat_template_args=chat_template_args,
        think_start_token=think_start_token,
        think_end_token=think_end_token,
        ignore_benchmark_eos_flags=ignore_benchmark_eos_flags,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        sampling_params=sampling_params,
        normalize_math_answers=normalize_math_answers,
        math_answer_task_hints=math_answer_task_hints,
    )
    resolved_logger.info("eSurge lm-eval adapter ready in %.2fs.", time.perf_counter() - adapter_start)

    if task_manager is None and (include_path is not None or not include_defaults):
        task_manager = _get_or_create_task_manager(
            verbosity=config.get("verbosity"),
            include_path=include_path,
            include_defaults=include_defaults,
            metadata=config.get("metadata"),
            summary_logger=resolved_logger,
        )

    task_list = maybe_resolve_instruct_task_variants(
        task_list,
        task_manager=task_manager,
        apply_chat_template=apply_chat_template,
        summary_logger=resolved_logger,
    )

    effective_task_manager = task_manager or _cached_task_manager
    apply_chat_template = maybe_disable_chat_template_for_prefilled_tasks(
        task_list,
        task_manager=effective_task_manager,
        apply_chat_template=apply_chat_template,
        summary_logger=resolved_logger,
    )
    config["apply_chat_template"] = apply_chat_template

    if code_eval_num_workers is not None or code_eval_timeout is not None:
        code_eval_override_ctx = override_lm_eval_code_exec(
            num_workers=code_eval_num_workers,
            timeout=code_eval_timeout,
        )
    else:
        code_eval_override_ctx = contextlib.nullcontext()

    try:
        resolved_logger.info(f"Starting evaluation on tasks: {task_list}")
        resolved_logger.info("Using eSurge engine")
        resolved_logger.info(
            "Batch size: %s, Few-shot: %s",
            batch_size,
            num_fewshot if num_fewshot is not None else "task-default",
        )
        eval_start = time.perf_counter()
        with code_eval_override_ctx:
            results = evaluator.simple_evaluate(
                model=eval_adapter,
                tasks=task_list,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                task_manager=effective_task_manager,
                **config,
            )
        resolved_logger.info("lm_eval.simple_evaluate finished in %.2fs.", time.perf_counter() - eval_start)
        resolved_logger.info("evaluation summary:")
        for task_name, metrics in results.get("results", {}).items():
            resolved_logger.info(f"{task_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    resolved_logger.info(
                        f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}"
                    )
        return results
    finally:
        if stop_engine and hasattr(eval_adapter, "stop"):
            eval_adapter.stop()

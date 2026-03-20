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

"""Evaluation and benchmark TypedDicts."""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

BenchmarkTask = str | dict[str, Any] | Any
"""A single benchmark task specification: a task name string, a task config dict, or an lm-eval task object."""

BenchmarkTasks = BenchmarkTask | collections.abc.Sequence[BenchmarkTask]
"""One or more benchmark task specifications."""


class EvalKwargs(TypedDict, total=False):
    """Keyword arguments passed to lm-evaluation-harness's ``simple_evaluate``
    function, plus EasyDeL-specific evaluation extensions.

    Attributes:
        num_fewshot: Number of few-shot examples. None for task default.
        max_new_tokens: Maximum tokens to generate per evaluation prompt.
        hard_max_new_tokens: If True, strictly enforce *max_new_tokens* even if
            the task requests more.
        enable_thinking: Enable chain-of-thought reasoning mode during evaluation.
        chat_template_args: Extra arguments passed to the chat template formatter.
            None for defaults.
        think_start_token: Token marking the start of a thinking block. None for
            model default.
        think_end_token: Token marking the end of a thinking block. None for
            model default.
        ignore_benchmark_eos_flags: Ignore EOS flags from the benchmark task
            definitions.
        temperature: Sampling temperature for generation (0.0 = greedy).
        top_p: Nucleus sampling probability threshold.
        normalize_math_answers: Whether to normalize mathematical answers before
            comparison.
        math_answer_task_hints: Task name patterns that indicate math answer
            normalization should apply.
        code_eval_num_workers: Number of parallel workers for code evaluation
            tasks. None for auto.
        code_eval_timeout: Per-sample timeout in seconds for code execution. None
            for default.
        batch_size: Evaluation batch size. Can be int or ``"auto"``. None for
            engine default.
        max_batch_size: Upper bound on auto batch size. None for no limit.
        device: Device string for lm-eval (e.g., ``"cpu"``). None for default.
        use_cache: Path to cache evaluation requests. None to disable caching.
        limit: Limit the number of evaluation examples. Float for fraction, int
            for count. None for all.
        cache_requests: Whether to cache individual evaluation requests.
        rewrite_requests_cache: Whether to overwrite existing request cache.
        delete_requests_cache: Whether to delete the request cache after
            evaluation.
        check_integrity: Whether to verify dataset integrity before evaluation.
        write_out: Whether to write detailed evaluation outputs.
        log_samples: Whether to log individual sample results.
        evaluation_tracker: Optional evaluation tracker object for logging. None
            to disable.
        system_instruction: System instruction prepended to all prompts. None for
            none.
        apply_chat_template: Whether to apply chat template formatting. Can be
            bool or template name string.
        fewshot_as_multiturn: Format few-shot examples as multi-turn conversation.
        gen_kwargs: Additional generation kwargs as string or dict. None for
            defaults.
        task_manager: Custom lm-eval TaskManager instance. None to create
            automatically.
        verbosity: Logging verbosity level for lm-eval.
        predict_only: If True, only run prediction without scoring.
        samples: Pre-computed samples dict. None to generate fresh samples.
        bootstrap_iters: Number of bootstrap iterations for confidence intervals.
        random_seed: Global random seed. None for non-deterministic.
        numpy_random_seed: NumPy random seed. None inherits from *random_seed*.
        torch_random_seed: PyTorch random seed. None inherits from *random_seed*.
        fewshot_random_seed: Seed for few-shot example selection. None inherits
            from *random_seed*.
        confirm_run_unsafe_code: Whether to allow execution of untrusted code in
            evaluation.
        metadata: Additional metadata dict attached to evaluation results. None
            for none.
        include_path: Path to additional lm-eval task definitions. None for
            built-in tasks only.
        include_defaults: Whether to include default lm-eval task definitions.
    """

    num_fewshot: NotRequired[int | None]
    max_new_tokens: NotRequired[int]
    hard_max_new_tokens: NotRequired[bool]
    enable_thinking: NotRequired[bool]
    chat_template_args: NotRequired[dict[str, Any] | None]
    think_start_token: NotRequired[str | None]
    think_end_token: NotRequired[str | None]
    ignore_benchmark_eos_flags: NotRequired[bool]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    normalize_math_answers: NotRequired[bool]
    math_answer_task_hints: NotRequired[collections.abc.Sequence[str] | str | None]
    code_eval_num_workers: NotRequired[int | None]
    code_eval_timeout: NotRequired[float | int | None]
    batch_size: NotRequired[int | str | None]
    max_batch_size: NotRequired[int | None]
    device: NotRequired[str | None]
    use_cache: NotRequired[str | None]
    limit: NotRequired[int | float | None]
    cache_requests: NotRequired[bool]
    rewrite_requests_cache: NotRequired[bool]
    delete_requests_cache: NotRequired[bool]
    check_integrity: NotRequired[bool]
    write_out: NotRequired[bool]
    log_samples: NotRequired[bool]
    evaluation_tracker: NotRequired[Any | None]
    system_instruction: NotRequired[str | None]
    apply_chat_template: NotRequired[bool | str]
    fewshot_as_multiturn: NotRequired[bool]
    gen_kwargs: NotRequired[str | dict[str, Any] | None]
    task_manager: NotRequired[Any | None]
    verbosity: NotRequired[Any]
    predict_only: NotRequired[bool]
    samples: NotRequired[dict[str, Any] | None]
    bootstrap_iters: NotRequired[int]
    random_seed: NotRequired[int | None]
    numpy_random_seed: NotRequired[int | None]
    torch_random_seed: NotRequired[int | None]
    fewshot_random_seed: NotRequired[int | None]
    confirm_run_unsafe_code: NotRequired[bool]
    metadata: NotRequired[dict[str, Any] | None]
    include_path: NotRequired[str | None]
    include_defaults: NotRequired[bool]


class BenchmarkConfig(EvalKwargs, total=False):
    """Configuration for a single named benchmark suite.

    Inherits all evaluation keyword arguments from :class:`EvalKwargs` and adds
    benchmark-level identity and task selection fields.

    Attributes:
        name: Human-readable name for this benchmark suite.
        tasks: Task specification(s) to evaluate.
    """

    name: NotRequired[str]
    tasks: NotRequired[BenchmarkTasks]


@dataclass(slots=True)
class ResolvedBenchmarkConfig:
    """Normalized, ready-to-run form of a :class:`BenchmarkConfig` after validation.

    Attributes:
        name: Resolved benchmark name (auto-generated if not provided).
        tasks: Flat list of resolved task specifications.
        eval_kwargs: Merged evaluation keyword arguments ready for lm-eval.
    """

    name: str
    tasks: list[BenchmarkTask]
    eval_kwargs: dict[str, Any]

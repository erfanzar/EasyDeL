# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from dataclasses import field
from functools import cached_property

import jax
from eformer.pytree import auto_pytree

from .logits_process import (
    FrequencyPenaltyLogitsProcessor,
    LogitsProcessorList,
    MinPLogitsWarper,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    hash_fn,
)


@auto_pytree
class SamplingParams:
    """
    Parameters controlling the sampling process during text generation.

    Attributes:
        max_tokens: The maximum number of tokens to generate (excluding the prompt).
            Defaults to 16.
        presence_penalty: Penalty applied to the logits of tokens already present in the
            generated sequence. Positive values discourage repetition. Defaults to 0.0.
        frequency_penalty: Penalty applied to the logits of tokens based on their frequency
            in the generated sequence so far. Positive values discourage verbatim repetition.
            Defaults to 0.0.
        repetition_penalty: Multiplicative penalty applied to the logits of previously seen tokens.
            Values > 1.0 discourage repetition, < 1.0 encourage it. Defaults to 1.0.
        temperature: Controls the randomness of the sampling. Higher values (e.g., > 1.0)
            make the distribution flatter (more random), lower values (e.g., < 1.0) make it
            peakier (more deterministic). A value of 0.0 effectively becomes greedy sampling.
            Defaults to 0.0.
        top_p: Nucleus sampling threshold. If set to a value < 1.0, only the most probable
            tokens with a cumulative probability exceeding `top_p` are considered for sampling.
            Defaults to 1.0 (no nucleus sampling).
        top_k: Top-k sampling threshold. If set to a value > 0, only the `top_k` most probable
            tokens are considered for sampling. Defaults to 0 (no top-k sampling).
        min_p: Minimum probability threshold. Filters out tokens with probability less than `min_p`.
            Defaults to 0.0 (no minimum probability filtering).
        suppress_tokens: A list of token IDs that should be completely suppressed (their logits
            set to -inf) during generation. Defaults to an empty list.
    """

    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = field(default_factory=lambda: list())
    stop: list[str] | str | None = field(default=None)
    n: int | None = field(default=1)

    def __post_init__(self):
        self.suppress_tokens = []  # not supported yet!

    def get_logits_warper(self):
        """
        Constructs a `LogitsProcessorList` containing the configured logits warpers.

        Logits warpers modify the probability distribution derived from logits, typically
        used for techniques like temperature scaling, top-k, top-p, and min-p sampling.

        Returns:
            A `LogitsProcessorList` containing the enabled logits warpers based on the
            sampling parameters.
        """
        warpers = LogitsProcessorList()
        warpers.append(TemperatureLogitsWarper(temperature=self.temperature))
        warpers.append(TopKLogitsWarper(top_k=self.top_k, min_tokens_to_keep=1))
        warpers.append(TopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=1))
        warpers.append(MinPLogitsWarper(min_p=self.min_p, min_tokens_to_keep=1))
        return warpers

    def get_logits_processor(self):
        """
        Constructs a `LogitsProcessorList` containing the configured logits processors.

        Logits processors modify the logits directly, often used for applying penalties
        (presence, frequency, repetition) or suppressing specific tokens.

        Returns:
            A `LogitsProcessorList` containing the enabled logits processors based on the
            sampling parameters.
        """
        processors = LogitsProcessorList()
        processors.append(SuppressTokensLogitsProcessor(self.suppress_tokens))
        processors.append(PresencePenaltyLogitsProcessor(self.presence_penalty))
        processors.append(FrequencyPenaltyLogitsProcessor(self.frequency_penalty))
        processors.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))

        return processors

    @cached_property
    def logits_processor(self):
        return self.get_logits_processor()

    @cached_property
    def logits_warper(self):
        return self.get_logits_warper()

    __hash__ = hash_fn


def lower_function(
    func,
    func_input_args,
    func_input_kwargs,
    mesh=None,
    in_shardings=None,
    out_shardings=None,
    static_argnums=None,
    donate_argnums=None,
):
    """
    Lowers a JAX function to its HLO (High-Level Optimizer) representation,
    optionally configuring sharding and device mesh for distributed execution.

    Lowering separates the definition of the computation from its compilation,
    allowing for inspection or manipulation of the HLO before final compilation.

    Args:
        func: The JAX function to lower.
        func_input_args: A tuple of positional arguments for the function.
        func_input_kwargs: A dictionary of keyword arguments for the function.
        mesh: An optional `jax.sharding.Mesh` object specifying the device topology
            for distributed execution. If provided, `jax.jit` is called within the
            mesh context.
        in_shardings: Optional sharding specifications for the input arguments.
            Can be a PyTree matching the structure of `func_input_args` and
            `func_input_kwargs`, containing `jax.sharding.PartitionSpec` objects.
        out_shardings: Optional sharding specifications for the output of the function.
            Can be a PyTree matching the function's output structure.
        static_argnums: Indices of positional arguments that should be treated as
            static (compile-time constants).
        donate_argnums: Indices of positional arguments whose underlying buffers can be
            donated (potentially modified in-place) to save memory.

    Returns:
        A `jax.Lowered` object representing the HLO computation.
    """
    if mesh is None:
        return jax.jit(
            func,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
        ).lower(*func_input_args, **func_input_kwargs)
    with mesh:
        return jax.jit(
            func,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
        ).lower(*func_input_args, **func_input_kwargs)


def compile_function(
    func,
    func_input_args,
    func_input_kwargs,
    mesh=None,
    in_shardings=None,
    out_shardings=None,
    static_argnums=None,
    donate_argnums=None,
):
    """
    Compiles a JAX function, potentially ready for distributed execution,
    after lowering it.

    This function first lowers the JAX function using `lower_function` and then
    calls `.compile()` on the lowered representation to produce an executable.

    Args:
        func: The JAX function to compile.
        func_input_args: A tuple of positional arguments for the function.
        func_input_kwargs: A dictionary of keyword arguments for the function.
        mesh: An optional `jax.sharding.Mesh` object for distributed execution.
        in_shardings: Optional sharding specifications for the input arguments.
        out_shardings: Optional sharding specifications for the output.
        static_argnums: Indices of static positional arguments.
        donate_argnums: Indices of positional arguments to donate.

    Returns:
        A compiled JAX function (typically a `jax.stages.Compiled` object).
    """
    return lower_function(
        func,
        func_input_args,
        func_input_kwargs,
        mesh=mesh,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    ).compile()

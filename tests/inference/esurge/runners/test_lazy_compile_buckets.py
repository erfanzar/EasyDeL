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

from easydel.inference.esurge.runners.execution_manager import ExecutionManager


class _SpmdMesh:
    is_mpmd = False


class _Model:
    mesh = _SpmdMesh()


def test_spmd_compile_warms_minimal_bucket_instead_of_full_shape_grid():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = _Model()
    manager.use_aot_forward = False
    manager.min_input_pad = 4
    manager._sampler_min_input_pad = 1
    manager._runtime_lazy_compile = True
    manager._model_num_tokens_paddings = []
    manager._runtime_compile_metadata = None
    manager._runtime_compile_max_num_reqs = None

    model_step_calls: list[tuple[int, int]] = []
    sampler_calls: list[int] = []

    manager._compile_model_step_variant = lambda **kwargs: model_step_calls.append(
        (int(kwargs["num_tokens"]), int(kwargs["padded_num_reqs"]))
    )
    manager._compile_sampler_variant = lambda **kwargs: sampler_calls.append(int(kwargs["padded_num_reqs"]))

    ExecutionManager.compile(
        manager,
        num_tokens_paddings=[4, 8, 16, 32],
        num_reqs_max_model_len=8,
        max_pages_per_req=1,
        max_num_reqs=8,
        metadata=object(),
        num_reqs_paddings=[4, 8],
        prune_infeasible_pairs=True,
    )

    assert model_step_calls == [(4, 4)]
    assert sampler_calls == [1]
    assert manager._model_num_tokens_paddings == [4, 8, 16, 32]


def test_spmd_runtime_ensure_compiles_exact_missing_bucket_once():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = _Model()
    manager._runtime_lazy_compile = True
    manager._runtime_compile_metadata = object()
    manager._runtime_compile_max_num_reqs = 8
    manager._runtime_compile_lock = type(
        "NoopLock",
        (),
        {"__enter__": lambda self: self, "__exit__": lambda self, *args: None},
    )()

    compiled_model_steps: set[tuple[int, int]] = {(4, 4)}
    compiled_samplers: set[tuple[int, int, str, str]] = {(0, 1, "sampler", "aot")}
    model_step_calls: list[tuple[int, int]] = []
    sampler_calls: list[int] = []

    class _ModelExecutor:
        supports_pipeline_model_step = False

        @staticmethod
        def has_model_step(num_tokens: int, padded_num_reqs: int) -> bool:
            return (int(num_tokens), int(padded_num_reqs)) in compiled_model_steps

    class _SamplerExecutor:
        @staticmethod
        def cache_key(*, padded_num_reqs: int) -> tuple[int, int, str, str]:
            return (0, int(padded_num_reqs), "sampler", "aot")

        @staticmethod
        def has(key: tuple[int, int, str, str]) -> bool:
            return key in compiled_samplers

    manager._model_executor = _ModelExecutor()
    manager._sampler_executor = _SamplerExecutor()

    def compile_model_step(**kwargs):
        key = (int(kwargs["num_tokens"]), int(kwargs["padded_num_reqs"]))
        model_step_calls.append(key)
        compiled_model_steps.add(key)

    def compile_sampler(**kwargs):
        reqs = int(kwargs["padded_num_reqs"])
        sampler_calls.append(reqs)
        compiled_samplers.add((0, reqs, "sampler", "aot"))

    manager._compile_model_step_variant = compile_model_step
    manager._compile_sampler_variant = compile_sampler

    manager._ensure_runtime_variants(num_tokens=16, padded_num_reqs=8, sampler_padded_num_reqs=4)
    manager._ensure_runtime_variants(num_tokens=16, padded_num_reqs=8, sampler_padded_num_reqs=4)

    assert model_step_calls == [(16, 8)]
    assert sampler_calls == [4]

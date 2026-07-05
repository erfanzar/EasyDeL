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

"""xla_compiler_options must actually reach the train-step jit.

Regression context: the knob was added to TrainingArguments and forwarded by
``compile_trainer_step`` only when ``arguments=`` is passed -- but no trainer
call site passed it, so the option was silently dead on every path. The
trainer call sites now pass ``arguments=self.arguments`` (and the bucketed
compile path includes it in ``_bucket_step_compile_kwargs``); these tests pin
the forwarding contract itself.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from easydel.trainers import training_utils
from easydel.trainers.training_utils import compile_trainer_step


def _capture_jit(monkeypatch):
    captured: dict = {}

    def fake_jit(fn, **kwargs):
        captured.update(kwargs)

        def compiled(*a, **k):
            return fn(*a, **k)

        return compiled

    monkeypatch.setattr(training_utils.spx, "jit", fake_jit)
    return captured


def test_arguments_xla_compiler_options_reach_jit(monkeypatch):
    captured = _capture_jit(monkeypatch)
    opts = {"xla_tpu_enable_latency_hiding_scheduler": "true"}
    arguments = SimpleNamespace(xla_compiler_options=opts, mpmd_scheduler=None)
    compile_trainer_step(lambda x: x, arguments=arguments)
    assert captured.get("compiler_options") == opts


def test_no_arguments_means_no_compiler_options(monkeypatch):
    captured = _capture_jit(monkeypatch)
    compile_trainer_step(lambda x: x)
    assert "compiler_options" not in captured


def test_unset_field_means_no_compiler_options(monkeypatch):
    captured = _capture_jit(monkeypatch)
    arguments = SimpleNamespace(xla_compiler_options=None, mpmd_scheduler=None)
    compile_trainer_step(lambda x: x, arguments=arguments)
    assert "compiler_options" not in captured


def test_explicit_jit_kwarg_wins_over_arguments(monkeypatch):
    captured = _capture_jit(monkeypatch)
    arguments = SimpleNamespace(xla_compiler_options={"a": "1"}, mpmd_scheduler=None)
    compile_trainer_step(lambda x: x, arguments=arguments, compiler_options={"b": "2"})
    assert captured.get("compiler_options") == {"b": "2"}


def test_every_trainer_flavor_passes_arguments_to_compile(monkeypatch):
    """Static wiring check: each call site that compiles a train/eval step with
    schedule=self.arguments.mpmd_scheduler must also pass arguments=self.arguments,
    or the xla_compiler_options knob silently regresses to a no-op there."""
    import pathlib

    import easydel.trainers as trainers_pkg

    root = pathlib.Path(trainers_pkg.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "schedule=self.arguments.mpmd_scheduler,":
                window = "\n".join(lines[max(0, i - 12) : i + 12])
                if "arguments=self.arguments," not in window and '"arguments": self.arguments,' not in window:
                    offenders.append(f"{path.relative_to(root)}:{i + 1}")
    assert not offenders, f"compile call sites missing arguments=self.arguments: {offenders}"

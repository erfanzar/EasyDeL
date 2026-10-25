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

"""Speculation config and engine-to-runner contracts without array computation."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.config import eSurgeConfig, eSurgeRuntimeConfig


@pytest.mark.cpu_ok
def test_speculation_defaults_and_config_roundtrip():
    for runtime in (eSurgeRuntimeConfig.coerce_config(None), eSurgeConfig.from_dict().runtime):
        assert runtime.spec_debug_max_traces == 0
        assert runtime.spec_adaptive_conf_threshold == 0.0
    values = {"spec_debug_max_traces": 5, "spec_adaptive_conf_threshold": 0.75}
    runtime = eSurgeRuntimeConfig.coerce_config(values)
    assert all(runtime[key] == value for key, value in values.items())
    aggregate = eSurgeConfig.from_dict(runtime=values)
    assert eSurgeConfig.from_dict(aggregate.to_dict()).runtime == runtime
    zeroed = runtime.replace(spec_debug_max_traces=0, spec_adaptive_conf_threshold=0.0)
    assert zeroed.spec_debug_max_traces == 0
    assert zeroed.spec_adaptive_conf_threshold == 0.0
    assert runtime.spec_debug_max_traces == 5
    overridden = eSurgeRuntimeConfig.from_dict(values, spec_debug_max_traces=0, spec_adaptive_conf_threshold=0.0)
    assert overridden == zeroed


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "field,value",
    [("spec_debug_max_traces", value) for value in (-1, 0.5, 1.0, True, "2", None, float("inf"), float("nan"))]
    + [
        ("spec_adaptive_conf_threshold", value)
        for value in (-0.1, 1.1, float("nan"), float("inf"), float("-inf"), None, "invalid", [])
    ],
)
def test_speculation_config_rejects_invalid_settings(field, value):
    settings = {field: value}
    for construct in (
        lambda: eSurgeRuntimeConfig.from_dict(settings),
        lambda: eSurgeRuntimeConfig.coerce_config(settings),
        lambda: eSurgeConfig.from_dict(runtime=settings),
        lambda: eSurgeRuntimeConfig.from_dict().replace(**settings),
    ):
        with pytest.raises(ValueError, match=field):
            construct()


@pytest.mark.cpu_ok
@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_speculation_confidence_valid_range(confidence):
    runtime = eSurgeRuntimeConfig.from_dict(spec_debug_max_traces=2, spec_adaptive_conf_threshold=confidence)
    assert runtime.spec_debug_max_traces == 2
    assert runtime.spec_adaptive_conf_threshold == confidence


class _RunnerReached(Exception):
    def __init__(self, options):
        self.options = options


@pytest.mark.cpu_ok
@pytest.mark.parametrize("runtime_form", ["mapping", "typed", "aggregate"])
@pytest.mark.parametrize("overrides,expected", [(None, (5, 0.75)), ((0, 0.0), (0, 0.0)), ((3, 0.25), (3, 0.25))])
def test_engine_inherits_runtime_and_preserves_explicit_overrides(monkeypatch, runtime_form, overrides, expected):
    """Run the real constructor up to a non-computing runner boundary."""
    engine_class = esurge_engine.eSurge
    monkeypatch.setattr(engine_class, "__del__", lambda self: None)
    assets = SimpleNamespace(
        model=SimpleNamespace(esurge_compatible_model=object()),
        processor=None,
        tokenizer=object(),
        tokenizer_source="test",
        tool_parser_name=None,
        reasoning_parser_name=None,
        tool_parser_class=None,
        reasoning_parser_class=None,
        drafter=None,
        model_name="test",
    )
    monkeypatch.setattr(esurge_engine, "build_engine_assets", lambda **kwargs: assets)
    monkeypatch.setattr(engine_class, "_apply_data_parallel_axis_to_model", lambda *args: None)
    monkeypatch.setattr(engine_class, "_warn_if_data_parallel_axis_shards_the_model", lambda *args: None)
    monkeypatch.setattr(
        esurge_engine,
        "WorkerManager",
        lambda *args, **kwargs: SimpleNamespace(
            start=lambda **kwargs: (None, None), tokenizer_endpoint=None, detokenizer_endpoint=None
        ),
    )

    def runner(**kwargs):
        raise _RunnerReached(kwargs)

    monkeypatch.setattr(esurge_engine, "eSurgeRunner", runner)
    settings = {"max_num_batched_tokens": 32, "spec_debug_max_traces": 5, "spec_adaptive_conf_threshold": 0.75}
    if runtime_form == "aggregate":
        options = {"config": eSurgeConfig.from_dict(runtime=settings)}
    else:
        options = {"runtime": eSurgeRuntimeConfig.from_dict(settings) if runtime_form == "typed" else settings}
    if overrides is not None:
        options.update(spec_debug_max_traces=overrides[0], spec_adaptive_conf_threshold=overrides[1])
    with pytest.raises(_RunnerReached) as captured:
        engine_class(model="test", **options)
    forwarded = captured.value.options
    assert (forwarded["spec_debug_max_traces"], forwarded["spec_adaptive_conf_threshold"]) == expected

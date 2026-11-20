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

"""Scheduler runtime configuration contracts without array computation."""

import pytest
from easydel.inference.esurge.config import eSurgeConfig, eSurgeRuntimeConfig


@pytest.mark.cpu_ok
def test_scheduler_runtime_defaults_preserve_existing_behavior():
    for runtime in (eSurgeRuntimeConfig.from_dict(), eSurgeConfig.from_dict().runtime):
        assert runtime.chunked_prefill_enabled is False
        assert runtime.token_safety_margin is None
        assert runtime.include_finished_set is False
        assert runtime.dp_admission_hold == 64


@pytest.mark.cpu_ok
@pytest.mark.parametrize("margin", [None, 0, 64])
def test_scheduler_settings_survive_nested_config_and_replacement(margin):
    settings = {
        "chunked_prefill_enabled": True,
        "token_safety_margin": margin,
        "include_finished_set": True,
    }
    config = eSurgeConfig.from_dict(runtime=settings)
    runtime = config.runtime
    assert all(runtime[key] == value for key, value in settings.items())
    restored = eSurgeConfig.from_dict(config.to_dict())
    assert restored.runtime == runtime
    updated = runtime.replace(max_num_seqs=8)
    assert all(updated[key] == value for key, value in settings.items())
    assert updated.max_num_seqs == 8
    assert runtime.max_num_seqs == 256


@pytest.mark.cpu_ok
def test_scheduler_runtime_keyword_overrides_mapping():
    runtime = eSurgeRuntimeConfig.from_dict(
        {"chunked_prefill_enabled": False, "token_safety_margin": 8, "include_finished_set": False},
        chunked_prefill_enabled=True,
        token_safety_margin=16,
        include_finished_set=True,
    )
    assert runtime.chunked_prefill_enabled is True
    assert runtime.token_safety_margin == 16
    assert runtime.include_finished_set is True


@pytest.mark.cpu_ok
@pytest.mark.parametrize("margin", [-1, 0.5, 1.0, float("inf"), float("nan"), True, "8"])
def test_scheduler_runtime_rejects_invalid_token_safety_margin(margin):
    with pytest.raises(ValueError, match="token_safety_margin must be a non-negative integer or None"):
        eSurgeRuntimeConfig.from_dict(token_safety_margin=margin)
    with pytest.raises(ValueError, match="token_safety_margin must be a non-negative integer or None"):
        eSurgeConfig.from_dict(runtime={"token_safety_margin": margin})
    with pytest.raises(ValueError, match="token_safety_margin must be a non-negative integer or None"):
        eSurgeRuntimeConfig.from_dict().replace(token_safety_margin=margin)


@pytest.mark.cpu_ok
def test_dp_admission_hold_survives_nested_config_and_zero_turns_it_off():
    config = eSurgeConfig.from_dict(runtime={"dp_admission_hold": 0})
    assert config.runtime.dp_admission_hold == 0
    assert eSurgeConfig.from_dict(config.to_dict()).runtime.dp_admission_hold == 0
    assert eSurgeRuntimeConfig.from_dict(dp_admission_hold=16).replace(max_num_seqs=8).dp_admission_hold == 16


@pytest.mark.cpu_ok
def test_freeze_startup_heap_defaults_on_and_survives_nested_config():
    assert eSurgeRuntimeConfig.from_dict().freeze_startup_heap is True
    assert eSurgeConfig.from_dict().runtime.freeze_startup_heap is True
    config = eSurgeConfig.from_dict(runtime={"freeze_startup_heap": False})
    assert config.runtime.freeze_startup_heap is False
    assert eSurgeConfig.from_dict(config.to_dict()).runtime.freeze_startup_heap is False
    assert config.runtime.replace(max_num_seqs=8).freeze_startup_heap is False


@pytest.mark.cpu_ok
@pytest.mark.parametrize("hold", [-1, 0.5, True, "8", None])
def test_scheduler_runtime_rejects_invalid_dp_admission_hold(hold):
    with pytest.raises(ValueError, match="dp_admission_hold must be a non-negative integer"):
        eSurgeRuntimeConfig.from_dict(dp_admission_hold=hold)

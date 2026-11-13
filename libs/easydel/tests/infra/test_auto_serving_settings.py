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

"""Automatic serving settings: tri-state config fields whose default ``None`` lets the reading code decide.

``moe_defer_tp_reduce``, ``gather_shared_projection_input``, ``kda_serving_ep_split`` and ``dsa_serving_ep_split``
default to ``None``. Both config entry points accept ``None``, ``True`` or ``False`` and nothing else. Configs saved
while the default was ``False`` store ``false`` for every one of them, so a saved ``false`` loads as ``None``; ``False``
passed when loading still turns a setting off.
"""

from __future__ import annotations

import json

import easydel as ed
import pytest
from easydel.modules.glm5_next import Glm5NextConfig

pytestmark = pytest.mark.cpu_ok

SETTINGS = ("moe_defer_tp_reduce", "gather_shared_projection_input", "kda_serving_ep_split", "dsa_serving_ep_split")


def _config(**kwargs) -> ed.LlamaConfig:
    return ed.LlamaConfig(hidden_size=64, num_attention_heads=4, **kwargs)


def _settings(config) -> dict[str, bool | None]:
    return {name: getattr(config, name) for name in SETTINGS}


def _save_with(tmp_path, values: dict[str, bool | None]) -> str:
    """Save a small config, then overwrite the given settings in its config.json, as an older save stored them."""
    _config().save_pretrained(str(tmp_path))
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.update(values)
    path.write_text(json.dumps(saved))
    return str(tmp_path)


def test_every_setting_defaults_to_none():
    assert _settings(_config()) == dict.fromkeys(SETTINGS)


@pytest.mark.parametrize("value", [None, True, False])
@pytest.mark.parametrize("name", SETTINGS)
def test_both_entry_points_keep_none_true_and_false(name, value):
    assert getattr(_config(**{name: value}), name) is value
    config = _config()
    config.add_basic_configurations(**{name: value})
    assert getattr(config, name) is value


@pytest.mark.parametrize("bad", ["yes", 1, 0])
@pytest.mark.parametrize("name", SETTINGS)
def test_both_entry_points_reject_anything_but_none_or_a_boolean(name, bad):
    with pytest.raises(TypeError, match=name):
        _config(**{name: bad})
    config = _config()
    with pytest.raises(TypeError, match=name):
        config.add_basic_configurations(**{name: bad})


def test_a_saved_false_loads_as_none_and_a_saved_true_stays_on(tmp_path):
    path = _save_with(tmp_path, {**dict.fromkeys(SETTINGS, False), "dsa_serving_ep_split": True})
    loaded = ed.LlamaConfig.from_pretrained(path)
    assert _settings(loaded) == {**dict.fromkeys(SETTINGS), "dsa_serving_ep_split": True}


def test_false_passed_when_loading_turns_a_setting_off(tmp_path):
    path = _save_with(tmp_path, dict.fromkeys(SETTINGS, False))
    loaded = ed.LlamaConfig.from_pretrained(path, moe_defer_tp_reduce=False)
    loaded.add_basic_configurations(kda_serving_ep_split=False)
    expected = {**dict.fromkeys(SETTINGS), "moe_defer_tp_reduce": False, "kda_serving_ep_split": False}
    assert _settings(loaded) == expected
    assert loaded.moe_defer_tp_reduce is False and loaded.kda_serving_ep_split is False


def test_true_and_none_survive_a_save_and_load(tmp_path):
    _config(moe_defer_tp_reduce=True, gather_shared_projection_input=True).save_pretrained(str(tmp_path))
    loaded = ed.LlamaConfig.from_pretrained(str(tmp_path))
    assert _settings(loaded) == {
        "moe_defer_tp_reduce": True,
        "gather_shared_projection_input": True,
        "kda_serving_ep_split": None,
        "dsa_serving_ep_split": None,
    }


def test_a_saved_false_inside_a_sub_config_loads_as_none():
    composite = Glm5NextConfig.from_dict({"text_config": dict.fromkeys(SETTINGS, False)})
    assert _settings(composite.text_config) == dict.fromkeys(SETTINGS)

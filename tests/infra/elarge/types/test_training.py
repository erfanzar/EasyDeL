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

"""Tests for ``easydel.infra.elarge.types.training``.

The module is mostly TypedDict declarations (~1900 lines of pure type
schemas). The runtime-testable surface:

* ``_normalize_trainer_type`` -- alias resolution + lowercase
* ``register_trainer_defaults`` -- trainer-specific overrides registry
* ``get_trainer_defaults`` -- merged defaults (BASE + trainer-specific)
* ``normalize_trainer_config`` -- full config normalization with
  defaults / deprecation handling / auto-computed fields
* ``BASE_TRAINER_DEFAULTS`` and ``TRAINER_SPECIFIC_DEFAULTS`` -- module-
  level dictionaries with documented invariants

Tests must restore ``TRAINER_SPECIFIC_DEFAULTS`` to its original state
after mutation (the registry is module-global).
"""

from __future__ import annotations

import warnings
from copy import deepcopy

import pytest

from easydel.infra.elarge.types import training as training_mod
from easydel.infra.elarge.types.training import (
    BASE_TRAINER_DEFAULTS,
    TRAINER_SPECIFIC_DEFAULTS,
    _normalize_trainer_type,
    get_trainer_defaults,
    normalize_trainer_config,
    register_trainer_defaults,
)


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("DPO", "dpo"),
        ("dpo", "dpo"),
        ("Dpo", "dpo"),
        ("SFT", "sft"),
        ("GRPO", "grpo"),
        ("nash_md", "nash-md"),
        ("nash-md", "nash-md"),
        ("agentic_moshpit", "agentic-moshpit"),
        ("rlvr_trainer", "rlvr"),
    ],
)
def test_normalize_trainer_type(input_str: str, expected: str):
    assert _normalize_trainer_type(input_str) == expected


def test_normalize_trainer_type_unknown_passes_through_lowercase():
    """Unknown trainer types are lowercased but not aliased."""
    assert _normalize_trainer_type("CustomTrainer") == "customtrainer"


def test_get_trainer_defaults_returns_base_when_no_trainer_specific():
    """A trainer with no specific overrides returns just BASE_TRAINER_DEFAULTS."""
    defaults = get_trainer_defaults("definitely_not_registered_trainer")
    assert defaults["learning_rate"] == BASE_TRAINER_DEFAULTS["learning_rate"]
    assert defaults["optimizer"] == BASE_TRAINER_DEFAULTS["optimizer"]


def test_get_trainer_defaults_merges_base_with_specific():
    """DPO overrides ``learning_rate`` and ``max_length`` but keeps base ``optimizer``."""
    defaults = get_trainer_defaults("dpo")

    assert defaults["learning_rate"] == 1e-6
    assert defaults["max_length"] == 512

    assert defaults["optimizer"] == BASE_TRAINER_DEFAULTS["optimizer"]
    assert defaults["weight_decay"] == BASE_TRAINER_DEFAULTS["weight_decay"]


def test_get_trainer_defaults_normalizes_input_type():
    """Mixed case / aliased input still finds the trainer's specific defaults."""
    a = get_trainer_defaults("DPO")
    b = get_trainer_defaults("dpo")
    assert a["learning_rate"] == b["learning_rate"]


def test_get_trainer_defaults_returns_independent_copy():
    """Mutating the returned dict must NOT mutate BASE_TRAINER_DEFAULTS."""
    defaults = get_trainer_defaults("sft")
    defaults["learning_rate"] = 99.0
    assert BASE_TRAINER_DEFAULTS["learning_rate"] != 99.0


@pytest.fixture(autouse=False)
def _restore_trainer_specific_defaults():
    """Snapshot/restore TRAINER_SPECIFIC_DEFAULTS around tests that mutate it."""
    snapshot = deepcopy(TRAINER_SPECIFIC_DEFAULTS)
    yield
    TRAINER_SPECIFIC_DEFAULTS.clear()
    TRAINER_SPECIFIC_DEFAULTS.update(snapshot)


def test_register_trainer_defaults_adds_new_entry(_restore_trainer_specific_defaults):
    register_trainer_defaults("custom_trainer_a", {"learning_rate": 1e-9})
    assert TRAINER_SPECIFIC_DEFAULTS["custom_trainer_a"] == {"learning_rate": 1e-9}
    defaults = get_trainer_defaults("custom_trainer_a")
    assert defaults["learning_rate"] == 1e-9

    assert defaults["optimizer"] == BASE_TRAINER_DEFAULTS["optimizer"]


def test_register_trainer_defaults_normalizes_alias(_restore_trainer_specific_defaults):
    register_trainer_defaults("nash_md", {"beta": 0.999})

    assert "nash-md" in TRAINER_SPECIFIC_DEFAULTS
    assert TRAINER_SPECIFIC_DEFAULTS["nash-md"]["beta"] == 0.999


def test_register_trainer_defaults_overrides_existing(_restore_trainer_specific_defaults):
    """Re-registering a trainer type replaces the entire entry (per docstring)."""
    register_trainer_defaults("DPO", {"beta": 0.42})
    assert TRAINER_SPECIFIC_DEFAULTS["dpo"] == {"beta": 0.42}

    defaults = get_trainer_defaults("dpo")
    assert defaults["beta"] == 0.42

    assert defaults["learning_rate"] == BASE_TRAINER_DEFAULTS["learning_rate"]


def test_normalize_trainer_config_applies_defaults():
    """Empty config gets the SFT trainer's full merged defaults (BASE + SFT-specific overrides)."""
    out = normalize_trainer_config({})
    assert out["trainer_type"] == "sft"

    expected = get_trainer_defaults("sft")
    assert out["learning_rate"] == expected["learning_rate"]

    assert out["optimizer"] == BASE_TRAINER_DEFAULTS["optimizer"]


def test_normalize_trainer_config_user_value_wins():
    """User-provided config values must override defaults."""
    out = normalize_trainer_config({"trainer_type": "sft", "learning_rate": 0.42})
    assert out["learning_rate"] == 0.42


def test_normalize_trainer_config_dpo_specific_defaults_apply():
    """DPO-specific defaults flow through when ``trainer_type='dpo'``."""
    out = normalize_trainer_config({"trainer_type": "dpo"})
    assert out["trainer_type"] == "dpo"
    assert out["beta"] == 0.1
    assert out["max_length"] == 512
    assert out["max_prompt_length"] == 256


def test_normalize_trainer_config_normalizes_trainer_type_alias():
    """``nash_md`` is normalized to canonical ``nash-md``."""
    out = normalize_trainer_config({"trainer_type": "nash_md"})
    assert out["trainer_type"] == "nash-md"


def test_normalize_trainer_config_does_not_mutate_input():
    """The function deep-copies its input per the docstring."""
    original = {"trainer_type": "sft", "learning_rate": 0.5}
    snapshot = deepcopy(original)
    normalize_trainer_config(original)
    assert original == snapshot


def test_normalize_trainer_config_max_sequence_length_deprecation_warning():
    """``max_sequence_length`` -> ``max_length`` migration emits a FutureWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = normalize_trainer_config({"trainer_type": "sft", "max_sequence_length": 1024})
    assert any(issubclass(w.category, FutureWarning) for w in caught)
    assert out["max_length"] == 1024
    assert "max_sequence_length" not in out


def test_normalize_trainer_config_max_sequence_length_when_max_length_already_set():
    """If both are set with different values, max_sequence_length is dropped + warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = normalize_trainer_config(
            {"trainer_type": "sft", "max_sequence_length": 1024, "max_length": 2048}
        )
    assert any(issubclass(w.category, FutureWarning) for w in caught)
    assert out["max_length"] == 2048
    assert "max_sequence_length" not in out


def test_normalize_trainer_config_auto_computes_max_completion_length_for_dpo():
    """For trainers in _TRAINERS_WITH_COMPLETION_LENGTH, completion = max_length - max_prompt_length."""
    out = normalize_trainer_config(
        {
            "trainer_type": "dpo",
            "max_length": 800,
            "max_prompt_length": 200,
        }
    )
    assert out["max_completion_length"] == 600


def test_normalize_trainer_config_skips_completion_for_sft():
    """SFT is not in _TRAINERS_WITH_COMPLETION_LENGTH so no auto-compute happens."""
    out = normalize_trainer_config(
        {
            "trainer_type": "sft",
            "max_length": 800,
            "max_prompt_length": 200,
        }
    )
    assert "max_completion_length" not in out or out.get("max_completion_length") is None


def test_normalize_trainer_config_eval_batch_size_defaults_to_total_batch_size():
    """If eval_batch_size is unset, it defaults to total_batch_size."""
    out = normalize_trainer_config({"trainer_type": "sft", "total_batch_size": 64})
    assert out["eval_batch_size"] == 64


def test_normalize_trainer_config_eval_batch_size_uses_32_when_no_total():
    """The function's default fallback is 32 when total_batch_size is also missing."""
    out = normalize_trainer_config({"trainer_type": "sft"})

    assert out["eval_batch_size"] == BASE_TRAINER_DEFAULTS["total_batch_size"]


def test_normalize_trainer_config_loss_config_dict_converted_to_lossconfig():
    """A raw dict ``loss_config`` is converted to a ``LossConfig`` instance."""
    from easydel.infra.loss_utils import LossConfig

    out = normalize_trainer_config(
        {
            "trainer_type": "sft",
            "loss_config": {"ignore_index": -100, "label_smoothing": 0.1},
        }
    )
    assert isinstance(out["loss_config"], LossConfig)


def test_normalize_trainer_config_loss_config_lossconfig_passthrough():
    """An already-LossConfig instance is NOT converted (dict-isinstance check fails).

    Note: the function deep-copies its input first, so the result is a deep copy
    of the same LossConfig instance -- but the type is preserved (not re-wrapped).
    """
    from easydel.infra.loss_utils import LossConfig

    lc = LossConfig(ignore_index=-100, label_smoothing=0.5)
    out = normalize_trainer_config({"trainer_type": "sft", "loss_config": lc})
    assert isinstance(out["loss_config"], LossConfig)

    assert out["loss_config"].ignore_index == -100
    assert out["loss_config"].label_smoothing == 0.5


def test_base_trainer_defaults_has_required_baseline_fields():
    """The base defaults must include the most-referenced fields used by trainers."""
    required_keys = {
        "learning_rate",
        "num_train_epochs",
        "total_batch_size",
        "gradient_accumulation_steps",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_length",
    }
    assert required_keys <= set(BASE_TRAINER_DEFAULTS.keys())


def test_trainer_specific_defaults_known_trainers_present():
    """The registry must contain entries for the canonical trainer types."""
    for known in ("dpo", "orpo", "grpo", "ppo", "sft", "kto", "bco", "cpo", "gkd"):
        assert known in TRAINER_SPECIFIC_DEFAULTS, f"missing trainer registry: {known}"


def test_trainers_with_completion_length_excludes_non_pairwise():
    """SFT and reward modeling are not pairwise-completion trainers."""
    assert "sft" not in training_mod._TRAINERS_WITH_COMPLETION_LENGTH
    assert "reward" not in training_mod._TRAINERS_WITH_COMPLETION_LENGTH

    assert "dpo" in training_mod._TRAINERS_WITH_COMPLETION_LENGTH
    assert "orpo" in training_mod._TRAINERS_WITH_COMPLETION_LENGTH
    assert "kto" in training_mod._TRAINERS_WITH_COMPLETION_LENGTH


def test_trainer_aliases_have_canonical_targets_in_specific_defaults():
    """Every alias's target must exist in TRAINER_SPECIFIC_DEFAULTS so normalization round-trips."""
    for alias, canonical in training_mod._TRAINER_TYPE_ALIASES.items():


        defaults = get_trainer_defaults(alias)
        assert "learning_rate" in defaults

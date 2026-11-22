"""Preference preparation compatibility imports (no array computation)."""

import pytest
from easydel.trainers._shared import concatenated_inputs
from easydel.trainers.contrastive_preference_optimization_trainer import _fn as cpo
from easydel.trainers.direct_preference_optimization_trainer import _fn as dpo

pytestmark = pytest.mark.cpu_ok


def test_preference_trainers_export_shared_batch_preparation():
    # Preserve the historical DPO import path as an alias, not a divergent copy.
    assert dpo.concatenated_inputs is concatenated_inputs
    assert cpo.concatenated_inputs is concatenated_inputs

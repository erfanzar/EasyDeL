"""Host-only checks that legacy SAO exports are the production objectives."""

from inspect import signature

import pytest
from easydel.trainers._online_rl import actor_critic
from easydel.trainers._shared import policy_losses
from easydel.trainers.single_rollout_asynchronous_optimization_trainer import _fn

pytestmark = pytest.mark.cpu_ok


def test_legacy_sao_objectives_resolve_to_production_exports():
    assert _fn.sao_policy_loss is actor_critic.sao_policy_loss is policy_losses.sao_policy_loss
    assert _fn.direct_importance_sampling_mask is policy_losses.direct_importance_sampling_mask
    assert set(_fn.__all__) == {"sao_policy_loss", "direct_importance_sampling_mask"}
    loss_args = signature(_fn.sao_policy_loss).bind(
        None, None, None, None, epsilon_low=0.2, epsilon_high=0.3, normalizer=10.0
    )
    assert loss_args.arguments["normalizer"] == 10.0
    assert signature(_fn.sao_policy_loss).parameters["return_keep_mask"].default is False
    mask_args = signature(_fn.direct_importance_sampling_mask).bind(None, None, None, epsilon_low=0.2, epsilon_high=0.3)
    assert mask_args.arguments["epsilon_low"] == 0.2


@pytest.mark.parametrize("epsilon_low, epsilon_high", [(0.0, 0.1), (1.1, 0.1), (0.1, 0.0)])
def test_legacy_dis_validation_precedes_array_computation(epsilon_low, epsilon_high):
    with pytest.raises(ValueError, match="epsilon_"):
        _fn.direct_importance_sampling_mask(None, None, None, epsilon_low=epsilon_low, epsilon_high=epsilon_high)

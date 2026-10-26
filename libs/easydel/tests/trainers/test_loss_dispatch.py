"""Pure-Python validation of loss-direction dispatch."""

import pytest
from easydel.trainers.direct_preference_optimization_trainer._fn import get_loss_function
from easydel.trainers.distillation_trainer.loss_utils import resolve_kl_direction

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("beta", "expected"),
    [(None, "forward"), (-1.0, "forward"), (0.0, "forward"), (0.5, "jsd"), (1.0, "reverse"), (2.0, "reverse")],
)
def test_kl_direction(beta, expected):
    assert resolve_kl_direction(beta) == expected


def test_invalid_preference_loss_is_rejected():
    with pytest.raises(ValueError):
        get_loss_function("not_a_loss", 0.1, 0.0)

"""Accelerator-only public-forward router padding invariance regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules._base import BaseCausalLMModule
from easydel.modules._base._features import RouterAuxLossFeature
from jax import numpy as jnp


@pytest.mark.parametrize("entrypoint", ["forward", "forward_moe"])
def test_real_router_aux_loss_is_padding_invariant_in_public_forward(entrypoint):
    class RouterFixture(BaseCausalLMModule):
        _auto_register = False

        def __init__(self, router_logits):
            self.config = SimpleNamespace(num_experts_per_tok=1, output_router_logits=True)
            self._router_aux_loss_feature = RouterAuxLossFeature(0.25)
            self._base_model_name = "backbone"
            self.router_logits = router_logits
            self.backbone = self.run_backbone

        def run_backbone(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=None,
                hidden_states=None,
                attentions=None,
                past_key_values=None,
                router_logits=(self.router_logits,),
            )

    real_logits = jnp.array([[3.0, 0.0], [2.0, 0.0]], dtype=jnp.float32)
    padded_logits = jnp.concatenate((real_logits, jnp.array([[-100.0, 100.0], [-100.0, 100.0]])))
    real = getattr(RouterFixture(real_logits), entrypoint)(
        attention_mask=jnp.array([[1, 1]]), apply_lm_head=False
    ).aux_loss
    padded_model = RouterFixture(padded_logits)
    padded = getattr(padded_model, entrypoint)(attention_mask=jnp.array([[1, 1, 0, 0]]), apply_lm_head=False).aux_loss
    unmasked = getattr(padded_model, entrypoint)(apply_lm_head=False).aux_loss
    # Both real tokens choose expert zero. Independently compute the Switch
    # loss from its mean softmax probability, two experts, and the coefficient.
    expected = 0.25 * 2.0 * np.mean(1.0 / (1.0 + np.exp(-np.array([3.0, 2.0]))))
    np.testing.assert_allclose(np.asarray(real), expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(padded), expected, rtol=1e-5)
    assert not np.isclose(np.asarray(unmasked), expected)

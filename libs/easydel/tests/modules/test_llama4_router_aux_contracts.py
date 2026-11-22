"""Host-only Llama4 task plumbing with independent masked router references.

The numerical loss backend is NumPy, not a JAX CPU execution.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from easydel.infra import loss_utils
from easydel.infra.modeling_outputs import MoeModelOutput
from easydel.modules._base import BaseCausalLMModule
from easydel.modules._base._base_task_module import BaseTaskModule
from easydel.modules.llama4 import Llama4TextConfig
from easydel.modules.llama4.modeling_llama4 import Llama4ForCausalLM

pytestmark = pytest.mark.cpu_ok


def _softmax(values, axis):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("padding", [False, True])
def test_llama4_public_forward_masked_auxiliary_matches_independent_reference(monkeypatch, top_k, padding):
    # Replace only the shared loss's numerical backend. The family forward,
    # shared task forwarding, and loss implementation remain the real methods.
    monkeypatch.setattr(loss_utils, "jnp", np)
    monkeypatch.setattr(
        loss_utils,
        "jax",
        SimpleNamespace(
            nn=SimpleNamespace(
                softmax=_softmax,
                one_hot=lambda indices, num_classes, dtype: np.eye(num_classes, dtype=dtype)[indices],
            ),
            lax=SimpleNamespace(
                top_k=lambda values, k: (
                    np.sort(values, axis=-1)[:, -k:][:, ::-1],
                    np.argsort(-values, axis=-1, kind="stable")[:, :k],
                )
            ),
        ),
    )
    config = Llama4TextConfig(num_local_experts=4, num_experts_per_tok=top_k, output_router_logits=True)
    routers = np.tile(np.asarray([[3, 0, -1, 1], [0, 2, 1, -1], [1, -1, 3, 0], [9, -9, -9, -9]], np.float32), (4, 1))
    mask = np.tile(np.asarray([[1, 1, 1, 0 if padding else 1]], np.int32), (4, 1))
    hidden = object()

    def backbone(**kwargs):
        return MoeModelOutput(
            last_hidden_state=hidden,
            router_logits=(routers,) if kwargs["output_router_logits"] else None,
        )

    model = SimpleNamespace(
        config=config,
        base_model=backbone,
        _router_aux_loss_feature=SimpleNamespace(coef=0.025),
    )
    model.forward_moe = MethodType(BaseCausalLMModule.forward_moe, model)
    model.compute_router_aux_loss = MethodType(BaseTaskModule.compute_router_aux_loss, model)
    model.compute_indexer_aux_loss = MethodType(BaseTaskModule.compute_indexer_aux_loss, model)
    model.compute_aux_loss = MethodType(BaseTaskModule.compute_aux_loss, model)
    model._router_top_k = MethodType(BaseTaskModule._router_top_k, model)
    output = Llama4ForCausalLM.forward(model, inputs_embeds=hidden, attention_mask=mask, apply_lm_head=False)
    assert output.last_hidden_state is hidden
    assert output.logits is None
    assert output.router_logits[0] is routers
    valid_routers = routers[mask.reshape(-1).astype(bool)].astype(np.float64)
    probabilities = _softmax(valid_routers, axis=-1)
    # Independently accumulate each expert's fraction across top-k slots.
    selected = np.argsort(-valid_routers, axis=-1)[:, :top_k]
    expected = sum(np.mean(selected == expert, axis=0).sum() * probabilities[:, expert].mean() for expert in range(4))
    np.testing.assert_allclose(output.aux_loss, 4 * 0.025 * expected, rtol=1e-6, atol=1e-7)
    disabled = Llama4ForCausalLM.forward(
        model, inputs_embeds=hidden, attention_mask=mask, apply_lm_head=False, output_router_logits=False
    )
    assert disabled.router_logits is None
    assert disabled.aux_loss is None

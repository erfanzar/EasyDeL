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

"""Regression tests for selective gradient-checkpointing targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from jax.ad_checkpoint import checkpoint_name, print_saved_residuals

import easydel  # noqa: F401
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.utils import get_gradient_checkpoint_policy
from easydel.modules.llama.llama_configuration import LlamaConfig
from easydel.modules.llama import modeling_llama


TARGETS = ["mlp_output", "residual"]


def _tiny_llama_config(**overrides):
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
    }
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def _build_llama_model(config):
    mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
    with mesh:
        return modeling_llama.LlamaModel(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )


def _install_fake_decoder_forward(monkeypatch):
    def unwrapped_decoder_forward(self, *args, **kwargs):
        raise AssertionError("This test only verifies construction-time remat wiring.")

    monkeypatch.setattr(modeling_llama.LlamaDecoderLayer, "forward", unwrapped_decoder_forward)
    return unwrapped_decoder_forward


@pytest.mark.parametrize(
    ("checkpointing", "jax_policy_attr"),
    [
        (EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES, "save_only_these_names"),
        (EasyDeLGradientCheckPointers.SAVE_ANYTHING_EXCEPT_THESE_NAMES, "save_any_names_but_these"),
    ],
)
def test_llama_decoder_remat_uses_configured_checkpoint_targets(monkeypatch, checkpointing, jax_policy_attr):
    captured_policy_names = []
    remat_calls = []
    sentinel_policy = object()

    def fake_target_policy(*names):
        captured_policy_names.append(names)
        return sentinel_policy

    def fake_remat(*, fn, prevent_cse, policy, mutable):
        remat_calls.append(
            {
                "fn": fn,
                "prevent_cse": prevent_cse,
                "policy": policy,
                "mutable": mutable,
            }
        )

        def wrapped_forward(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped_forward

    unwrapped_decoder_forward = _install_fake_decoder_forward(monkeypatch)
    monkeypatch.setattr(jax.checkpoint_policies, jax_policy_attr, fake_target_policy)
    monkeypatch.setattr(modeling_llama.spx, "remat", fake_remat)

    config = _tiny_llama_config(
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_targets=TARGETS,
    )

    _build_llama_model(config)

    assert captured_policy_names == [tuple(TARGETS)]
    assert len(remat_calls) == 1
    assert remat_calls[0]["fn"] is unwrapped_decoder_forward
    assert remat_calls[0]["policy"] is sentinel_policy
    assert remat_calls[0]["prevent_cse"] is True
    assert remat_calls[0]["mutable"] == ["rng"]


def test_llama_decoder_remat_is_class_level_and_shared_by_all_layers(monkeypatch):
    remat_calls = []
    sentinel_policy = object()

    def fake_save_only_these_names(*names):
        return sentinel_policy

    def fake_remat(*, fn, prevent_cse, policy, mutable):
        remat_calls.append(fn)

        def wrapped_forward(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped_forward

    _install_fake_decoder_forward(monkeypatch)
    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)
    monkeypatch.setattr(modeling_llama.spx, "remat", fake_remat)

    config = _tiny_llama_config(
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        gradient_checkpointing=EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES,
        gradient_checkpointing_targets=TARGETS,
    )

    model = _build_llama_model(config)

    assert len(model.layers) == config.num_hidden_layers
    assert len(remat_calls) == 1
    assert getattr(modeling_llama.LlamaDecoderLayer.forward, "_easydel_auto_remat_wrapped", False)


def test_gradient_checkpointing_none_bypasses_target_policy_and_remat(monkeypatch):
    def fail_policy(*names):
        raise AssertionError("target policy should not be built when checkpointing is disabled")

    def fail_remat(**kwargs):
        raise AssertionError("spx.remat should not be called when checkpointing is disabled")

    _install_fake_decoder_forward(monkeypatch)
    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fail_policy)
    monkeypatch.setattr(modeling_llama.spx, "remat", fail_remat)

    config = _tiny_llama_config(
        gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
        gradient_checkpointing_targets=TARGETS,
    )

    _build_llama_model(config)

    assert not getattr(modeling_llama.LlamaDecoderLayer.forward, "_easydel_auto_remat_wrapped", False)


@pytest.mark.parametrize(
    "checkpointing",
    [
        EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES,
        EasyDeLGradientCheckPointers.SAVE_ANYTHING_EXCEPT_THESE_NAMES,
    ],
)
def test_selective_checkpointing_requires_targets(monkeypatch, checkpointing):
    _install_fake_decoder_forward(monkeypatch)

    config = _tiny_llama_config(
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_targets=None,
    )

    with pytest.raises(ValueError, match="names must be provided"):
        _build_llama_model(config)


def _jaxpr_checkpoint_names(jaxpr):
    names = set()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "name":
            names.add(eqn.params["name"])
        for param in eqn.params.values():
            if hasattr(param, "jaxpr"):
                names.update(_jaxpr_checkpoint_names(param.jaxpr))
            elif isinstance(param, (tuple, list)):
                for item in param:
                    if hasattr(item, "jaxpr"):
                        names.update(_jaxpr_checkpoint_names(item.jaxpr))
    return names


def test_llama_mlp_exposes_checkpoint_names_used_by_targets():
    config = _tiny_llama_config()
    mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
    with mesh:
        mlp = modeling_llama.LlamaMLP(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )

    jaxpr = jax.make_jaxpr(mlp.forward)(jnp.ones((1, 2, config.hidden_size), dtype=jnp.float32))

    assert {"mlp_gate", "mlp_up", "mlp_down", "mlp_output"}.issubset(_jaxpr_checkpoint_names(jaxpr.jaxpr))


def test_target_policy_selects_the_named_residual(capsys):
    def toy_loss(x):
        keep = checkpoint_name(jnp.arange(2, dtype=jnp.float32) * x, "keep_me")
        drop = checkpoint_name(jnp.arange(3, dtype=jnp.float32) * jnp.sum(keep), "drop_me")
        return jnp.sum(drop * drop)

    keep_policy = get_gradient_checkpoint_policy(
        EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES,
        save_names=["keep_me"],
    )
    print_saved_residuals(jax.checkpoint(toy_loss, policy=keep_policy), jnp.float32(2.0))
    keep_output = capsys.readouterr().out

    drop_policy = get_gradient_checkpoint_policy(
        EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES,
        save_names=["drop_me"],
    )
    print_saved_residuals(jax.checkpoint(toy_loss, policy=drop_policy), jnp.float32(2.0))
    drop_output = capsys.readouterr().out

    assert "f32[2]" in keep_output
    assert "f32[3]" not in keep_output
    assert "f32[3]" in drop_output
    assert "f32[2]" not in drop_output

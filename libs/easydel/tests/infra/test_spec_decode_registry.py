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

"""Unit tests for the extensible drafter registry and ``SpecDecodeBase``.

These are light-weight (no heavy models): they exercise the registry
resolution / alias semantics, custom registration, the ``SpecDecodeBase``
capability-flag defaults and shared ``draft(...)`` tail, and the multi-layer
seed-hidden build-time gate (5b).
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from easydel.inference.speculative import DraftStep
from easydel.infra import DrafterRegistry, SpecDecodeBase, register_drafter
from easydel.infra import spec_decode as _spec_decode


def test_infra_reexports_are_identical():
    """The ``easydel.infra`` re-exports are the same objects as the module ones."""
    assert DrafterRegistry is _spec_decode.DrafterRegistry
    assert SpecDecodeBase is _spec_decode.SpecDecodeBase
    assert register_drafter is _spec_decode.register_drafter


def test_builtin_drafters_registered():
    """Importing/using the registry lazily registers all built-in families."""
    names = DrafterRegistry.names()
    for family in ("mtp", "gemma4_assistant", "eagle3", "dspark", "dflash"):
        assert family in names, f"{family} not registered: {names}"
        assert DrafterRegistry.has(family)


def test_alias_resolution_maps_to_same_builder():
    """Aliases resolve to the same builder as the canonical name."""
    assert DrafterRegistry.resolve("mtp") is DrafterRegistry.resolve("qwen3_5_mtp")
    assert DrafterRegistry.resolve("mtp") is DrafterRegistry.resolve("inline_mtp")
    # Slug normalization: dots/spaces/case collapse to the canonical slug.
    assert DrafterRegistry.resolve("mtp") is DrafterRegistry.resolve("Qwen3.5 MTP")
    assert DrafterRegistry.resolve("gemma4_assistant") is DrafterRegistry.resolve("assistant")
    assert DrafterRegistry.resolve("gemma4_assistant") is DrafterRegistry.resolve("gemma4")


def test_unknown_drafter_raises_valueerror_listing_names():
    """An unknown method raises ``ValueError`` naming the registered drafters."""
    with pytest.raises(ValueError) as excinfo:
        DrafterRegistry.resolve("does_not_exist")
    message = str(excinfo.value)
    assert "does_not_exist" in message
    assert "mtp" in message  # lists registered names


def test_custom_register_drafter_resolves_by_name_and_alias():
    """A custom ``@register_drafter`` becomes resolvable by name and alias."""
    sentinel = object()

    @register_drafter("unit_custom_drafter", "unit_custom_alias")
    def _builder(target_model, *, num_draft_tokens=1, drafter_model=None, **kwargs):
        del target_model, num_draft_tokens, kwargs
        return drafter_model if drafter_model is not None else sentinel

    assert DrafterRegistry.has("unit_custom_drafter")
    assert DrafterRegistry.resolve("unit custom drafter") is _builder
    assert DrafterRegistry.resolve("unit_custom_alias") is _builder
    assert DrafterRegistry.resolve("unit_custom_drafter")(object()) is sentinel


def test_spec_decode_base_capability_flag_defaults():
    """``SpecDecodeBase`` exposes the documented capability-flag defaults."""
    assert SpecDecodeBase.supports_return_full_log_probs is True
    assert SpecDecodeBase.supports_prefix_draft is False
    assert SpecDecodeBase.requires_target_kv_cache is False
    assert SpecDecodeBase.num_draft_tokens == 1


class _ToyDrafter(SpecDecodeBase):
    """Minimal drafter exercising the shared ``draft`` tail."""

    def __init__(self, logits: jnp.ndarray):
        self._logits = logits

    def _draft_logits(
        self, input_ids, *, target_hidden_states, position_ids=None, target_kv_cache=None, attention_mask=None
    ):
        del input_ids, target_hidden_states, position_ids, target_kv_cache, attention_mask
        return self._logits, None


def test_spec_decode_base_draft_greedy_argmax_no_logprobs():
    """Greedy ``draft`` returns the argmax token and skips the log-prob tail."""
    # (batch=2, seq=3, vocab=4); last position argmax is index 2 for both rows.
    logits = jnp.zeros((2, 3, 4), dtype=jnp.float32).at[:, -1, 2].set(5.0)
    step = _ToyDrafter(logits).draft(input_ids=jnp.zeros((2, 3), dtype=jnp.int32))
    assert isinstance(step, DraftStep)
    assert step.token_ids.shape == (2,)
    assert step.token_ids.dtype == jnp.int32
    assert np.array_equal(np.asarray(step.token_ids), np.array([2, 2]))
    assert step.log_probs is None
    assert step.full_log_probs is None


def test_spec_decode_base_draft_full_log_probs_shapes():
    """``return_full_log_probs`` materializes ``(B, vocab)`` + ``(B,)`` tensors."""
    logits = jnp.zeros((2, 3, 4), dtype=jnp.float32).at[:, -1, 1].set(5.0)
    step = _ToyDrafter(logits).draft(
        input_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        return_full_log_probs=True,
    )
    assert step.full_log_probs.shape == (2, 4)
    assert step.log_probs.shape == (2,)
    # token_log_probs is the log-prob assigned to the chosen (argmax) token.
    chosen = np.asarray(step.token_ids)
    picked = np.asarray(step.full_log_probs)[np.arange(2), chosen]
    assert np.allclose(picked, np.asarray(step.log_probs), atol=1e-5)


def test_spec_decode_base_draft_sample_requires_rng():
    """``sample=True`` without an ``rng_key`` fails loudly."""
    logits = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    with pytest.raises(ValueError):
        _ToyDrafter(logits).draft(input_ids=jnp.zeros((1, 2), dtype=jnp.int32), sample=True)


def test_model_native_multi_layer_seed_gate():
    """The generic model-native builder rejects a multi-layer standalone drafter (5b)."""

    class _FakeConfig:
        target_layer_ids = (0, 1, 2, 3, 4)

    class _FakeMultiLayerDrafter(SpecDecodeBase):
        config = _FakeConfig()

    builder = DrafterRegistry.resolve("eagle3")
    with pytest.raises(ValueError) as excinfo:
        builder(object(), drafter_model=_FakeMultiLayerDrafter(), num_draft_tokens=2)
    assert "target layers" in str(excinfo.value)


def test_model_native_builder_requires_spec_decode_base():
    """A non-``SpecDecodeBase`` drafter_model is rejected with ``TypeError``."""
    builder = DrafterRegistry.resolve("dspark")
    with pytest.raises(TypeError):
        builder(object(), drafter_model=object(), num_draft_tokens=1)

    with pytest.raises(ValueError):
        builder(object(), num_draft_tokens=1)  # neither drafter_model nor assistant_model


def test_model_native_builder_configures_single_layer_drafter():
    """A single-layer standalone drafter passes the gate and records draft tokens."""

    class _FakeConfig:
        target_layer_ids = (0,)

    class _FakeSingleLayerDrafter(SpecDecodeBase):
        config = _FakeConfig()

    builder = DrafterRegistry.resolve("dspark")
    drafter = _FakeSingleLayerDrafter()
    returned = builder(object(), drafter_model=drafter, num_draft_tokens=4, target_config="TCFG")
    assert returned is drafter
    assert drafter.num_draft_tokens == 4
    assert drafter.target_config == "TCFG"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

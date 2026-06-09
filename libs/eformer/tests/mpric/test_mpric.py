# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for mixed precision utilities."""

import jax.numpy as jnp
import pytest

from eformer.mpric.dtypes.precision_types import get_platform_default_half, put_dtype
from eformer.mpric.handler.precision_handler import PrecisionHandler
from eformer.mpric.loss_scaling.loss_scaler import DynamicLossScale, LossScaleConfig, NoOpLossScale
from eformer.mpric.policy.policy import Policy


def test_policy_from_string_and_half():
    policy = Policy.from_string("p=f32,c=f16,o=bf16")
    assert policy.param_dtype == jnp.float32
    assert policy.compute_dtype == jnp.float16
    assert policy.output_dtype == jnp.bfloat16

    default_half = get_platform_default_half()
    policy_half = Policy.from_string("half")
    assert policy_half.param_dtype == default_half
    assert policy_half.compute_dtype == default_half
    assert policy_half.output_dtype == default_half

    with pytest.raises(ValueError):
        Policy.from_string("p=unknown")


def test_put_dtype():
    x = jnp.ones((2, 2), dtype=jnp.float16)
    y = put_dtype(x, "fp32")
    assert y.dtype == jnp.float32

    z = put_dtype(x, None)
    assert z.dtype == x.dtype

    with pytest.raises(ValueError):
        put_dtype(x, "not_a_dtype")


def test_dynamic_loss_scale_adjust():
    scale = DynamicLossScale(
        loss_scale=jnp.array(4.0),
        period=2,
        factor=2,
        min_loss_scale=jnp.array(1.0),
    )

    scale = scale.adjust(jnp.array(True))
    assert float(scale.loss_scale) == 4.0
    assert int(scale.counter) == 1

    scale = scale.adjust(jnp.array(True))
    assert float(scale.loss_scale) == 8.0
    assert int(scale.counter) == 0

    scale = scale.adjust(jnp.array(False))
    assert float(scale.loss_scale) == 4.0


def test_noop_loss_scale():
    scale = NoOpLossScale()
    tree = {"a": jnp.array(1.0)}
    assert scale.scale(tree)["a"] == tree["a"]
    assert scale.unscale(tree)["a"] == tree["a"]
    assert scale.adjust(jnp.array(True)) is scale


def test_precision_handler_wrappers():
    handler = PrecisionHandler("p=f32,c=f16,o=f32", use_dynamic_scale=False, loss_scale_config=LossScaleConfig())

    x = jnp.ones((2, 2), dtype=jnp.float32)
    cast_x = handler.cast_for_compute(x)
    assert cast_x.dtype == jnp.float16

    cast_out = handler.cast_for_output(cast_x)
    assert cast_out.dtype == jnp.float32

    def training_step(inp):
        loss = jnp.sum(inp)
        grads = {"w": inp}
        return loss, grads

    wrapped = handler.training_step_wrapper(training_step)
    loss, grads, grads_finite = wrapped(x)

    assert loss.dtype == jnp.float32
    assert grads["w"].dtype == jnp.float32
    assert bool(grads_finite) is True

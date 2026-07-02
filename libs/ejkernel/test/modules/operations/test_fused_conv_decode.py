# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Module-level tests for fused conv-state decode operation."""

from __future__ import annotations

import jax.numpy as jnp

from ejkernel.modules.operations import (
    FusedConvDecode,
    FusedConvDecodeConfig,
    fused_conv_decode,
    fused_conv_decode_op,
)


def test_fused_conv_decode_function_class_and_op_match():
    """Check the three fused-conv-decode entry points produce identical results.

    Exercises the module-level :func:`fused_conv_decode` function, the
    :class:`FusedConvDecode` kernel's ``run`` method, and the :data:`fused_conv_decode_op` alias on
    the same zero state, all-ones tokens, and ``0.25`` kernel, with ``activation="none"`` (provided
    directly to the function and via :class:`FusedConvDecodeConfig` for the class/op paths).
    Asserts the function's outputs have the expected ``[2, 4, 3]`` state and ``[2, 4]`` output
    shapes, then asserts the class- and op-routed states and outputs match the function's within
    :func:`jnp.allclose` tolerance.

    Raises:
        AssertionError: If any of the three paths disagree in shape or value.
    """
    conv_state = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    new_tokens = jnp.ones((2, 4), dtype=jnp.float32)
    kernel = jnp.ones((4, 3), dtype=jnp.float32) * 0.25
    cfg = FusedConvDecodeConfig(d_conv=3, activation="none")

    direct_state, direct_output = fused_conv_decode(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
        activation="none",
    )
    class_state, class_output = FusedConvDecode().run(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
        cfg=cfg,
    )
    op_state, op_output = fused_conv_decode_op(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
        cfg=cfg,
    )

    assert direct_state.shape == (2, 4, 3)
    assert direct_output.shape == (2, 4)
    assert jnp.allclose(class_state, direct_state)
    assert jnp.allclose(class_output, direct_output)
    assert jnp.allclose(op_state, direct_state)
    assert jnp.allclose(op_output, direct_output)


def test_fused_conv_decode_config_roundtrips():
    """Check :class:`FusedConvDecodeConfig` survives a JSON serialization round-trip.

    Builds a config with ``d_conv=3``, ``activation="none"``, and ``platform="xla"``, serializes it
    via ``to_json`` and reconstructs it with ``from_json``, then asserts the ``d_conv``,
    ``activation``, and ``platform`` fields are preserved.

    Raises:
        AssertionError: If any field differs after the serialize/deserialize round-trip.
    """
    cfg = FusedConvDecodeConfig(d_conv=3, activation="none", platform="xla")
    rt = FusedConvDecodeConfig.from_json(cfg.to_json())
    assert rt.d_conv == 3
    assert rt.activation == "none"
    assert rt.platform == "xla"

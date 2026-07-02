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

"""EasyDeL adapter checks for GDN schedule construction."""

from __future__ import annotations

import inspect
from pathlib import Path

import jax.numpy as jnp
from ejkernel.modules.operations import (
    KERNEL_TILE_POLICIES,
    GDNComputeScheduleV2,
    GDNComputeScheduleV2Config,
    compute_schedule_table_v2,
    normalize_kernel_tile_policy,
)

from easydel.inference.esurge import config as esurge_config
from easydel.operations._operation_impl import OperationImpl
from easydel.operations.kernels import gated_delta_rule as easydel_gdr
from easydel.operations.kernels import gdn_compute_schedule_v2 as easydel_schedule
from easydel.operations.kernels import inference_gdn as easydel_gdn
from easydel.operations.kernels.gated_delta_rule import GatedDeltaRuleOp


def test_gdn_schedule_adapter_delegates_to_ejkernel_operation():
    """Assert the EasyDeL GDN schedule surface re-exports and delegates to eJKernel.

    Confirms that the operation class and config object exposed under
    ``easydel.operations.kernels.gdn_compute_schedule_v2`` are the exact same
    objects as the public eJKernel symbols, then runs
    ``compute_schedule_table_v2`` through both the EasyDeL adapter and the
    eJKernel implementation on identical CSR offsets and checks the returned
    schedule table and block count are element-wise equal. Also inspects the
    adapter source to ensure it imports from ``ejkernel.modules.operations``
    rather than any private ``_pallas`` path and preserves the ``Column
    layout`` documentation in the wrapper docstring.
    """
    assert easydel_schedule.GDNComputeScheduleV2 is GDNComputeScheduleV2
    assert easydel_schedule.GDNComputeScheduleV2Config is GDNComputeScheduleV2Config

    query_start_loc = jnp.array([0, 1, 5, 9], dtype=jnp.int32)
    cfg = GDNComputeScheduleV2Config(platform="xla", backend="any")
    easydel_table, easydel_blocks = easydel_schedule.compute_schedule_table_v2(
        query_start_loc,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        max_tokens=9,
        chunk_size=4,
        BT=4,
        alignment=4,
        cfg=cfg,
    )
    ejkernel_table, ejkernel_blocks = compute_schedule_table_v2(
        query_start_loc,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        max_tokens=9,
        chunk_size=4,
        BT=4,
        alignment=4,
        cfg=cfg,
    )
    assert jnp.array_equal(easydel_table, ejkernel_table)
    assert jnp.array_equal(easydel_blocks, ejkernel_blocks)

    source = inspect.getsource(easydel_schedule)
    assert "ejkernel.modules.operations" in source
    assert "_pallas" not in source
    assert "Column layout" in easydel_schedule.compute_schedule_table_v2.__doc__


def test_gdn_policy_is_imported_from_ejkernel_public_surface():
    """Assert the kernel-tile policy helpers come from the eJKernel public surface.

    Checks that ``"auto"`` is a recognized policy in ``KERNEL_TILE_POLICIES``,
    that ``normalize_kernel_tile_policy(None)`` falls back to ``"auto"``, and
    that the eSurge config module re-exports the very same
    ``normalize_kernel_tile_policy`` callable rather than defining its own copy.
    """
    assert "auto" in KERNEL_TILE_POLICIES
    assert normalize_kernel_tile_policy(None) == "auto"
    assert esurge_config.normalize_kernel_tile_policy is normalize_kernel_tile_policy


def test_ragged_gdn_is_real_easydel_operation_adapter():
    """Assert ragged GDN is a genuine EasyDeL OperationImpl adapter over eJKernel.

    Verifies ``RaggedGatedDeltaRule`` subclasses ``OperationImpl`` and reports
    the impl name ``"ragged_gated_delta_rule_v2"``, then inspects the module
    source to ensure it dispatches to the public
    ``ejkernel.modules.operations.ragged_gated_delta_rule_v2`` operation and
    contains no references to private eJKernel kernels (``ejkernel.kernels._``)
    or raw ``jax.experimental.pallas`` usage.
    """
    assert issubclass(easydel_gdn.RaggedGatedDeltaRule, OperationImpl)
    assert easydel_gdn.RaggedGatedDeltaRule.get_impl_name() == "ragged_gated_delta_rule_v2"

    source = inspect.getsource(easydel_gdn)
    assert "ejkernel.modules.operations.ragged_gated_delta_rule_v2" in source
    assert "ejkernel.kernels._" not in source
    assert "jax.experimental.pallas" not in source


def test_no_private_gdn_backend_shims_remain_in_easydel_operations():
    """Assert no private GDN backend shim modules remain under EasyDeL kernels.

    Walks the ``easydel/operations/kernels`` directory and asserts the legacy
    backend shim files (``_gdn_policy.py``, ``fused_gdn_decode.py``,
    ``gdn_recurrent_scan_v2.py``) have been deleted. For the surviving adapter
    modules (``gdn_compute_schedule_v2.py``, ``inference_gdn.py``) it reads the
    source text and confirms neither references private eJKernel kernels
    (``ejkernel.kernels._``) nor raw ``jax.experimental.pallas`` imports.
    """
    kernel_dir = Path(__file__).parents[3] / "easydel" / "operations" / "kernels"
    for removed in ("_gdn_policy.py", "fused_gdn_decode.py", "gdn_recurrent_scan_v2.py"):
        assert not (kernel_dir / removed).exists()

    for path in (kernel_dir / "gdn_compute_schedule_v2.py", kernel_dir / "inference_gdn.py"):
        source = path.read_text()
        assert "ejkernel.kernels._" not in source
        assert "jax.experimental.pallas" not in source


def test_gated_delta_rule_adapter_imports_only_public_ejkernel_ops():
    """Assert the gated-delta-rule adapter touches only the public eJKernel ops.

    Inspects the ``gated_delta_rule`` adapter source and asserts it imports from
    ``ejkernel.modules.operations`` while containing none of the private/raw
    backend markers: ``ejkernel.kernels._``, ``jax.experimental.pallas``,
    ``pallas_call``, the ``pl.`` Pallas alias, ``_gdr_grouped_decode_kernel``,
    or ``_fused_conv_decode_kernel``.
    """
    source = inspect.getsource(easydel_gdr)
    assert "ejkernel.modules.operations" in source
    assert "ejkernel.kernels._" not in source
    assert "jax.experimental.pallas" not in source
    assert "pallas_call" not in source
    assert "pl." not in source
    assert "_gdr_grouped_decode_kernel" not in source
    assert "_fused_conv_decode_kernel" not in source


def test_gated_delta_rule_adapter_delegates_grouped_decode_and_fused_conv():
    """Assert the gated-delta-rule op delegates grouped decode and fused conv.

    Verifies ``GatedDeltaRuleOp`` is an ``OperationImpl`` subclass and that its
    ``fused_conv_decode`` and ``grouped_gdr_decode`` method bodies reference the
    corresponding public eJKernel operations (``fused_conv_decode`` and
    ``gated_delta_rule_grouped_decode``) rather than reimplementing the kernels.
    """
    assert issubclass(easydel_gdr.GatedDeltaRuleOp, OperationImpl)
    assert "fused_conv_decode" in inspect.getsource(easydel_gdr.GatedDeltaRuleOp.fused_conv_decode)
    assert "gated_delta_rule_grouped_decode" in inspect.getsource(easydel_gdr.GatedDeltaRuleOp.grouped_gdr_decode)


def test_gated_delta_rule_fused_conv_decode_static_method_works():
    """Assert the fused-conv-decode static method runs and shifts the conv state.

    Calls ``GatedDeltaRuleOp.fused_conv_decode`` on a zero conv state of shape
    ``[batch=2, channels=4, kernel_width=3]`` with all-ones new tokens of shape
    ``[batch=2, channels=4]`` and a constant ``0.25`` depthwise kernel. Checks
    the returned updated state keeps shape ``[2, 4, 3]``, the conv output keeps
    shape ``[2, 4]``, both are finite, and that the updated state equals the old
    state shifted left by one with the new token appended in the last position
    (the expected causal-window update).
    """
    conv_state = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    new_tokens = jnp.ones((2, 4), dtype=jnp.float32)
    kernel = jnp.ones((4, 3), dtype=jnp.float32) * 0.25

    updated_state, conv_output = GatedDeltaRuleOp.fused_conv_decode(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
    )

    assert updated_state.shape == (2, 4, 3)
    assert conv_output.shape == (2, 4)
    assert jnp.all(jnp.isfinite(updated_state))
    assert jnp.all(jnp.isfinite(conv_output))

    expected_state = jnp.concatenate(
        [conv_state[..., 1:], new_tokens[..., None]],
        axis=-1,
    )
    assert jnp.allclose(updated_state, expected_state)

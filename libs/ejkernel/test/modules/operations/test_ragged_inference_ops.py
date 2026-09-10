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

"""Module-level tests for packed/ragged inference operations migrated from EasyDeL.

This test module exercises the public ``ejkernel.modules.operations`` surface for the
inference kernels that operate on packed (concatenated, variable-length) token batches
and per-slot recurrent/convolution state. Each kernel is validated through three call
styles that must agree numerically:

* the bare functional entry point (e.g. ``ragged_causal_conv1d``),
* the :class:`Operation` subclass via its ``.run(...)`` method (e.g. ``RaggedCausalConv1D``),
* the registered ``*_op`` adapter (e.g. ``ragged_causal_conv1d_op``).

The covered kernels are:

* ``RaggedCausalConv1D`` -- ragged causal depthwise 1D convolution with per-slot state.
* ``RaggedGatedDeltaRuleV2`` -- ragged gated delta-rule linear attention (v2 schedule).
* ``GDNComputeScheduleV2`` -- the chunk-schedule table builder used by the GDN kernels.
* ``FusedConvDecode`` -- fused single-step decode convolution update.
* ``GatedDeltaRuleGroupedDecode`` -- grouped (GQA-style) single-step gated delta-rule decode.

The suite also checks: that every symbol is re-exported from both the ``operations`` and
top-level ``modules`` namespaces; the tile-policy public surface; config JSON round-trips;
and the ``create_shard_map_wrapper`` sharded execution paths (single-device, two-device
head sharding, and the split-qkv head-shard callback variant).

Helpers prefixed with an underscore build meshes and synthetic kernel inputs; tests are
plain ``pytest`` functions and rely on ``assert_allclose`` from the sibling ``_utils``
module for tolerance-aware array comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel import modules
from ejkernel.modules import operations
from ejkernel.modules.operations import (
    KERNEL_TILE_POLICIES,
    FusedConvDecode,
    FusedConvDecodeConfig,
    GatedDeltaRuleGroupedDecode,
    GatedDeltaRuleGroupedDecodeConfig,
    GDNComputeScheduleV2,
    GDNComputeScheduleV2Config,
    RaggedCausalConv1D,
    RaggedCausalConv1DConfig,
    RaggedGatedDeltaRuleV2,
    RaggedGatedDeltaRuleV2Config,
    compute_schedule_table_v2,
    fused_conv_decode,
    fused_conv_decode_op,
    gated_delta_rule_grouped_decode,
    gdn_compute_schedule_v2,
    normalize_kernel_tile_policy,
    ragged_causal_conv1d,
    ragged_causal_conv1d_op,
    ragged_gated_delta_rule_v2,
    ragged_gated_delta_rule_v2_op,
)
from jax.sharding import Mesh, PartitionSpec

from ._utils import assert_allclose


def _single_device_mesh() -> Mesh:
    """Build a one-device JAX mesh with a single axis named ``"x"``.

    Returns:
        Mesh: A :class:`jax.sharding.Mesh` containing the first local device on axis ``"x"``.
            Used by ``create_shard_map_wrapper`` tests to exercise the sharded code path on a
            single device (where the result must equal the unsharded direct computation).
    """
    return Mesh(np.asarray(jax.devices()[:1]), ("x",))


def _two_device_mesh_or_skip() -> Mesh:
    """Build a two-device JAX mesh, skipping the test if fewer than two devices exist.

    Returns:
        Mesh: A :class:`jax.sharding.Mesh` over the first two local devices on axis ``"x"``,
            used to exercise true head-sharded execution.

    Raises:
        Skipped: The test is skipped via :func:`pytest.skip` when fewer than two local JAX
            devices are available.
    """
    if len(jax.devices()) < 2:
        pytest.skip("requires at least two local devices")
    return Mesh(np.asarray(jax.devices()[:2]), ("x",))


def test_ragged_inference_ops_are_exported_from_modules_init_files():
    """Assert the ragged inference symbols are re-exported from both module namespaces.

    Iterates over ``ejkernel.modules.operations`` and the top-level ``ejkernel.modules``
    package and asserts that every public ragged-inference class, config, functional entry
    point, ``*_op`` adapter, and tile-policy helper is reachable as an attribute on each.
    Also asserts that the three sharded operations (``RaggedCausalConv1D``,
    ``RaggedGatedDeltaRuleV2``, ``GatedDeltaRuleGroupedDecode``) define their own
    ``create_shard_map_wrapper`` method directly in their class ``__dict__`` (i.e. they
    override it rather than only inheriting it).

    Raises:
        AssertionError: If any expected symbol is missing from a namespace, or if a
            ``create_shard_map_wrapper`` override is absent from a class ``__dict__``.
    """
    for namespace in (operations, modules):
        assert hasattr(namespace, "RaggedCausalConv1D")
        assert hasattr(namespace, "RaggedCausalConv1DConfig")
        assert hasattr(namespace, "GDNComputeScheduleV2")
        assert hasattr(namespace, "GDNComputeScheduleV2Config")
        assert hasattr(namespace, "compute_schedule_table_v2")
        assert hasattr(namespace, "gdn_compute_schedule_v2")
        assert hasattr(namespace, "ragged_causal_conv1d")
        assert hasattr(namespace, "ragged_causal_conv1d_head_sharded")
        assert hasattr(namespace, "ragged_causal_conv1d_op")
        assert hasattr(namespace, "RaggedGatedDeltaRuleV2")
        assert hasattr(namespace, "RaggedGatedDeltaRuleV2Config")
        assert hasattr(namespace, "KERNEL_TILE_POLICIES")
        assert hasattr(namespace, "normalize_kernel_tile_policy")
        assert hasattr(namespace, "ragged_gated_delta_rule_v2")
        assert hasattr(namespace, "ragged_gated_delta_rule_v2_op")
        assert hasattr(namespace, "set_gdn_kernel_tile_policy")
        assert hasattr(namespace, "FusedConvDecode")
        assert hasattr(namespace, "FusedConvDecodeConfig")
        assert hasattr(namespace, "fused_conv_decode")
        assert hasattr(namespace, "fused_conv_decode_op")
        assert hasattr(namespace, "GatedDeltaRuleGroupedDecode")
        assert hasattr(namespace, "GatedDeltaRuleGroupedDecodeConfig")
        assert hasattr(namespace, "gated_delta_rule_grouped_decode")

    assert "create_shard_map_wrapper" in RaggedCausalConv1D.__dict__
    assert "create_shard_map_wrapper" in RaggedGatedDeltaRuleV2.__dict__
    assert "create_shard_map_wrapper" in GatedDeltaRuleGroupedDecode.__dict__


def test_gdn_tile_policy_is_public_module_surface():
    """Assert the GDN kernel tile-policy enum and normalizer are exposed and behave correctly.

    Checks that ``KERNEL_TILE_POLICIES`` is exactly the frozenset of supported policy names,
    that ``normalize_kernel_tile_policy`` maps ``None`` to the default ``"auto"`` and lowercases
    a valid uppercase policy (``"B8"`` -> ``"b8"``), and that an unknown policy string raises.

    Raises:
        AssertionError: If the policy set or normalization results differ from the expected values.
        ValueError: Expected (caught via :func:`pytest.raises`) when normalizing an invalid policy
            name such as ``"nope"``.
    """
    assert KERNEL_TILE_POLICIES == frozenset(("auto", "b16", "b8", "b4", "b2", "b1"))
    assert normalize_kernel_tile_policy(None) == "auto"
    assert normalize_kernel_tile_policy("B8") == "b8"
    with pytest.raises(ValueError):
        normalize_kernel_tile_policy("nope")


def test_ragged_inference_config_roundtrips():
    """Assert each operation config survives a JSON serialize/deserialize round-trip.

    For every kernel config (``RaggedCausalConv1DConfig``, ``RaggedGatedDeltaRuleV2Config``,
    ``GDNComputeScheduleV2Config``, ``FusedConvDecodeConfig``, ``GatedDeltaRuleGroupedDecodeConfig``)
    this builds an instance with non-default field values, serializes it via ``to_json()`` and
    rebuilds it via ``from_json(...)``, then asserts the round-tripped fields (e.g. ``d_conv``,
    ``apply_silu``, ``chunk_size``, ``activation``, ``platform``, ``backend``) match the originals.

    Raises:
        AssertionError: If any field value is not preserved across the JSON round-trip.
    """
    conv_cfg = RaggedCausalConv1DConfig(d_conv=3, apply_silu=False, platform="xla")
    conv_roundtrip = RaggedCausalConv1DConfig.from_json(conv_cfg.to_json())
    assert conv_roundtrip.d_conv == 3
    assert conv_roundtrip.apply_silu is False
    assert conv_roundtrip.platform == "xla"

    gdn_cfg = RaggedGatedDeltaRuleV2Config(chunk_size=2, platform="xla")
    gdn_roundtrip = RaggedGatedDeltaRuleV2Config.from_json(gdn_cfg.to_json())
    assert gdn_roundtrip.chunk_size == 2
    assert gdn_roundtrip.platform == "xla"

    schedule_cfg = GDNComputeScheduleV2Config(platform="xla")
    schedule_roundtrip = GDNComputeScheduleV2Config.from_json(schedule_cfg.to_json())
    assert schedule_roundtrip.platform == "xla"

    fused_cfg = FusedConvDecodeConfig(d_conv=4, activation="silu", platform="xla")
    fused_roundtrip = FusedConvDecodeConfig.from_json(fused_cfg.to_json())
    assert fused_roundtrip.d_conv == 4
    assert fused_roundtrip.activation == "silu"
    assert fused_roundtrip.platform == "xla"

    grouped_cfg = GatedDeltaRuleGroupedDecodeConfig(platform="pallas", backend="tpu")
    grouped_roundtrip = GatedDeltaRuleGroupedDecodeConfig.from_json(grouped_cfg.to_json())
    assert grouped_roundtrip.platform == "pallas"
    assert grouped_roundtrip.backend == "tpu"


def test_gdn_compute_schedule_v2_function_class_and_alias_match():
    """Assert the GDN schedule builder gives identical results across its three entry points.

    Computes the chunk-schedule table from a ragged ``query_start_loc`` cumulative-offset
    vector (4 entries describing 3 sequences over 9 tokens) using:
    the functional ``compute_schedule_table_v2``, the ``GDNComputeScheduleV2().run(...)`` class
    method, and the ``gdn_compute_schedule_v2`` alias. The functional form takes ``max_tokens``,
    ``chunk_size``, ``BT``, and ``alignment`` as keyword args while the class/alias take them
    positionally. Each call returns a ``(table, num_blocks)`` pair; the test asserts the three
    tables are element-wise equal, the three block counts are equal, and that the resulting
    block count is exactly 3 for this fixture.

    Raises:
        AssertionError: If the block count is not 3 or any of the three tables/counts disagree.
    """
    query_start_loc = jnp.array([0, 1, 5, 9], dtype=jnp.int32)
    cfg = GDNComputeScheduleV2Config(platform="xla")

    fn_table, fn_blocks = compute_schedule_table_v2(
        query_start_loc,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        max_tokens=9,
        chunk_size=4,
        BT=4,
        alignment=4,
        cfg=cfg,
    )
    class_table, class_blocks = GDNComputeScheduleV2().run(
        query_start_loc,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        9,
        4,
        4,
        4,
        cfg=cfg,
    )
    alias_table, alias_blocks = gdn_compute_schedule_v2(
        query_start_loc,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        9,
        4,
        4,
        4,
        cfg=cfg,
    )

    assert int(fn_blocks) == 3
    assert jnp.array_equal(fn_table, class_table)
    assert jnp.array_equal(fn_table, alias_table)
    assert jnp.array_equal(fn_blocks, class_blocks)
    assert jnp.array_equal(fn_blocks, alias_blocks)


def test_ragged_causal_conv1d_function_class_and_op_match():
    """Assert ragged causal conv1d agrees across functional, class, and op entry points.

    Runs the ragged causal depthwise 1D convolution over a packed input of 3 tokens x 4
    channels (``x``), a per-channel kernel of width 3 (``kernel`` shaped ``(4, 3)``), a packed
    layout described by ``query_start_loc``/``state_indices``/``distribution``, and a freshly
    zeroed per-slot state of shape ``(2, 4, 3)``. The same call is made through
    ``ragged_causal_conv1d`` (functional), ``RaggedCausalConv1D().run(...)`` (class), and
    ``ragged_causal_conv1d_op`` (op adapter), each producing an ``(output, new_state)`` pair.
    The test asserts the direct output/state have the expected shapes ``(3, 4)`` and ``(2, 4, 3)``
    and that the class and op results match the direct result exactly (``atol=0.0``).
    A nested ``fresh_state`` helper supplies an independent zeroed state to each call so the
    in/out state argument is not shared between the three invocations.

    Raises:
        AssertionError: If output/state shapes are wrong or any entry point's result diverges.
    """
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    kernel = jnp.ones((4, 3), dtype=jnp.float32) * 0.25
    query_start_loc = jnp.array([0, 1, 3], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
    cfg = RaggedCausalConv1DConfig(d_conv=3, apply_silu=False)

    def fresh_state():
        """Return a freshly zeroed conv state of shape ``(2, 4, 3)`` (slots, channels, d_conv-1+1).

        Returns:
            jax.Array: A new zero-filled float32 per-slot convolution state, so each entry-point
                call receives an independent state buffer.
        """
        return jnp.zeros((2, 4, 3), dtype=jnp.float32)

    direct_out, direct_state = ragged_causal_conv1d(
        x,
        fresh_state(),
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=3,
        apply_silu=False,
    )
    class_out, class_state = RaggedCausalConv1D().run(
        x,
        fresh_state(),
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=3,
        apply_silu=False,
        cfg=cfg,
    )
    op_out, op_state = ragged_causal_conv1d_op(
        x,
        fresh_state(),
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=3,
        apply_silu=False,
        cfg=cfg,
    )

    assert direct_out.shape == (3, 4)
    assert direct_state.shape == (2, 4, 3)
    assert_allclose(class_out, direct_out, atol=0.0)
    assert_allclose(class_state, direct_state, atol=0.0)
    assert_allclose(op_out, direct_out, atol=0.0)
    assert_allclose(op_state, direct_state, atol=0.0)


def test_ragged_causal_conv1d_create_shard_map_wrapper_matches_direct():
    """Assert the conv1d ``create_shard_map_wrapper`` output equals the direct ``run`` output.

    Builds the wrapper with a single-device mesh and fully replicated ``in_specs``/``out_specs``,
    invokes the returned ``shard_fn`` over the returned ``call_args``, and asserts the sharded
    ``(output, state)`` pair matches the unsharded ``RaggedCausalConv1D().run(...)`` result
    exactly (``atol=0.0``). On a single device the shard-map path must be a numeric no-op.

    Raises:
        AssertionError: If the sharded output or state differs from the direct result.
    """
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    kernel = jnp.ones((4, 3), dtype=jnp.float32) * 0.25
    query_start_loc = jnp.array([0, 1, 3], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
    cfg = RaggedCausalConv1DConfig(d_conv=3, apply_silu=False, platform="xla")

    def fresh_state():
        """Return a freshly zeroed conv state of shape ``(2, 4, 3)``.

        Returns:
            jax.Array: A new zero-filled float32 per-slot convolution state for each call.
        """
        return jnp.zeros((2, 4, 3), dtype=jnp.float32)

    direct_out, direct_state = RaggedCausalConv1D().run(
        x,
        fresh_state(),
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=3,
        apply_silu=False,
        cfg=cfg,
    )
    replicated = PartitionSpec()
    shard_fn, call_args = RaggedCausalConv1D().create_shard_map_wrapper(
        x,
        fresh_state(),
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=3,
        apply_silu=False,
        cfg=cfg,
        mesh=_single_device_mesh(),
        in_specs=(replicated, replicated, replicated, replicated, replicated, replicated),
        out_specs=(replicated, replicated),
    )
    sharded_out, sharded_state = shard_fn(*call_args)

    assert_allclose(sharded_out, direct_out, atol=0.0)
    assert_allclose(sharded_state, direct_state, atol=0.0)


def test_ragged_gated_delta_rule_v2_function_class_and_op_match():
    """Assert ragged gated-delta-rule v2 agrees across functional, class, and op entry points.

    Builds a small packed gated-delta-rule fixture: a fused ``mixed_qkv`` of shape
    ``(tokens, 2*key_dim + value_dim)`` (query+key+value packed along the last axis), zero gate
    tensors ``b``/``a``, zero ``A_log``/``dt_bias`` decay parameters, the
    ``query_start_loc``/``state_indices``/``distribution`` packing metadata, and a zeroed per-slot
    recurrent state of shape ``(2, n_v, d_k, d_v)``. The same computation runs through
    ``ragged_gated_delta_rule_v2`` (functional), ``RaggedGatedDeltaRuleV2().run(...)`` (class),
    and ``ragged_gated_delta_rule_v2_op`` (op adapter), each returning a ``(new_state, output)``
    pair. Asserts the direct output has shape ``(tokens, n_v*d_v)``, the state has shape
    ``(2, n_v, d_k, d_v)``, the output is all-finite, and the class/op results match the direct
    result exactly (``atol=0.0``).

    Raises:
        AssertionError: If shapes are wrong, the output is non-finite, or any entry point diverges.
    """
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 1, 4, 3
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(0),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)
    cfg = RaggedGatedDeltaRuleV2Config(chunk_size=2)

    def fresh_state():
        """Return a freshly zeroed recurrent state of shape ``(2, n_v, d_k, d_v)``.

        Returns:
            jax.Array: A new zero-filled float32 per-slot recurrent state, independent per call.
        """
        return jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)

    direct_state, direct_out = ragged_gated_delta_rule_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
    )
    class_state, class_out = RaggedGatedDeltaRuleV2().run(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
    )
    op_state, op_out = ragged_gated_delta_rule_v2_op(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
    )

    assert direct_out.shape == (tokens, n_v * d_v)
    assert direct_state.shape == (2, n_v, d_k, d_v)
    assert jnp.all(jnp.isfinite(direct_out))
    assert_allclose(class_out, direct_out, atol=0.0)
    assert_allclose(class_state, direct_state, atol=0.0)
    assert_allclose(op_out, direct_out, atol=0.0)
    assert_allclose(op_state, direct_state, atol=0.0)


def test_ragged_gated_delta_rule_v2_create_shard_map_wrapper_matches_direct():
    """Assert the gated-delta-rule v2 shard-map wrapper output equals the direct ``run`` output.

    Provides an explicit ``has_initial_state`` boolean mask and builds the wrapper on a
    single-device mesh with fully replicated specs for all ten inputs. Runs the returned
    ``shard_fn`` over ``call_args`` and asserts the sharded ``(state, output)`` pair matches
    the direct ``RaggedGatedDeltaRuleV2().run(...)`` result exactly (``atol=0.0``).

    Raises:
        AssertionError: If the sharded output or state differs from the direct result.
    """
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 1, 4, 3
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(0),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)
    has_initial_state = jnp.ones((2,), dtype=jnp.bool_)
    cfg = RaggedGatedDeltaRuleV2Config(chunk_size=2, platform="xla")

    def fresh_state():
        """Return a freshly zeroed recurrent state of shape ``(2, n_v, d_k, d_v)``.

        Returns:
            jax.Array: A new zero-filled float32 per-slot recurrent state, independent per call.
        """
        return jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)

    direct_state, direct_out = RaggedGatedDeltaRuleV2().run(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
    )
    replicated = PartitionSpec()
    shard_fn, call_args = RaggedGatedDeltaRuleV2().create_shard_map_wrapper(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
        mesh=_single_device_mesh(),
        in_specs=(
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_specs=(replicated, replicated),
    )
    sharded_state, sharded_out = shard_fn(*call_args)

    assert_allclose(sharded_out, direct_out, atol=0.0)
    assert_allclose(sharded_state, direct_state, atol=0.0)


def test_ragged_gated_delta_rule_v2_split_wrapper_generates_initial_state_mask():
    """Assert the split-qkv head-shard wrapper reproduces the direct result via its callback.

    Exercises ``create_shard_map_wrapper`` with ``split_qkv_for_head_shard=True`` and without an
    explicit ``has_initial_state`` argument: in this mode the wrapper is responsible for
    generating the initial-state mask itself and returns a third value, a ``callback``, used to
    post-process the shard-map output. The reference is computed with an explicit all-True
    ``has_initial_state`` mask passed to the direct ``run``. The test runs ``shard_fn(*call_args)``,
    pipes the raw result through ``callback(..., cfg)`` to recover ``(state, output)``, and asserts
    those match the direct result exactly (``atol=0.0``).

    Raises:
        AssertionError: If the callback-postprocessed sharded output or state differs from direct.
    """
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 1, 4, 3
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(3),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)
    cfg = RaggedGatedDeltaRuleV2Config(chunk_size=2, platform="xla")

    def fresh_state():
        """Return a freshly zeroed recurrent state of shape ``(2, n_v, d_k, d_v)``.

        Returns:
            jax.Array: A new zero-filled float32 per-slot recurrent state, independent per call.
        """
        return jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)

    direct_state, direct_out = RaggedGatedDeltaRuleV2().run(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        jnp.ones((2,), dtype=jnp.bool_),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
    )
    replicated = PartitionSpec()
    shard_fn, call_args, callback = RaggedGatedDeltaRuleV2().create_shard_map_wrapper(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
        split_qkv_for_head_shard=True,
        mesh=_single_device_mesh(),
        in_specs=(
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_specs=(replicated, replicated),
    )
    sharded_state, sharded_out = callback(shard_fn(*call_args), cfg)

    assert_allclose(sharded_out, direct_out, atol=0.0)
    assert_allclose(sharded_state, direct_state, atol=0.0)


def test_ragged_gated_delta_rule_v2_split_head_sharded_path_matches_direct():
    """Assert the split (non-flat) head-sharded GDR v2 path matches the unsharded result.

    Requires two devices (skips otherwise). Computes the reference with a plain
    ``ragged_gated_delta_rule_v2`` call, then re-runs the same functional entry point with
    ``mesh=<two-device>``, ``head_axis="x"``, and ``flat_tp_shard=False`` to drive the split
    head-sharded code path (heads distributed across the mesh axis without flattening into a
    single tensor-parallel layout). Uses ``n_v=2`` value heads so head sharding is meaningful.
    Asserts the sharded ``(state, output)`` pair matches the direct result within
    ``atol=1e-5, rtol=1e-5``.

    Raises:
        AssertionError: If the head-sharded output or state diverges beyond tolerance.
        Skipped: When fewer than two local devices are available.
    """
    mesh = _two_device_mesh_or_skip()
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 2, 4, 3
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(1),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)
    has_initial_state = jnp.ones((2,), dtype=jnp.bool_)
    cfg = RaggedGatedDeltaRuleV2Config(chunk_size=2, platform="xla")

    def fresh_state():
        """Return a freshly zeroed recurrent state of shape ``(2, n_v, d_k, d_v)``.

        Returns:
            jax.Array: A new zero-filled float32 per-slot recurrent state, independent per call.
        """
        return jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)

    direct_state, direct_out = ragged_gated_delta_rule_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
    )
    sharded_state, sharded_out = ragged_gated_delta_rule_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        mesh=mesh,
        head_axis="x",
        flat_tp_shard=False,
        cfg=cfg,
    )

    assert_allclose(sharded_out, direct_out, atol=1e-5, rtol=1e-5)
    assert_allclose(sharded_state, direct_state, atol=1e-5, rtol=1e-5)


def test_ragged_gated_delta_rule_v2_flat_head_sharded_path_matches_direct():
    """Assert the flat (tensor-parallel) head-sharded GDR v2 path matches the unsharded result.

    Requires two devices (skips otherwise). Mirrors the split head-sharded test but with
    ``n_kq=2`` query/key heads and ``flat_tp_shard=True`` so the heads are sharded in the flat
    tensor-parallel layout across the ``head_axis="x"`` mesh axis. Computes a reference via a
    plain ``ragged_gated_delta_rule_v2`` call and asserts the flat-sharded ``(state, output)``
    pair matches within ``atol=1e-5, rtol=1e-5``.

    Raises:
        AssertionError: If the flat head-sharded output or state diverges beyond tolerance.
        Skipped: When fewer than two local devices are available.
    """
    mesh = _two_device_mesh_or_skip()
    tokens, n_kq, n_v, d_k, d_v = 2, 2, 2, 4, 3
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(2),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)
    has_initial_state = jnp.ones((2,), dtype=jnp.bool_)
    cfg = RaggedGatedDeltaRuleV2Config(chunk_size=2, platform="xla")

    def fresh_state():
        """Return a freshly zeroed recurrent state of shape ``(2, n_v, d_k, d_v)``.

        Returns:
            jax.Array: A new zero-filled float32 per-slot recurrent state, independent per call.
        """
        return jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)

    direct_state, direct_out = ragged_gated_delta_rule_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        cfg=cfg,
    )
    sharded_state, sharded_out = ragged_gated_delta_rule_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
        mesh=mesh,
        head_axis="x",
        flat_tp_shard=True,
        cfg=cfg,
    )

    assert_allclose(sharded_out, direct_out, atol=1e-5, rtol=1e-5)
    assert_allclose(sharded_state, direct_state, atol=1e-5, rtol=1e-5)


def _make_fused_conv_decode_inputs(num_slots: int = 2, conv_dim: int = 4, d_conv: int = 3):
    """Build random inputs for the fused single-step decode convolution kernel.

    Args:
        num_slots: Number of independent cache/decode slots (batch of state buffers). Defaults to 2.
        conv_dim: Number of convolution channels (feature width). Defaults to 4.
        d_conv: Convolution kernel width / state depth. Defaults to 3.

    Returns:
        tuple: A 3-tuple ``(conv_state, new_tokens, kernel)`` of float32 arrays, all derived from a
            fixed PRNG seed (7) so the fixture is deterministic:

            * ``conv_state``: shape ``(num_slots, conv_dim, d_conv)`` -- the per-slot rolling conv state.
            * ``new_tokens``: shape ``(num_slots, conv_dim)`` -- the new per-slot decode-step inputs.
            * ``kernel``: shape ``(conv_dim, d_conv)`` -- the per-channel depthwise convolution weights.
    """
    rng = jax.random.PRNGKey(7)
    conv_state = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32)
    new_tokens = jax.random.normal(jax.random.fold_in(rng, 1), (num_slots, conv_dim), dtype=jnp.float32)
    kernel = jax.random.normal(jax.random.fold_in(rng, 2), (conv_dim, d_conv), dtype=jnp.float32)
    return conv_state, new_tokens, kernel


def _make_grouped_gdr_decode_inputs(
    batch: int = 2,
    num_k_heads: int = 3,
    expand_ratio: int = 2,
    head_dim: int = 4,
    value_dim: int = 5,
):
    """Build random inputs for the grouped (GQA-style) gated-delta-rule single-step decode kernel.

    The number of value heads is derived as ``num_v_heads = num_k_heads * expand_ratio``, so each
    key/query head is shared across ``expand_ratio`` value heads. Gate values are passed through
    activations to keep them in valid ranges: ``beta`` via sigmoid (in (0, 1)) and ``decay`` via the
    negated softplus (non-positive log-decay).

    Args:
        batch: Number of independent decode positions in the batch. Defaults to 2.
        num_k_heads: Number of key/query heads. Defaults to 3.
        expand_ratio: Number of value heads grouped per key head (GQA expansion). Defaults to 2.
        head_dim: Per-head key/query feature dimension. Defaults to 4.
        value_dim: Per-head value feature dimension. Defaults to 5.

    Returns:
        tuple: A 6-tuple ``(query, key, value, beta, decay, recurrent_state)`` of float32 arrays,
            derived from a fixed PRNG seed (11):

            * ``query``: shape ``(batch, num_k_heads, head_dim)``.
            * ``key``: shape ``(batch, num_k_heads, head_dim)``.
            * ``value``: shape ``(batch, num_k_heads, expand_ratio, value_dim)``.
            * ``beta``: shape ``(batch, num_k_heads, expand_ratio)``, sigmoid-activated update gate.
            * ``decay``: shape ``(batch, num_k_heads, expand_ratio)``, negative log-decay rate.
            * ``recurrent_state``: shape ``(batch, num_v_heads, head_dim, value_dim)``.
    """
    num_v_heads = num_k_heads * expand_ratio
    rng = jax.random.PRNGKey(11)
    query = jax.random.normal(rng, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, num_k_heads, head_dim), dtype=jnp.float32)
    value = jax.random.normal(
        jax.random.fold_in(rng, 2),
        (batch, num_k_heads, expand_ratio, value_dim),
        dtype=jnp.float32,
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    )
    decay = -jax.nn.softplus(
        jax.random.normal(jax.random.fold_in(rng, 4), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    )
    recurrent_state = jax.random.normal(
        jax.random.fold_in(rng, 5),
        (batch, num_v_heads, head_dim, value_dim),
        dtype=jnp.float32,
    )
    return query, key, value, beta, decay, recurrent_state


def test_fused_conv_decode_function_class_and_op_match():
    """Assert fused conv decode agrees across functional, class, and op entry points.

    Generates a ``(num_slots=2, conv_dim=4, d_conv=3)`` fixture and runs the fused single-step
    decode convolution with ``output_dtype=jnp.float32`` through ``fused_conv_decode`` (functional),
    ``FusedConvDecode().run(...)`` (class), and ``fused_conv_decode_op`` (op adapter), each returning
    a ``(new_state, output)`` pair. Asserts the direct output has shape ``(2, 4)``, the new state has
    shape ``(2, 4, 3)``, and the class/op results match the direct result exactly (``atol=0.0``).

    Raises:
        AssertionError: If output/state shapes are wrong or any entry point's result diverges.
    """
    conv_state, new_tokens, kernel = _make_fused_conv_decode_inputs()
    cfg = FusedConvDecodeConfig(d_conv=3, activation="silu", platform="xla")

    direct_state, direct_out = fused_conv_decode(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
    )
    class_state, class_out = FusedConvDecode().run(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
        cfg=cfg,
    )
    op_state, op_out = fused_conv_decode_op(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
        cfg=cfg,
    )

    assert direct_out.shape == (2, 4)
    assert direct_state.shape == (2, 4, 3)
    assert_allclose(class_out, direct_out, atol=0.0)
    assert_allclose(class_state, direct_state, atol=0.0)
    assert_allclose(op_out, direct_out, atol=0.0)
    assert_allclose(op_state, direct_state, atol=0.0)


def test_gated_delta_rule_grouped_decode_function_class_and_op_match():
    """Assert grouped gated-delta-rule decode agrees across functional and class entry points.

    Builds a grouped GDR decode fixture and runs the kernel through
    ``gated_delta_rule_grouped_decode`` (functional) and ``GatedDeltaRuleGroupedDecode().run(...)``
    (class), using ``platform="xla", backend="any"``. Each returns an ``(output, new_state)`` pair.
    Asserts the output has shape ``(batch, num_v_heads, value_dim)`` derived from the recurrent state
    shape ``(batch, num_v_heads, head_dim, value_dim)``, the new state shape equals the input state
    shape, and the class result matches the direct result within ``atol=1e-5, rtol=1e-5``.

    Despite the name, only the functional and class forms are compared here (there is no ``*_op``
    call in this test).

    Raises:
        AssertionError: If output/state shapes are wrong or the class result diverges beyond tolerance.
    """
    query, key, value, beta, decay, recurrent_state = _make_grouped_gdr_decode_inputs()
    cfg = GatedDeltaRuleGroupedDecodeConfig(platform="xla", backend="any")

    direct_out, direct_state = gated_delta_rule_grouped_decode(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        cfg=cfg,
    )
    class_out, class_state = GatedDeltaRuleGroupedDecode().run(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        cfg=cfg,
    )

    assert direct_out.shape == (recurrent_state.shape[0], recurrent_state.shape[1], recurrent_state.shape[3])
    assert direct_state.shape == recurrent_state.shape
    assert_allclose(class_out, direct_out, atol=1e-5, rtol=1e-5)
    assert_allclose(class_state, direct_state, atol=1e-5, rtol=1e-5)


def test_gated_delta_rule_grouped_decode_create_shard_map_wrapper_matches_direct():
    """Assert the grouped GDR decode shard-map wrapper output equals the direct ``run`` output.

    Builds the wrapper on a single-device mesh with fully replicated ``in_specs``/``out_specs`` and
    ``check_vma=False``, runs the returned ``shard_fn`` over ``call_args``, and asserts the sharded
    ``(output, state)`` pair matches the direct ``GatedDeltaRuleGroupedDecode().run(...)`` result
    within ``atol=1e-5, rtol=1e-5``.

    Raises:
        AssertionError: If the sharded output or state diverges from the direct result beyond tolerance.
    """
    query, key, value, beta, decay, recurrent_state = _make_grouped_gdr_decode_inputs()
    cfg = GatedDeltaRuleGroupedDecodeConfig(platform="xla", backend="any")
    replicated = PartitionSpec()

    direct_out, direct_state = GatedDeltaRuleGroupedDecode().run(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        cfg=cfg,
    )
    shard_fn, call_args = GatedDeltaRuleGroupedDecode().create_shard_map_wrapper(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        cfg=cfg,
        mesh=_single_device_mesh(),
        in_specs=(replicated, replicated, replicated, replicated, replicated, replicated),
        out_specs=(replicated, replicated),
        check_vma=False,
    )
    sharded_out, sharded_state = shard_fn(*call_args)

    assert_allclose(sharded_out, direct_out, atol=1e-5, rtol=1e-5)
    assert_allclose(sharded_state, direct_state, atol=1e-5, rtol=1e-5)

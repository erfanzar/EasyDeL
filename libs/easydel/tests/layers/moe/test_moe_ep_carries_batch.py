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

"""MaxText-style "ep carries batch" fused-MoE dispatch (``moe_ep_carries_batch``).

With the knob on, the token batch shards over the expert axis as well
((dp, fsdp, ep) instead of (dp, fsdp)) and tokens travel to their
expert-owning shard through a ragged all-to-all DISPATCH, are locally
re-sorted by local expert, run the grouped-matmul FFN, and return through a
COMBINE all-to-all (the transposed traffic matrix). This is the port of
MaxText's ``is_batch_sharded_by_ep`` training path
(maxtext/layers/moe.py: ``get_all_to_all_params`` / ``local_permute`` /
``ra2a_and_route``).

XLA:CPU cannot *execute* ``jax.lax.ragged_all_to_all`` (unsupported by the
CPU thunk emitter), so runtime coverage here has three layers:

* the pure routing/offset/permute logic (traffic matrix, a2a parameters in
  both directions, receive-buffer bound, chunk group sizes) is exercised
  numerically against hand-computed values and a full numpy-emulated
  multi-shard dispatch that must match a dense single-device reference;
* the end-to-end module path (forward AND backward, chunked and unchunked)
  is traced via ``jax.eval_shape`` on the fake 8-device mesh, which
  validates every shard_map spec, buffer shape, and transpose rule without
  running the collective;
* knob-off (and knob-on-but-inactive) configurations must remain
  bit-identical to the historical path.

True multi-shard runtime numerics require TPU
(``scripts/tpu_probe_ep_carries_batch.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _require_fake_mesh():
    """Skip unless the fake 8-device CPU host mesh is available."""
    if jax.device_count() < 8:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")


def _build_gptoss_block(
    *,
    sharding_axis_dims,
    ep_carries_batch: bool,
    ring: bool = False,
    fsdp_is_ep_bound: bool = False,
    moe_chunk_size: int | None = None,
    num_experts: int = 8,
    seed: int = 123,
):
    """Instantiate a GptOssMLP (fused MoE) under a mesh, mirroring the fsdp-batch tests."""
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=num_experts,
        num_experts_per_tok=2,
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(
        use_ring_of_experts=ring,
        fsdp_is_ep_bound=fsdp_is_ep_bound,
        sp_is_ep_bound=False,
        moe_ep_carries_batch=ep_carries_batch,
        moe_chunk_size=moe_chunk_size,
    )
    return GptOssMLP(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))


def _input(batch: int = 8, hidden_size: int = 32):
    """Deterministic (batch, 4, hidden) test batch shared by all meshes."""
    return jax.random.normal(jax.random.PRNGKey(0), (batch, 4, hidden_size), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Config-level lever: PartitionAxis.batch_axis gains the ep axis.
# ---------------------------------------------------------------------------


def test_config_extends_activation_batch_axis_with_ep():
    """moe_ep_carries_batch extends PartitionAxis.batch_axis model-wide; decode stays put."""
    _require_fake_mesh()

    on = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True)
    assert on.config.partition_axis.batch_axis == ("fsdp", "dp", "ep")
    assert "ep" not in tuple(
        on.config.partition_axis.decode_batch_axis
        if isinstance(on.config.partition_axis.decode_batch_axis, (tuple, list))
        else (on.config.partition_axis.decode_batch_axis,)
    ), "decode_batch_axis must keep its historical ep-replicated placement"

    off = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=False)
    assert off.config.partition_axis.batch_axis == ("fsdp", "dp")

    # ep-bound folding contradicts ep-carried batch: the lever must not fire.
    bound = _build_gptoss_block(
        sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True, fsdp_is_ep_bound=True
    )
    assert bound.config.partition_axis.batch_axis == ("fsdp", "dp")


def test_batch_axis_names_resolution():
    """The fused-MoE shard_map batch axes gain ep exactly when the gate is active."""
    from easydel.layers.moe._communication_utils import DP, resolve_eformer_axis

    _require_fake_mesh()

    def batch_axes(block):
        with block.config.mesh:
            expert_mesh = block._active_auto_expert_mesh(arr=None)
            dp_axis = resolve_eformer_axis(DP, block.runtime_sharding_resolver)
            return block._moe_batch_axis_names(expert_mesh, dp_axis)

    on = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True)
    assert batch_axes(on) == ("dp", "fsdp", "ep")

    off = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=False)
    assert batch_axes(off) == ("dp", "fsdp")

    # ring all-gathers the batch over ep — the opposite placement; gate stays off.
    ring = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True, ring=True)
    assert batch_axes(ring) == ("dp", "fsdp")

    # ep=1: nothing to carry.
    ep1 = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ep_carries_batch=True)
    assert batch_axes(ep1) == ("dp", "fsdp")

    # folded 3-D expert mesh (fsdp ep-bound): no distinct fsdp axis, gate off.
    folded = _build_gptoss_block(
        sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True, fsdp_is_ep_bound=True
    )
    assert batch_axes(folded) == ("dp",)


def test_knob_inactive_is_bit_identical():
    """knob=True on an ep=1 mesh must produce bit-identical outputs to knob=False.

    The gate deactivates (ep_size == 1), and the extended (fsdp, dp, ep)
    batch spec splits over a size-1 axis — a placement no-op. Any numeric
    drift here means the knob is not gated purely.
    """
    _require_fake_mesh()

    off = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ep_carries_batch=False)
    on = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ep_carries_batch=True)
    x = _input()

    with off.config.mesh:
        out_off, logits_off = off(x)
    with on.config.mesh:
        out_on, logits_on = on(x)

    assert out_on.shape == out_off.shape
    assert bool(jnp.array_equal(out_on, out_off)), "inactive moe_ep_carries_batch changed the ep=1 output"
    assert bool(jnp.array_equal(logits_on, logits_off))


# ---------------------------------------------------------------------------
# Pure logic: a2a parameters, traffic matrix, buffer bound, chunk group sizes.
# ---------------------------------------------------------------------------


def test_all_to_all_params_batch_sharded_hand_reference():
    """Dispatch and combine parameter vectors match a hand-computed 2-shard example."""
    from easydel.layers.moe._communication_utils import get_all_to_all_params

    # matrix[i, j] = rows batch shard i sends expert shard j.
    matrix = jnp.array([[3, 1], [2, 4]], dtype=jnp.int32)

    in_off, send, out_off, recv = get_all_to_all_params(matrix, 0, 2, is_batch_sharded=True, is_dispatch=True)
    np.testing.assert_array_equal(np.asarray(in_off), [0, 3])
    np.testing.assert_array_equal(np.asarray(send), [3, 1])
    np.testing.assert_array_equal(np.asarray(out_off), [0, 0])
    np.testing.assert_array_equal(np.asarray(recv), [3, 2])

    in_off, send, out_off, recv = get_all_to_all_params(matrix, 1, 2, is_batch_sharded=True, is_dispatch=True)
    np.testing.assert_array_equal(np.asarray(in_off), [0, 2])
    np.testing.assert_array_equal(np.asarray(send), [2, 4])
    np.testing.assert_array_equal(np.asarray(out_off), [3, 1])
    np.testing.assert_array_equal(np.asarray(recv), [1, 4])

    # Combine is the same exchange on the transposed matrix.
    in_off, send, out_off, recv = get_all_to_all_params(matrix, 0, 2, is_batch_sharded=True, is_dispatch=False)
    np.testing.assert_array_equal(np.asarray(in_off), [0, 3])
    np.testing.assert_array_equal(np.asarray(send), [3, 2])
    np.testing.assert_array_equal(np.asarray(out_off), [0, 0])
    np.testing.assert_array_equal(np.asarray(recv), [3, 1])

    in_off, send, out_off, recv = get_all_to_all_params(matrix, 1, 2, is_batch_sharded=True, is_dispatch=False)
    np.testing.assert_array_equal(np.asarray(in_off), [0, 1])
    np.testing.assert_array_equal(np.asarray(send), [1, 4])
    np.testing.assert_array_equal(np.asarray(out_off), [3, 2])
    np.testing.assert_array_equal(np.asarray(recv), [2, 4])


def test_all_to_all_params_replicated_batch_unchanged():
    """The is_batch_sharded=False layout (existing production path) is untouched.

    ``is_dispatch`` transposes the traffic matrix, which is a no-op on the
    1-D replicated-batch layout — knob-off bit-identity at the parameter
    level.
    """
    from easydel.layers.moe._communication_utils import get_all_to_all_params

    sizes = jnp.array([5, 3, 7, 1], dtype=jnp.int32)
    base = get_all_to_all_params(sizes, 2, 4, is_batch_sharded=False)
    with_flag = get_all_to_all_params(sizes, 2, 4, is_batch_sharded=False, is_dispatch=False)
    for a, b in zip(base, with_flag, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _emulate_ragged_all_to_all(buffers, params_per_shard, recv_rows, width):
    """Numpy emulation of jax.lax.ragged_all_to_all across a virtual ep axis.

    ``buffers[s]`` is shard *s*'s operand; ``params_per_shard[s]`` its
    (input_offsets, send_sizes, output_offsets, recv_sizes). Sender *s*
    writes its segment for destination *j* at ``output_offsets[j]`` of *j*'s
    output buffer — the operational semantics of the primitive.
    """
    out = [np.zeros((recv_rows, width), dtype=buffers[0].dtype) for _ in buffers]
    for s, (in_off, send, out_off, _recv) in enumerate(params_per_shard):
        for j in range(len(buffers)):
            n = int(send[j])
            src = int(in_off[j])
            dst = int(out_off[j])
            out[j][dst : dst + n] = buffers[s][src : src + n]
    return out


def test_receive_buffer_bound_dominates_all_routings():
    """The static receive bound is never exceeded by any routing (brute force)."""
    from easydel.layers.moe._communication_utils import (
        build_ep_traffic_matrix,
        ep_batch_receive_buffer_rows,
    )

    rng = np.random.default_rng(0)
    ep, les, k, tokens = 2, 2, 2, 5
    num_experts = ep * les
    bound = ep_batch_receive_buffer_rows(tokens, k, ep, les)
    assert bound == ep * tokens * min(k, les)

    worst = 0
    for _ in range(200):
        # k distinct experts per token, per shard.
        group_sizes = np.zeros((ep, num_experts), dtype=np.int64)
        for s in range(ep):
            for _t in range(tokens):
                chosen = rng.choice(num_experts, size=k, replace=False)
                for e in chosen:
                    group_sizes[s, e] += 1
        matrix = np.asarray(build_ep_traffic_matrix(jnp.asarray(group_sizes), ep, les))
        worst = max(worst, int(matrix.sum(axis=0).max()))
        assert matrix.sum(axis=0).max() <= bound
    # The bound must be attainable-ish, not vacuous: all tokens CAN route to one shard.
    assert worst > tokens * k // 2

    # les < k tightens the bound below ep * tokens * k.
    assert ep_batch_receive_buffer_rows(16, 8, 4, 2) == 4 * 16 * 2
    assert ep_batch_receive_buffer_rows(16, 8, 4, 64) == 4 * 16 * 8


def test_chunk_group_sizes_partition_the_valid_prefix():
    """Chunk group sizes tile the ragged groups exactly and exclude the buffer tail."""
    from easydel.layers.moe._communication_utils import chunk_group_sizes

    group_sizes = jnp.array([3, 0, 5, 2], dtype=jnp.int32)  # valid rows: 10
    ends = jnp.cumsum(group_sizes)
    starts = ends - group_sizes
    chunk_rows = 4
    buffer_rows = 16  # 6 rows of worst-case slack past the valid prefix

    total = np.zeros(4, dtype=np.int64)
    for c0 in range(0, buffer_rows, chunk_rows):
        gs_c = np.asarray(chunk_group_sizes(starts, ends, jnp.int32(c0), chunk_rows))
        assert gs_c.sum() <= chunk_rows
        total += gs_c
    np.testing.assert_array_equal(total, np.asarray(group_sizes))  # tail contributed nothing


def test_ep_batch_dispatch_simulation_matches_dense_reference():
    """Full numpy-emulated 2-shard dispatch == dense per-token reference.

    Runs the exact pure building blocks the module uses — ``permute``,
    ``build_ep_traffic_matrix``, ``get_all_to_all_params`` (both
    directions), ``local_permute``, ``sort_activations`` unsort,
    ``unpermute`` — with the two collectives emulated in numpy, against a
    dense reference computing ``sum_k w_k * (x @ W[e_k])`` per token. Any
    offset, ordering, or masking bug in the dispatch flow breaks this.
    The chunked FFN variant (``chunk_group_sizes``) must match the
    unchunked one bit-for-bit.
    """
    from easydel.layers.moe._communication_utils import (
        build_ep_traffic_matrix,
        chunk_group_sizes,
        ep_batch_receive_buffer_rows,
        get_all_to_all_params,
        local_permute,
        permute,
        sort_activations,
        unpermute,
    )

    rng = np.random.default_rng(42)
    ep, les, k, tokens, hidden = 2, 2, 2, 6, 8
    num_experts = ep * les
    W = rng.normal(size=(num_experts, hidden, hidden)).astype(np.float32)

    X = rng.normal(size=(ep, tokens, hidden)).astype(np.float32)
    logits = rng.normal(size=(ep, tokens, num_experts)).astype(np.float32)

    # Dense reference over the concatenated batch.
    ref = np.zeros((ep, tokens, hidden), dtype=np.float32)
    for s in range(ep):
        w_top, sel = jax.lax.top_k(jnp.asarray(logits[s]), k)
        w_top, sel = np.asarray(w_top), np.asarray(sel)
        for t in range(tokens):
            for j in range(k):
                ref[s, t] += w_top[t, j] * (X[s, t] @ W[sel[t, j]])

    # --- per-shard permute (pure, no collective) ---
    shard_state = []
    for s in range(ep):
        xs, sorted_sel, weights, group_sizes, _sorted_experts = permute(
            inputs=jnp.asarray(X[s][None]),
            gate_logits=jnp.asarray(logits[s]),
            num_experts_per_tok=k,
            num_experts=num_experts,
            dtype=jnp.float32,
        )
        shard_state.append((np.asarray(xs), np.asarray(sorted_sel), np.asarray(weights), np.asarray(group_sizes)))

    global_group_sizes = jnp.asarray(np.stack([st[3] for st in shard_state]))  # emulated all_gather
    matrix = build_ep_traffic_matrix(global_group_sizes, ep, les)
    recv_rows = ep_batch_receive_buffer_rows(tokens, k, ep, les)

    dispatch_params = [
        tuple(np.asarray(p) for p in get_all_to_all_params(matrix, s, ep, is_batch_sharded=True, is_dispatch=True))
        for s in range(ep)
    ]
    received = _emulate_ragged_all_to_all([st[0] for st in shard_state], dispatch_params, recv_rows, hidden)

    combine_params = [
        tuple(np.asarray(p) for p in get_all_to_all_params(matrix, s, ep, is_batch_sharded=True, is_dispatch=False))
        for s in range(ep)
    ]

    outputs = []
    for s in range(ep):
        x_sorted, sorted_idx, local_gs, _local_ids = local_permute(
            jnp.asarray(received[s]),
            global_group_sizes,
            les,
            shard_index=s,
            is_offset=False,
        )
        x_sorted = np.asarray(x_sorted)
        local_gs = np.asarray(local_gs)
        valid = int(local_gs.sum())

        def grouped_ffn(rows, gs, base_expert):
            """Reference grouped matmul: rows past sum(gs) stay zero (tail mask)."""
            y = np.zeros_like(rows)
            r = 0
            for e, n in enumerate(np.asarray(gs)):
                n = int(n)
                y[r : r + n] = rows[r : r + n] @ W[base_expert + e]
                r += n
            return y

        y = grouped_ffn(x_sorted, local_gs, s * les)

        # Chunked variant must agree exactly (chunk boundaries cut groups).
        ends = jnp.cumsum(jnp.asarray(local_gs))
        starts = ends - jnp.asarray(local_gs)
        chunk_rows_n = 5  # deliberately not a divisor of recv_rows
        padded = -(-recv_rows // chunk_rows_n) * chunk_rows_n
        x_p = np.pad(x_sorted, ((0, padded - recv_rows), (0, 0)))
        y_chunked = np.zeros_like(x_p)
        for c0 in range(0, padded, chunk_rows_n):
            gs_c = chunk_group_sizes(starts, ends, jnp.int32(c0), chunk_rows_n)
            # Base expert of the chunk: experts whose range ends at/before c0 are done.
            y_chunked[c0 : c0 + chunk_rows_n] = grouped_ffn(
                x_p[c0 : c0 + chunk_rows_n],
                np.asarray(gs_c),
                s * les,
            )
        np.testing.assert_allclose(y_chunked[:recv_rows], y, rtol=1e-6, atol=1e-6)

        # Mask tail, local unsort, mask again (mirrors _dispatch_ep_batch).
        y[valid:] = 0.0
        y_unsorted = np.array(sort_activations(jnp.asarray(y), jnp.argsort(jnp.asarray(sorted_idx)), True))
        y_unsorted[valid:] = 0.0
        outputs.append(y_unsorted)

    combined = _emulate_ragged_all_to_all(outputs, combine_params, tokens * k, hidden)

    for s in range(ep):
        _xs, sorted_sel, weights, _gs = shard_state[s]
        out = unpermute(
            jnp.asarray(combined[s]),
            jnp.asarray(sorted_sel),
            jnp.asarray(weights),
            batch_size=1,
            sequence_length=tokens,
            num_experts_per_tok=k,
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(np.asarray(out)[0], ref[s], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end module traces (fwd + bwd) on the fake 8-device mesh.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [None, 4], ids=["unchunked", "chunked"])
def test_ep_carries_batch_traces_fwd_and_bwd(chunk):
    """The full module path traces forward and backward under the (dp,fsdp,ep) specs.

    XLA:CPU cannot execute ragged_all_to_all, but abstract evaluation
    validates every shard_map in/out spec, the a2a parameter shapes, the
    worst-case receive buffer shape, and — via ``jax.grad`` — the transpose
    rules of the dispatch/combine collectives and the tail masks.
    """
    from easydel.layers.moe._communication_utils import MoeFusedHooks

    _require_fake_mesh()

    block = _build_gptoss_block(
        sharding_axis_dims=(1, 1, 2, 2, 2, 1),
        ep_carries_batch=True,
        moe_chunk_size=chunk,
    )
    seen: list[tuple[int, ...]] = []

    def spy(inputs_2d, weights, inputs_shape):
        seen.append(tuple(inputs_2d.shape))
        return inputs_2d

    block.moe_hooks = MoeFusedHooks(refine_inputs_hook=spy)
    x = _input()

    with block.config.mesh:
        out_struct = jax.eval_shape(lambda xin: block(xin)[0], x)
        grad_struct = jax.eval_shape(jax.grad(lambda xin: jnp.sum(block(xin)[0].astype(jnp.float32) ** 2)), x)

    assert out_struct.shape == x.shape
    assert grad_struct.shape == x.shape

    # Batch must be sharded over dp*fsdp*ep = 4 inside the shard_map: the
    # permute hook sees B/(dp*fsdp*ep) * S = 8 local rows (not 16).
    batch, seq, hidden = x.shape
    expected_rows = (batch // 4) * seq
    assert seen, "refine_inputs_hook never ran inside the fused MoE shard_map"
    assert set(seen) == {(expected_rows, hidden)}, (
        f"per-shard token rows {set(seen)} != {{({expected_rows}, {hidden})}}: the batch is not "
        "sharded over (dp, fsdp, ep) inside the ep-carries-batch shard_map"
    )


def test_ep_carries_batch_emits_remat_tags():
    """The dispatch/combine a2a outputs carry checkpoint_name tags for remat policies."""
    _require_fake_mesh()

    block = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True)
    x = _input()
    with block.config.mesh:
        jaxpr = jax.make_jaxpr(lambda xin: block(xin)[0])(x)
    text = str(jaxpr)
    assert "moe_dispatch" in text, "dispatch a2a output lost its checkpoint_name tag"
    assert "moe_combine" in text, "combine a2a output lost its checkpoint_name tag"
    assert "ragged_all_to_all" in text


def test_non_divisible_batch_falls_back():
    """A batch not divisible by dp*fsdp*ep deactivates the path instead of erroring."""
    from easydel.layers.moe._communication_utils import MoeFusedHooks

    _require_fake_mesh()

    block = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ep_carries_batch=True)
    seen: list[tuple[int, ...]] = []

    def spy(inputs_2d, weights, inputs_shape):
        seen.append(tuple(inputs_2d.shape))
        return inputs_2d

    block.moe_hooks = MoeFusedHooks(refine_inputs_hook=spy)
    x = _input(batch=6)  # divisible by dp*fsdp=2, not by dp*fsdp*ep=4

    with block.config.mesh:
        out_struct = jax.eval_shape(lambda xin: block(xin)[0], x)

    assert out_struct.shape == x.shape
    batch, seq, hidden = x.shape
    expected_rows = (batch // 2) * seq  # (dp, fsdp) fallback sharding
    assert set(seen) == {(expected_rows, hidden)}, (
        f"non-divisible batch must fall back to (dp, fsdp) sharding; hook saw {set(seen)}"
    )


def test_remat_targets_registered():
    """moe_dispatch/moe_combine are registered checkpoint targets consumable by policies."""
    from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS
    from easydel.infra.utils import get_gradient_checkpoint_policy

    assert "moe_dispatch" in GRADIENT_CHECKPOINT_TARGETS
    assert "moe_combine" in GRADIENT_CHECKPOINT_TARGETS
    policy = get_gradient_checkpoint_policy("save_only_these_names", save_names=["moe_dispatch", "moe_combine"])
    assert callable(policy)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])

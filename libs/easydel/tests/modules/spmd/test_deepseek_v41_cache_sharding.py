# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Slot-sharded DeepSeek-V4.1 decode cache.

On a multi-device mesh the compressed entries and index keys of :class:`SharedCompressedWindowCache` are split over
the slot (batch) axis and the shared indexer selects slot-locally. Checks: the axis-selection rule (pure Python), the
buffers really are slot-sharded with ``1/N`` rows per device, and prefill + cached decode still reproduces the
stateless forward (itself pinned to DeepSeek's reference by ``test_deepseek_v41.py``) on dp and tp meshes.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.caching.compressed_window import slot_partition_axes
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]


def _fake_mesh(**axes):
    return types.SimpleNamespace(shape=dict(axes), axis_names=tuple(axes), is_mpmd=False)


@pytest.mark.cpu_ok
def test_slot_partition_axes_rule():
    """Axes of size > 1 are taken in mesh order while their product divides the slot count; pp disables it."""
    mesh = _fake_mesh(pp=1, dp=1, fsdp=1, ep=1, tp=4, sp=1)
    assert slot_partition_axes(mesh, 8) == ("tp",)
    assert slot_partition_axes(mesh, 6) == ()  # 6 % 4 != 0: replicated
    assert slot_partition_axes(_fake_mesh(pp=1, dp=2, fsdp=1, ep=1, tp=2, sp=1), 4) == ("dp", "tp")
    assert slot_partition_axes(_fake_mesh(pp=1, dp=2, fsdp=1, ep=1, tp=2, sp=1), 2) == ("dp",)
    assert slot_partition_axes(_fake_mesh(pp=2, dp=1, fsdp=1, ep=1, tp=2, sp=1), 8) == ()
    assert slot_partition_axes(None, 8) == ()


MESHES = {"dp4": (1, 4, 1, 1, 1, 1), "tp4": (1, 1, 1, 1, 4, 1)}


@pytest.mark.parametrize("mesh_name", sorted(MESHES))
def test_slot_sharded_cached_decode_matches_stateless(mesh_name):
    if jax.device_count() != 4:
        pytest.skip("needs exactly 4 accelerator devices")
    kwargs = ref.case_config_kwargs("plain")
    kwargs["sharding_axis_dims"] = MESHES[mesh_name]
    config = DeepseekV41Config(**kwargs)
    config.moe_force_xla_gmm = True
    config.fsdp_is_ep_bound = False
    config.attach_custom_arguments()
    batch, total, prefill_len = 4, 16, 7
    ids = jnp.asarray(np.random.default_rng(3).integers(8, ref.VOCAB, size=(batch, total)), jnp.int32)

    with config.mesh:
        model = DeepseekV41ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(11),
        )
        model.eval()
        graphdef, state = spx.export(model)

        @jax.jit
        def cached_step(state, input_ids, cache):
            out = spx.bind(graphdef, state)(input_ids=input_ids, past_key_values=cache)
            return out.logits[:, -1], out.past_key_values

        stateless = np.asarray(model(input_ids=ids).logits)
        cache = model.init_cache(batch_size=batch, max_length=total + 8)
        # Executors derive compiled cache shardings from the leaves: every allocated leaf must carry a NamedSharding.
        for leaf in jax.tree_util.tree_leaves(cache):
            assert isinstance(leaf.sharding, jax.sharding.NamedSharding), (mesh_name, leaf.shape, leaf.sharding)
        expected_axes = slot_partition_axes(config.mesh, batch)
        assert expected_axes, "a 4-device mesh with 4 slots must slot-shard the cache"
        last, cache = cached_step(state, ids[:, :prefill_len], cache)
        steps = [np.asarray(last)]
        for pos in range(prefill_len, total):
            last, cache = cached_step(state, ids[:, pos : pos + 1], cache)
            steps.append(np.asarray(last))

    worst = max(float(np.abs(stateless[:, prefill_len - 1 + k] - step).max()) for k, step in enumerate(steps))
    assert worst < 2e-4, f"slot-sharded cached decode diverged from the stateless forward: max|d|={worst:.3e}"
    owned = 0
    for view in cache.views:
        assert view.metadata.slot_axes == expected_axes
        # Every other buffer must come back in its replicated allocation layout, or the next compiled step shape
        # (eSurge buckets) rejects the cache with an input-sharding mismatch.
        for buf in (view.window_kv, view.cache_position, view.compressor_kv_state, view.compressor_score_state):
            if buf is not None:
                assert buf.sharding.is_fully_replicated, (mesh_name, view.layer_index, buf.sharding)
        for buf in (view.compressed_kv, view.index_k):
            if buf is None:
                continue
            owned += 1
            spec = buf.sharding.spec
            entry = spec[0]
            names = (entry,) if isinstance(entry, str) else tuple(entry or ())
            assert names == expected_axes, (mesh_name, spec)
            assert buf.addressable_shards[0].data.shape[0] == batch // 4
    assert owned > 0

    # eSurge frees finished slots through reset_slots between compiled steps: the reset must return every leaf in
    # the layout it came in, or the next compiled step rejects the cache.
    before = [leaf.sharding for leaf in jax.tree_util.tree_leaves(cache)]
    reset = cache.reset_slots(np.asarray([0, 2], dtype=np.int32))
    after = jax.tree_util.tree_leaves(reset)
    assert len(after) == len(before)
    for sharding, leaf in zip(before, after, strict=True):
        assert leaf.sharding.is_equivalent_to(sharding, leaf.ndim), (mesh_name, leaf.shape, sharding, leaf.sharding)
    for view in reset.views:
        positions = np.asarray(view.cache_position)
        assert positions[0] == 0 and positions[2] == 0 and positions[1] == total and positions[3] == total

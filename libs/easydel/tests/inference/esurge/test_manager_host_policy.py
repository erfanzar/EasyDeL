"""Manager-owned host policy and seed transport, with NumPy placement/executable boundaries only."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners import execution_manager as execution
from easydel.inference.esurge.runners import host_sync
from easydel.inference.esurge.runners.executors import batch_preparer as preparation
from easydel.inference.esurge.runners.executors import sampler_executor as sampling
from jax.experimental import multihost_utils

from .test_grouped_pool_execution import _config, _groups, _prepare
from .test_runner_execution import _host_put

pytestmark = pytest.mark.cpu_ok


class _HostModel:
    """Complete host-only model boundary for the manager's public constructor."""

    mesh = None

    def __init__(self):
        text = SimpleNamespace(
            attn_mechanism="ragged_page_attention_v3", kv_cache_quantization_config=None, vocab_size=8
        )
        self.config = SimpleNamespace(get_text_config=lambda: text)

    @staticmethod
    def _quant_class(*, quantization_config):
        return None

    @staticmethod
    def init_operations_cache(**kwargs):
        return None

    @staticmethod
    def split_module():
        return None, None, None


def _host_boundaries(monkeypatch):
    def zeros(shape, *, device=None, out_sharding=None, **kwargs):
        return np.zeros(shape, **kwargs)

    host_jax = SimpleNamespace(
        device_put=_host_put,
        Array=jax.Array,
        random=SimpleNamespace(PRNGKey=lambda seed: np.asarray([0, seed], dtype=np.uint32)),
        jit=jax.jit,
        named_scope=jax.named_scope,
    )
    host_numpy = SimpleNamespace(zeros=zeros, int32=np.int32, uint32=np.uint32)
    monkeypatch.setattr(execution, "jax", host_jax)
    monkeypatch.setattr(execution, "jnp", host_numpy)
    monkeypatch.setattr(execution, "replicated_named_sharding", lambda mesh: None)
    # Forward execution is outside these host-policy tests. Keep the actual
    # preparer, sampler executor, and sampler runtime constructor paths.
    monkeypatch.setattr(execution, "ModelStepExecutor", lambda **kwargs: None)
    monkeypatch.setattr(preparation, "jax", host_jax)
    monkeypatch.setattr(sampling, "jax", host_jax)
    monkeypatch.setattr(sampling, "jnp", host_numpy)
    monkeypatch.setattr(sampling, "_device_put_tree_uniform", lambda values, sharding: _host_put(values))
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)

    def compiled(packed_f32, packed_i32, packed_misc, logits, key, counts):
        # Expose seed transport through a deterministic executable stand-in:
        # seeded rows return the low seed word; others return the row fold-in.
        tokens = np.where(packed_i32[9] != 0, packed_i32[7], packed_i32[0]).copy()
        return key, tokens, np.ones(tokens.shape, dtype=bool), counts

    monkeypatch.setattr(sampling.SamplerExecutor, "get_compiled", lambda *args, **kwargs: compiled)


def _construct(**kwargs):
    metadata = _config()
    return execution.ExecutionManager(
        model=_HostModel(),
        metadata=metadata,
        min_input_pad=1,
        max_num_reqs=4,
        max_num_tokens=4,
        max_model_len=8,
        cache_group_configs=(metadata, _config(page_size=2)),
        layer_to_cache_group=(0, 1),
        **kwargs,
    )


def _sample(manager, rows=(0, 1, 2, 3)):
    integers = np.ones(4, dtype=np.int32)
    floats = np.ones(4, dtype=np.float32)
    indices = np.arange(4, dtype=np.int32)
    return manager.sample_tokens(
        num_tokens=4,
        padded_num_reqs=4,
        sampler_padded_num_reqs=4,
        sampler_num_reqs=4,
        sampler_total_tokens=4,
        req_num_tokens_full_cpu=integers,
        logits=np.ones((4, 8), dtype=np.float32),
        rng_key=np.asarray([0, 0], dtype=np.uint32),
        gather_positions_cpu=indices,
        sampling_seeds_cpu=indices,
        scatter_positions_cpu=indices,
        compact_window_row_indices_cpu=np.asarray(rows, dtype=np.int32),
        compact_scheduled_cpu=integers,
        compact_seq_lens_cpu=integers,
        compact_active_mask_cpu=np.ones(4, dtype=bool),
        compact_temperature_cpu=floats,
        compact_top_p_cpu=floats,
        compact_top_k_cpu=np.zeros(4, dtype=np.int32),
        compact_min_p_cpu=floats,
        compact_frequency_penalties_cpu=np.zeros(4, dtype=np.float32),
        compact_presence_penalties_cpu=np.zeros(4, dtype=np.float32),
        compact_repetition_penalties_cpu=floats,
        need_penalties=False,
    )


def test_constructor_preserves_live_seed_mapping_through_moves_and_removals(monkeypatch):
    _host_boundaries(monkeypatch)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", lambda value: value)
    seeds = {0: 91, 2: 0, 3: -7}
    manager = _construct(request_seeds=seeds)
    assert manager.host_payloads_replicated is False
    assert manager.sampler_runtime.request_seeds is seeds
    np.testing.assert_array_equal(_sample(manager)[1], [91, 1, 0, -7])
    seeds.clear()
    seeds.update({1: -7, 2: 91, 0: 0})
    np.testing.assert_array_equal(_sample(manager, rows=(2, 0, 3, 1))[1], [91, 0, 2, -7])
    seeds.clear()
    np.testing.assert_array_equal(_sample(manager)[1], [0, 1, 2, 3])
    # Old callers omit both additions and retain unseeded, synchronized behavior.
    legacy = _construct()
    assert legacy.host_payloads_replicated is False
    assert legacy.sampler_runtime.request_seeds is None
    np.testing.assert_array_equal(_sample(legacy)[1], [0, 1, 2, 3])


@pytest.mark.parametrize("initial_policy", [False, True])
def test_late_policy_changes_prepared_tokens_and_sampling_without_affecting_other_managers(monkeypatch, initial_policy):
    _host_boundaries(monkeypatch)

    def root_host_payload(payload):
        # Simulate root-host values that differ from this host. Both preparation
        # and sampling must use root values unless this manager has its own
        # coordinator guarantee. Shapes distinguish the two existing wire formats.
        payload = list(_host_put(tuple(payload)))
        if payload[0].ndim == 1:
            payload[0].fill(77)
            # Context lengths are refreshed per step, unlike version-cached page tables.
            payload[2][1].fill(77)
        else:
            payload[1][0].fill(88)
        return tuple(payload)

    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", root_host_payload)
    manager = _construct(host_payloads_replicated=initial_policy)
    other = _construct()
    groups = _groups()

    def check(instance, replicated):
        batch, tokens, *_ = _prepare(instance, groups)
        expected = [3, 11, 19, 27] if replicated else [77] * 4
        np.testing.assert_array_equal(tokens, expected)
        for group, source in zip(batch.cache_groups, groups, strict=True):
            np.testing.assert_array_equal(group.pages_tables, source.page_table_cpu)
            np.testing.assert_array_equal(group.context_lens, [4] * 4 if replicated else [77] * 4)
        np.testing.assert_array_equal(_sample(instance)[1], [0, 1, 2, 3] if replicated else [88] * 4)
        assert instance.host_payloads_replicated is replicated
        assert instance.sampler_runtime.host_payloads_replicated is replicated

    # Warm the grouped child preparers before every late policy assignment.
    check(manager, initial_policy)
    for policy in (not initial_policy, initial_policy, not initial_policy):
        manager.host_payloads_replicated = policy
        check(manager, policy)
        check(other, False)

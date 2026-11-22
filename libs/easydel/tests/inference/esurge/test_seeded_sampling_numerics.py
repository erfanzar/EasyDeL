"""Accelerator-only seeded serving parity; intentionally not marked cpu_ok."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerExecutor, SamplerRuntime
from jax import numpy as jnp


@pytest.mark.parametrize("typed_key", [False, True])
@pytest.mark.parametrize("use_aot_forward", [False, True])
@pytest.mark.parametrize("named_sharding", [False, True])
def test_public_sampler_seeded_streams_match_independent_keys_after_compaction(
    typed_key, use_aot_forward, named_sharding
):
    count, vocab = 4, 32
    device = jax.devices()[0]
    if named_sharding:
        mesh = jax.sharding.Mesh(np.asarray([device]), ("sampler",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    else:
        sharding = jax.sharding.SingleDeviceSharding(device)
    model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=vocab)), dtype=jnp.float32
    )
    executor = SamplerExecutor(model=model, max_model_len=32, empty_sharding=sharding, use_aot_forward=use_aot_forward)
    key = jax.random.key(1) if typed_key else jax.random.PRNGKey(1)
    executor.compile(
        num_tokens=4,
        padded_num_reqs=count,
        inputs=SimpleNamespace(rng_key=key, req_num_tokens_full=np.ones(count, dtype=np.int32)),
        metadata=None,
    )
    seeds = {0: 0, 2: (1 << 40) + 3, 3: -123}
    runtime = SamplerRuntime(
        sampler_executor=executor,
        vocab_size=vocab,
        max_num_reqs=count,
        sampler_sharding=sharding,
        empty_sharding=sharding,
        request_seeds=seeds,
    )
    logits = jnp.tile(jnp.linspace(-0.25, 0.25, vocab, dtype=jnp.float32), (count, 1))
    progress = np.asarray([8, 9, 10, 11], dtype=np.int32)
    kwargs = dict(
        num_tokens=4,
        padded_num_reqs=count,
        sampler_padded_num_reqs=count,
        sampler_num_reqs=count,
        sampler_total_tokens=count,
        req_num_tokens_full_cpu=np.ones(count, dtype=np.int32),
        logits=logits,
        rng_key=key,
        gather_positions_cpu=np.arange(count, dtype=np.int32),
        sampling_seeds_cpu=np.arange(count, dtype=np.int32),
        scatter_positions_cpu=np.arange(count, dtype=np.int32),
        compact_window_row_indices_cpu=np.arange(count, dtype=np.int32),
        compact_scheduled_cpu=np.ones(count, dtype=np.int32),
        compact_seq_lens_cpu=progress,
        compact_active_mask_cpu=np.ones(count, dtype=np.bool_),
        compact_temperature_cpu=np.ones(count, dtype=np.float32),
        compact_top_p_cpu=np.ones(count, dtype=np.float32),
        compact_top_k_cpu=np.zeros(count, dtype=np.int32),
        compact_min_p_cpu=np.zeros(count, dtype=np.float32),
        compact_frequency_penalties_cpu=np.zeros(count, dtype=np.float32),
        compact_presence_penalties_cpu=np.zeros(count, dtype=np.float32),
        compact_repetition_penalties_cpu=np.ones(count, dtype=np.float32),
        need_penalties=False,
    )
    _, actual, valid, _ = runtime.sample_tokens(**kwargs)
    expected = []
    for row in range(count):
        if row in seeds:
            unsigned_seed = seeds[row] & ((1 << 64) - 1)
            request_key = jax.random.PRNGKey(np.uint32(unsigned_seed & 0xFFFFFFFF))
            request_key = jax.random.fold_in(request_key, np.uint32(unsigned_seed >> 32))
            request_key = jax.random.fold_in(request_key, progress[row])
            request_key = jax.random.fold_in(request_key, 0)
        else:
            request_key = jax.random.fold_in(key, row)
        expected.append(jax.random.categorical(request_key, logits[row]))
    np.testing.assert_array_equal(actual, np.asarray(expected))
    np.testing.assert_array_equal(valid, np.ones(count, dtype=np.bool_))

    # Reverse the logical requests and consume an unrelated engine key. Seeded
    # streams retain their original result; the unseeded row follows the engine.
    permutation = np.asarray([3, 2, 1, 0], dtype=np.int32)
    kwargs["rng_key"] = jax.random.fold_in(key, 9001)
    kwargs["compact_window_row_indices_cpu"] = permutation
    kwargs["compact_seq_lens_cpu"] = progress[permutation]
    _, replay, _, _ = runtime.sample_tokens(**kwargs)
    for local_row, original_row in enumerate(permutation):
        if int(original_row) in seeds:
            assert int(replay[local_row]) == int(actual[original_row])

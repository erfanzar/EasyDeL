"""Accelerator-only parity from SamplingParams through runner buffers to sampling."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerExecutor, SamplerRuntime
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

from .test_request_seed_wiring import sampler_args, setup_runner


@pytest.mark.parametrize("use_aot_forward", [False, True])
def test_runner_admission_seeded_streams_follow_requests_across_reordering_and_reset(use_aot_forward):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("sampler",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    key = jax.random.key(123)
    executor = SamplerExecutor(
        model=SimpleNamespace(
            config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=32)), dtype=jnp.float32
        ),
        max_model_len=16,
        empty_sharding=sharding,
        use_aot_forward=use_aot_forward,
    )
    executor.compile(
        num_tokens=4,
        padded_num_reqs=4,
        inputs=SimpleNamespace(rng_key=key, req_num_tokens_full=np.ones(4, dtype=np.int32)),
        metadata=None,
    )
    runtime = SamplerRuntime(
        sampler_executor=executor,
        vocab_size=32,
        max_num_reqs=4,
        sampler_sharding=sharding,
        empty_sharding=sharding,
    )
    runner = setup_runner(runtime, sharding)
    seeds = {"zero": 0, "wide": (1 << 40) + 3, "negative": -123, "unseeded": None}
    for name, seed in seeds.items():
        runner.sequence_buffer.begin_sequence(name, [1, 2], SamplingParams(seed=seed))
    logits = jax.device_put(jnp.tile(jnp.linspace(-0.25, 0.25, 32, dtype=jnp.float32), (4, 1)), sharding)

    def sample_and_check(engine_key, rows):
        kwargs = sampler_args(rows)
        kwargs["rng_key"] = engine_key
        kwargs["logits"] = logits
        _, actual, valid, _ = runtime.sample_tokens(**kwargs)
        expected = []
        for local_row, row in enumerate(rows):
            seed = seeds[runner.sequence_buffer.req_ids[row]]
            if seed is None:
                request_key = jax.random.fold_in(engine_key, local_row)
            else:
                unsigned = seed & ((1 << 64) - 1)
                request_key = jax.random.PRNGKey(np.uint32(unsigned & 0xFFFFFFFF))
                request_key = jax.random.fold_in(request_key, np.uint32(unsigned >> 32))
                request_key = jax.random.fold_in(request_key, np.uint32(3))
                request_key = jax.random.fold_in(request_key, np.uint32(0))
            expected.append(jax.random.categorical(request_key, logits[local_row]))
        np.testing.assert_array_equal(actual, np.asarray(expected))
        np.testing.assert_array_equal(valid, np.ones(4, dtype=bool))
        return np.asarray(actual)

    original = sample_and_check(key, [0, 1, 2, 3])
    runner.sequence_buffer.swap_states(0, 2)
    runner.sequence_buffer.swap_states(1, 3)
    reordered = sample_and_check(jax.random.fold_in(key, 9001), [3, 2, 1, 0])
    for local_row, row in enumerate([3, 2, 1, 0]):
        name = runner.sequence_buffer.req_ids[row]
        if seeds[name] is not None:
            assert reordered[local_row] == original[list(seeds).index(name)]
    runner._setup_variables()
    for name, seed in reversed(list(seeds.items())):
        runner.sequence_buffer.begin_sequence(name, [1, 2], SamplingParams(seed=seed))
    sample_and_check(jax.random.fold_in(key, 42), [0, 1, 2, 3])

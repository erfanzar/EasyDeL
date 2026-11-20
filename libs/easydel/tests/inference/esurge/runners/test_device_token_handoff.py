from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.runners.async_types import AsyncPreResults, AsyncWindowResult, DeviceInputTokenHandoff
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.execution_types import BatchMetadata
from easydel.inference.esurge.runners.model_runner import eSurgeRunner


def _metadata(input_ids: jax.Array) -> BatchMetadata:
    return BatchMetadata(
        packed_qsl_seqlens=jnp.zeros((2, 5), dtype=jnp.int32),
        packed_i32_padded=jnp.zeros((3, 4), dtype=jnp.int32),
        packed_f32_padded=jnp.zeros((6, 4), dtype=jnp.float32),
        packed_misc_i32=jnp.zeros((5,), dtype=jnp.int32),
        pages_tables=jnp.zeros((4, 2), dtype=jnp.int32),
        input_ids_buf=input_ids,
        position_ids_buf=jnp.arange(input_ids.shape[0], dtype=jnp.int32),
    )


def test_device_token_handoff_patches_flattened_input_ids_without_host_tokens():
    metadata = _metadata(jnp.array([11, 0, 22, 0, 33, 44], dtype=jnp.int32))
    handoff = DeviceInputTokenHandoff(
        input_positions=jnp.array([1, 3, 0, 0], dtype=jnp.int32),
        token_ids=jnp.array([101, 202, 0, 0], dtype=jnp.int32),
        count=jnp.array(2, dtype=jnp.int32),
    )

    patched = ExecutionManager._apply_device_token_handoff(metadata, handoff)

    np.testing.assert_array_equal(
        np.asarray(patched.model_input_ids),
        np.array([11, 101, 22, 202, 33, 44], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.input_ids_buf),
        np.array([11, 0, 22, 0, 33, 44], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(patched.input_ids_buf),
        np.array([11, 0, 22, 0, 33, 44], dtype=np.int32),
    )
    assert patched.position_ids_buf is metadata.position_ids_buf


def test_device_token_handoff_skips_prefill_rows_of_a_mixed_step():
    """A prompt's final prefill chunk ends on a real prompt token: only the decode row is patched."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    runner = object.__new__(eSurgeRunner)
    runner.async_scheduling = True
    runner.metadata = SimpleNamespace(data_parallel_size=1)
    runner.num_reqs_max_model_len = 4
    runner.sequence_buffer = SimpleNamespace(
        # Row 0 decodes at its placeholder (position 5); row 1 is a new request whose 3-token prompt is one chunk.
        num_computed_tokens=np.array([0, 5, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([3, 6, 0, 0], dtype=np.int32),
    )
    runner.executor_manager = SimpleNamespace(max_num_reqs=4, _empty_sharding=sharding)
    runner._handoff_positions_cache = {}
    runner._handoff_scalar_cache = {}
    pre_results = AsyncPreResults(
        windows=[
            AsyncWindowResult(
                req_ids=["decode"],
                row_positions=[0],
                sampled_token_ids=jnp.array([111, 0, 0, 0], dtype=jnp.int32),
                valid_mask=[True],
            )
        ],
        request_seq_lens=[],
    )

    handoff = runner._build_device_token_handoff(
        pre_results=pre_results,
        req_ids_window=["new", "decode"],
        scheduled_list=[3, 1],
        window_row_indices=np.array([0, 1], dtype=np.int32),
        num_tokens_static=8,
    )

    assert handoff is not None
    assert int(np.asarray(handoff.count)) == 1
    metadata = _metadata(jnp.array([7, 8, 9, 0, 0, 0, 0, 0], dtype=jnp.int32))
    patched = ExecutionManager._apply_device_token_handoff(metadata, handoff)
    np.testing.assert_array_equal(
        np.asarray(patched.model_input_ids),
        np.array([7, 8, 9, 111, 0, 0, 0, 0], dtype=np.int32),
    )


def test_device_token_handoff_gathers_rows_sampled_in_two_row_windows():
    """The previous step ran as two row windows: each row's token comes from its own window's sampled tokens."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    runner = object.__new__(eSurgeRunner)
    runner.async_scheduling = True
    runner.metadata = SimpleNamespace(data_parallel_size=1)
    runner.num_reqs_max_model_len = 4
    runner.sequence_buffer = SimpleNamespace(
        num_computed_tokens=np.array([5, 7, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 8, 0, 0], dtype=np.int32),
    )
    runner.executor_manager = SimpleNamespace(max_num_reqs=4, _empty_sharding=sharding)
    runner._handoff_positions_cache = {}
    runner._handoff_scalar_cache = {}
    pre_results = AsyncPreResults(
        windows=[
            AsyncWindowResult(
                req_ids=["a"],
                row_positions=[1],
                sampled_token_ids=jnp.array([0, 111], dtype=jnp.int32),
                valid_mask=[True],
            ),
            AsyncWindowResult(
                req_ids=["b"],
                row_positions=[0],
                sampled_token_ids=jnp.array([222, 0, 0, 0], dtype=jnp.int32),
                valid_mask=[True],
            ),
        ],
        request_seq_lens=[],
    )

    handoff = runner._build_device_token_handoff(
        pre_results=pre_results,
        req_ids_window=["a", "b"],
        scheduled_list=[1, 1],
        window_row_indices=np.array([0, 1], dtype=np.int32),
        num_tokens_static=8,
    )

    assert handoff is not None
    assert int(np.asarray(handoff.count)) == 2
    np.testing.assert_array_equal(np.asarray(handoff.token_ids), np.array([111, 222, 0, 0], dtype=np.int32))
    metadata = _metadata(jnp.array([0, 0, 9, 9, 9, 9, 9, 9], dtype=jnp.int32))
    patched = ExecutionManager._apply_device_token_handoff(metadata, handoff)
    np.testing.assert_array_equal(
        np.asarray(patched.model_input_ids),
        np.array([111, 222, 9, 9, 9, 9, 9, 9], dtype=np.int32),
    )


def test_device_token_handoff_uses_rank_major_spmd_offsets():
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    runner = object.__new__(eSurgeRunner)
    runner.async_scheduling = True
    runner.metadata = SimpleNamespace(data_parallel_size=2)
    runner.num_reqs_max_model_len = 4
    runner.sequence_buffer = SimpleNamespace(
        num_computed_tokens=np.array([5, 0, 0, 7], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 0, 0, 8], dtype=np.int32),
    )
    runner.executor_manager = SimpleNamespace(max_num_reqs=4, _empty_sharding=sharding)
    runner._handoff_positions_cache = {}
    runner._handoff_scalar_cache = {}

    pre_results = AsyncPreResults(
        windows=[
            AsyncWindowResult(
                req_ids=["r0", "r1"],
                row_positions=[0, 1],
                sampled_token_ids=jnp.array([111, 222, 0, 0], dtype=jnp.int32),
                valid_mask=[True, True],
            )
        ],
        request_seq_lens=[],
    )

    handoff = runner._build_device_token_handoff(
        pre_results=pre_results,
        req_ids_window=["r0", "r1"],
        scheduled_list=[1, 1],
        window_row_indices=np.array([0, 3], dtype=np.int32),
        num_tokens_static=8,
    )

    assert handoff is not None
    np.testing.assert_array_equal(np.asarray(handoff.input_positions), np.array([0, 4, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(handoff.token_ids), np.array([111, 222, 0, 0], dtype=np.int32))
    assert int(np.asarray(handoff.count)) == 2

    metadata = _metadata(jnp.array([0, 10, 10, 10, 0, 20, 20, 20], dtype=jnp.int32))
    patched = ExecutionManager._apply_device_token_handoff(metadata, handoff)

    np.testing.assert_array_equal(
        np.asarray(patched.model_input_ids),
        np.array([111, 10, 10, 10, 222, 20, 20, 20], dtype=np.int32),
    )

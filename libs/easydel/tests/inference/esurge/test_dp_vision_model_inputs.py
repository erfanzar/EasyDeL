"""Rank-major vision payloads at the shared compiled model-call boundary."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.execution_types import BatchMetadata
from easydel.inference.esurge.runners.executors import model_executor as execution
from jax import numpy as jnp


def _metadata(array_module, *, mrope, prefill):
    xp = array_module
    ids = xp.arange(1, 9, dtype=xp.int32)
    masks = xp.asarray([True, False, False, True, True, False, False, False])
    return BatchMetadata(
        packed_qsl_seqlens=xp.zeros((2, 5), dtype=xp.int32),
        packed_i32_padded=xp.zeros((3, 4), dtype=xp.int32),
        packed_f32_padded=xp.zeros((6, 4), dtype=xp.float32),
        packed_misc_i32=xp.asarray([4, 4, 0, 4, 4], dtype=xp.int32),
        pages_tables=xp.zeros((4, 2), dtype=xp.int32),
        input_ids_buf=ids,
        position_ids_buf=xp.asarray([0, 1, 10, 11, 20, 21, 0, 0], dtype=xp.int32),
        model_side_inputs={
            "token_types": xp.asarray([1, 0, 0, 2, 3, 0, 0, 0], dtype=xp.int32),
            "token_features": xp.arange(32, dtype=xp.float32).reshape(8, 2, 2),
        },
        mrope_position_ids=xp.arange(24, dtype=xp.int32).reshape(3, 8) if mrope else None,
        prefill_embeds=xp.arange(16, dtype=xp.float32).reshape(8, 2) + 100 if prefill else None,
        prefill_embeds_mask=masks if prefill else None,
        visual_pos_masks=masks,
        deepstack_visual_embeds=(xp.arange(16, dtype=xp.float32).reshape(8, 2),),
    )


class _Embedding:
    def __init__(self, array_module):
        self.xp = array_module

    def compute_embedding(self, ids):
        return self.xp.stack((ids, -ids), axis=-1).astype(self.xp.float32)


def _executor(dp_size):
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.metadata = SimpleNamespace(data_parallel_size=dp_size)
    return executor


def _assert_inputs(result, metadata, dp_size, *, mrope, prefill):
    inputs, positions, external = result
    rows_per_rank = 8 // dp_size
    ids = np.arange(1, 9, dtype=np.int32).reshape(dp_size, rows_per_rank)
    if prefill:
        expected = np.stack((ids, -ids), axis=-1).astype(np.float32)
        overrides = np.arange(16, dtype=np.float32).reshape(dp_size, rows_per_rank, 2) + 100
        mask = np.asarray(metadata.prefill_embeds_mask).reshape(dp_size, rows_per_rank)
        expected[mask] = overrides[mask]
        assert inputs["input_ids"] is None
        np.testing.assert_array_equal(inputs["inputs_embeds"], expected)
    else:
        np.testing.assert_array_equal(inputs["input_ids"], ids)
    expected_positions = (
        np.arange(24, dtype=np.int32).reshape(3, dp_size, rows_per_rank)
        if mrope
        else np.asarray([0, 1, 10, 11, 20, 21, 0, 0]).reshape(dp_size, rows_per_rank)
    )
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(
        external["token_types"], np.asarray([1, 0, 0, 2, 3, 0, 0, 0]).reshape(dp_size, rows_per_rank)
    )
    np.testing.assert_array_equal(
        external["token_features"], np.arange(32, dtype=np.float32).reshape(dp_size, rows_per_rank, 2, 2)
    )
    np.testing.assert_array_equal(
        external["visual_pos_masks"],
        np.asarray([True, False, False, True, True, False, False, False]).reshape(dp_size, rows_per_rank),
    )
    # DeepStack's established contract is flat token-major rows, not a batched
    # side input. Only its matching visual position mask acquires a batch axis.
    assert len(external["deepstack_visual_embeds"]) == 1
    np.testing.assert_array_equal(external["deepstack_visual_embeds"][0], np.arange(16).reshape(8, 2))


@pytest.mark.cpu_ok
@pytest.mark.parametrize("dp_size", [1, 4])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_rank_major_vision_call_host_layout(monkeypatch, dp_size, mrope, prefill):
    # Host-only layout/control-flow regression: no JAX arrays or compilation.
    monkeypatch.setattr(execution, "jnp", np)
    metadata = _metadata(np, mrope=mrope, prefill=prefill)
    result = _executor(dp_size)._prepare_model_call(_Embedding(np), metadata, scope="host_layout")
    _assert_inputs(result, metadata, dp_size, mrope=mrope, prefill=prefill)


@pytest.mark.parametrize("dp_size", [1, 4])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_rank_major_vision_call_compiles_without_cross_rank_broadcast(dp_size, mrope, prefill):
    """Accelerator-only parity with different payloads and padding on each rank."""
    metadata = _metadata(jnp, mrope=mrope, prefill=prefill)
    executor = _executor(dp_size)
    model = _Embedding(jnp)
    prepare = jax.jit(lambda batch: executor._prepare_model_call(model, batch, scope="compiled_layout"))
    _assert_inputs(prepare(metadata), metadata, dp_size, mrope=mrope, prefill=prefill)

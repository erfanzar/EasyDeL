"""Tiny accelerator regressions. Do not run these on a CPU backend."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.sequence_buffer import (
    SequenceBuffer,
    build_allowed_mask,
    build_sampling_arrays,
    fill_slice,
    pack_prompts,
)
from easydel.inference.esurge.runners.spec import build_target_kv_pairs
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp


def test_public_padding_helpers_keep_documented_shapes_and_defaults():
    tokens = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    prompts = pack_prompts(tokens, jnp.asarray([2, 1]), 3, 4, 9)
    np.testing.assert_array_equal(prompts, [[1, 2, 9, 9], [4, 9, 9, 9], [9, 9, 9, 9]])
    assert prompts.shape == (3, 4)
    actual = build_sampling_arrays(
        jnp.asarray([0.7, 0.8]),
        jnp.asarray([0.2, 0.3]),
        jnp.asarray([0.9, 0.95]),
        jnp.asarray([4, 5]),
        jnp.asarray(1),
        3,
    )
    for result, expected in zip(actual, [[0.7, -1, -1], [0.2, 0, 0], [0.9, 1, 1], [4, 0, 0]], strict=True):
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert [arr.dtype for arr in actual] == [jnp.float32, jnp.float32, jnp.float32, jnp.int32]
    np.testing.assert_array_equal(fill_slice(tokens, 9, 1, 3), [[1, 2, 3], [9, 9, 9], [9, 9, 9]])


def test_allowed_mask_handles_dynamic_lengths_duplicate_ids_and_padding_under_jit():
    actual = build_allowed_mask(jnp.asarray([[1, 1, -1], [2, 4, 0]]), jnp.asarray([2, 1]), 5, 3)
    np.testing.assert_array_equal(actual, [[True, False, True, True, True], [True, True, False, True, True]])


def test_public_penalty_and_allowed_token_accessors():
    buffer = SequenceBuffer(2, 8, 8, 8, [2])
    buffer.begin_sequence("a", [1], SamplingParams(temperature=0, frequency_penalty=0.5, allowed_token_ids=[1, 3]))
    buffer.begin_sequence("b", [2])
    assert not buffer.no_allowed_token_ids
    np.testing.assert_array_equal(buffer.get_request_indices_with_penalty(), [0])
    buffer.remove_request("a")
    assert buffer.no_allowed_token_ids
    assert buffer.get_request_indices_with_penalty().shape == (0,)


def test_paged_public_helper_and_live_assistant_payload_gather_the_same_request():
    # Logical positions 0, 1, 2 map to physical pages 2, 2, 0.
    keys = jnp.asarray(np.arange(8).reshape(4, 2, 1, 1))
    values = keys + 100
    table = np.asarray([[1, 3], [2, 0]], dtype=np.int32)
    view = SimpleNamespace(key_pages=keys, value_pages=values)  # Missing metadata exercises page_size fallback.
    cache = SimpleNamespace(views=[view, SimpleNamespace()])
    page_table = SimpleNamespace(get_cpu_tensor=lambda: table)
    pairs = build_target_kv_pairs(cache, [0, 1], page_tables=[page_table], batch_index=1, kv_len=3, page_size=2)
    np.testing.assert_array_equal(pairs[0][0].reshape(-1), [4, 5, 0])
    np.testing.assert_array_equal(pairs[0][1].reshape(-1), [104, 105, 100])
    assert pairs[1] is None
    runner = SimpleNamespace(
        max_model_len=4,
        page_size=2,
        kv_cache_groups=[],
        executor_manager=SimpleNamespace(kv_pages=cache),
        sequence_buffer=SimpleNamespace(req_id_to_index={"a": 1}, page_table=SimpleNamespace(page_tables=[page_table])),
    )
    spec = DrafterSpeculation(
        runner=runner,
        drafter=SimpleNamespace(requires_target_kv_cache=True, resolve_layer_mapping=lambda _: [0, 1]),
        num_draft_tokens=2,
    )
    actual, mask = spec.target_kv_payload_for_drafter(req_id="a", seed_position=2)
    np.testing.assert_array_equal(actual[0][0].reshape(-1), [4, 5, 0, 1])
    assert actual[1] is None
    np.testing.assert_array_equal(mask.reshape(-1), [0, 0, 0, -1e10])


def test_dense_assistant_payload_does_not_require_paged_cache_fields():
    keys = jnp.asarray(np.arange(8).reshape(1, 4, 1, 2))
    runner = SimpleNamespace(
        max_model_len=4,
        executor_manager=SimpleNamespace(kv_pages=SimpleNamespace(views=[SimpleNamespace(key=keys, value=keys + 10)])),
        sequence_buffer=SimpleNamespace(req_id_to_index={"a": 0}),
    )
    spec = DrafterSpeculation(runner=runner, drafter=SimpleNamespace(requires_target_kv_cache=True), num_draft_tokens=2)
    pairs, mask = spec.target_kv_payload_for_drafter(req_id="a", seed_position=1)
    np.testing.assert_array_equal(pairs[0][0], keys)
    np.testing.assert_array_equal(pairs[0][1], keys + 10)
    np.testing.assert_array_equal(mask.reshape(-1), [0, 0, -1e10, -1e10])


@pytest.mark.parametrize("persist_override,expected_rollbacks", [(None, [0]), ("0", []), ("true", [0])])
def test_explicit_adaptive_drafting_and_registered_persistence_override(
    monkeypatch, persist_override, expected_rollbacks
):
    if persist_override is None:
        monkeypatch.delenv("EASYDEL_MTP_PERSIST_KV", raising=False)
    else:
        monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", persist_override)

    class AdaptiveDrafter:
        persist_kv = True

        def __init__(self):
            self.rollbacks = []
            self.threshold = None

        def rollback_to(self, length, *, row_pos):
            self.rollbacks.append(length)

        def draft_adaptive(self, *, conf_threshold, **kwargs):
            self.threshold = conf_threshold
            return jnp.asarray([3, 4, 0]), jnp.asarray(2)

    drafter = AdaptiveDrafter()
    runner = SimpleNamespace(max_num_reqs=1)
    spec = DrafterSpeculation(runner=runner, drafter=drafter, num_draft_tokens=3, adaptive_conf_threshold=0.8)
    request = CachedRequestState("a", [1], SamplingParams(temperature=0), None, ([],), 0, [])
    actual = spec.draft_next(req_id="a", seed_token=1, seed_position=0, seed_hidden=jnp.zeros((2,)), req_state=request)
    assert actual == [3, 4]
    assert spec.num_drafts_generated == 2
    assert drafter.threshold == 0.8
    assert drafter.rollbacks == expected_rollbacks

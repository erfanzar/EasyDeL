from __future__ import annotations

import jax

from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.sampling_params import SamplingParams


def _make_request(req_id: str, token_id: int) -> CachedRequestState:
    """Build a minimal cached request state for compaction tests."""
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[token_id],
        sampling_params=SamplingParams(max_tokens=8),
        generator=jax.random.PRNGKey(token_id),
        page_ids=([token_id],),
        num_computed_tokens=1,
        output_token_ids=[],
    )


def test_sequence_buffer_condense_preserves_all_live_requests_with_multiple_holes() -> None:
    """Compaction should preserve request order while removing multiple gaps."""
    buffer = SequenceBuffer(
        max_num_reqs=8,
        max_model_len=16,
        max_num_batched_tokens=32,
        vocab_size=128,
        page_sizes=[8],
    )
    requests = [_make_request(f"req-{i}", i + 1) for i in range(5)]

    for request in requests:
        buffer.add_request(request)

    removed = [buffer.remove_request("req-1"), buffer.remove_request("req-3")]
    buffer.condense([idx for idx in removed if idx is not None])

    assert buffer.num_reqs == 3
    assert buffer.req_ids == ["req-0", "req-2", "req-4"]
    assert buffer.req_id_to_index == {"req-0": 0, "req-2": 1, "req-4": 2}

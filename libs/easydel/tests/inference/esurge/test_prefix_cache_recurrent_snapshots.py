"""Prefix-cache hits on a GDN hybrid restore the recurrent state of the cached prefix (issue #296)."""

import jax
import pytest
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

try:
    from ._common import make_tiny_qwen35
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35

PAGE_SIZE = 16
PROMPT = [((7 * i + 3) % 250) + 2 for i in range(53)]
OTHER = [((11 * i + 5) % 250) + 2 for i in range(37)]


@pytest.fixture(scope="module")
def model():
    return make_tiny_qwen35(mtp_layers=0, precision=jax.lax.Precision.HIGHEST)


def _runner(model, snapshots: int) -> eSurgeRunner:
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=PAGE_SIZE,
        max_cache_tokens=2048,
        max_model_len=128,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=4,
        max_num_seq_buckets=[4],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        recurrent_prefix_snapshots=snapshots,
    )
    runner.compile(max_num_batched_tokens=64)
    return runner


def _generate(runner, scheduler, prompts: dict[str, list[int]]) -> list[EngineRequest]:
    requests = [
        EngineRequest(
            request_id=name,
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        for name, prompt in prompts.items()
    ]
    for request in requests:
        scheduler.add_request(request)
    for _ in range(400):
        output = scheduler.schedule()
        scheduler.update_from_output(output, runner.execute_model(output))
        if all(request.is_finished() for request in requests):
            return requests
    raise AssertionError("generation did not finish")


def _run(model, *, enable_prefix_caching: bool, snapshots: int):
    runner = _runner(model, snapshots)
    scheduler = Scheduler.from_runner(
        runner, max_num_batched_tokens=64, enable_prefix_caching=enable_prefix_caching, async_scheduling=False
    )
    finished = _generate(runner, scheduler, {"seq-0": PROMPT})
    finished += _generate(runner, scheduler, {"seq-1": PROMPT})
    finished += _generate(runner, scheduler, {"batch-0": PROMPT, "batch-1": PROMPT, "batch-2": PROMPT, "other": OTHER})
    return (
        scheduler.enable_prefix_caching,
        {request.request_id: list(request.output_token_ids) for request in finished},
        {request.request_id: int(request.num_cached_tokens) for request in finished},
    )


def test_prefix_hits_on_recurrent_hybrid_match_uncached_decoding(model):
    _, reference, _ = _run(model, enable_prefix_caching=False, snapshots=0)
    effective, outputs, cached = _run(model, enable_prefix_caching=True, snapshots=16)

    assert effective is True
    last_boundary = (len(PROMPT) - 1) // PAGE_SIZE * PAGE_SIZE
    assert cached["seq-0"] == 0
    assert all(cached[name] == last_boundary for name in ("seq-1", "batch-0", "batch-1", "batch-2"))
    assert cached["other"] == 0
    assert outputs == reference


def test_zero_snapshot_slots_turn_prefix_caching_off_for_recurrent_hybrid(model):
    _, reference, _ = _run(model, enable_prefix_caching=False, snapshots=0)
    effective, outputs, cached = _run(model, enable_prefix_caching=True, snapshots=0)

    assert effective is False
    assert set(cached.values()) == {0}
    assert outputs == reference

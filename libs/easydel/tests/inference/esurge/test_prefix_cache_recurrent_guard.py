"""Repeated prompts on a GDN hybrid must decode identically whether or not prefix caching was requested."""

import jax
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

try:
    from ._common import make_tiny_qwen35
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35

PROMPT = [((7 * i + 3) % 250) + 2 for i in range(53)]


def _runner(model) -> eSurgeRunner:
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=16,
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
    )
    runner.compile(max_num_batched_tokens=64)
    return runner


def _generate(runner, scheduler, names: list[str]) -> list[EngineRequest]:
    requests = [
        EngineRequest(
            request_id=name,
            prompt_token_ids=list(PROMPT),
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        for name in names
    ]
    for request in requests:
        scheduler.add_request(request)
    for _ in range(400):
        output = scheduler.schedule()
        scheduler.update_from_output(output, runner.execute_model(output))
        if all(request.is_finished() for request in requests):
            return requests
    raise AssertionError("generation did not finish")


def _outputs(enable_prefix_caching: bool, model) -> tuple[bool, list[list[int]], list[int]]:
    runner = _runner(model)
    scheduler = Scheduler.from_runner(
        runner, max_num_batched_tokens=64, enable_prefix_caching=enable_prefix_caching, async_scheduling=False
    )
    finished = _generate(runner, scheduler, ["seq-0"])
    finished += _generate(runner, scheduler, ["seq-1"])
    finished += _generate(runner, scheduler, ["batch-0", "batch-1", "batch-2"])
    return (
        scheduler.enable_prefix_caching,
        [list(request.output_token_ids) for request in finished],
        [int(request.num_cached_tokens) for request in finished],
    )


def test_repeated_prompts_on_recurrent_hybrid_ignore_requested_prefix_caching():
    model = make_tiny_qwen35(mtp_layers=0, precision=jax.lax.Precision.HIGHEST)

    _, reference, _ = _outputs(False, model)
    effective, outputs, cached = _outputs(True, model)

    assert effective is False
    assert cached == [0] * len(cached)
    assert all(tokens == reference[0] for tokens in reference)
    assert outputs == reference

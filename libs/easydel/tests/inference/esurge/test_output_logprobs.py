"""Host-only output transport, token alignment and streaming regressions."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import wire
from easydel.inference.esurge.engine import output_pipeline
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.esurge.outputs import LogprobsLists
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def pipeline_factory(monkeypatch):
    monkeypatch.setattr(output_pipeline, "get_metrics_collector", lambda: None)

    def build(*, n=1, stop=None, interval=1):
        registry = RequestRegistry()
        output = RequestOutput(
            request_id="parent",
            prompt="prompt",
            prompt_token_ids=[1],
            outputs=[CompletionOutput(index=i, text="", token_ids=[]) for i in range(n)],
        )
        registry.outputs["parent"] = output
        for index in range(n):
            request_id = "parent" if n == 1 else f"parent-{index}"
            registry.register(
                request_id,
                RequestRecord(
                    prompt_token_ids=[1],
                    sampling_params=SamplingParams(max_tokens=32, stop=stop),
                    sample_index=index,
                    parent_request_id="parent" if n > 1 else None,
                ),
                output,
            )
        # Real multi-sample admission retains a parent wake event too.
        registry.events.setdefault("parent", registry.events[request_id])
        stop_matches = []
        pipeline = OutputPipeline(
            registry=registry,
            detokenizer_client=SimpleNamespace(
                decode=lambda request_id, tokens, **kwargs: SimpleNamespace(
                    accumulated_text="".join(chr(t) for t in tokens),
                    delta_text="".join(chr(t) for t in tokens),
                    last_decoded_index=len(tokens),
                ),
                reset=lambda request_id: None,
            ),
            eos_token_ids=[0],
            decode_interval_tokens=interval,
            decode_interval_secs=1000,
            on_stop_strings=stop_matches.append,
            on_activity=lambda: None,
            on_fatal=lambda *args: pytest.fail("Unexpected output worker failure"),
        )
        return pipeline, registry, output, stop_matches

    return build


def submit(pipeline, *outputs, finished_requests=None):
    # Exercise the actual distributed payload codec before the local pipeline.
    bundle = {0: EngineCoreOutputs(outputs=list(outputs), timestamp=1.0, finished_requests=finished_requests)}
    pipeline.submit(wire.decode_payload(wire.encode_payload(bundle)))


def probabilities(tokens, values):
    # The selected id is deliberately NOT the first candidate or the top rank.
    return LogprobsLists([[999, token] for token in tokens], [[-0.01, value] for value in values], [2] * len(tokens))


def test_multisample_transport_accumulates_probabilities_through_eos_and_empty_finalization(pipeline_factory):
    pipeline, _, output, _ = pipeline_factory(n=2, interval=100)
    submit(
        pipeline,
        EngineCoreOutput("parent-1", [67], new_logprobs=probabilities([67], [-0.7])),
        EngineCoreOutput("parent-0", [65, 66], new_logprobs=probabilities([65, 66], [-0.1, -0.2])),
    )
    # Decode gating cannot hide tokens while probabilities have advanced.
    assert output.outputs[0].text == ""
    assert output.outputs[0].token_ids == [65, 66]
    assert output.outputs[0].logprobs == [{999: -0.01, 65: -0.1}, {999: -0.01, 66: -0.2}]
    submit(
        pipeline,
        EngineCoreOutput("parent-1", [0], new_logprobs=probabilities([0], [-0.8]), finish_reason=FinishReason.STOP),
    )
    assert not output.finished
    assert output.outputs[1].text == "C"
    assert output.outputs[1].token_ids == [67, 0]
    assert output.outputs[1].cumulative_logprob == pytest.approx(-1.5)
    submit(pipeline, EngineCoreOutput("parent-0", [], finish_reason=FinishReason.LENGTH))
    assert output.finished
    assert output.outputs[0].text == "AB"
    assert output.outputs[0].cumulative_logprob == pytest.approx(-0.3)
    for completion, expected in zip(output.outputs, [[-0.1, -0.2], [-0.7, -0.8]], strict=True):
        # This is the exact public payload consumed by trainer rollout reconstruction.
        assert [row[token] for token, row in zip(completion.token_ids, completion.logprobs, strict=True)] == expected


def test_eos_only_completion_retains_token_and_probability(pipeline_factory):
    pipeline, _, output, _ = pipeline_factory()
    submit(
        pipeline,
        EngineCoreOutput("parent", [0], new_logprobs=probabilities([0], [-0.4]), finish_reason=FinishReason.STOP),
    )
    assert output.finished
    assert output.outputs[0].text == ""
    assert output.outputs[0].token_ids == [0]
    assert output.outputs[0].logprobs == [{999: -0.01, 0: -0.4}]
    assert output.outputs[0].cumulative_logprob == -0.4


def test_stop_trimming_does_not_shift_sampled_token_maps(pipeline_factory):
    pipeline, _, output, stops = pipeline_factory(stop=["!"])
    tokens = [65, 66, 33, 67]
    submit(pipeline, EngineCoreOutput("parent", tokens, new_logprobs=probabilities(tokens, [-0.1, -0.2, -0.3, -0.4])))
    completion = output.outputs[0]
    assert output.finished
    assert completion.finish_reason == "stop"
    assert completion.text == "AB"
    assert completion.token_ids == tokens
    assert [row[token] for token, row in zip(tokens, completion.logprobs, strict=True)] == [-0.1, -0.2, -0.3, -0.4]
    assert completion.cumulative_logprob == pytest.approx(-1.0)
    assert stops == [{"parent": "!"}]


def test_streaming_snapshots_keep_token_probability_pairs_stable(pipeline_factory):
    pipeline, registry, output, _ = pipeline_factory()
    engine = eSurge.__new__(eSurge)
    engine._request_lock = registry.request_lock
    engine._output_lock = registry.output_lock
    engine._request_events = registry.events
    engine._request_outputs = registry.outputs
    engine._max_request_outputs = None
    engine._scheduler_running = True
    engine._raise_if_scheduler_failed = lambda: None
    engine._recover_orphaned_request = lambda request_id: False
    stream = engine._iter_request_snapshots("parent", stopped_message="stopped")
    submit(pipeline, EngineCoreOutput("parent", [65], new_logprobs=probabilities([65], [-0.2])))
    first = next(stream)
    submit(pipeline, EngineCoreOutput("parent", [66], new_logprobs=probabilities([66], [-0.4])))
    second = next(stream)
    assert first.outputs[0].token_ids == [65]
    assert first.outputs[0].logprobs == [{999: -0.01, 65: -0.2}]
    assert first.outputs[0].cumulative_logprob == -0.2
    assert second.outputs[0].token_ids == [65, 66]
    assert second.outputs[0].cumulative_logprob == pytest.approx(-0.6)
    submit(pipeline, EngineCoreOutput("parent", [], finish_reason=FinishReason.STOP))
    final = next(stream)
    with pytest.raises(StopIteration):
        next(stream)
    assert final.finished and output.finished
    assert len(final.outputs[0].token_ids) == len(final.outputs[0].logprobs) == 2


def test_absent_probabilities_stay_absent_through_scheduler_only_finish(pipeline_factory):
    pipeline, _, output, _ = pipeline_factory(interval=100)
    submit(pipeline, EngineCoreOutput("parent", [65]))
    submit(pipeline, finished_requests={"parent"})
    assert output.finished
    assert output.outputs[0].token_ids == [65]
    assert output.outputs[0].logprobs is None
    assert output.outputs[0].cumulative_logprob is None


@pytest.mark.parametrize(
    "malformed",
    [
        LogprobsLists([], [], []),
        LogprobsLists([[65]], [], [1]),
        LogprobsLists([[65, 66]], [[-0.2]], [1]),
        LogprobsLists([[66]], [[-0.2]], [1]),
    ],
)
def test_malformed_probability_rows_fail_before_token_state_changes(pipeline_factory, malformed):
    pipeline, registry, output, _ = pipeline_factory()
    with pytest.raises(ValueError, match=r"[Ee]ngine log probabilit"):
        submit(pipeline, EngineCoreOutput("parent", [65], new_logprobs=malformed))
    assert registry.records["parent"].generated_tokens == []
    assert output.outputs[0].token_ids == []
    assert output.outputs[0].logprobs is None


@pytest.mark.parametrize("first_has_probabilities", [True, False])
def test_partially_missing_probabilities_are_not_silently_misaligned(pipeline_factory, first_has_probabilities):
    pipeline, _, output, _ = pipeline_factory()
    submit(
        pipeline,
        EngineCoreOutput("parent", [65], new_logprobs=probabilities([65], [-0.2]) if first_has_probabilities else None),
    )
    with pytest.raises(ValueError, match="part of a completion"):
        submit(
            pipeline,
            EngineCoreOutput(
                "parent", [66], new_logprobs=None if first_has_probabilities else probabilities([66], [-0.3])
            ),
        )
    assert output.outputs[0].token_ids == [65]

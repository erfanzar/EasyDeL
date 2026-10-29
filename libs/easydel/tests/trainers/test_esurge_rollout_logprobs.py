"""Host-only rollout payload and tokenizer contracts; no JAX device execution."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.trainers import base_trainer

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def reconstruct(monkeypatch):
    # Only replace the final array materialization; exercise the real token,
    # candidate-lookup, prompt-order and padding code using host data.
    monkeypatch.setattr(base_trainer, "jnp", SimpleNamespace(array=np.array))

    def convert(outputs):
        return base_trainer.BaseTrainer._esurge_outputs_to_arrays(
            object(),
            outputs,
            input_ids=[[0, 10], [0, 20]],
            attention_mask=[[0, 1], [0, 1]],
            return_prompts=["first", "second"],
            pad_token_id=0,
            max_seq_len=2,
            max_new_tokens=3,
        )

    return convert


def request(prompt_token, completions):
    return RequestOutput(
        request_id=str(prompt_token),
        prompt="first" if prompt_token == 10 else "second",
        prompt_token_ids=[[prompt_token]],
        outputs=completions,
        finished=True,
    )


def test_esurge_sampled_logprobs_follow_prompt_reorder_padding_and_multiple_completions(reconstruct):
    # Inputs arrive in engine completion order, not caller prompt order. Candidate
    # maps deliberately put a non-sampled token first; score lookup must use ids.
    second = CompletionOutput(0, "second answer", [8], logprobs=[{99: -9.0, 8: -0.5}])
    first = CompletionOutput(0, "first answer", [3, 2], logprobs=[{99: -9.0, 3: -0.25}, {2: -1.0}])
    alternative = CompletionOutput(
        1,
        "alternative",
        [4, 5, 6, 7],
        logprobs=[{4: -0.5}, {5: -1.0}, {6: -1.5}, {7: -2.0}],
    )
    arrays = reconstruct([request(20, [second]), request(10, [first, alternative])])
    assert arrays.completion_ids.tolist() == [[3, 2, 0], [4, 5, 6], [8, 0, 0]]
    assert arrays.completion_mask.tolist() == [[1, 1, 0], [1, 1, 1], [1, 0, 0]]
    assert arrays.sampling_per_token_logps.tolist() == [[-0.25, -1.0, 0.0], [-0.5, -1.0, -1.5], [-0.5, 0.0, 0.0]]
    assert arrays.sampling_per_token_logps.dtype == np.float32
    assert arrays.output_records == ["first answer", "alternative", "second answer"]
    assert arrays.completion_prompts == ["first", "first", "second"]
    # Reconstruction must not mutate public engine outputs or truncate their payload.
    assert alternative.token_ids == [4, 5, 6, 7]
    assert alternative.logprobs[-1] == {7: -2.0}


@pytest.mark.parametrize(
    "logprobs,error",
    [([], "fewer sampled-token"), ([{99: -1.0}], "omit sampled token 3")],
)
def test_esurge_malformed_sampled_logprobs_are_rejected(reconstruct, logprobs, error):
    completion = CompletionOutput(0, "answer", [3], logprobs=logprobs)
    with pytest.raises(ValueError, match=error):
        reconstruct([request(10, [completion])])


def test_esurge_partial_sampled_logprob_payload_is_not_silently_misaligned(reconstruct):
    first = CompletionOutput(0, "first answer", [3], logprobs=[{3: -0.5}])
    second = CompletionOutput(0, "second answer", [8])
    with pytest.raises(ValueError, match="only some completions"):
        reconstruct([request(10, [first]), request(20, [second])])


def test_esurge_absent_sampled_logprobs_remain_backward_compatible(reconstruct):
    arrays = reconstruct([request(10, [CompletionOutput(0, "answer", [3])])])
    assert arrays.sampling_per_token_logps is None
    assert arrays.completion_ids.tolist() == [[3, 0, 0]]


def test_async_runner_scores_reach_rollout_arrays_without_injected_completion_logprobs(reconstruct):
    from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
    from easydel.inference.esurge.outputs import ModelRunnerOutput
    from easydel.inference.esurge.runners.async_types import AsyncWindowResult
    from easydel.inference.esurge.runners.model_runner import _AsyncExecutionHandle, _materialize_logprobs

    # Supply host equivalents of the runner's in-flight window arrays, not a
    # manufactured completion.logprobs payload. Sparse row positions distinguish
    # runner layout from request/output order. No device or model is initialized.
    output = ModelRunnerOutput(
        req_ids=["10", "20"],
        req_id_to_index={"10": 0, "20": 1},
        sampled_token_ids=[],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    window = AsyncWindowResult(
        req_ids=["10", "20"],
        row_positions=[2, 0],
        sampled_token_ids=np.asarray([8, 99, 3]),
        valid_mask=[True, True],
        token_logprobs=np.asarray([-0.5, -99.0, -0.25], dtype=np.float32),
    )
    # execute_model wires this probability materializer independently of
    # sampler metrics. Match that caller: an unwired handle would not turn
    # metric-only scores into sampling scores.
    payloads = {
        req_id: (
            window.sampled_token_ids[[row], None],
            window.token_logprobs[[row], None],
            np.asarray([1]),
        )
        for req_id, row in zip(window.req_ids, window.row_positions, strict=True)
    }
    handle = _AsyncExecutionHandle(
        output,
        [window],
        resolve_logprobs=partial(
            _materialize_logprobs, output.req_ids, payloads=payloads, requested=set(output.req_ids)
        ),
    )
    resolved = handle.get_output()
    assert resolved.sampled_token_ids == [[3], [8]]
    assert resolved.token_logprobs == {"10": -0.25, "20": -0.5}
    assert handle.get_output() is resolved

    requests = []
    offset = 0
    for prompt_token, tokens in zip([10, 20], resolved.sampled_token_ids, strict=True):
        completion = CompletionOutput(0, "answer", tokens)
        # Use the exact field consumed by the scheduler, sliced by emitted-token
        # offset, and the real completion accumulator. No score is injected here.
        logprobs = None if resolved.logprobs is None else resolved.logprobs.slice(offset, offset + len(tokens))
        OutputPipeline._append_token_logprobs(completion, tokens, logprobs, previous_count=0)
        requests.append(request(prompt_token, [completion]))
        offset += len(tokens)
    arrays = reconstruct(requests)
    assert arrays.sampling_per_token_logps is not None
    assert arrays.sampling_per_token_logps.tolist() == [[-0.25, 0.0, 0.0], [-0.5, 0.0, 0.0]]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("pad_token,expected_pad", [(None, "new-eos"), ("new-pad", "new-pad")])
def test_prm_shared_tokenizer_overrides_preserve_processor_and_pad_fallback(
    monkeypatch, wrapped, pad_token, expected_pad
):
    from easydel.trainers.prm_trainer import prm_trainer

    class HostState:
        pass

    # Bypass only parent training setup, which would allocate model/device state.
    monkeypatch.setattr(prm_trainer, "EasyDeLState", HostState)
    monkeypatch.setattr(prm_trainer.Trainer, "__init__", lambda self, **kwargs: None)
    tokenizer = SimpleNamespace(eos_token="old-eos", pad_token=None, chat_template="keep-template")
    processor = SimpleNamespace(tokenizer=tokenizer) if wrapped else tokenizer
    arguments = prm_trainer.PRMConfig(disable_dropout=False, eos_token="new-eos", pad_token=pad_token)
    trainer = prm_trainer.PRMTrainer(arguments, model=HostState(), processing_class=processor)
    assert trainer.processing_class is processor
    assert tokenizer.eos_token == "new-eos"
    assert tokenizer.pad_token == expected_pad
    assert tokenizer.chat_template == "keep-template"

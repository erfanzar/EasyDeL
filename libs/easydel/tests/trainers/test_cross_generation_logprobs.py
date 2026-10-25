"""Host-only parsing/alignment tests; the token-array adapter uses NumPy, never JAX numerics."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference import SamplingParams
from easydel.trainers import base_trainer as module
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


def completion(tokens, logprobs):
    return SimpleNamespace(token_ids=tokens, logprobs=logprobs, text="visible", finish_reason="stop")


def request(prompt, completions):
    return SimpleNamespace(prompt=prompt, prompt_token_ids=[[prompt]], outputs=completions)


def reconstruct(monkeypatch, outputs, **kwargs):
    # Only the host-to-device conversion at the end is replaced: parsing,
    # sampled-token selection, padding/truncation, and sorting are production code.
    monkeypatch.setattr(module, "jnp", np)
    return object.__new__(Trainer)._esurge_outputs_to_arrays(
        outputs,
        input_ids=kwargs.pop("input_ids", None),
        attention_mask=None,
        return_prompts=[10, 20],
        pad_token_id=0,
        max_seq_len=2,
        max_new_tokens=3,
        **kwargs,
    )


def test_logprobs_follow_sampled_token_not_top_candidate_and_align_n_sampling(monkeypatch):
    outputs = [
        request(
            20,
            [
                completion([7], [{1: -0.1, 7: -2.0}]),
                completion([8, 9, 10, 11], [{8: -3.0}, {9: -4.0}, {10: -5.0}, {11: -6.0}]),
            ],
        ),
        request(10, [completion([5, 6], [{5: -0.5}, {6: SimpleNamespace(logprob=-0.75)}]), completion([], [])]),
    ]
    arrays = reconstruct(monkeypatch, outputs, input_ids=np.array([[0, 10], [0, 20]]))
    assert arrays.completion_ids.tolist() == [[5, 6, 0], [0, 0, 0], [7, 0, 0], [8, 9, 10]]
    assert arrays.completion_mask.tolist() == [[1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1]]
    assert arrays.sampling_per_token_logps.dtype == np.float32
    assert arrays.sampling_per_token_logps.tolist() == [
        [-0.5, -0.75, 0.0],
        [0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-3.0, -4.0, -5.0],
    ]
    assert arrays.completion_prompts == [10, 10, 20, 20]


def test_legacy_absent_logprobs_keep_optional_payload_none(monkeypatch):
    arrays = reconstruct(monkeypatch, [request(10, [completion([5], None)])])
    assert arrays.sampling_per_token_logps is None
    legacy = GenerationResults("text", None, None, None, None, None, "prompt")
    assert legacy.sampling_per_token_logps is None


@pytest.mark.parametrize("bad_payload", [[{7: -1.0}], []])
def test_incomplete_sampled_token_payload_is_rejected(monkeypatch, bad_payload):
    with pytest.raises(ValueError, match="eSurge"):
        reconstruct(monkeypatch, [request(10, [completion([5], bad_payload)])])


def test_mixed_present_and_missing_payload_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="only some completions"):
        reconstruct(monkeypatch, [request(10, [completion([5], [{5: -1.0}]), completion([6], None)])])


def test_backend_requests_and_returns_logprobs_without_mutating_params(monkeypatch):
    monkeypatch.setattr(module, "jnp", np)
    trainer = object.__new__(Trainer)
    trainer.arguments = SimpleNamespace(max_length=2, esurge_use_tqdm=False)
    trainer._normalize_esurge_prompts = lambda prompts, apply_chat_template: prompts
    trainer._esurge_init_kwargs = lambda: {"reserve_tokens": 0}
    trainer.maybe_release_generation_runtime = lambda results, **kwargs: results
    received = []
    gathered = []

    def call_engine(engine, **kwargs):
        received.append(kwargs["sampling_params"])
        return [request(10, [completion([5], [{5: -0.5}])])]

    def gather(value):
        gathered.append(value)
        return value

    trainer._all_gather = gather
    state = SimpleNamespace(model=SimpleNamespace(get_esurge=lambda **kwargs: object(), _call_esurge_engine=call_engine))
    params = SamplingParams(max_tokens=3)
    for _ in range(2):  # Initial and follow-up tool rollouts use the same entry point.
        result = trainer._generate_via_esurge(
            input_ids=np.array([[0, 10]]),
            attention_mask=np.array([[0, 1]]),
            prompts=[10],
            processor=None,
            pad_token_id=0,
            sampling_params=params,
            state=state,
            apply_chat_template=False,
            all_gather=True,
            release_runtime_after_generation=False,
            clear_esurge_compiled_cache_after_generation=False,
        )
        assert result.sampling_per_token_logps.tolist() == [[-0.5, 0.0, 0.0]]
    assert params.logprobs is None
    assert all(sampling.logprobs == 0 and sampling is not params for sampling in received)
    assert len(gathered) == 12  # Includes sampled logprobs, aligned with the five token/mask arrays.

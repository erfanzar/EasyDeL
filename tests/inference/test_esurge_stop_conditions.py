from easydel.inference.esurge.mixins.parsing import EngineParsingMixin
from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.utils import check_stop
from easydel.inference.sampling_params import SamplingParams


class _StopPolicyHarness(EngineParsingMixin, EngineUtilsMixin):
    pass


def test_check_stop_with_custom_stop_token_id():
    sampling_params = SamplingParams(max_tokens=16, stop_token_ids=[42], ignore_eos=True)
    request = EngineRequest(
        request_id="req-stop-token",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        eos_token_id=0,
    )
    request.status = EngineRequestStatus.RUNNING
    request.append_output_token_ids(42)

    assert check_stop(request, max_model_len=4096) is True
    assert request.status == EngineRequestStatus.FINISHED_STOPPED
    assert request.stop_reason == 42


def test_check_stop_ignores_eos_when_ignore_eos_true():
    sampling_params = SamplingParams(max_tokens=16, stop_token_ids=[2], ignore_eos=True)
    request = EngineRequest(
        request_id="req-ignore-eos",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        eos_token_id=2,
    )
    request.status = EngineRequestStatus.RUNNING
    request.append_output_token_ids(2)

    assert check_stop(request, max_model_len=4096) is False
    assert request.status == EngineRequestStatus.RUNNING


def test_stop_string_policy_trims_on_match():
    harness = _StopPolicyHarness()
    sampling_params = SamplingParams(max_tokens=32, stop=["<user>"])
    rd = {"sampling_params": sampling_params, "decoder_visible_text": "Hello "}

    visible_text, visible_delta, stop_triggered, stop_reason = harness._apply_stop_string_policy(
        rd,
        accumulated_text="Hello world<user>ignored",
        fallback_delta="world<user>ignored",
    )

    assert stop_triggered is True
    assert stop_reason == "<user>"
    assert visible_text == "Hello world"
    assert visible_delta == "world"


def test_stop_string_policy_passes_through_without_match():
    harness = _StopPolicyHarness()
    sampling_params = SamplingParams(max_tokens=32, stop=["abcd"])
    rd = {"sampling_params": sampling_params, "decoder_visible_text": ""}

    visible_text, visible_delta, stop_triggered, stop_reason = harness._apply_stop_string_policy(
        rd,
        accumulated_text="abcx",
        fallback_delta="abcx",
    )

    assert stop_triggered is False
    assert stop_reason is None
    assert visible_text == "abcx"
    assert visible_delta == "abcx"


def test_stop_string_policy_can_include_stop_string_when_requested():
    harness = _StopPolicyHarness()
    sampling_params = SamplingParams(max_tokens=32, stop=["<user>"], include_stop_str_in_output=True)
    rd = {"sampling_params": sampling_params, "decoder_visible_text": ""}

    visible_text, visible_delta, stop_triggered, stop_reason = harness._apply_stop_string_policy(
        rd,
        accumulated_text="ans<user>tail",
        fallback_delta="ans<user>tail",
    )

    assert stop_triggered is True
    assert stop_reason == "<user>"
    assert visible_text == "ans<user>"
    assert visible_delta == "ans<user>"

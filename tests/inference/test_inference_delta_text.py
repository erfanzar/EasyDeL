from easydel.inference.inference_engine_interface import BaseInferenceApiServer


def test_compute_delta_text_uses_accumulated_growth():
    delta = BaseInferenceApiServer._compute_delta_text("hello world", "hello ", "fallback")
    assert delta == "world"


def test_compute_delta_text_does_not_replay_fallback_when_text_unchanged():
    delta = BaseInferenceApiServer._compute_delta_text("same text", "same text", "same text")
    assert delta == ""


def test_compute_delta_text_keeps_fallback_for_mismatch():
    delta = BaseInferenceApiServer._compute_delta_text("new branch", "old branch", "fallback")
    assert delta == "fallback"

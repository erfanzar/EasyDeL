from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.sampling_params import SamplingParams


def _make_server(extra_stops):
    server = object.__new__(eSurgeApiServer)
    server._extra_stops = eSurgeApiServer._normalize_stop_sequences(extra_stops)
    return server


def test_normalize_stop_sequences_handles_common_inputs():
    assert eSurgeApiServer._normalize_stop_sequences(None) == []
    assert eSurgeApiServer._normalize_stop_sequences("<user>") == ["<user>"]
    assert eSurgeApiServer._normalize_stop_sequences(["", "<user>", "<user>", 42, None]) == ["<user>", "42"]


def test_apply_extra_stops_appends_and_deduplicates():
    server = _make_server(["<user>", "</assistant>"])
    sampling_params = SamplingParams(max_tokens=32, stop=["</assistant>", "DONE"])

    updated = server._apply_extra_stops_to_sampling_params(sampling_params)

    assert updated.stop == ["</assistant>", "DONE", "<user>"]


def test_apply_extra_stops_populates_empty_stop_list():
    server = _make_server("<user>")
    sampling_params = SamplingParams(max_tokens=16)

    updated = server._apply_extra_stops_to_sampling_params(sampling_params)

    assert updated.stop == ["<user>"]

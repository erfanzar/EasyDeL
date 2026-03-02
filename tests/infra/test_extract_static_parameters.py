import inspect

from easydel.infra.utils import extract_static_parameters


def test_extract_static_parameters_handles_wrapped_cycle():
    def call(self, hidden_states, mode, frequencies=None, output_attentions=False):
        return hidden_states

    # Simulate a broken decorator chain that inspect.unwrap() cannot traverse.
    call.__wrapped__ = call

    class DummyModule:
        __call__ = call

    try:
        inspect.signature(DummyModule.__call__)
    except ValueError as err:
        assert "wrapper loop when unwrapping" in str(err)

    assert extract_static_parameters(DummyModule) == (2, 3, 4)

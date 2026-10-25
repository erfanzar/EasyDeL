"""Pure-Python LM-head adapter contracts; no parameter arrays or numerical execution."""

from types import SimpleNamespace

import pytest
from easydel.infra.base_module import EasyDeLBaseModule


class _Head:
    """Report projection inputs without constructing or computing arrays."""

    def __call__(self, hidden_states, *, w=None):
        return {"hidden_states": hidden_states, "weight": w}

    native_forward = __call__


class _Transposed(str):
    """Stand-in transposed embedding: compares as its label and exposes a kernel shape."""

    shape = (4, 8)


class _ProjectionModel:
    _tie_embeddings_enabled = EasyDeLBaseModule._tie_embeddings_enabled
    _lm_head_kernel_spec = EasyDeLBaseModule._lm_head_kernel_spec
    _tied_lm_head_weight = EasyDeLBaseModule._tied_lm_head_weight
    apply_lm_head = EasyDeLBaseModule.apply_lm_head
    make_lm_head_fn = EasyDeLBaseModule.make_lm_head_fn

    def __init__(self, config, feature_tie, native):
        self.config = SimpleNamespace(**config)
        if feature_tie is not None:
            self._tie_embeddings_feature = SimpleNamespace(tie=feature_tie)
        self.head = _Head()
        if not native:
            self.head.native_forward = None

    def get_lm_head(self):
        return self.head

    def get_embedding(self):
        return SimpleNamespace(weight=SimpleNamespace(value=SimpleNamespace(T=_Transposed("embedding-transpose"))))


@pytest.mark.cpu_ok
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "config,feature_tie,expected_weight",
    [
        ({"tie_word_embeddings": False}, True, "embedding-transpose"),
        ({"tie_word_embeddings": True}, False, "embedding-transpose"),
        ({"tie_word_embeddings": False}, False, None),
        ({"tie_word_embeddings": False, "use_lm_head": True}, None, None),
        ({"use_lm_head": True}, None, "embedding-transpose"),
        ({"share_input_output_layers": True}, None, "embedding-transpose"),
        ({}, None, None),
    ],
)
def test_projection_adapters_preserve_constructor_tying_and_legacy_config_precedence(
    config, feature_tie, expected_weight, native
):
    model = _ProjectionModel(config, feature_tie, native)
    expected = {"hidden_states": "hidden-states", "weight": expected_weight}
    assert model.apply_lm_head("hidden-states") == expected
    assert model.make_lm_head_fn()("hidden-states") == expected


@pytest.mark.cpu_ok
def test_trace_safe_projection_uses_native_head_without_entering_remat_wrapper():
    model = _ProjectionModel({"tie_word_embeddings": False}, True, native=True)

    class _RematHead(_Head):
        def __call__(self, *args, **kwargs):
            pytest.fail("trace-safe projection must not enter the mutable head wrapper")

    model.head = _RematHead()
    assert model.make_lm_head_fn()("hidden-states") == {
        "hidden_states": "hidden-states",
        "weight": "embedding-transpose",
    }

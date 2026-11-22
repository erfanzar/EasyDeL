"""CPU-safe coverage of Xerxes' parameter-free normalization configuration."""

from types import SimpleNamespace

import pytest
from easydel.modules.xerxes import modeling_xerxes
from spectrax import nn

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("mlp_norm", [False, True])
def test_kvnorm_dense_decoder_preserves_identity_normalization(monkeypatch, mlp_norm):
    """Build the decoder configuration without allocating any model parameters."""

    class ConfiguredNorm(nn.Identity):
        def __init__(self, **kwargs):
            self.dim = kwargs["dim"]

    # Only parameter-bearing dependencies are replaced; decoder construction and
    # its parameter-free identity layers are the actual production classes.
    monkeypatch.setattr(modeling_xerxes, "XerxesAttention", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(modeling_xerxes, "XerxesMLP", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(modeling_xerxes, "RMSNorm", ConfiguredNorm)
    config = SimpleNamespace(xe_kvnorm=True, xe_moe=False, xe_mlpnorm=mlp_norm, hidden_size=8, rms_norm_eps=1e-6)
    layer = modeling_xerxes.XerxesDecoderLayer(config, layer_idx=0, rngs=None)
    assert layer.identity is (not mlp_norm)
    assert isinstance(layer.input_layernorm, ConfiguredNorm)
    assert isinstance(layer.post_attention_layernorm, ConfiguredNorm)
    expected_type = ConfiguredNorm if mlp_norm else nn.Identity
    assert type(layer.pre_feedforward_layernorm) is expected_type
    assert type(layer.post_feedforward_layernorm) is expected_type
    if not mlp_norm:
        token = object()
        assert layer.pre_feedforward_layernorm(token) is token
        assert layer.post_feedforward_layernorm(token) is token

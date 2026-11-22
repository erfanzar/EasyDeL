"""Public image-feature wrapper contracts without array computation."""

from types import SimpleNamespace

import pytest
from easydel.modules.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("return_valid_length", [None, False, True])
def test_public_image_features_preserves_optional_valid_length(return_valid_length):
    pixel_values, image_sizes, features, valid_length = (object() for _ in range(4))
    expected_pixel_values, expected_image_sizes = pixel_values, image_sizes

    def backbone_image_features(*, pixel_values, image_sizes, return_valid_length=False):
        assert pixel_values is expected_pixel_values
        assert image_sizes is expected_image_sizes
        # Stand-ins preserve the actual public return contract without creating
        # a vision model or executing numerical array operations on the CPU.
        return (features, valid_length) if return_valid_length else features

    model = SimpleNamespace(base_model=SimpleNamespace(get_image_features=backbone_image_features))
    kwargs = {} if return_valid_length is None else {"return_valid_length": return_valid_length}
    result = Mistral3ForConditionalGeneration.get_image_features(model, pixel_values, image_sizes, **kwargs)

    if return_valid_length:
        assert isinstance(result, tuple)
        assert result[0] is features
        assert result[1] is valid_length
    else:
        assert result is features

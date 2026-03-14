from __future__ import annotations

import numpy as np

from easydel.infra.elarge_model.utils import make_serializable


def _sample_callback() -> str:
    """Return a stable callable payload for serialization tests."""
    return "ok"


def test_make_serializable_handles_callable_and_array_like_values() -> None:
    """Serialize callables and array-likes into JSON-safe structures."""
    payload = {
        "callable": _sample_callback,
        "matrix": np.array([[1, 2], [3, 4]], dtype=np.int32),
    }

    assert make_serializable(payload) == {
        "callable": f"{__name__}._sample_callback",
        "matrix": [[1, 2], [3, 4]],
    }

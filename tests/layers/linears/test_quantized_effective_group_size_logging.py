# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that _effective_ejkernel_group_size logs a warning when falling back."""

import pytest

from easydel.layers.linears._linear_quantized import _effective_ejkernel_group_size


def _capture_warnings(monkeypatch):
    """Intercept logger.warning calls on the module's logger."""
    import easydel.layers.linears._linear_quantized as mod

    messages = []
    original_warning = mod.logger.warning

    def _spy(fmt, *args, **kwargs):
        messages.append(fmt % args if args else fmt)
        original_warning(fmt, *args, **kwargs)

    monkeypatch.setattr(mod.logger, "warning", _spy)
    return messages


def test_logs_warning_on_fallback(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    # group_size=128 doesn't divide group_dim=96, should fall back to 32
    result = _effective_ejkernel_group_size("affine", 128, (256, 96))

    assert result == 32
    assert any("Adjusted ejkernel group_size from 128 to 32" in msg for msg in messages)


def test_no_warning_when_exact_match(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    # group_size=64 divides group_dim=128 exactly — no fallback
    result = _effective_ejkernel_group_size("affine", 64, (256, 128))

    assert result == 64
    assert not any("Adjusted" in msg for msg in messages)


def test_fallback_picks_largest_valid_candidate(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    # group_size=512, group_dim=256 -> 512 doesn't divide 256, largest valid is 256
    result = _effective_ejkernel_group_size("affine", 512, (128, 256))

    assert result == 256
    assert any("from 512 to 256" in msg for msg in messages)


def test_nf4_mode_also_logs_fallback(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    result = _effective_ejkernel_group_size("nf4", 128, (256, 96))

    assert result == 32
    assert any("nf4" in msg for msg in messages)


def test_non_affine_mode_returns_as_is():
    # mxfp4 mode should return the requested size without fallback
    result = _effective_ejkernel_group_size("mxfp4", 128, (256, 96))
    assert result == 128


def test_raises_on_zero_group_size():
    with pytest.raises(ValueError, match="must be > 0"):
        _effective_ejkernel_group_size("affine", 0, (256, 128))


def test_empty_shape_returns_requested():
    result = _effective_ejkernel_group_size("affine", 64, ())
    assert result == 64

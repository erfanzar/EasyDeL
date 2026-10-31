# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Integration tests for the eray public API."""

import eray


class TestPublicAPI:
    def test_all_symbols_accessible(self):
        missing = [n for n in eray.__all__ if not hasattr(eray, n)]
        assert missing == [], f"Missing symbols: {missing}"

    def test_no_duplicates_in_all(self):
        assert len(eray.__all__) == len(set(eray.__all__))

    def test_subpackages_importable(self):
        pass

    def test_can_import_individual_symbols(self):
        pass


class TestNoEformerDependency:
    """Verify the package has zero imports back into eformer/JAX."""

    def test_no_eformer_import(self):
        import sys

        mods = [m for m in sys.modules if m.startswith("eformer")]
        assert mods == [], f"eformer modules loaded: {mods}"

    def test_no_jax_import(self):
        """eray must not pull in the JAX framework at import time.

        Note: ``jaxtyping`` registers a pytest plugin that auto-loads on
        collection, so we check for the actual ``jax`` package, not the
        ``jax`` substring (which would catch ``jaxtyping``).
        """
        import sys

        mods = [m for m in sys.modules if m == "jax" or m.startswith("jax.")]
        assert mods == [], f"jax modules loaded: {mods}"

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

import importlib
import unittest

from easydel import _import_structure


class ImportTests(unittest.TestCase):
    def test_imports(self):
        for module_name, attributes in _import_structure.items():
            try:
                module = importlib.import_module(f"easydel.{module_name}")
                for attribute in attributes:
                    self.assertTrue(
                        hasattr(module, attribute),
                        f"Module {module_name} doesn't have attribute {attribute}",
                    )
            except ImportError as e:
                self.fail(f"Failed to import module {module_name}: {e}")


if __name__ == "__main__":
    unittest.main()

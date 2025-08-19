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

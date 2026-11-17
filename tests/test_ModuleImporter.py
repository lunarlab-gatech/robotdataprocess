import os
import unittest
from robotdataprocess.utils.ModuleImporter import ModuleImporter


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestModuleImporter(unittest.TestCase):

    def test_get_module_invalid_module(self):
        """ Test ImportError for invalid module name. """
        # Clear cache to ensure we test the error path
        ModuleImporter._module_cache.pop('nonexistent_fake_module_xyz', None)
        with self.assertRaises(ImportError):
            ModuleImporter.get_module('nonexistent_fake_module_xyz')

    def test_get_module_attribute_invalid_attr(self):
        """ Test ImportError for valid module but invalid attribute. """
        # Clear caches to ensure we test the error path
        ModuleImporter._attr_cache.pop('os.definitely_not_a_real_attr_xyz', None)
        with self.assertRaises(ImportError):
            ModuleImporter.get_module_attribute('os', 'definitely_not_a_real_attr_xyz')

    def test_get_module_caching(self):
        """ Test that modules are cached after first load. """
        ModuleImporter._module_cache.pop('os.path', None)
        mod1 = ModuleImporter.get_module('os.path')
        mod2 = ModuleImporter.get_module('os.path')
        self.assertIs(mod1, mod2)

    def test_get_module_attribute_caching(self):
        """ Test that attributes are cached after first load. """
        ModuleImporter._attr_cache.pop('os.path.join', None)
        attr1 = ModuleImporter.get_module_attribute('os.path', 'join')
        attr2 = ModuleImporter.get_module_attribute('os.path', 'join')
        self.assertIs(attr1, attr2)


if __name__ == "__main__":
    unittest.main()

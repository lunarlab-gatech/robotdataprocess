import dataclasses
import os
from robotdataprocess.eval.SLAMMethod import SLAMMethod
import unittest


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSLAMMethod(unittest.TestCase):
    """SLAMMethod: positional field order (callers construct it positionally), required base_param_config and
    color, param_overrides defaulting to None, and immutability."""

    def test_positional_field_order(self):
        method = SLAMMethod("MG_SM_MS35", "MeronomyGraph (HMO, Max Size 35)", "MG_SM", "#FF8C00",
                            {"submap_align_params.submap_max_size": 35})
        self.assertEqual(method.name, "MG_SM_MS35")
        self.assertEqual(method.display_name, "MeronomyGraph (HMO, Max Size 35)")
        self.assertEqual(method.base_param_config, "MG_SM")
        self.assertEqual(method.color, "#FF8C00")
        self.assertEqual(method.param_overrides, {"submap_align_params.submap_max_size": 35})

    def test_param_overrides_defaults_to_none(self):
        method = SLAMMethod("ROMAN_O", "ROMAN", "ROMAN_O", "#FF0000")
        self.assertIsNone(method.param_overrides)

    def test_base_param_config_is_required(self):
        with self.assertRaises(TypeError):
            SLAMMethod("ROMAN_O", "ROMAN", color="#FF0000")

    def test_color_is_required(self):
        with self.assertRaises(TypeError):
            SLAMMethod("ROMAN_O", "ROMAN", "ROMAN_O")

    def test_frozen(self):
        method = SLAMMethod("ROMAN_O", "ROMAN", "ROMAN_O", "#FF0000")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            method.base_param_config = "MG_SM"


if __name__ == "__main__":
    unittest.main()

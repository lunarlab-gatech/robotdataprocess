import numpy as np
import os
import unittest
from robotdataprocess.utils import math_utils

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestNearestIndex(unittest.TestCase):

    def test_exact_matches(self):
        """ Test that a query equal to a sample returns that sample's index. """
        ts = [0.0, 1.0, 2.0, 3.0]
        for i, t in enumerate(ts):
            self.assertEqual(math_utils.nearest_index(ts, t), i)

    def test_picks_closer_neighbour(self):
        """ Test that a query between samples snaps to whichever is nearer. """
        ts = [0.0, 1.0, 2.0]
        self.assertEqual(math_utils.nearest_index(ts, 0.4), 0)
        self.assertEqual(math_utils.nearest_index(ts, 0.6), 1)
        self.assertEqual(math_utils.nearest_index(ts, 1.9), 2)

    def test_midpoint_resolves_to_earlier(self):
        """ Test that an exactly-between query takes the earlier sample. """
        self.assertEqual(math_utils.nearest_index([0.0, 1.0], 0.5), 0)

    def test_clamps_outside_range(self):
        """ Test that queries beyond either end clamp instead of raising. """
        ts = [10.0, 11.0, 12.0]
        self.assertEqual(math_utils.nearest_index(ts, -100.0), 0)
        self.assertEqual(math_utils.nearest_index(ts, 100.0), 2)

    def test_accepts_ndarray(self):
        """ Test that a numpy array works as well as a list. """
        self.assertEqual(math_utils.nearest_index(np.array([0.0, 5.0, 9.0]), 4.9), 1)

    def test_single_element(self):
        """ Test that a one-sample array always returns index 0. """
        self.assertEqual(math_utils.nearest_index([7.0], 0.0), 0)
        self.assertEqual(math_utils.nearest_index([7.0], 70.0), 0)

    def test_empty_raises(self):
        """ Test that an empty array is rejected rather than silently clamped. """
        with self.assertRaises(ValueError):
            math_utils.nearest_index([], 0.0)


if __name__ == '__main__':
    unittest.main()

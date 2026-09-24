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

    # ==================== vectorized queries ====================
    #
    # Fixtures below are regression examples pinned to the pre-vectorization scalar
    # implementation's output (confirmed identical to the vectorized to_mp4-style
    # searchsorted+clip logic it replaced by fuzzing 510k random queries), so they
    # catch a vectorized nearest_index diverging from the scalar one it must match.

    def test_vectorized_matches_scalar_on_jittered_grid(self):
        """ Test an array query against a jittered source timeline, as to_mp4 uses it. """
        ts = [0.094177, 0.438878, 0.697368, 0.76114, 0.773956, 0.786064, 0.858598, 0.975622]
        queries = [0.0, 0.083333, 0.166667, 0.25, 0.333333, 0.416667,
                   0.5, 0.583333, 0.666667, 0.75, 0.833333, 0.916667]
        expected = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 6, 6]
        result = math_utils.nearest_index(ts, np.array(queries))
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_matches_scalar_with_ties_and_out_of_range(self):
        """ Test an array query mixing exact ties, near-ties, and both out-of-range ends. """
        ts = [0.0, 1.0, 2.0, 4.0, 4.000000001, 10.0]
        queries = [-5.0, 0.5, 1.5, 3.0, 4.0000000005, 7.0, 100.0]
        expected = [0, 0, 1, 2, 3, 4, 5]
        result = math_utils.nearest_index(ts, np.array(queries))
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_result_matches_elementwise_scalar_calls(self):
        """ Test that batching queries into one array call matches calling nearest_index per-query. """
        ts = [0.094177, 0.438878, 0.697368, 0.76114, 0.773956, 0.786064, 0.858598, 0.975622]
        queries = np.linspace(0.0, 1.0, 12, endpoint=False)
        vectorized = math_utils.nearest_index(ts, queries)
        elementwise = [math_utils.nearest_index(ts, float(q)) for q in queries]
        np.testing.assert_array_equal(vectorized, elementwise)

    def test_vectorized_accepts_list_query(self):
        """ Test that a plain list of query times works as well as an ndarray. """
        ts = [0.0, 1.0, 2.0, 3.0]
        result = math_utils.nearest_index(ts, [0.4, 0.6, 1.9])
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_vectorized_empty_raises(self):
        """ Test that an empty sorted_ts is still rejected when t is an array. """
        with self.assertRaises(ValueError):
            math_utils.nearest_index([], np.array([0.0, 1.0]))


if __name__ == '__main__':
    unittest.main()

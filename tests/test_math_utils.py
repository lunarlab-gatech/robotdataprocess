import numpy as np
import os
from robotdataprocess.utils import math_utils
from scipy.spatial import ConvexHull, QhullError
import unittest

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


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestMaxPairwiseDistance(unittest.TestCase):

    def test_3d_farthest_pair_in_middle(self):
        """ Test a 3D set whose farthest pair is neither first nor last and differs from the bounding-box diagonal. """
        # Farthest pair (3, 4, 0)-(0, 0, 12) = 13; bbox diagonal ~13.60, z range 12, first-to-last ~1.41
        points = np.array([[1, 1, 1], [3, 4, 0], [-2, 0, 0], [0, 0, 12], [1, 0, 0]], dtype=float)
        self.assertAlmostEqual(math_utils.max_pairwise_distance(points), 13.0)

    def test_planar_exact_despite_joggle(self):
        """ Test that a ground-robot (all z = 0) set, which crashes a plain hull, returns the exact unjoggled distance. """
        # Farthest pair (0, 0)-(6, 8) = 10 exactly; the 1e4 offset would make any joggle visible in the result
        points = np.array([[2, 2, 0], [0, 0, 0], [6, 0, 0], [6, 8, 0], [1, 5, 0]], dtype=float) + 1e4
        with self.assertRaises(QhullError):
            ConvexHull(points)
        self.assertEqual(math_utils.max_pairwise_distance(points), 10.0)

    def test_collinear_exact_despite_joggle(self):
        """ Test that an unordered line of points, which crashes a plain hull, returns the exact unjoggled distance. """
        # Endpoints (0, 0, 0)-(9, 12, 0) = 15 exactly; the 1e4 offset would make any joggle visible in the result
        points = np.array([[6, 8, 0], [0, 0, 0], [9, 12, 0], [3, 4, 0], [1.5, 2, 0]], dtype=float) + 1e4
        with self.assertRaises(QhullError):
            ConvexHull(points)
        self.assertEqual(math_utils.max_pairwise_distance(points), 15.0)

    def test_many_points_with_planted_diameter(self):
        """ Test 300 random points strictly inside a radius-4.9 ball plus two planted points 10 m apart. """
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(300, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = directions * 4.9 * rng.uniform(0.0, 1.0, size=(300, 1)) ** (1 / 3)
        points = np.insert(points, 117, [5.0, 0.0, 0.0], axis=0)
        points = np.insert(points, 42, [-5.0, 0.0, 0.0], axis=0)
        self.assertAlmostEqual(math_utils.max_pairwise_distance(points), 10.0)

    def test_two_and_three_points(self):
        """ Test the small-N inputs that are too few for a 3D hull. """
        self.assertAlmostEqual(math_utils.max_pairwise_distance(np.array([[0, 0, 0], [3, 4, 0]], dtype=float)), 5.0)
        self.assertAlmostEqual(math_utils.max_pairwise_distance(
            np.array([[1, 1, 0], [0, 0, 0], [3, 4, 0]], dtype=float)), 5.0)

    def test_identical_points(self):
        """ Test that coincident points have zero extent. """
        self.assertAlmostEqual(math_utils.max_pairwise_distance(np.ones((5, 3))), 0.0)

    def test_fewer_than_two_points_raises(self):
        """ Test that 0 or 1 points, which have no meaningful extent, are rejected. """
        for n in (0, 1):
            with self.subTest(n=n), self.assertRaises(ValueError):
                math_utils.max_pairwise_distance(np.zeros((n, 3)))


if __name__ == '__main__':
    unittest.main()

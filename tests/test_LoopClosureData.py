import matplotlib
matplotlib.use('Agg')

from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.Data import CoordinateFrame
from robotdataprocess.data_types.LoopClosureData import LoopClosureData
from robotdataprocess.data_types.PathData import PathData
from robotdataprocess.math_utils import interpolate_poses
from scipy.spatial.transform import Rotation as R
import unittest


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestInterpolatePoses(unittest.TestCase):
    """Test the shared interpolate_poses utility."""

    def test_exact_timestamps(self):
        """Interpolation at exact source timestamps should return source values."""
        ts = np.array([0.0, 1.0, 2.0])
        pos = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0]])
        quat = np.array([[0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0]])

        new_pos, new_quat = interpolate_poses(ts, pos, quat, ts)

        np.testing.assert_array_almost_equal(new_pos, pos)
        np.testing.assert_array_almost_equal(new_quat, quat)

    def test_midpoint_interpolation(self):
        """Interpolation at midpoint should give average position."""
        ts = np.array([0.0, 2.0])
        pos = np.array([[0.0, 0.0, 0.0],
                         [4.0, 6.0, 8.0]])
        quat = np.array([[0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0]])

        target = np.array([1.0])
        new_pos, new_quat = interpolate_poses(ts, pos, quat, target)

        np.testing.assert_array_almost_equal(new_pos[0], [2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(new_quat[0], [0.0, 0.0, 0.0, 1.0])

    def test_orientation_slerp(self):
        """SLERP between identity and 90 deg Z rotation at midpoint should give 45 deg."""
        ts = np.array([0.0, 1.0])
        pos = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
        # Identity quaternion and 90 deg rotation about Z
        r0 = R.identity()
        r1 = R.from_euler('z', 90, degrees=True)
        quat = np.array([r0.as_quat(), r1.as_quat()])

        target = np.array([0.5])
        new_pos, new_quat = interpolate_poses(ts, pos, quat, target)

        # Should be 45 deg about Z
        expected = R.from_euler('z', 45, degrees=True)
        angle_diff = (R.from_quat(new_quat[0]) * expected.inv()).magnitude()
        self.assertAlmostEqual(angle_diff, 0.0, places=10)

    def test_out_of_range_raises(self):
        """Timestamps outside source range should raise ValueError."""
        ts = np.array([1.0, 2.0])
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

        with self.assertRaises(ValueError):
            interpolate_poses(ts, pos, quat, np.array([0.5]))

        with self.assertRaises(ValueError):
            interpolate_poses(ts, pos, quat, np.array([2.5]))

    def test_empty_target(self):
        """Empty target timestamps should raise ValueError."""
        ts = np.array([0.0, 1.0])
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

        with self.assertRaises(ValueError):
            interpolate_poses(ts, pos, quat, np.array([]))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataFromJson(unittest.TestCase):
    """Test from_json loading."""

    def test_from_json(self):
        """Load the test align.json and verify count and first entry values."""
        json_path = Path(__file__).parent / 'files' / 'test_LoopClosureData' / 'test_from_json' / 'align.json'
        lc_data = LoopClosureData.from_json(json_path)

        # Verify count
        self.assertEqual(lc_data.num_loop_closures, 131)

        # Verify first entry
        # seconds=[0, 98], nanoseconds=[50000000, 549999999]
        # ts_a = 0 + 50000000/1e9 = 0.05
        # ts_b = 98 + 549999999/1e9 = 98.549999999
        expected_ts_a = Decimal("0") + Decimal("50000000") / Decimal("1000000000")
        expected_ts_b = Decimal("98") + Decimal("549999999") / Decimal("1000000000")
        self.assertEqual(lc_data.timestamps_a[0], expected_ts_a)
        self.assertEqual(lc_data.timestamps_b[0], expected_ts_b)

        # Names
        self.assertEqual(lc_data.names[0], ("Husky1", "Husky2"))

        # Translation
        np.testing.assert_almost_equal(
            float(lc_data.translations[0][0]), 5.661624933379803, 10)
        np.testing.assert_almost_equal(
            float(lc_data.translations[0][1]), 2.73853263907489, 10)
        np.testing.assert_almost_equal(
            float(lc_data.translations[0][2]), 1.0984062303330102, 10)

        # Orientation (xyzw)
        np.testing.assert_almost_equal(
            float(lc_data.orientations[0][0]), 0.023114926598089634, 10)
        np.testing.assert_almost_equal(
            float(lc_data.orientations[0][3]), 0.9997169950776631, 10)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataCalculateErrors(unittest.TestCase):
    """Test calculate_errors with known poses."""

    def _make_path_data(self, timestamps, positions, orientations):
        """Helper to create a PathData object."""
        return PathData(
            frame_id="world",
            timestamps=np.array(timestamps, dtype=object),
            positions=np.array(positions, dtype=object),
            orientations=np.array(orientations, dtype=object),
            frame=CoordinateFrame.FLU,
        )

    def test_zero_error(self):
        """When estimated LC matches GT exactly, errors should be zero."""
        # Robot A at origin with identity orientation at t=0,1,2
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0, 2.0],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )

        # Robot B offset by [3, 0, 0] with identity orientation at t=0,1,2
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0, 2.0],
            positions=[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )

        # GT relative transform at t_a=1.0, t_b=1.0:
        # T_A^{-1} * T_B: R_A.inv() * (pos_B - pos_A) = [4-1, 0, 0] = [3, 0, 0]
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("RobotA", "RobotB")],
            translations=np.array([[3.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        errors = lc.calculate_errors({"RobotA": path_a, "RobotB": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 10)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 10)

    def test_known_error(self):
        """Test with a known translation error."""
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0, 2.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )

        path_b = self._make_path_data(
            timestamps=[0.0, 1.0, 2.0],
            positions=[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )

        # GT: T_A^{-1} * T_B at t=1.0 => translation = [10, 0, 0], rotation = identity
        # Estimated: translation = [11, 0, 0] (off by 1 meter), rotation = identity
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[11.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 1.0, 10)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 10)

    def test_rotation_error(self):
        """Test with a known rotation error."""
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0, 2.0],
            positions=[[0.0, 0.0, 0.0]] * 3,
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0, 2.0],
            positions=[[0.0, 0.0, 0.0]] * 3,
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )

        # GT: identity rotation. Estimated: 90 deg about Z
        r_90z = R.from_euler('z', 90, degrees=True).as_quat()
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[0.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([r_90z.tolist()], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 10)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 90.0, 5)

    def test_with_interpolation(self):
        """Test that interpolation is used correctly for non-exact timestamps."""
        path_a = self._make_path_data(
            timestamps=[0.0, 2.0],
            positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 2.0],
            positions=[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        # At t_a=1.0 (midpoint): pos_a = [2, 0, 0], ori_a = identity
        # At t_b=1.0 (midpoint): pos_b = [10, 0, 0], ori_b = identity
        # GT: translation = [10-2, 0, 0] = [8, 0, 0]
        # Estimated: [8, 0, 0] => zero error
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[8.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 8)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 8)

    def test_missing_robot_name_raises(self):
        """Should raise ValueError if robot name is not in name_to_path."""
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[0.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        with self.assertRaises(ValueError):
            lc.calculate_errors({"A": None})  # B is missing


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataVisualization(unittest.TestCase):
    """Test visualization methods don't crash."""

    def _make_errors(self):
        return {
            "translation_errors": np.array([0.1, 0.5, 1.0, 2.0, 3.0, 0.3]),
            "rotation_errors": np.array([1.0, 5.0, 10.0, 20.0, 45.0, 2.0]),
        }

    def test_visualize_errors(self):
        errors = self._make_errors()
        fig1, fig2 = LoopClosureData.visualize_errors([errors], labels=["A"], show_plots=False)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_visualize_success_rate(self):
        errors = self._make_errors()
        fig1, fig2, fig3, fig4, fig5, fig6 = LoopClosureData.visualize_success_rate([errors], labels=["A"], show_plots=False)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(fig3)
        self.assertIsNotNone(fig4)
        self.assertIsNotNone(fig5)
        self.assertIsNotNone(fig6)
        import matplotlib.pyplot as plt
        plt.close('all')


if __name__ == "__main__":
    unittest.main()

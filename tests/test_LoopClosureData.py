import matplotlib
matplotlib.use('Agg')

from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.Data import CoordinateFrame
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
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
class TestLoopClosureDataFromJsonNamesOverride(unittest.TestCase):
    """Test from_json names_override parameter."""

    JSON_PATH = Path(__file__).parent / 'files' / 'test_LoopClosureData' / 'test_from_json' / 'align.json'

    def test_names_override_replaces_all_names(self):
        """names_override dict replaces both names for every loop closure."""
        lc_data = LoopClosureData.from_json(
            self.JSON_PATH,
            names_override={"Husky1": "aerial-07", "Husky2": "ground-03"},
        )

        for name_pair in lc_data.names:
            self.assertEqual(name_pair, ("aerial-07", "ground-03"))

    def test_names_override_partial(self):
        """Only the mapped name is replaced; unmapped names are kept as-is."""
        lc_data = LoopClosureData.from_json(
            self.JSON_PATH,
            names_override={"Husky1": "aerial-07"},
        )

        for name_pair in lc_data.names:
            self.assertEqual(name_pair, ("aerial-07", "Husky2"))

    def test_names_override_empty_dict_keeps_original(self):
        """An empty dict leaves all names unchanged."""
        lc_data_no_override = LoopClosureData.from_json(self.JSON_PATH)
        lc_data_empty = LoopClosureData.from_json(self.JSON_PATH, names_override={})

        self.assertEqual(lc_data_no_override.names, lc_data_empty.names)

    def test_names_override_none_keeps_original(self):
        """Passing names_override=None (the default) leaves names unchanged."""
        lc_data = LoopClosureData.from_json(self.JSON_PATH, names_override=None)

        self.assertEqual(lc_data.names[0], ("Husky1", "Husky2"))

    def test_names_override_flipped_entries(self):
        """When some JSON entries have names in reverse order, the override
        maps each name independently so the output order is also reversed."""
        import json
        import tempfile

        entries = [
            {
                "seconds": [0, 10], "nanoseconds": [0, 0],
                "names": ["Husky1", "Husky2"],
                "translation": [1.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "seconds": [1, 11], "nanoseconds": [0, 0],
                "names": ["Husky2", "Husky1"],   # flipped
                "translation": [2.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            },
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(entries, f)
            tmp_path = f.name

        lc_data = LoopClosureData.from_json(
            tmp_path,
            names_override={"Husky1": "aerial-07", "Husky2": "ground-03"},
        )

        self.assertEqual(lc_data.names[0], ("aerial-07", "ground-03"))
        self.assertEqual(lc_data.names[1], ("ground-03", "aerial-07"))

    def test_names_override_does_not_affect_other_fields(self):
        """names_override must not change timestamps, translations, or orientations."""
        lc_base = LoopClosureData.from_json(self.JSON_PATH)
        lc_override = LoopClosureData.from_json(
            self.JSON_PATH,
            names_override={"Husky1": "aerial-07", "Husky2": "ground-03"},
        )

        self.assertEqual(lc_base.num_loop_closures, lc_override.num_loop_closures)
        np.testing.assert_array_equal(lc_base.timestamps_a, lc_override.timestamps_a)
        np.testing.assert_array_equal(lc_base.timestamps_b, lc_override.timestamps_b)
        np.testing.assert_array_equal(lc_base.translations, lc_override.translations)
        np.testing.assert_array_equal(lc_base.orientations, lc_override.orientations)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataFromG2o(unittest.TestCase):
    """Test from_g2o loading."""

    TEST_DIR = Path(__file__).parent / 'files' / 'test_LoopClosureData' / 'test_from_g20'
    G2O_PATH = TEST_DIR / 'inlier_lc_inter_robot.g2o'
    TIME_PATH = TEST_DIR / 'odom_all.time.txt'

    def test_from_g2o_count(self):
        """Load the test g2o file and verify loop closure count."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)
        self.assertEqual(lc_data.num_loop_closures, 8)

    def test_from_g2o_first_entry(self):
        """Verify first entry values: key1=a:0, key2=b:0."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)

        # a:0 -> robot_id=0, keyframe=0 -> 50000000 ns = 0.05 s
        # b:0 -> robot_id=1, keyframe=0 -> 50000000 ns = 0.05 s
        expected_ts_a = Decimal("50000000") / Decimal("1000000000")
        expected_ts_b = Decimal("50000000") / Decimal("1000000000")
        self.assertEqual(lc_data.timestamps_a[0], expected_ts_a)
        self.assertEqual(lc_data.timestamps_b[0], expected_ts_b)
        self.assertEqual(lc_data.names[0], ("a", "b"))

        # Translation
        np.testing.assert_almost_equal(float(lc_data.translations[0][0]), -0.344539, 5)
        np.testing.assert_almost_equal(float(lc_data.translations[0][1]), -9.20917, 5)
        np.testing.assert_almost_equal(float(lc_data.translations[0][2]), 2.02169, 5)

        # Orientation (xyzw)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][0]), 0.0171068, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][1]), 0.0252338, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][2]), -0.00721471, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][3]), 0.999509, 5)

    def test_from_g2o_timestamps(self):
        """Verify timestamp lookup for multiple entries."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)

        # Second entry: a:601 -> (0,601) -> 100200000000 ns = 100.2 s
        #               b:601 -> (1,601) -> 100200000000 ns = 100.2 s
        expected_ts_a_1 = Decimal("100200000000") / Decimal("1000000000")
        expected_ts_b_1 = Decimal("100200000000") / Decimal("1000000000")
        self.assertEqual(lc_data.timestamps_a[1], expected_ts_a_1)
        self.assertEqual(lc_data.timestamps_b[1], expected_ts_b_1)

        # Fourth entry: a:1638 -> (0,1638) -> 273050000000 ns = 273.05 s
        #               b:0    -> (1,0)    -> 50000000 ns = 0.05 s
        expected_ts_a_3 = Decimal("273050000000") / Decimal("1000000000")
        expected_ts_b_3 = Decimal("50000000") / Decimal("1000000000")
        self.assertEqual(lc_data.timestamps_a[3], expected_ts_a_3)
        self.assertEqual(lc_data.timestamps_b[3], expected_ts_b_3)

    def test_from_g2o_all_names(self):
        """All loop closures should be between robots 'a' and 'b'."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)

        for name_pair in lc_data.names:
            self.assertEqual(name_pair, ("a", "b"))

    def test_from_g2o_names_override(self):
        """names_override dict replaces decoded character keys for all loop closures."""
        lc_data = LoopClosureData.from_g2o(
            self.G2O_PATH, self.TIME_PATH,
            names_override={"a": "aerial-07", "b": "ground-03"},
        )

        for name_pair in lc_data.names:
            self.assertEqual(name_pair, ("aerial-07", "ground-03"))

        # Other fields should be unaffected
        self.assertEqual(lc_data.num_loop_closures, 8)

    def test_from_g2o_names_override_flipped_entries(self):
        """When some g2o entries have keys in b->a order, the override maps
        each character independently so the output pair is also flipped."""
        import tempfile

        # key encoding: char << 56 | index
        key_a0 = 97 * (1 << 56)   # a:0
        key_b0 = 98 * (1 << 56)   # b:0

        g2o_lines = (
            # normal a->b entry
            f"EDGE_SE3:QUAT {key_a0} {key_b0} 1.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
            # flipped b->a entry
            f"EDGE_SE3:QUAT {key_b0} {key_a0} 2.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
        )
        time_lines = (
            "0 0 100000000 xxx\n"   # robot 0 (a), keyframe 0 -> 0.1 s
            "1 0 200000000 xxx\n"   # robot 1 (b), keyframe 0 -> 0.2 s
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.g2o', delete=False) as gf:
            gf.write(g2o_lines)
            g2o_tmp = gf.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write(time_lines)
            time_tmp = tf.name

        lc_data = LoopClosureData.from_g2o(
            g2o_tmp, time_tmp,
            names_override={"a": "aerial-07", "b": "ground-03"},
        )

        self.assertEqual(lc_data.names[0], ("aerial-07", "ground-03"))
        self.assertEqual(lc_data.names[1], ("ground-03", "aerial-07"))

    def test_from_g2o_invalid_edge_type_raises(self):
        """Lines not starting with EDGE_SE3:QUAT should raise ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.g2o') as f:
            f.write("VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n")
            f.flush()
            with self.assertRaises(ValueError):
                LoopClosureData.from_g2o(f.name, self.TIME_PATH)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataFromMaplabJson(unittest.TestCase):
    """Test from_maplab_json loading."""

    JSON_PATH = Path(__file__).parent / 'files' / 'test_LoopClosureData' / 'test_from_maplab_json' / 'lc.json'

    def test_from_maplab_json_count(self):
        """Load the test lc.json and verify loop closure count."""
        lc_data = LoopClosureData.from_maplab_json(self.JSON_PATH)
        self.assertEqual(lc_data.num_loop_closures, 51)

    def test_from_maplab_json_first_entry(self):
        """Verify first entry timestamps, names, translation, and orientation."""
        lc_data = LoopClosureData.from_maplab_json(self.JSON_PATH)

        # from_timestamp_ns=124050000000, to_timestamp_ns=356850000000
        self.assertEqual(lc_data.timestamps_a[0], Decimal("124.05"))
        self.assertEqual(lc_data.timestamps_b[0], Decimal("356.85"))

        # Mission UUIDs used as names by default
        self.assertEqual(lc_data.names[0], (
            "99a8765349fea5180b00000000000000",
            "3f67cb5349fea5180b00000000000000",
        ))

        # Translation
        np.testing.assert_almost_equal(
            float(lc_data.translations[0][0]), 2.4478760718183228, 10)
        np.testing.assert_almost_equal(
            float(lc_data.translations[0][1]), 3.5392956138660558, 10)
        np.testing.assert_almost_equal(
            float(lc_data.translations[0][2]), -2.0999611172827564, 10)

        # Orientation (xyzw)
        np.testing.assert_almost_equal(
            float(lc_data.orientations[0][0]), 0.0012377342867899051, 10)
        np.testing.assert_almost_equal(
            float(lc_data.orientations[0][1]), -0.031413439155310738, 10)
        np.testing.assert_almost_equal(
            float(lc_data.orientations[0][2]), 0.12681357197506921, 10)
        np.testing.assert_almost_equal(
            float(lc_data.orientations[0][3]), 0.99142825348947716, 10)

    def test_from_maplab_json_detected_inliers_none(self):
        """detected_inliers should be None since switch_variable is not parsed."""
        lc_data = LoopClosureData.from_maplab_json(self.JSON_PATH)
        self.assertFalse(hasattr(lc_data, 'detected_inliers'))

    def test_from_maplab_json_names_override(self):
        """names_override replaces mission UUIDs with friendly names."""
        lc_data = LoopClosureData.from_maplab_json(
            self.JSON_PATH,
            names_override={
                "99a8765349fea5180b00000000000000": "robot-01",
                "3f67cb5349fea5180b00000000000000": "robot-02",
            },
        )

        self.assertEqual(lc_data.names[0], ("robot-01", "robot-02"))

    def test_from_maplab_json_names_override_partial(self):
        """Only mapped UUIDs are replaced; unmapped UUIDs are kept as-is."""
        lc_data = LoopClosureData.from_maplab_json(
            self.JSON_PATH,
            names_override={"99a8765349fea5180b00000000000000": "robot-01"},
        )

        self.assertEqual(lc_data.names[0], ("robot-01", "3f67cb5349fea5180b00000000000000"))


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

    def test_timestamp_before_range_clamped_to_first(self):
        """A loop closure timestamp before the path range is clamped to the
        first timestamp and does not raise."""
        # Path spans t=1.0 to t=2.0
        path_a = self._make_path_data(
            timestamps=[1.0, 2.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[1.0, 2.0],
            positions=[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        # LC timestamp 0.0 is before the path start (1.0); should clamp to 1.0.
        # GT relative transform at t=1.0: [5,0,0] - [0,0,0] = [5,0,0], identity rot.
        # Estimated matches exactly -> zero error.
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("0.0")], dtype=object),
            timestamps_b=np.array([Decimal("0.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[5.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 8)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 8)

    def test_timestamp_after_range_clamped_to_last(self):
        """A loop closure timestamp after the path range is clamped to the
        last timestamp and does not raise."""
        # Path spans t=0.0 to t=1.0
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        # LC timestamp 9.0 is after the path end (1.0); should clamp to 1.0.
        # GT relative transform at t=1.0: [3,0,0], identity rot.
        # Estimated matches exactly -> zero error.
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("9.0")], dtype=object),
            timestamps_b=np.array([Decimal("9.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[3.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 8)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 8)

    def test_one_timestamp_in_range_one_clamped(self):
        """Only the out-of-range timestamp is clamped; the in-range one is
        interpolated normally."""
        # Path spans t=0.0 to t=2.0 with position moving linearly.
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

        # ts_a=1.0 is in range -> pos_a = [2, 0, 0] (midpoint interpolation)
        # ts_b=5.0 is out of range -> clamped to 2.0 -> pos_b = [10, 0, 0]
        # GT: [10 - 2, 0, 0] = [8, 0, 0], identity rot.
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("5.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[8.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 8)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 8)

    def test_flipped_names(self):
        """When names are (B, A) instead of (A, B) the GT relative transform is
        computed from B's frame, so the estimated LC must reflect that to get
        zero error."""
        # Robot A stationary at origin with identity rotation.
        # Robot B stationary at [5, 0, 0] with a 90 deg Z rotation.
        r_b = R.from_euler('z', 90, degrees=True)
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[R.identity().as_quat().tolist()] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            orientations=[r_b.as_quat().tolist()] * 2,
        )

        # names=(B, A): GT = T_B^{-1} * T_A
        # R_rel = R_B^{-1} * R_A = R_z(-90) * identity = R_z(-90)
        # t_rel = R_z(-90).apply([0,0,0] - [5,0,0]) = R_z(-90).apply([-5,0,0]) = [0, 5, 0]
        r_rel = r_b.inv()
        t_rel = r_rel.apply([-5.0, 0.0, 0.0])
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("B", "A")],
            translations=np.array([t_rel.tolist()], dtype=object),
            orientations=np.array([r_rel.as_quat().tolist()], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 10)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 10)

    def test_same_robot_both_names(self):
        """When both names refer to the same robot the GT relative transform is
        computed between two timestamps of that robot's own trajectory."""
        # Robot A moves from [0,0,0] with identity rotation at t=0 to
        # [4,0,0] with a 90 deg Z rotation at t=2.
        r_end = R.from_euler('z', 90, degrees=True)
        path_a = self._make_path_data(
            timestamps=[0.0, 2.0],
            positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            orientations=[R.identity().as_quat().tolist(), r_end.as_quat().tolist()],
        )

        # GT: T_A(0)^{-1} * T_A(2)
        # R_rel = identity^{-1} * R_z(90) = R_z(90)
        # t_rel = identity^{-1}.apply([4,0,0] - [0,0,0]) = [4, 0, 0]
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("0.0")], dtype=object),
            timestamps_b=np.array([Decimal("2.0")], dtype=object),
            names=[("A", "A")],
            translations=np.array([[4.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([r_end.as_quat().tolist()], dtype=object),
        )

        errors = lc.calculate_errors({"A": path_a})

        np.testing.assert_almost_equal(errors["translation_errors"][0], 0.0, 10)
        np.testing.assert_almost_equal(errors["rotation_errors"][0], 0.0, 10)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLabelInliersViaOtherLoopClosureData(unittest.TestCase):
    """Test label_inliers_via_other_LoopClosureData."""

    def _make_lc(self, timestamps_a, timestamps_b, names, translations, orientations):
        return LoopClosureData(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=names,
            translations=np.array(translations, dtype=object),
            orientations=np.array(orientations, dtype=object),
        )

    def test_exact_match(self):
        """Loop closures with identical fields are labelled as inliers."""
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0"), Decimal("2.0")],
            timestamps_b=[Decimal("1.5"), Decimal("2.5")],
            names=[("A", "B"), ("A", "B")],
            translations=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )

        lc_self.label_inliers_via_other_LoopClosureData(lc_other)

        np.testing.assert_array_equal(lc_self.detected_inliers, [True, False])

    def test_no_match_raises(self):
        """Unmatched loop closures in other should raise ValueError."""
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("9.0")],
            timestamps_b=[Decimal("9.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )

        with self.assertRaises(ValueError):
            lc_self.label_inliers_via_other_LoopClosureData(lc_other)

    def test_different_translation_raises(self):
        """Same timestamps and names but different translation should raise."""
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[99.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )

        with self.assertRaises(ValueError):
            lc_self.label_inliers_via_other_LoopClosureData(lc_other)

    def test_different_orientation_raises(self):
        """Same timestamps and names but different orientation should raise."""
        r_90z = R.from_euler('z', 90, degrees=True).as_quat().tolist()
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[r_90z],
        )

        with self.assertRaises(ValueError):
            lc_self.label_inliers_via_other_LoopClosureData(lc_other)

    def test_swapped_names_raises(self):
        """Swapped name pairs (A,B) vs (B,A) should raise."""
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("1.5")],
            timestamps_b=[Decimal("1.0")],
            names=[("B", "A")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )

        with self.assertRaises(ValueError):
            lc_self.label_inliers_via_other_LoopClosureData(lc_other)

    def test_numerical_imprecision_still_matches(self):
        """Values differing by tiny floating-point noise should still match."""
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0 + 1e-12, 2.0 - 1e-12, 3.0]],
            orientations=[[0.0, 0.0, 1e-13, 1.0]],
        )

        lc_self.label_inliers_via_other_LoopClosureData(lc_other)

        np.testing.assert_array_equal(lc_self.detected_inliers, [True])

    def test_negated_quaternion_matches(self):
        """q and -q represent the same rotation and should match."""
        q = [0.1, 0.2, 0.3, 0.9327379053088816]  # normalized quaternion
        q_neg = [-x for x in q]
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[q],
        )
        lc_other = self._make_lc(
            timestamps_a=[Decimal("1.0")],
            timestamps_b=[Decimal("1.5")],
            names=[("A", "B")],
            translations=[[1.0, 2.0, 3.0]],
            orientations=[q_neg],
        )

        lc_self.label_inliers_via_other_LoopClosureData(lc_other)

        np.testing.assert_array_equal(lc_self.detected_inliers, [True])

    def test_multiple_inliers(self):
        """Multiple loop closures can be labelled as inliers at once."""
        lc_self = self._make_lc(
            timestamps_a=[Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
            timestamps_b=[Decimal("1.5"), Decimal("2.5"), Decimal("3.5")],
            names=[("A", "B"), ("A", "B"), ("A", "B")],
            translations=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 3,
        )
        # other contains first and third but not second
        lc_other = self._make_lc(
            timestamps_a=[Decimal("3.0"), Decimal("1.0")],
            timestamps_b=[Decimal("3.5"), Decimal("1.5")],
            names=[("A", "B"), ("A", "B")],
            translations=[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        lc_self.label_inliers_via_other_LoopClosureData(lc_other)

        np.testing.assert_array_equal(lc_self.detected_inliers, [True, False, True])


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataRoundTimestamps(unittest.TestCase):
    """Test round_timestamps method."""

    def _make_lc(self, timestamps_a, timestamps_b):
        return LoopClosureData(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=[("A", "B")] * len(timestamps_a),
            translations=np.array([[0.0, 0.0, 0.0]] * len(timestamps_a), dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]] * len(timestamps_a), dtype=object),
        )

    def test_round_to_two_decimals(self):
        """Timestamps should be rounded to the specified number of decimal places."""
        lc = self._make_lc(
            timestamps_a=[Decimal("1.23456"), Decimal("2.78901")],
            timestamps_b=[Decimal("3.45678"), Decimal("4.12345")],
        )

        lc.round_timestamps(2)

        self.assertEqual(lc.timestamps_a[0], Decimal("1.23"))
        self.assertEqual(lc.timestamps_a[1], Decimal("2.79"))
        self.assertEqual(lc.timestamps_b[0], Decimal("3.46"))
        self.assertEqual(lc.timestamps_b[1], Decimal("4.12"))

    def test_round_to_zero_decimals(self):
        """Rounding to 0 decimal places should give whole numbers."""
        lc = self._make_lc(
            timestamps_a=[Decimal("1.5"), Decimal("2.4")],
            timestamps_b=[Decimal("3.6"), Decimal("4.3")],
        )

        lc.round_timestamps(0)

        self.assertEqual(lc.timestamps_a[0], Decimal("2"))
        self.assertEqual(lc.timestamps_a[1], Decimal("2"))
        self.assertEqual(lc.timestamps_b[0], Decimal("4"))
        self.assertEqual(lc.timestamps_b[1], Decimal("4"))

    def test_num_loop_closures_unchanged(self):
        """Rounding should not change the number of loop closures."""
        lc = self._make_lc(
            timestamps_a=[Decimal("1.111"), Decimal("2.222"), Decimal("3.333")],
            timestamps_b=[Decimal("4.444"), Decimal("5.555"), Decimal("6.666")],
        )

        lc.round_timestamps(1)

        self.assertEqual(lc.num_loop_closures, 3)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestPruneIntraRobotLoopClosures(unittest.TestCase):
    """Test prune_intra_robot_loop_closures method."""

    def _make_lc(self, names, detected_inliers=None):
        n = len(names)
        return LoopClosureData(
            timestamps_a=np.array([Decimal(str(i)) for i in range(n)], dtype=object),
            timestamps_b=np.array([Decimal(str(i + 10)) for i in range(n)], dtype=object),
            names=names,
            translations=np.array([[float(i), 0.0, 0.0] for i in range(n)], dtype=object),
            orientations=np.array([[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)], dtype=object),
            detected_inliers=detected_inliers,
        )

    def test_removes_intra_robot(self):
        """Intra-robot loop closures (same name pair) are removed."""
        lc = self._make_lc([("A", "B"), ("A", "A"), ("B", "C"), ("B", "B")])
        lc.prune_intra_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.names, [("A", "B"), ("B", "C")])
        # entries 0 and 2 survive: translations [0,0,0] and [2,0,0], orientations [1,0,0,0] and [3,0,0,0]
        np.testing.assert_almost_equal(float(lc.translations[0][0]), 0.0)
        np.testing.assert_almost_equal(float(lc.translations[1][0]), 2.0)
        np.testing.assert_almost_equal(float(lc.orientations[0][0]), 1.0)
        np.testing.assert_almost_equal(float(lc.orientations[1][0]), 3.0)

    def test_updates_all_fields(self):
        """timestamps, translations, and orientations are pruned consistently."""
        lc = self._make_lc([("A", "A"), ("A", "B"), ("C", "C")])
        lc.prune_intra_robot_loop_closures()

        # Only the middle entry (index 1) survives
        self.assertEqual(lc.num_loop_closures, 1)
        self.assertEqual(lc.timestamps_a[0], Decimal("1"))
        self.assertEqual(lc.timestamps_b[0], Decimal("11"))
        np.testing.assert_almost_equal(float(lc.translations[0][0]), 1.0)
        np.testing.assert_almost_equal(float(lc.orientations[0][0]), 2.0)

    def test_prunes_detected_inliers_when_set(self):
        """detected_inliers array is pruned in sync with the other fields."""
        lc = self._make_lc(
            [("A", "B"), ("A", "A"), ("B", "C")],
            detected_inliers=[True, True, False],
        )
        lc.prune_intra_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 2)
        np.testing.assert_array_equal(lc.detected_inliers, [True, False])

    def test_no_intra_robot_is_noop(self):
        """When there are no intra-robot loop closures, nothing changes."""
        lc = self._make_lc([("A", "B"), ("B", "C")])
        lc.prune_intra_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.names, [("A", "B"), ("B", "C")])

    def test_all_intra_robot_gives_empty(self):
        """When all loop closures are intra-robot, the result is empty."""
        lc = self._make_lc([("A", "A"), ("B", "B")])
        lc.prune_intra_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 0)
        self.assertEqual(lc.names, [])
        self.assertEqual(len(lc.timestamps_a), 0)
        self.assertEqual(len(lc.translations), 0)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataVisualization(unittest.TestCase):
    """Test visualization methods don't crash."""

    def _make_errors(self):
        return {
            "translation_errors": np.array([0.1, 0.5, 1.0, 2.0, 3.0, 0.3]),
            "rotation_errors": np.array([1.0, 5.0, 10.0, 20.0, 45.0, 2.0]),
        }

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

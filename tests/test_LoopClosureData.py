import matplotlib
matplotlib.use('Agg')

from decimal import Decimal
import numpy as np
import os
from pathlib import Path
import tempfile
from robotdataprocess.data_types.Data import CoordinateFrame
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
from robotdataprocess.data_types.LoopClosureData.LoopClosureDataResult import LoopClosureDataResult
from robotdataprocess.data_types.PathData import PathData
from robotdataprocess.utils.math_utils import interpolate_poses
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
class TestLoopClosureDataToJson(unittest.TestCase):
    """Test to_json round-trips exactly through from_json."""

    def test_round_trip_matches_original(self):
        lc_data = LoopClosureData(
            timestamps_a=np.array([Decimal("0.05"), Decimal("98.549999999")], dtype=object),
            timestamps_b=np.array([Decimal("1.123456789"), Decimal("200")], dtype=object),
            names=[("Husky1", "Husky2"), ("aerial-07", "ground-03")],
            translations=np.array(
                [[5.661624933379803, 2.73853263907489, 1.0984062303330102], [1.0, 2.0, 3.0]], dtype=object),
            orientations=np.array(
                [[0.023114926598089634, 0.0, 0.0, 0.9997169950776631], [0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / 'roundtrip.json'
            lc_data.to_json(json_path)
            reloaded = LoopClosureData.from_json(json_path)

        self.assertEqual(reloaded.num_loop_closures, lc_data.num_loop_closures)
        np.testing.assert_array_equal(reloaded.timestamps_a, lc_data.timestamps_a)
        np.testing.assert_array_equal(reloaded.timestamps_b, lc_data.timestamps_b)
        self.assertEqual(reloaded.names, lc_data.names)
        np.testing.assert_array_equal(reloaded.translations, lc_data.translations)
        np.testing.assert_array_equal(reloaded.orientations, lc_data.orientations)


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
            "# LC:\n"
            f"EDGE_SE3:QUAT {key_a0} {key_b0} 1.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
            # flipped b->a entry
            "# LC:\n"
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

    def test_from_g2o_non_edge_lines_skipped(self):
        """VERTEX_SE3:QUAT lines, comments, and blank lines are silently
        skipped rather than raising an error, so odom_and_lc.g2o files
        (which contain vertex and comment lines) are accepted."""
        import tempfile

        key_a0 = 97 * (1 << 56)   # a:0
        key_b0 = 98 * (1 << 56)   # b:0

        mixed_lines = (
            "# this is a comment\n"
            "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
            "\n"
            "# LC:\n"
            f"EDGE_SE3:QUAT {key_a0} {key_b0} 1.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
        )
        time_lines = (
            "0 0 100000000 xxx\n"
            "1 0 200000000 xxx\n"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.g2o', delete=False) as gf:
            gf.write(mixed_lines)
            g2o_tmp = gf.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write(time_lines)
            time_tmp = tf.name

        lc_data = LoopClosureData.from_g2o(g2o_tmp, time_tmp)
        self.assertEqual(lc_data.num_loop_closures, 1)

    def test_from_g2o_odometry_edges_skipped(self):
        """Consecutive same-robot keyframe edges (odometry) are skipped;
        only loop closures are returned."""
        import tempfile

        key_a0 = 97 * (1 << 56)        # a:0
        key_a1 = 97 * (1 << 56) | 1    # a:1  <- consecutive, odometry
        key_b0 = 98 * (1 << 56)        # b:0
        key_b1 = 98 * (1 << 56) | 1    # b:1  <- consecutive, odometry
        key_a5 = 97 * (1 << 56) | 5    # a:5  <- non-consecutive, loop closure

        g2o_lines = (
            # odometry edges (consecutive, same robot) — should be skipped
            f"EDGE_SE3:QUAT {key_a0} {key_a1} 1.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
            f"EDGE_SE3:QUAT {key_b0} {key_b1} 1.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
            # inter-robot loop closure — should be kept
            "# LC:\n"
            f"EDGE_SE3:QUAT {key_a0} {key_b0} 2.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
            # intra-robot non-consecutive loop closure — should be kept
            "# LC:\n"
            f"EDGE_SE3:QUAT {key_a0} {key_a5} 3.0 0.0 0.0 0.0 0.0 0.0 1.0 "
            "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n"
        )
        time_lines = (
            "0 0 100000000 xxx\n"   # a:0 -> 0.1 s
            "0 1 200000000 xxx\n"   # a:1 -> 0.2 s
            "0 5 600000000 xxx\n"   # a:5 -> 0.6 s
            "1 0 100000000 xxx\n"   # b:0 -> 0.1 s
            "1 1 200000000 xxx\n"   # b:1 -> 0.2 s
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.g2o', delete=False) as gf:
            gf.write(g2o_lines)
            g2o_tmp = gf.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write(time_lines)
            time_tmp = tf.name

        lc_data = LoopClosureData.from_g2o(g2o_tmp, time_tmp)
        self.assertEqual(lc_data.num_loop_closures, 2)
        self.assertEqual(lc_data.names[0], ("a", "b"))
        self.assertEqual(lc_data.names[1], ("a", "a"))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataFromG2oOdomAndLC(unittest.TestCase):
    """Test from_g2o with a combined odometry + loop closure g2o file."""

    TEST_DIR = Path(__file__).parent / 'files' / 'test_LoopClosureData' / 'test_from_g20_odom_and_lc'
    G2O_PATH = TEST_DIR / 'odom_and_lc.g2o'
    TIME_PATH = TEST_DIR / 'odom_all.time.txt'

    def test_count(self):
        """Only LC-marked edges are returned; odometry edges are skipped."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)
        self.assertEqual(lc_data.num_loop_closures, 368)

    def test_first_entry(self):
        """Verify first LC entry: a:1 -> a:28."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)

        # a:1  -> robot 0, keyframe 1  -> 1670533757500283136 ns
        # a:28 -> robot 0, keyframe 28 -> 1670533808477174784 ns
        self.assertEqual(lc_data.timestamps_a[0], Decimal("1670533757500283136") / Decimal("1000000000"))
        self.assertEqual(lc_data.timestamps_b[0], Decimal("1670533808477174784") / Decimal("1000000000"))
        self.assertEqual(lc_data.names[0], ("a", "a"))
        np.testing.assert_almost_equal(float(lc_data.translations[0][0]), -12.265847440823814, 5)
        np.testing.assert_almost_equal(float(lc_data.translations[0][1]),   6.175869150286243, 5)
        np.testing.assert_almost_equal(float(lc_data.translations[0][2]),  -1.3800860633298973, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][0]),   0.020047445030492685, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][1]),  -0.06214035164221092, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][2]),  -0.690281167642425, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[0][3]),   0.7205890550402095, 5)

    def test_second_entry(self):
        """Verify second LC entry: a:1 -> a:32."""
        lc_data = LoopClosureData.from_g2o(self.G2O_PATH, self.TIME_PATH)

        # a:1  -> robot 0, keyframe 1  -> 1670533757500283136 ns
        # a:32 -> robot 0, keyframe 32 -> 1670533815327563776 ns
        self.assertEqual(lc_data.timestamps_a[1], Decimal("1670533757500283136") / Decimal("1000000000"))
        self.assertEqual(lc_data.timestamps_b[1], Decimal("1670533815327563776") / Decimal("1000000000"))
        self.assertEqual(lc_data.names[1], ("a", "a"))
        np.testing.assert_almost_equal(float(lc_data.translations[1][0]), -10.919073293432417, 5)
        np.testing.assert_almost_equal(float(lc_data.translations[1][1]),  -2.314908451609168, 5)
        np.testing.assert_almost_equal(float(lc_data.translations[1][2]),  -0.8056660578935075, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[1][0]),   0.015723572846883128, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[1][1]),  -0.06143692111768608, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[1][2]),  -0.6344300948836249, 5)
        np.testing.assert_almost_equal(float(lc_data.orientations[1][3]),   0.7703744081201443, 5)


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

        lc.calculate_errors({"RobotA": path_a, "RobotB": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 10)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 10)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 1.0, 10)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 10)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 10)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 90.0, 5)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 8)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 8)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 8)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 8)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 8)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 8)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 8)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 8)

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

        lc.calculate_errors({"A": path_a, "B": path_b})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 10)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 10)

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

        lc.calculate_errors({"A": path_a})

        np.testing.assert_almost_equal(lc.results.translation_errors[0], 0.0, 10)
        np.testing.assert_almost_equal(lc.results.rotation_errors[0], 0.0, 10)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataLabelSuccessful(unittest.TestCase):
    """Test the label_successful delegate method."""

    def _make_path_data(self, timestamps, positions, orientations):
        return PathData(
            frame_id="world",
            timestamps=np.array(timestamps, dtype=object),
            positions=np.array(positions, dtype=object),
            orientations=np.array(orientations, dtype=object),
            frame=CoordinateFrame.FLU,
        )

    def test_raises_before_calculate_errors(self):
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0")], dtype=object),
            names=[("A", "B")],
            translations=np.array([[0.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=object),
        )

        with self.assertRaises(ValueError):
            lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)

    def test_translation_threshold_is_tight(self):
        # Robot A stationary at origin; robot B stationary at [10, 0, 0], both
        # identity rotation -> GT relative transform is [10, 0, 0], identity
        # rotation (zero rotation error) for every loop closure below.
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        lc = LoopClosureData(
            # First LC estimates [10.99, 0, 0] -> 0.99 m translation error (just within target)
            # Second LC estimates [11.001, 0, 0] -> 1.001 m translation error (just outside target)
            timestamps_a=np.array([Decimal("1.0"), Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0"), Decimal("1.0")], dtype=object),
            names=[("A", "B"), ("A", "B")],
            translations=np.array([[10.99, 0.0, 0.0], [11.001, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=object),
        )

        lc.calculate_errors({"A": path_a, "B": path_b})
        lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)

        np.testing.assert_array_equal(lc.results.successful, [True, False])
        self.assertEqual(lc.results.trans_err_in_target, 1.0)
        self.assertEqual(lc.results.rot_err_in_target, 5.0)

    def test_rotation_threshold_is_tight(self):
        # Both robots stationary at the same position -> GT relative transform
        # has identity rotation (zero translation error) for every loop closure below.
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        # First LC estimates a 4.99 deg rotation about Z (just within target)
        # Second LC estimates a 5.001 deg rotation about Z (just outside target)
        r_within = R.from_euler('z', 4.99, degrees=True).as_quat()
        r_outside = R.from_euler('z', 5.001, degrees=True).as_quat()
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0"), Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0"), Decimal("1.0")], dtype=object),
            names=[("A", "B"), ("A", "B")],
            translations=np.array([[0.0, 0.0, 0.0]] * 2, dtype=object),
            orientations=np.array([r_within.tolist(), r_outside.tolist()], dtype=object),
        )

        lc.calculate_errors({"A": path_a, "B": path_b})
        lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)

        np.testing.assert_array_equal(lc.results.successful, [True, False])

    def test_both_thresholds_required(self):
        # Both robots stationary at the same position -> GT relative transform
        # is identity (zero translation and rotation error).
        path_a = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )
        path_b = self._make_path_data(
            timestamps=[0.0, 1.0],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        )

        # First LC: translation error within target (0.5 m), rotation error outside target (10 deg)
        # Second LC: translation error outside target (2.0 m), rotation error within target (1 deg)
        r_10deg = R.from_euler('z', 10.0, degrees=True).as_quat()
        r_1deg = R.from_euler('z', 1.0, degrees=True).as_quat()
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("1.0"), Decimal("1.0")], dtype=object),
            timestamps_b=np.array([Decimal("1.0"), Decimal("1.0")], dtype=object),
            names=[("A", "B"), ("A", "B")],
            translations=np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([r_10deg.tolist(), r_1deg.tolist()], dtype=object),
        )

        lc.calculate_errors({"A": path_a, "B": path_b})
        lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)

        np.testing.assert_array_equal(lc.results.successful, [False, False])


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

    def test_invalidates_results(self):
        """round_timestamps should invalidate any previously computed results,
        since rounding can shift which GT poses are interpolated."""
        lc = self._make_lc(
            timestamps_a=[Decimal("1.111")],
            timestamps_b=[Decimal("2.222")],
        )
        lc.results = LoopClosureDataResult(
            translation_errors=np.array([0.1]),
            rotation_errors=np.array([1.0]),
        )

        lc.round_timestamps(1)

        self.assertIsNone(lc.results)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestPruneIntraRobotLoopClosures(unittest.TestCase):
    """Test prune_intra_robot_loop_closures method."""

    def _make_lc(self, names):
        n = len(names)
        return LoopClosureData(
            timestamps_a=np.array([Decimal(str(i)) for i in range(n)], dtype=object),
            timestamps_b=np.array([Decimal(str(i + 10)) for i in range(n)], dtype=object),
            names=names,
            translations=np.array([[float(i), 0.0, 0.0] for i in range(n)], dtype=object),
            orientations=np.array([[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)], dtype=object),
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

    def test_empty_input_is_noop(self):
        """When there are no loop closures at all, nothing changes (regression: an
        empty mask defaults to dtype=float64, which can't be used to index)."""
        lc = self._make_lc([])
        lc.prune_intra_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 0)
        self.assertEqual(lc.names, [])
        self.assertEqual(len(lc.timestamps_a), 0)
        self.assertEqual(len(lc.translations), 0)


class TestPruneInterRobotLoopClosures(unittest.TestCase):
    """Test prune_inter_robot_loop_closures method."""

    def _make_lc(self, names):
        n = len(names)
        return LoopClosureData(
            timestamps_a=np.array([Decimal(str(i)) for i in range(n)], dtype=object),
            timestamps_b=np.array([Decimal(str(i + 10)) for i in range(n)], dtype=object),
            names=names,
            translations=np.array([[float(i), 0.0, 0.0] for i in range(n)], dtype=object),
            orientations=np.array([[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)], dtype=object),
        )

    def test_removes_inter_robot(self):
        """Inter-robot loop closures (different name pair) are removed."""
        lc = self._make_lc([("A", "B"), ("A", "A"), ("B", "C"), ("B", "B")])
        lc.prune_inter_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.names, [("A", "A"), ("B", "B")])
        # entries 1 and 3 survive: translations [1,0,0] and [3,0,0], orientations [2,0,0,0] and [4,0,0,0]
        np.testing.assert_almost_equal(float(lc.translations[0][0]), 1.0)
        np.testing.assert_almost_equal(float(lc.translations[1][0]), 3.0)
        np.testing.assert_almost_equal(float(lc.orientations[0][0]), 2.0)
        np.testing.assert_almost_equal(float(lc.orientations[1][0]), 4.0)

    def test_updates_all_fields(self):
        """timestamps, translations, and orientations are pruned consistently."""
        lc = self._make_lc([("A", "A"), ("A", "B"), ("C", "C")])
        lc.prune_inter_robot_loop_closures()

        # Entries 0 and 2 survive
        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.timestamps_a[0], Decimal("0"))
        self.assertEqual(lc.timestamps_b[0], Decimal("10"))
        self.assertEqual(lc.timestamps_a[1], Decimal("2"))
        self.assertEqual(lc.timestamps_b[1], Decimal("12"))
        np.testing.assert_almost_equal(float(lc.translations[0][0]), 0.0)
        np.testing.assert_almost_equal(float(lc.translations[1][0]), 2.0)

    def test_no_inter_robot_is_noop(self):
        """When there are no inter-robot loop closures, nothing changes."""
        lc = self._make_lc([("A", "A"), ("B", "B")])
        lc.prune_inter_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.names, [("A", "A"), ("B", "B")])

    def test_all_inter_robot_gives_empty(self):
        """When all loop closures are inter-robot, the result is empty."""
        lc = self._make_lc([("A", "B"), ("B", "C")])
        lc.prune_inter_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 0)
        self.assertEqual(lc.names, [])
        self.assertEqual(len(lc.timestamps_a), 0)
        self.assertEqual(len(lc.translations), 0)

    def test_empty_input_is_noop(self):
        """When there are no loop closures at all, nothing changes (regression: an
        empty mask defaults to dtype=float64, which can't be used to index)."""
        lc = self._make_lc([])
        lc.prune_inter_robot_loop_closures()

        self.assertEqual(lc.num_loop_closures, 0)
        self.assertEqual(lc.names, [])
        self.assertEqual(len(lc.timestamps_a), 0)
        self.assertEqual(len(lc.translations), 0)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoopClosureDataMerge(unittest.TestCase):
    """Test LoopClosureData.merge static method."""

    def _make_lc(self, timestamps_a, timestamps_b, names, translations, orientations):
        return LoopClosureData(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=names,
            translations=np.array(translations, dtype=object),
            orientations=np.array(orientations, dtype=object),
        )

    def test_merge_count(self):
        """Merged result has num_loop_closures equal to the sum of all inputs."""
        lc1 = self._make_lc(
            [Decimal("1.0"), Decimal("2.0")], [Decimal("1.5"), Decimal("2.5")],
            [("A", "B"), ("A", "B")],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )
        lc2 = self._make_lc(
            [Decimal("3.0")], [Decimal("3.5")],
            [("A", "C")],
            [[3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        )

        merged = LoopClosureData.merge([lc1, lc2])

        self.assertEqual(merged.num_loop_closures, 3)

    def test_merge_fields_concatenated(self):
        """timestamps, names, translations, and orientations are concatenated in order."""
        lc1 = self._make_lc(
            [Decimal("1.0")], [Decimal("1.5")],
            [("A", "B")],
            [[1.0, 2.0, 3.0]],
            [[0.1, 0.2, 0.3, 0.9]],
        )
        lc2 = self._make_lc(
            [Decimal("2.0")], [Decimal("2.5")],
            [("C", "D")],
            [[4.0, 5.0, 6.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        )

        merged = LoopClosureData.merge([lc1, lc2])

        self.assertEqual(merged.timestamps_a[0], Decimal("1.0"))
        self.assertEqual(merged.timestamps_a[1], Decimal("2.0"))
        self.assertEqual(merged.timestamps_b[0], Decimal("1.5"))
        self.assertEqual(merged.timestamps_b[1], Decimal("2.5"))
        self.assertEqual(merged.names[0], ("A", "B"))
        self.assertEqual(merged.names[1], ("C", "D"))
        np.testing.assert_almost_equal(float(merged.translations[0][0]), 1.0)
        np.testing.assert_almost_equal(float(merged.translations[1][0]), 4.0)
        np.testing.assert_almost_equal(float(merged.orientations[0][3]), 0.9)
        np.testing.assert_almost_equal(float(merged.orientations[1][3]), 1.0)

    def test_merge_single_item(self):
        """Merging a single-item list returns an equivalent object."""
        lc = self._make_lc(
            [Decimal("1.0"), Decimal("2.0")], [Decimal("1.5"), Decimal("2.5")],
            [("A", "B"), ("A", "C")],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )

        merged = LoopClosureData.merge([lc])

        self.assertEqual(merged.num_loop_closures, 2)
        np.testing.assert_array_equal(merged.timestamps_a, lc.timestamps_a)
        np.testing.assert_array_equal(merged.timestamps_b, lc.timestamps_b)
        self.assertEqual(merged.names, lc.names)
        np.testing.assert_array_equal(merged.translations, lc.translations)
        np.testing.assert_array_equal(merged.orientations, lc.orientations)

    def test_merge_does_not_modify_inputs(self):
        """Inputs and the input list are not modified by merge."""
        lc1 = self._make_lc(
            [Decimal("1.0")], [Decimal("1.5")],
            [("A", "B")],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        )
        lc2 = self._make_lc(
            [Decimal("2.0")], [Decimal("2.5")],
            [("C", "D")],
            [[2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        )
        input_list = [lc1, lc2]

        LoopClosureData.merge(input_list)

        self.assertEqual(lc1.num_loop_closures, 1)
        self.assertEqual(lc2.num_loop_closures, 1)
        self.assertEqual(lc1.names, [("A", "B")])
        self.assertEqual(lc2.names, [("C", "D")])
        self.assertEqual(len(input_list), 2)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestPrintDuplicateInfo(unittest.TestCase):
    """Test print_duplicate_info method."""

    def _make_lc(self, names, timestamps_a, timestamps_b):
        n = len(names)
        return LoopClosureData(
            timestamps_a=np.array([Decimal(str(t)) for t in timestamps_a], dtype=object),
            timestamps_b=np.array([Decimal(str(t)) for t in timestamps_b], dtype=object),
            names=names,
            translations=np.zeros((n, 3), dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]] * n, dtype=object),
        )

    def test_no_duplicates(self):
        lc = self._make_lc([("A", "B"), ("A", "C")], [0, 1], [10, 11])
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            lc.print_duplicate_info("run")
        finally:
            sys.stdout = sys.__stdout__
        out = buf.getvalue()
        self.assertIn("2 total loop closures", out)
        self.assertIn("0 duplicates", out)
        self.assertIn("2 if deduplicated", out)
        self.assertIn("run:", out)

    def test_with_duplicates(self):
        # Entry 0 and entry 1 are the same canonical LC
        lc = self._make_lc([("A", "B"), ("A", "B"), ("A", "C")], [0, 0, 1], [10, 10, 11])
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            lc.print_duplicate_info()
        finally:
            sys.stdout = sys.__stdout__
        out = buf.getvalue()
        self.assertIn("3 total loop closures", out)
        self.assertIn("1 duplicates", out)
        self.assertIn("2 if deduplicated", out)
        self.assertIn("avg duplicate transform diff: 0.0000 m, 0.0000 deg", out)

    def test_avg_duplicate_transform_diff_nonzero(self):
        # Entries 0 and 1 are the same canonical LC but have different translations
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal("0"), Decimal("0")], dtype=object),
            timestamps_b=np.array([Decimal("10"), Decimal("10")], dtype=object),
            names=[("A", "B"), ("A", "B")],
            translations=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=object),
        )
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            lc.print_duplicate_info()
        finally:
            sys.stdout = sys.__stdout__
        out = buf.getvalue()
        self.assertIn("avg duplicate transform diff: 5.0000 m, 0.0000 deg", out)

    def test_avg_duplicate_transform_diff_omitted_without_duplicates(self):
        lc = self._make_lc([("A", "B"), ("A", "C")], [0, 1], [10, 11])
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            lc.print_duplicate_info()
        finally:
            sys.stdout = sys.__stdout__
        out = buf.getvalue()
        self.assertNotIn("avg duplicate transform diff", out)

    def test_swapped_pair_counted_as_duplicate(self):
        # (A, B, 0, 10) and (B, A, 10, 0) are canonical-equal
        lc = self._make_lc([("A", "B"), ("B", "A")], [0, 10], [10, 0])
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            lc.print_duplicate_info()
        finally:
            sys.stdout = sys.__stdout__
        out = buf.getvalue()
        self.assertIn("2 total loop closures", out)
        self.assertIn("1 duplicates", out)
        self.assertIn("1 if deduplicated", out)
        # Both entries represent the same identity transform once un-swapped, so diff is zero
        self.assertIn("avg duplicate transform diff: 0.0000 m, 0.0000 deg", out)

    def test_no_label(self):
        lc = self._make_lc([("A", "B")], [0], [10])
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            lc.print_duplicate_info()
        finally:
            sys.stdout = sys.__stdout__
        out = buf.getvalue()
        self.assertFalse(out.startswith(":"))


class TestPruneDuplicates(unittest.TestCase):
    """Test prune_duplicates method."""

    def _make_lc(self, names, timestamps_a, timestamps_b):
        n = len(names)
        return LoopClosureData(
            timestamps_a=np.array([Decimal(str(t)) for t in timestamps_a], dtype=object),
            timestamps_b=np.array([Decimal(str(t)) for t in timestamps_b], dtype=object),
            names=names,
            translations=np.array([[float(i), 0.0, 0.0] for i in range(n)], dtype=object),
            orientations=np.array([[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)], dtype=object),
        )

    def test_no_duplicates_is_noop(self):
        lc = self._make_lc([("A", "B"), ("A", "C")], [0, 1], [10, 11])
        lc.prune_duplicates()

        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.names, [("A", "B"), ("A", "C")])

    def test_removes_duplicates_keeps_first(self):
        # Entry 0 and entry 1 are the same canonical LC; entry 1 should be dropped.
        lc = self._make_lc([("A", "B"), ("A", "B"), ("A", "C")], [0, 0, 1], [10, 10, 11])
        lc.prune_duplicates()

        self.assertEqual(lc.num_loop_closures, 2)
        self.assertEqual(lc.names, [("A", "B"), ("A", "C")])
        # First occurrence (index 0) survives, not the duplicate (index 1)
        np.testing.assert_almost_equal(float(lc.translations[0][0]), 0.0)
        np.testing.assert_almost_equal(float(lc.orientations[0][0]), 1.0)

    def test_swapped_pair_treated_as_duplicate(self):
        # (A, B, 0, 10) and (B, A, 10, 0) are canonical-equal
        lc = self._make_lc([("A", "B"), ("B", "A")], [0, 10], [10, 0])
        lc.prune_duplicates()

        self.assertEqual(lc.num_loop_closures, 1)
        self.assertEqual(lc.names, [("A", "B")])

    def test_matches_print_duplicate_info_unique_count(self):
        lc = self._make_lc(
            [("A", "B"), ("A", "B"), ("A", "C"), ("B", "A")], [0, 0, 1, 10], [10, 10, 11, 0]
        )
        lc.prune_duplicates()

        # ("A", "B", 0, 10) appears 3 times (once swapped as ("B", "A", 10, 0)); ("A", "C", 1, 11) once
        self.assertEqual(lc.num_loop_closures, 2)

    def test_prunes_results_when_set(self):
        lc = self._make_lc([("A", "B"), ("A", "B"), ("A", "C")], [0, 0, 1], [10, 10, 11])
        lc.results = LoopClosureDataResult(
            translation_errors=np.array([0.1, 0.2, 0.3]),
            rotation_errors=np.array([1.0, 2.0, 3.0]),
        )
        lc.results.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)
        lc.prune_duplicates()

        # Duplicate at index 1 is dropped; indices 0 and 2 survive.
        self.assertEqual(lc.num_loop_closures, 2)
        np.testing.assert_array_equal(lc.results.translation_errors, [0.1, 0.3])
        np.testing.assert_array_equal(lc.results.rotation_errors, [1.0, 3.0])
        np.testing.assert_array_equal(lc.results.successful, [True, True])


class TestLoopClosureDataVisualization(unittest.TestCase):
    """Test visualization methods don't crash."""

    def _make_lc_with_results(self):
        n = 6
        lc = LoopClosureData(
            timestamps_a=np.array([Decimal(str(i)) for i in range(n)], dtype=object),
            timestamps_b=np.array([Decimal(str(i)) for i in range(n)], dtype=object),
            names=[("A", "B")] * n,
            translations=np.array([[0.0, 0.0, 0.0]] * n, dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0]] * n, dtype=object),
        )
        lc.results = LoopClosureDataResult(
            translation_errors=np.array([0.1, 0.5, 1.0, 2.0, 3.0, 0.3]),
            rotation_errors=np.array([1.0, 5.0, 10.0, 20.0, 45.0, 2.0]),
        )
        return lc

    def test_visualize_success_rate(self):
        lc = self._make_lc_with_results()
        fig = LoopClosureData.visualize_success_rate([lc], labels=["A"], show_plots=False)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close('all')


if __name__ == "__main__":
    unittest.main()

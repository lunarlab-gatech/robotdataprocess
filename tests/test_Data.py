from decimal import Decimal
from io import StringIO
import numpy as np
import os
import sys
import unittest
from unittest.mock import patch
from robotdataprocess.data_types.Data import CoordinateFrame, ROSMsgLibType
from robotdataprocess.data_types.SequentialData import SequentialData


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestCoordinateFrame(unittest.TestCase):
    """ Test the CoordinateFrame enum. """

    def test_enum_values(self):
        """ Test all CoordinateFrame enum values exist and have correct values. """
        self.assertEqual(CoordinateFrame.FLU.value, 0)
        self.assertEqual(CoordinateFrame.NED.value, 1)
        self.assertEqual(CoordinateFrame.ENU.value, 2)
        self.assertEqual(CoordinateFrame.NONE.value, 3)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestROSMsgLibType(unittest.TestCase):
    """ Test the ROSMsgLibType enum. """

    def test_enum_values(self):
        """ Test all ROSMsgLibType enum values exist and have correct values. """
        self.assertEqual(ROSMsgLibType.ROSBAGS.value, 0)
        self.assertEqual(ROSMsgLibType.RCLPY.value, 1)
        self.assertEqual(ROSMsgLibType.ROSPY.value, 2)
        self.assertEqual(ROSMsgLibType.NONE.value, 3)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSequentialData(unittest.TestCase):
    """ Test the Data base class. """

    def test_init_valid_timestamps(self):
        """ Test Data initialization with valid sequential timestamps. """
        timestamps = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]
        data = SequentialData("test_frame", timestamps)

        self.assertEqual(data.frame_id, "test_frame")
        self.assertEqual(len(data.timestamps), 3)
        np.testing.assert_array_equal(data.timestamps, timestamps)

    def test_init_with_numpy_array(self):
        """ Test Data initialization with numpy array timestamps. """
        timestamps = np.array([0.1, 0.2, 0.3])
        data = SequentialData("test_frame", timestamps)

        self.assertEqual(data.frame_id, "test_frame")
        self.assertEqual(len(data.timestamps), 3)

    def test_init_non_sequential_timestamps_raises(self):
        """ Test that non-sequential timestamps raise ValueError. """
        # Timestamps out of order
        timestamps = [Decimal("0.3"), Decimal("0.2"), Decimal("0.1")]
        with self.assertRaises(ValueError):
            SequentialData("test_frame", timestamps)

    def test_init_duplicate_timestamps_raises(self):
        """ Test that duplicate timestamps raise ValueError. """
        timestamps = [Decimal("0.1"), Decimal("0.1"), Decimal("0.2")]
        with self.assertRaises(ValueError):
            SequentialData("test_frame", timestamps)

    def test_init_single_timestamp(self):
        """ Test Data initialization with single timestamp (no validation needed). """
        timestamps = [Decimal("0.1")]
        data = SequentialData("test_frame", timestamps)
        self.assertEqual(data.len(), 1)

    def test_init_empty_timestamps(self):
        """ Test Data initialization with empty timestamps. """
        timestamps = []
        data = SequentialData("test_frame", timestamps)
        self.assertEqual(data.len(), 0)

    def test_len(self):
        """ Test len() returns correct number of timestamps. """
        timestamps = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3"), Decimal("0.4")]
        data = SequentialData("test_frame", timestamps)
        self.assertEqual(data.len(), 4)

    def test_get_ros_msg_type_raises(self):
        """ Test get_ros_msg_type raises NotImplementedError. """
        with self.assertRaises(NotImplementedError):
            SequentialData.get_ros_msg_type(ROSMsgLibType.ROSBAGS)

    def test_get_ros_msg_raises(self):
        """ Test get_ros_msg raises NotImplementedError. """
        timestamps = [Decimal("0.1"), Decimal("0.2")]
        data = SequentialData("test_frame", timestamps)
        with self.assertRaises(NotImplementedError):
            data.get_ros_msg(ROSMsgLibType.ROSBAGS, 0)

    def test_crop_data_raises(self):
        """ Test crop_data raises NotImplementedError. """
        timestamps = [Decimal("0.1"), Decimal("0.2")]
        data = SequentialData("test_frame", timestamps)
        with self.assertRaises(NotImplementedError):
            data.crop_data(Decimal("0.1"), Decimal("0.2"))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestDataHertzAnalysis(unittest.TestCase):
    """ Test the hertz analysis methods of the Data class. """

    def test_compute_hertz_stats_basic_no_trim(self):
        """ Test compute_hertz_stats with basic valid data without trimming. """
        # Create timestamps at 10 Hz (0.1s intervals)
        timestamps = [Decimal(f"{i * 0.1:.1f}") for i in range(20)]
        data = SequentialData("test_frame", timestamps)

        hertz_diffs, hertz_values, num_zero_diffs = data.compute_hertz_stats(trim_outliers=False)

        self.assertEqual(len(hertz_diffs), 19)  # n-1 differences
        self.assertEqual(len(hertz_values), 19)  # All non-zero
        self.assertEqual(num_zero_diffs, 0)

        # All diffs should be ~0.1
        expected_diffs = np.full(19, 0.1)
        np.testing.assert_array_almost_equal([float(d) for d in hertz_diffs], expected_diffs, decimal=5)

        # All hertz values should be ~10
        expected_hz = np.full(19, 10.0)
        np.testing.assert_array_almost_equal([float(h) for h in hertz_values], expected_hz, decimal=5)

    def test_compute_hertz_stats_with_trimming(self):
        """ Test compute_hertz_stats trims outliers when there are enough samples. """
        # Create 20 timestamps (19 differences, should trim to 9 after removing first/last 5)
        timestamps = [Decimal(f"{i * 0.1:.1f}") for i in range(20)]
        data = SequentialData("test_frame", timestamps)

        hertz_diffs, hertz_values, num_zero_diffs = data.compute_hertz_stats(trim_outliers=True)

        # 19 - 10 = 9 after trimming
        self.assertEqual(len(hertz_diffs), 9)
        self.assertEqual(len(hertz_values), 9)

    def test_compute_hertz_stats_trim_insufficient_data_raises(self):
        """ Test compute_hertz_stats raises ValueError when trimming requested but not enough data. """
        # Only 4 timestamps (3 differences, not enough to trim)
        timestamps = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3"), Decimal("0.4")]
        data = SequentialData("test_frame", timestamps)

        with self.assertRaises(ValueError):
            data.compute_hertz_stats(trim_outliers=True)

    def test_compute_hertz_stats_trim_boundary_raises(self):
        """ Test compute_hertz_stats raises ValueError at boundary (exactly 10 differences). """
        # 11 timestamps = 10 differences, should still raise (need > 10)
        timestamps = [Decimal(f"{i * 0.1:.1f}") for i in range(11)]
        data = SequentialData("test_frame", timestamps)

        with self.assertRaises(ValueError):
            data.compute_hertz_stats(trim_outliers=True)

    def test_compute_hertz_stats_insufficient_data(self):
        """ Test compute_hertz_stats raises ValueError with < 2 samples. """
        timestamps = [Decimal("0.1")]
        data = SequentialData("test_frame", timestamps)

        with self.assertRaises(ValueError):
            data.compute_hertz_stats()

    def test_compute_hertz_stats_empty_data(self):
        """ Test compute_hertz_stats raises ValueError with empty data. """
        timestamps = []
        data = SequentialData("test_frame", timestamps)

        with self.assertRaises(ValueError):
            data.compute_hertz_stats()

    def test_hertz_analysis_without_plots(self):
        """ Test hertz_analysis returns correct data with show_plots=False. """
        timestamps = [Decimal(f"{i * 0.1:.1f}") for i in range(20)]
        data = SequentialData("test_frame", timestamps)

        hertz_diffs, hertz_values = data.hertz_analysis(show_plots=False)

        # Should return trimmed data (19 - 10 = 9)
        self.assertEqual(len(hertz_diffs), 9)
        self.assertEqual(len(hertz_values), 9)

    def test_hertz_analysis_insufficient_data(self):
        """ Test hertz_analysis raises ValueError with < 2 samples. """
        timestamps = [Decimal("0.1")]
        data = SequentialData("test_frame", timestamps)

        with self.assertRaises(ValueError):
            data.hertz_analysis(show_plots=False)

    def test_compute_hertz_stats_varying_rates(self):
        """ Test compute_hertz_stats with varying time intervals. """
        # Timestamps with varying intervals
        timestamps = [Decimal("0.0"), Decimal("0.1"), Decimal("0.3"), Decimal("0.4"), Decimal("1.0")]
        data = SequentialData("test_frame", timestamps)

        hertz_diffs, hertz_values, num_zero_diffs = data.compute_hertz_stats(trim_outliers=False)

        self.assertEqual(len(hertz_diffs), 4)
        self.assertEqual(num_zero_diffs, 0)

        # Diffs should be sorted: 0.1, 0.1, 0.2, 0.6
        expected_diffs = [0.1, 0.1, 0.2, 0.6]
        np.testing.assert_array_almost_equal([float(d) for d in hertz_diffs], expected_diffs, decimal=5)

    def test_compute_hertz_stats_sorted_output(self):
        """ Test that hertz_diffs and hertz_values are sorted. """
        # Create timestamps with varying intervals that will result in unsorted diffs
        timestamps = [Decimal("0.0"), Decimal("0.5"), Decimal("0.6"), Decimal("0.65"), Decimal("1.0")]
        data = SequentialData("test_frame", timestamps)

        hertz_diffs, hertz_values, _ = data.compute_hertz_stats(trim_outliers=False)

        # Verify sorted in ascending order
        for i in range(len(hertz_diffs) - 1):
            self.assertLessEqual(hertz_diffs[i], hertz_diffs[i + 1])
        for i in range(len(hertz_values) - 1):
            self.assertLessEqual(hertz_values[i], hertz_values[i + 1])

    def test_hertz_analysis_zero_diffs_warning(self):
        """ Test hertz_analysis prints warning when there are zero differences. """
        # We can't easily create zero differences in Data due to validation,
        # but we can test the warning path by mocking compute_hertz_stats
        timestamps = [Decimal(f"{i * 0.1:.1f}") for i in range(20)]
        data = SequentialData("test_frame", timestamps)

        # Mock compute_hertz_stats to return data with zero diffs
        with patch.object(data, 'compute_hertz_stats', return_value=(
            [Decimal("0.1")] * 9,
            [Decimal("10.0")] * 9,
            3  # Simulate 3 zero differences
        )):
            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            data.hertz_analysis(show_plots=False)

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            self.assertIn("Warning", output)
            self.assertIn("3", output)

    @patch('matplotlib.pyplot.show')
    def test_hertz_analysis_with_plots_mocked(self, mock_show):
        """ Test hertz_analysis visualization code path with mocked matplotlib. """
        timestamps = [Decimal(f"{i * 0.1:.1f}") for i in range(20)]
        data = SequentialData("test_frame", timestamps)

        hertz_diffs, hertz_values = data.hertz_analysis(show_plots=True)

        # Verify plt.show was called (once per histogram = 2 times)
        self.assertEqual(mock_show.call_count, 2)

        # Verify return values
        self.assertEqual(len(hertz_diffs), 9)
        self.assertEqual(len(hertz_values), 9)


if __name__ == "__main__":
    unittest.main()

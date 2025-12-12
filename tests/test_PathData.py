import numpy as np
from pathlib import Path
from robotdataprocess.conversion_utils import col_to_dec_arr
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from robotdataprocess.data_types.PathData import PathData
import unittest

class TestPathData(unittest.TestCase):

    def test_calculate_trajectory_errors(self):
        """ Verify via regression test on a couple calculated metrics. """

        # Load the poseData files
        file_path = Path(__file__).parent / 'files' / 'test_PathData' / 'test_calculate_trajectory_errors'
        gt_data = OdometryData.from_csv(file_path / 'poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        est_data = OdometryData.from_csv(file_path / 'poseEst.csv', "world", "robot", CoordinateFrame.FLU, True, None)

        # Calculate all metrics
        results_dict: dict = PathData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1)
        
        # Make sure the values match what we expect
        np.testing.assert_almost_equal(results_dict['APE']['translation_part']['rmse'], 0.43900241699624326, 12)
        np.testing.assert_almost_equal(results_dict['APE']['translation_part']['max'], 0.5769000332405032, 12)
        np.testing.assert_almost_equal(results_dict['APE']['rotation_angle_deg']['mean'], 35.1468632257006, 12)

        # TODO: Write test cases to verify that RPE metrics are good.

    def test_to_OdometryData(self):

        # Create a PathData object
        ts_expected = np.array([0.0, 1.0, 2.0], dtype=object)
        pos_expected = np.array([[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]], dtype=object)
        ori_expected = np.array([[1.0, 0.0, 0.0, 0.0],
                                 [0.7071, 0.0, 0.7071, 0.0],
                                 [0.0, 0.0, 1.0, 0.0]], dtype=object)
        path_data = PathData(
            frame_id="robot",
            timestamps=ts_expected,
            positions=pos_expected,
            orientations=ori_expected,
            frame=CoordinateFrame.FLU
        )

        # Convert to OdometryData
        odom_data = path_data.to_OdometryData('world', 'robot')

        # Verify values
        np.testing.assert_array_equal(odom_data.timestamps, col_to_dec_arr(ts_expected))
        np.testing.assert_array_equal(odom_data.positions, col_to_dec_arr(pos_expected))
        np.testing.assert_array_equal(odom_data.orientations, col_to_dec_arr(ori_expected))
        np.testing.assert_equal(odom_data.frame_id, 'world')
        np.testing.assert_equal(odom_data.child_frame_id, 'robot')
        np.testing.assert_equal(odom_data.frame, CoordinateFrame.FLU)

    def test_make_start_and_end_times_match(self):
        # Create two PathData objects with non-matching start and end times
        path1 = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path2 = PathData(
            frame_id="robot",
            timestamps=np.array([1.0, 2.0, 3.0], dtype=object),
            positions=np.array([[0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 2.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )

        # Make their start and end times match
        path1_matched, path2_matched = PathData.make_start_and_end_times_match([path1], [path2])
        path1_matched = path1_matched[0]
        path2_matched = path2_matched[0]

        # Ensure that the start and end times now match
        self.assertEqual(path1_matched.timestamps[0], path2_matched.timestamps[0])
        self.assertEqual(path1_matched.timestamps[-1], path2_matched.timestamps[-1])
        self.assertEqual(len(path1_matched.timestamps), 4)
        self.assertEqual(len(path2_matched.timestamps), 4)

        # Check that the data is properly set for both as well
        np.testing.assert_array_equal(path1_matched.timestamps, np.array([0.0, 1.0, 2.0, 3.0], dtype=object))
        np.testing.assert_array_equal(path1_matched.positions, np.array([[0.0, 0.0, 0.0],
                                                                        [1.0, 0.0, 0.0],
                                                                        [2.0, 0.0, 0.0],
                                                                        [2.0, 0.0, 0.0]], dtype=object))
        np.testing.assert_array_equal(path1_matched.orientations, np.array([[1.0, 0.0, 0.0, 0.0],
                                                                           [1.0, 0.0, 0.0, 0.0],
                                                                           [1.0, 0.0, 0.0, 0.0],
                                                                           [1.0, 0.0, 0.0, 0.0]], dtype=object))

        np.testing.assert_array_equal(path2_matched.timestamps, np.array([0.0, 1.0, 2.0, 3.0], dtype=object))
        np.testing.assert_array_equal(path2_matched.positions, np.array([[0.0, 0.0, 1.0],
                                                                        [0.0, 0.0, 1.0],
                                                                        [0.0, 1.0, 0.0],
                                                                        [0.0, 2.0, 0.0]], dtype=object))
        np.testing.assert_array_equal(path2_matched.orientations, np.array([[0.0, 0.0, 0.0, 1.0],
                                                                           [0.0, 0.0, 0.0, 1.0],
                                                                           [0.0, 0.0, 0.0, 1.0],
                                                                           [0.0, 0.0, 0.0, 1.0]], dtype=object))
        
        # Test the method for the other way around
        path2_matched, path1_matched = PathData.make_start_and_end_times_match([path2], [path1])
        path1_matched = path1_matched[0]
        path2_matched = path2_matched[0]
        self.assertEqual(path1_matched.timestamps[0], path2_matched.timestamps[0])
        self.assertEqual(path1_matched.timestamps[-1], path2_matched.timestamps[-1])
        self.assertEqual(len(path1_matched.timestamps), 4)
        self.assertEqual(len(path2_matched.timestamps), 4)
        np.testing.assert_array_equal(path1_matched.timestamps, np.array([0.0, 1.0, 2.0, 3.0], dtype=object))
        np.testing.assert_array_equal(path1_matched.positions, np.array([[0.0, 0.0, 0.0],
                                                                        [1.0, 0.0, 0.0],
                                                                        [2.0, 0.0, 0.0],
                                                                        [2.0, 0.0, 0.0]], dtype=object))
        np.testing.assert_array_equal(path1_matched.orientations, np.array([[1.0, 0.0, 0.0, 0.0],
                                                                           [1.0, 0.0, 0.0, 0.0],
                                                                           [1.0, 0.0, 0.0, 0.0],
                                                                           [1.0, 0.0, 0.0, 0.0]], dtype=object))

        np.testing.assert_array_equal(path2_matched.timestamps, np.array([0.0, 1.0, 2.0, 3.0], dtype=object))
        np.testing.assert_array_equal(path2_matched.positions, np.array([[0.0, 0.0, 1.0],
                                                                        [0.0, 0.0, 1.0],
                                                                        [0.0, 1.0, 0.0],
                                                                        [0.0, 2.0, 0.0]], dtype=object))
        np.testing.assert_array_equal(path2_matched.orientations, np.array([[0.0, 0.0, 0.0, 1.0],
                                                                           [0.0, 0.0, 0.0, 1.0],
                                                                           [0.0, 0.0, 0.0, 1.0],
                                                                           [0.0, 0.0, 0.0, 1.0]], dtype=object))

        
    def test_to_evo_and_from_evo(self):
        path = PathData(
            frame_id="robot",
            timestamps=np.array([1.0, 2.0, 3.0], dtype=object),
            positions=np.array([[1.0, 1.0, 1.0],
                                [1.0, 2.0, 1.0],
                                [1.0, 2.0, 2.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0]], dtype=object),
            frame=CoordinateFrame.FLU)

        evo_traj = path.to_evo()

        np.testing.assert_array_equal(evo_traj.timestamps, path.timestamps)
        np.testing.assert_array_equal(evo_traj.positions_xyz, path.positions)
        np.testing.assert_array_equal(evo_traj.orientations_quat_wxyz, path.orientations[:, [3, 0, 1, 2]])
        path_converted = PathData.from_evo(evo_traj, "robot", CoordinateFrame.FLU)

        np.testing.assert_array_equal(path.timestamps, path_converted.timestamps)
        np.testing.assert_array_equal(path.positions, path_converted.positions)
        np.testing.assert_array_equal(path.orientations, path_converted.orientations)
        self.assertEqual(path.frame_id, path_converted.frame_id)
        self.assertEqual(path.frame, path_converted.frame)

    def test_concatenate_PathData(self):

        # Create two PathData objects
        path1 = PathData(
            frame_id="robot",
            timestamps=np.array([10.1, 11.1, 12.1], dtype=object),
            positions=np.array([[0.0, 4.0, 6.8],
                                [1.0, 4.0, 6.8],
                                [2.0, 4.0, 6.8]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0]], dtype=object),
            frame=CoordinateFrame.FLU)
        path2 = PathData(
            frame_id="robot2",
            timestamps=np.array([1.1, 2.1, 4.1], dtype=object),
            positions=np.array([[3.0, 4.0, 6.8],
                                [4.0, 4.0, 6.8],
                                [5.0, 4.0, 6.8]], dtype=object),
            orientations=np.array([[0.0, 1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0]], dtype=object),
            frame=CoordinateFrame.ENU)
        
        # Make sure concatenation fails when not enough PathData objects are given
        with self.assertRaises(ValueError):
            PathData.concatenate_PathData([])
        with self.assertRaises(ValueError):
            PathData.concatenate_PathData([path1])
        
        # Concatenate the PathData objects
        concatenated_path = PathData.concatenate_PathData([path1, path2])

        # Verify the concatenated values
        expected_id = "robot"
        expected_timestamps = np.array([10.1, 11.1, 12.1, 13.1, 14.1, 16.1], dtype=object)
        expected_positions = np.array([[0.0, 4.0, 6.8],
                                       [1.0, 4.0, 6.8],
                                       [2.0, 4.0, 6.8],
                                       [3.0, 4.0, 6.8],
                                       [4.0, 4.0, 6.8],
                                       [5.0, 4.0, 6.8]], dtype=object)
        expected_orientations = np.array([[0.0, 0.0, 0.0, 1.0],
                                          [0.0, 0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [1.0, 0.0, 0.0, 0.0]], dtype=object)
        expected_frame = CoordinateFrame.FLU

        self.assertEqual(concatenated_path.frame_id, expected_id)
        np.testing.assert_array_equal(concatenated_path.timestamps, col_to_dec_arr(expected_timestamps))
        np.testing.assert_array_equal(concatenated_path.positions, col_to_dec_arr(expected_positions))
        np.testing.assert_array_equal(concatenated_path.orientations, col_to_dec_arr(expected_orientations))
        self.assertEqual(concatenated_path.frame, expected_frame)

if __name__ == "__main__":
    unittest.main()
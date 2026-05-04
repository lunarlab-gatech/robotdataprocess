import matplotlib
matplotlib.use('Agg')

from copy import deepcopy
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess.conversion_utils import col_to_dec_arr
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from robotdataprocess.data_types.PathData import PathData
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import tempfile
import unittest
import unittest.mock

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestPathData(unittest.TestCase):

    def test_calculate_trajectory_errors(self):
        """ Verify via regression test on a couple calculated metrics. """

        # Load the poseData files
        file_path = Path(__file__).parent / 'files' / 'test_PathData' / 'test_calculate_trajectory_errors'
        gt_data = OdometryData.from_csv(file_path / 'poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        est_data = OdometryData.from_csv(file_path / 'poseEst.csv', "world", "robot", CoordinateFrame.FLU, True, None)

        # Calculate all metrics
        results_dict, _, _ = PathData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1)
        
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

    def test_from_evo(self):
        from evo.core.trajectory import PoseTrajectory3D

        timestamps = np.array([1.0, 2.0, 2.0, 3.0])
        positions  = np.array([[0.1, 0.2, 0.3],
                                [1.1, 2.2, 3.3],
                                [1.1, 2.2, 3.3],   # exact duplicate of index 1
                                [4.4, 5.5, 6.6]])
        # orientations in wxyz (normalized, non-trivial)
        orientations_wxyz = np.array([[1.0,  0.0,  0.0,  0.0],
                                      [0.5,  0.5,  0.5,  0.5],
                                      [0.5,  0.5,  0.5,  0.5],   # exact duplicate of index 1
                                      [0.0,  0.0,  0.0,  1.0]])

        # Identical duplicate should be silently removed
        traj = PoseTrajectory3D(positions_xyz=positions, orientations_quat_wxyz=orientations_wxyz, timestamps=timestamps)
        result = PathData.from_evo(traj, "map", CoordinateFrame.FLU)
        self.assertEqual(result.len(), 3)
        np.testing.assert_array_equal(result.timestamps, np.array([1.0, 2.0, 3.0]))

        # Position differs by 1e-8 at duplicate timestamp → must raise
        positions_bad_pos = positions.copy()
        positions_bad_pos[2, 0] += 1e-8
        traj_bad_pos = PoseTrajectory3D(positions_xyz=positions_bad_pos,
                                        orientations_quat_wxyz=orientations_wxyz,
                                        timestamps=timestamps)
        with self.assertRaises(ValueError):
            PathData.from_evo(traj_bad_pos, "map", CoordinateFrame.FLU)

        # Orientation differs by 1e-8 at duplicate timestamp → must raise
        orientations_bad_ori = orientations_wxyz.copy()
        orientations_bad_ori[2, 1] += 1e-8
        traj_bad_ori = PoseTrajectory3D(positions_xyz=positions,
                                        orientations_quat_wxyz=orientations_bad_ori,
                                        timestamps=timestamps)
        with self.assertRaises(ValueError):
            PathData.from_evo(traj_bad_ori, "map", CoordinateFrame.FLU)

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

    @unittest.mock.patch('robotdataprocess.data_types.PathData.plt')
    def test_visualize_basic(self, mock_plt):
        """ Test that visualize runs without error (mocked matplotlib). """
        mock_fig = unittest.mock.MagicMock()
        mock_ax = unittest.mock.MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        path1 = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path2 = PathData(
            frame_id="robot2",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 1.0, 0.0],
                                [1.0, 1.0, 0.0],
                                [2.0, 1.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )

        # Should not raise
        path1.visualize_3D([path2], ['Path1', 'Path2'], axes_length=1.0, axes_interval=1)
        mock_plt.show.assert_called()

    def test_visualize_error_cases(self):
        """ Test that visualize raises errors for mismatched titles, axes_length, axes_interval. """
        path1 = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )

        # Wrong number of titles
        with self.assertRaises(ValueError):
            path1.visualize_3D([], ['Title1', 'Title2'])

    @unittest.mock.patch('robotdataprocess.data_types.PathData.plt')
    def test_visualize_list_axes_params(self, mock_plt):
        """ Test that visualize raises errors for mismatched list sizes of axes_length and axes_interval. """
        mock_fig = unittest.mock.MagicMock()
        mock_ax = unittest.mock.MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        path1 = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )

        # axes_length list wrong size
        with self.assertRaises(ValueError):
            path1.visualize_3D([], ['Title1'], axes_length=[1.0, 2.0])

        # axes_interval list wrong size
        with self.assertRaises(ValueError):
            path1.visualize_3D([], ['Title1'], axes_length=[1.0],axes_interval=[1, 2])

    @unittest.mock.patch('robotdataprocess.data_types.PathData.plt')
    def test_calculate_trajectory_errors_with_visualization(self, mock_plt):
        """ Test the visualize=True path in calculate_trajectory_errors. """
        mock_fig = unittest.mock.MagicMock()
        mock_ax = unittest.mock.MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        file_path = Path(__file__).parent / 'files' / 'test_PathData' / 'test_calculate_trajectory_errors'
        gt_data = OdometryData.from_csv(file_path / 'poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        est_data = OdometryData.from_csv(file_path / 'poseEst.csv', "world", "robot", CoordinateFrame.FLU, True, None)

        results_dict, _, _ = PathData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=True)
        # Verify we still get results
        self.assertIn('APE', results_dict)
        # Verify matplotlib was invoked
        mock_plt.show.assert_called()

    # =========================================================================
    # ====== Post-refactor tests: moved methods tested on PathData directly ===
    # =========================================================================

    def test_crop_data_pathdata(self):
        """ Test crop_data on PathData directly. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=object),
            positions=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
                                   [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path.crop_data(Decimal('1.0'), Decimal('2.0'))
        self.assertEqual(path.len(), 3)
        np.testing.assert_array_equal(path.timestamps.astype(float), [1.0, 1.5, 2.0])
        np.testing.assert_array_equal(path.positions.astype(float),
                                      [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    def test_shift_position_pathdata(self):
        """ Test shift_position on PathData directly. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0], dtype=object),
            positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path.shift_position(10.0, -20.0, 5.0)
        np.testing.assert_array_almost_equal(
            path.positions.astype(float),
            [[11.0, -18.0, 8.0], [14.0, -15.0, 11.0]])

    def test_shift_to_start_at_identity_pathdata(self):
        """ Test shift_to_start_at_identity on PathData directly. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[5.0, 5.0, 5.0],
                                [6.0, 5.0, 5.0],
                                [7.0, 5.0, 5.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path.shift_to_start_at_identity()
        # First position should be at origin
        np.testing.assert_array_almost_equal(path.positions[0].astype(float), [0.0, 0.0, 0.0])
        # Second position should be shifted by [1, 0, 0]
        np.testing.assert_array_almost_equal(path.positions[1].astype(float), [1.0, 0.0, 0.0])

    def test_apply_transformation_pathdata(self):
        """ Test apply_transformation_left_side and right_side on PathData directly. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0], dtype=object),
            positions=np.array([[1.0, 2.0, 3.0]], dtype=object),
            orientations=np.array([[-0.7071068, 0, 0, 0.7071068]], dtype=object),
            frame=CoordinateFrame.FLU
        )

        H = np.array([[0.0, -1.0,  0.0,  1.0],
                       [1.0,  0.0,  0.0,  0.0],
                       [0.0,  0.0,  1.0,  0.0],
                       [0.0,  0.0,  0.0,  1.0]])

        path.apply_transformation_left_side(H)
        np.testing.assert_array_almost_equal(
            path.positions[0].astype(float), [-1.0, 1.0, 3.0])
        np.testing.assert_array_almost_equal(
            path.orientations[0].astype(float), [0.5, 0.5, -0.5, -0.5], decimal=5)

        # Test right side
        path2 = PathData(
            frame_id="robot",
            timestamps=np.array([0.0], dtype=object),
            positions=np.array([[1.0, 2.0, 3.0]], dtype=object),
            orientations=np.array([[-0.7071068, 0, 0, 0.7071068]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path2.apply_transformation_right_side(H)
        np.testing.assert_array_almost_equal(
            path2.positions[0].astype(float), [2.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(
            path2.orientations[0].astype(float), [0.5, -0.5, -0.5, -0.5], decimal=5)

    def test_round_timestamps(self):
        """ Test that round_timestamps correctly rounds Decimal timestamps. """

        # Create a PathData object with high-precision timestamps
        path = PathData(
            frame_id="robot",
            timestamps=np.array([1.123456789, 2.987654321, 3.555555555], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        original_positions = path.positions.copy()
        original_orientations = path.orientations.copy()

        # Round to 3 decimal places
        path.round_timestamps(3)

        # Verify timestamps are rounded
        self.assertEqual(path.timestamps[0], Decimal('1.123'))
        self.assertEqual(path.timestamps[1], Decimal('2.988'))
        self.assertEqual(path.timestamps[2], Decimal('3.556'))

        # Verify positions and orientations are unchanged
        np.testing.assert_array_equal(path.positions, original_positions)
        np.testing.assert_array_equal(path.orientations, original_orientations)

    def test_round_timestamps_zero_decimals(self):
        """ Test rounding to 0 decimal places. """

        path = PathData(
            frame_id="robot",
            timestamps=np.array([1.4, 2.5, 3.6], dtype=object),
            positions=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        path.round_timestamps(0)
        self.assertEqual(path.timestamps[0], Decimal('1'))
        self.assertEqual(path.timestamps[1], Decimal('2'))
        self.assertEqual(path.timestamps[2], Decimal('4'))

    def test_round_timestamps_invalidates_odom_cache(self):
        """ Test that round_timestamps calls _invalidate_cache, clearing OdometryData's cached poses. """

        odom = OdometryData(
            frame_id="world",
            child_frame_id="robot",
            timestamps=np.array([1.111111, 2.222222, 3.333333], dtype=object),
            positions=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU
        )

        # Simulate cached poses
        odom.poses = ["fake_pose_1", "fake_pose_2", "fake_pose_3"]
        odom.poses_rclpy = ["fake_rclpy_1", "fake_rclpy_2"]

        # Round timestamps should clear the cache
        odom.round_timestamps(2)

        self.assertEqual(odom.poses, [])
        self.assertEqual(odom.poses_rclpy, [])
        self.assertEqual(odom.timestamps[0], Decimal('1.11'))
        self.assertEqual(odom.timestamps[1], Decimal('2.22'))
        self.assertEqual(odom.timestamps[2], Decimal('3.33'))

    def test_invalidate_cache_is_noop_on_pathdata(self):
        """ Verify _invalidate_cache is a no-op on PathData (no crash, no side effects). """
        path = PathData("r", np.array([0.0], dtype=object),
                         np.array([[0, 0, 0]], dtype=object),
                         np.array([[0, 0, 0, 1]], dtype=object),
                         CoordinateFrame.FLU)
        # Should not raise
        path._invalidate_cache()
        self.assertFalse(hasattr(path, 'poses'))

    # =========================================================================
    # ================ visualize_2D Tests (lines 353-555) =====================
    # =========================================================================

    def test_visualize_2D_mismatched_list_lengths(self):
        """ Test ValueError when input lists have different lengths. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        with self.assertRaises(ValueError):
            PathData.visualize_2D([path], [True, False], ['#FF0000'], ['robot'])
        with self.assertRaises(ValueError):
            PathData.visualize_2D([path], [True], ['#FF0000', '#0000FF'], ['robot'])
        with self.assertRaises(ValueError):
            PathData.visualize_2D([path], [True], ['#FF0000'], ['robot', 'robot2'])

    def test_visualize_2D_invalid_gt_color_lightness(self):
        """ Test ValueError for out-of-range gt_color_lightness_range_val. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        with self.assertRaises(ValueError):
            PathData.visualize_2D([path], [True], ['#FF0000'], ['robot'],
                                  gt_color_lightness_range_val=-1)
        with self.assertRaises(ValueError):
            PathData.visualize_2D([path], [True], ['#FF0000'], ['robot'],
                                  gt_color_lightness_range_val=20)

    def test_visualize_2D_save_to_file(self):
        """ Test 2D visualization saving to PDF with GT and Est paths. """
        path1 = PathData(
            frame_id="robot1",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)
        path2 = PathData(
            frame_id="robot2",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            tmp_path = f.name
        try:
            axs = PathData.visualize_2D(
                [path1, path2], [True, False], ['#FF0000', '#0000FF'],
                ['Robot1', 'Robot2'], save_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertIsNotNone(axs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_visualize_2D_no_save_path(self):
        """ Test the plt.show() code path (no save, no external axes). """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        axs = PathData.visualize_2D([path], [False], ['#FF0000'], ['Robot'])
        self.assertIsNotNone(axs)

    def test_visualize_2D_external_axes_with_options(self):
        """ Test 2D vis on externally provided axes with display options. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        fig, ax = plt.subplots()
        result = PathData.visualize_2D(
            [path], [True], ['#00FF00'], ['Robot'],
            no_background=True, show_grid=True, legend=False,
            no_border=True, disable_x_label=True, disable_y_label=True, ax=ax)
        self.assertIs(result, ax)
        plt.close(fig)

    def test_visualize_2D_background_image(self):
        """ Test 2D vis with background image, and error when x_edge missing. """
        import matplotlib.image as mpimg_saver
        img_data = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            mpimg_saver.imsave(f.name, img_data)
            tmp_img = f.name

        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0, 2.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        try:
            # Should succeed with x_edge provided
            fig, ax = plt.subplots()
            PathData.visualize_2D([path], [False], ['#FF0000'], ['Robot'],
                                  background_image_path=tmp_img,
                                  background_image_x_edge=10.0, ax=ax)
            plt.close(fig)

            # Should fail without x_edge
            fig2, ax2 = plt.subplots()
            with self.assertRaises(ValueError):
                PathData.visualize_2D([path], [False], ['#FF0000'], ['Robot'],
                                      background_image_path=tmp_img, ax=ax2)
            plt.close(fig2)
        finally:
            os.remove(tmp_img)

    # =========================================================================
    # ======= make_start_and_end_times_match error (line 704) =================
    # =========================================================================

    def test_make_start_and_end_times_match_error_cases(self):
        """ Test ValueError for empty or mismatched-length lists. """
        path = PathData(
            frame_id="robot",
            timestamps=np.array([0.0, 1.0], dtype=object),
            positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=object),
            frame=CoordinateFrame.FLU)

        with self.assertRaises(ValueError):
            PathData.make_start_and_end_times_match([], [])
        with self.assertRaises(ValueError):
            PathData.make_start_and_end_times_match([path], [])
        with self.assertRaises(ValueError):
            PathData.make_start_and_end_times_match([], [path])
        with self.assertRaises(ValueError):
            PathData.make_start_and_end_times_match([path, path], [path])

    # =========================================================================
    # ============= seperate_PathData Tests (lines 783-800) ===================
    # =========================================================================

    def test_seperate_PathData(self):
        """ Test that seperate_PathData correctly splits a concatenated PathData. """
        path1 = PathData(
            frame_id="robot",
            timestamps=np.array([10.1, 11.1, 12.1], dtype=object),
            positions=np.array([[0.0, 4.0, 6.8], [1.0, 4.0, 6.8], [2.0, 4.0, 6.8]], dtype=object),
            orientations=np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=object),
            frame=CoordinateFrame.FLU)
        path2 = PathData(
            frame_id="robot2",
            timestamps=np.array([1.1, 2.1, 4.1], dtype=object),
            positions=np.array([[3.0, 4.0, 6.8], [4.0, 4.0, 6.8], [5.0, 4.0, 6.8]], dtype=object),
            orientations=np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=object),
            frame=CoordinateFrame.ENU)

        concatenated = PathData.concatenate_PathData([path1, path2])
        separated = PathData.seperate_PathData([path1, path2], concatenated)

        self.assertEqual(len(separated), 2)
        self.assertEqual(separated[0].len(), 3)
        self.assertEqual(separated[1].len(), 3)
        np.testing.assert_array_equal(separated[0].positions, path1.positions)
        np.testing.assert_array_equal(separated[0].orientations, path1.orientations)
        np.testing.assert_array_equal(separated[1].positions, path2.positions)
        np.testing.assert_array_equal(separated[1].orientations, path2.orientations)


    # =========================================================================
    # =================== to_txt_file / to_tum Tests ==========================
    # =========================================================================

    def _make_path(self):
        """ Helper: a small PathData with known values for txt/tum tests. """
        return PathData(
            frame_id="world",
            timestamps=np.array([1.0, 2.0, 3.0], dtype=object),
            positions=np.array([[1.1, 2.2, 3.3],
                                [4.4, 5.5, 6.6],
                                [7.7, 8.8, 9.9]], dtype=object),
            orientations=np.array([[0.1, 0.2, 0.3, 0.9],
                                   [0.0, 0.0, 0.0, 1.0],
                                   [0.5, 0.5, 0.5, 0.5]], dtype=object),
            frame=CoordinateFrame.FLU
        )

    def test_to_txt_file_default_round_trip(self):
        """
        Writing with default data_to_column and reading back with default
        column_to_data (from_txt) should reproduce the original data.
        """
        path = self._make_path()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.txt"
            path.to_txt_file(out)

            # Verify column order: ts x y z qw qx qy qz
            first_row = open(out).readline().split()
            self.assertEqual(len(first_row), 8)
            self.assertAlmostEqual(float(first_row[0]), 1.0)   # timestamp
            self.assertAlmostEqual(float(first_row[1]), 1.1)   # x
            self.assertAlmostEqual(float(first_row[4]), 0.9)   # qw at index 4

            loaded = PathData.from_txt(out, "world", CoordinateFrame.FLU, header_included=False)
            np.testing.assert_array_almost_equal(
                loaded.timestamps.astype(float), path.timestamps.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.positions.astype(float), path.positions.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.orientations.astype(float), path.orientations.astype(float), decimal=9)

    def test_to_txt_file_custom_data_to_column(self):
        """
        A custom data_to_column permutation should place fields in the specified
        columns and be recoverable via the inverse column_to_data.
        """
        path = self._make_path()
        # Swap qw (index 4) and timestamp (index 0): data_to_column[0]=4, [4]=0
        # Full permutation: ts→4, x→1, y→2, z→3, qw→0, qx→5, qy→6, qz→7
        data_to_column = [4, 1, 2, 3, 0, 5, 6, 7]
        # Inverse (column_to_data): col0=qw(4), col1=x(1), col2=y(2), col3=z(3),
        #                            col4=ts(0), col5=qx(5), col6=qy(6), col7=qz(7)
        column_to_data = [4, 1, 2, 3, 0, 5, 6, 7]  # same as data_to_column here

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "custom.txt"
            path.to_txt_file(out, data_to_column=data_to_column)

            # Column 0 should now hold qw of first row
            first_row = open(out).readline().split()
            self.assertAlmostEqual(float(first_row[0]), 0.9)   # qw
            self.assertAlmostEqual(float(first_row[4]), 1.0)   # timestamp

            loaded = PathData.from_txt(out, "world", CoordinateFrame.FLU,
                                            header_included=False,
                                            column_to_data=column_to_data)
            np.testing.assert_array_almost_equal(
                loaded.timestamps.astype(float), path.timestamps.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.positions.astype(float), path.positions.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.orientations.astype(float), path.orientations.astype(float), decimal=9)

    def test_to_txt_file_existing_file_raises(self):
        """ to_txt_file should raise ValueError when the output file already exists. """
        path = self._make_path()
        existing = Path(__file__).absolute()
        with self.assertRaises(ValueError):
            path.to_txt_file(existing)

    def test_to_tum_format_and_round_trip(self):
        """
        to_tum() should write TUM format (ts x y z qx qy qz qw) and be
        recoverable via from_txt with column_to_data=[0,1,2,3,7,4,5,6].
        """
        path = self._make_path()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "tum.txt"
            path.to_tum(out)

            # Verify column order: ts x y z qx qy qz qw
            first_row = open(out).readline().split()
            self.assertEqual(len(first_row), 8)
            self.assertAlmostEqual(float(first_row[0]), 1.0)   # timestamp
            self.assertAlmostEqual(float(first_row[1]), 1.1)   # x
            self.assertAlmostEqual(float(first_row[4]), 0.1)   # qx at index 4
            self.assertAlmostEqual(float(first_row[7]), 0.9)   # qw at index 7

            # Round-trip via from_txt with TUM column_to_data
            loaded = PathData.from_txt(out, "world", CoordinateFrame.FLU,
                                            header_included=False,
                                            column_to_data=[0, 1, 2, 3, 7, 4, 5, 6])
            np.testing.assert_array_almost_equal(
                loaded.timestamps.astype(float), path.timestamps.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.positions.astype(float), path.positions.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.orientations.astype(float), path.orientations.astype(float), decimal=9)

    def test_to_tum_existing_file_raises(self):
        """ to_tum() should raise ValueError when the output file already exists. """
        path = self._make_path()
        existing = Path(__file__).absolute()
        with self.assertRaises(ValueError):
            path.to_tum(existing)

    def test_from_tum_round_trip(self):
        """
        Writing with to_tum() and reading back with from_tum() should reproduce
        the original data exactly.
        """
        path = self._make_path()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "tum.txt"
            path.to_tum(out)

            loaded = PathData.from_tum(out, "world", CoordinateFrame.FLU)
            np.testing.assert_array_almost_equal(
                loaded.timestamps.astype(float), path.timestamps.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.positions.astype(float), path.positions.astype(float), decimal=9)
            np.testing.assert_array_almost_equal(
                loaded.orientations.astype(float), path.orientations.astype(float), decimal=9)
            self.assertEqual(loaded.frame_id, "world")
            self.assertEqual(loaded.frame, CoordinateFrame.FLU)

    def test_from_tum_reads_column_order(self):
        """
        from_tum() must interpret qw as the last (8th) column, not the 5th.
        Use a hand-crafted file where qw != qx to confirm the mapping is correct.
        """
        with tempfile.TemporaryDirectory() as d:
            tum_file = Path(d) / "manual.txt"
            # Row: ts=5.0  x=1  y=2  z=3  qx=0.1  qy=0.2  qz=0.3  qw=0.9
            tum_file.write_text("5.0 1.0 2.0 3.0 0.1 0.2 0.3 0.9\n")

            loaded = PathData.from_tum(tum_file, "map", CoordinateFrame.FLU)
            self.assertEqual(loaded.len(), 1)
            self.assertAlmostEqual(float(loaded.timestamps[0]), 5.0)
            np.testing.assert_array_almost_equal(
                loaded.positions[0].astype(float), [1.0, 2.0, 3.0])
            # orientations stored as (qx, qy, qz, qw)
            np.testing.assert_array_almost_equal(
                loaded.orientations[0].astype(float), [0.1, 0.2, 0.3, 0.9])

    def test_from_tum_multiple_rows(self):
        """ from_tum() loads all rows and assigns the correct frame_id and frame. """
        with tempfile.TemporaryDirectory() as d:
            tum_file = Path(d) / "multi.txt"
            tum_file.write_text(
                "1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
                "2.0 1.0 2.0 3.0 0.5 0.5 0.5 0.5\n"
                "3.0 4.0 5.0 6.0 0.0 0.0 1.0 0.0\n"
            )

            loaded = PathData.from_tum(tum_file, "odom", CoordinateFrame.NED)
            self.assertEqual(loaded.len(), 3)
            self.assertEqual(loaded.frame_id, "odom")
            self.assertEqual(loaded.frame, CoordinateFrame.NED)
            np.testing.assert_array_almost_equal(
                loaded.timestamps.astype(float), [1.0, 2.0, 3.0])
            np.testing.assert_array_almost_equal(
                loaded.positions[1].astype(float), [1.0, 2.0, 3.0])
            # Second row: qx=0.5, qy=0.5, qz=0.5, qw=0.5
            np.testing.assert_array_almost_equal(
                loaded.orientations[1].astype(float), [0.5, 0.5, 0.5, 0.5])
            # Third row: qx=0.0, qy=0.0, qz=1.0, qw=0.0
            np.testing.assert_array_almost_equal(
                loaded.orientations[2].astype(float), [0.0, 0.0, 1.0, 0.0])

    def test_from_tum_odometrydata(self):
        """
        OdometryData.from_tum() should return an OdometryData with the correct
        child_frame_id set and correctly parsed TUM column order.
        """
        from robotdataprocess.data_types.OdometryData import OdometryData
        with tempfile.TemporaryDirectory() as d:
            tum_file = Path(d) / "odom.txt"
            tum_file.write_text("10.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n")

            loaded = OdometryData.from_tum(tum_file, "world", "robot", CoordinateFrame.FLU)
            self.assertIsInstance(loaded, OdometryData)
            self.assertEqual(loaded.child_frame_id, "robot")
            np.testing.assert_array_almost_equal(
                loaded.positions[0].astype(float), [1.0, 2.0, 3.0])
            np.testing.assert_array_almost_equal(
                loaded.orientations[0].astype(float), [0.0, 0.0, 0.0, 1.0])

    def test_to_txt_file_odometrydata_inherits(self):
        """ OdometryData should inherit to_txt_file and to_tum from PathData. """
        from robotdataprocess.data_types.OdometryData import OdometryData
        odom = OdometryData(
            frame_id="world", child_frame_id="robot",
            timestamps=np.array([10.0, 11.0], dtype=object),
            positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU
        )
        with tempfile.TemporaryDirectory() as d:
            txt_out = Path(d) / "odom.txt"
            tum_out = Path(d) / "odom_tum.txt"
            odom.to_txt_file(txt_out)
            odom.to_tum(tum_out)
            self.assertTrue(txt_out.exists())
            self.assertTrue(tum_out.exists())
            # Verify TUM qw is last for OdometryData as well
            first_row = open(tum_out).readline().split()
            self.assertAlmostEqual(float(first_row[7]), 1.0)   # qw


if __name__ == "__main__":
    unittest.main()
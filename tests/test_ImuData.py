from copy import deepcopy
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame
from robotdataprocess.data_types.ImuData import ImuData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
import unittest
import unittest.mock

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImuData(unittest.TestCase):
    
    # TODO: Write test cases for: 
    # - Orientation loading for from_ros2_bag
    # - Orientation conversion from NED to ROS frame works as well.
    # These are technically written, but don't test on any orientation other than identity.
    
    def test_from_txt(self):
        """
        Test that we can load IMU data from a txt file 
        and save it into a ROS2 bag.
        """

        # Load the IMU data and save it into a ROS2 bag
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_from_txt_file', 'imu.txt').absolute()
        imu_data = ImuData.from_txt(file_path, '/Husky1/base_link', CoordinateFrame.FLU)
        bag_path = Path(Path('.'), 'tests', 'temporary_files', 'test_ImuData', 'test_from_txt_file', 'imu_bag').absolute()
        if os.path.isdir(bag_path):
            os.remove(bag_path / 'imu_bag.db3')
            os.remove(bag_path / 'metadata.yaml')
            os.rmdir(bag_path)
        Ros2BagWrapper.write_data_to_ros2_bag(bag_path, [imu_data], ['/imu'], [None], None)

        # Load the data back again
        ros_data = ImuData.from_ros2_bag(bag_path, '/imu', '/Husky1/base_link')

        # Make sure this data matches what we expect
        np.testing.assert_equal(float(ros_data.timestamps[85]), 29.8)
        np.testing.assert_array_equal(ros_data.lin_acc[85].astype(np.float128), [-4.311637, 0.022841, -9.319456])
        np.testing.assert_array_equal(ros_data.ang_vel[85].astype(np.float128), [0.001708, -0.065869, -0.002687])
        np.testing.assert_array_equal(ros_data.orientations[85].astype(np.float128), [0, 0, 0, 1])
        np.testing.assert_equal(ros_data.frame_id, '/Husky1/base_link')

        # ======== Additionaly test when Orientation data is provided
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_from_txt_file', 'synthetic_imu_9axis.txt').absolute()
        imu_data = ImuData.from_txt(file_path, '/Husky2/robot', CoordinateFrame.NED, nine_axis=True)
        np.testing.assert_equal(float(imu_data.timestamps[84]), 331.79)
        np.testing.assert_array_equal(imu_data.lin_acc[84].astype(np.float128), [-8.800791, -0.004754, -9.927985])
        np.testing.assert_array_equal(imu_data.ang_vel[84].astype(np.float128), [ 0.000035,  0.001181, -0.005946])
        np.testing.assert_array_equal(imu_data.orientations[84].astype(np.float128), [0.001867, -0.000799, 0.927410, 0.374042])
        np.testing.assert_equal(imu_data.frame_id, '/Husky2/robot')

        bag_path_2 = Path(Path('.'), 'tests', 'temporary_files', 'test_ImuData', 'test_from_txt_file', 'imu_ori_bag').absolute()
        if os.path.isdir(bag_path_2):
            os.remove(bag_path_2 / 'imu_ori_bag.db3')
            os.remove(bag_path_2 / 'metadata.yaml')
            os.rmdir(bag_path_2)
        Ros2BagWrapper.write_data_to_ros2_bag(bag_path_2, [imu_data], ['/imu'], [None], None)

        ros_data_2 = ImuData.from_ros2_bag(bag_path_2, '/imu', '/Husky1/base_link')
        np.testing.assert_equal(float(ros_data_2.timestamps[84]), 331.79)
        np.testing.assert_array_equal(ros_data_2.lin_acc[84].astype(np.float128), [-8.800791, -0.004754, -9.927985])
        np.testing.assert_array_equal(ros_data_2.ang_vel[84].astype(np.float128), [ 0.000035,  0.001181, -0.005946])
        np.testing.assert_array_equal(ros_data_2.orientations[84].astype(np.float128), [0.001867, -0.000799, 0.927410, 0.374042])
        np.testing.assert_equal(ros_data_2.frame_id, '/Husky1/base_link')

    def test_crop_data(self):
        """ Make sure data is successfully cropped. """

        # Load the IMU data
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_crop_data', 'imu.txt').absolute()
        imu_data = ImuData.from_txt(file_path, '/Husky1/base_link', CoordinateFrame.FLU)

        # Crop it and make sure it matches what we expect
        imu_data_cropped = deepcopy(imu_data)
        imu_data_cropped.crop_data(Decimal('257.745'), Decimal('258.050000'))
        np.testing.assert_array_equal(imu_data_cropped.timestamps, imu_data.timestamps[13:75])
        np.testing.assert_array_equal(imu_data_cropped.lin_acc, imu_data.lin_acc[13:75])
        np.testing.assert_array_equal(imu_data_cropped.ang_vel, imu_data.ang_vel[13:75])
        np.testing.assert_equal(imu_data_cropped.orientations, None)

    def test_crop_to_matched_raises(self):
        """ crop_to_matched raises NotImplementedError. """
        timestamps = [Decimal('0.1'), Decimal('0.2')]
        lin_acc = np.zeros((2, 3))
        ang_vel = np.zeros((2, 3))
        imu1 = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel, None)
        imu2 = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel, None)
        with self.assertRaises(NotImplementedError):
            ImuData.crop_to_matched(imu1, imu2, Decimal('0.01'))

    @unittest.mock.patch('robotdataprocess.data_types.ImuData.plt')
    def test_visualize_mocked(self, mock_plt):
        """ Test that ImuData.visualize runs without error (mocked matplotlib). """
        mock_fig = unittest.mock.MagicMock()
        mock_axes = [unittest.mock.MagicMock() for _ in range(4)]
        mock_plt.subplots.return_value = (mock_fig, mock_axes[:3])  # lin_acc/ang_vel have 3 columns

        # subplots is called 3 times: lin_acc (3 cols), ang_vel (3 cols), orientations (4 cols)
        mock_plt.subplots.side_effect = [
            (mock_fig, [unittest.mock.MagicMock() for _ in range(3)]),
            (mock_fig, [unittest.mock.MagicMock() for _ in range(3)]),
            (mock_fig, [unittest.mock.MagicMock() for _ in range(4)]),
        ]

        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_from_txt_file', 'synthetic_imu_9axis.txt').absolute()
        imu = ImuData.from_txt(file_path, '/robot', CoordinateFrame.NED, nine_axis=True)
        imu.visualize(float(imu.timestamps[0]), float(imu.timestamps[-1]))
        mock_plt.show.assert_called()

    def test_to_PathData_no_orientation_raises(self):
        """ Test error when use_ang_vel=False and no orientations. """
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_from_txt_file', 'imu.txt').absolute()
        imu = ImuData.from_txt(file_path, '/robot', CoordinateFrame.FLU, nine_axis=False)
        with self.assertRaises(ValueError):
            imu.to_PathData(
                initial_pos=np.array([0, 0, 0], dtype=float),
                initial_vel=np.array([0, 0, 0], dtype=float),
                initial_ori=np.array([0, 0, 0, 1], dtype=float),
                use_ang_vel=False)

    def test_eq(self):
        """ Test __eq__ compares frame, lin_acc, ang_vel, and orientations, not just frame_id/timestamps. """
        timestamps = [Decimal('0.1'), Decimal('0.2')]
        lin_acc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ang_vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        orientations = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

        imu = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel, orientations)

        # Identical values should be equal
        imu_same = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel, orientations)
        self.assertEqual(imu, imu_same)

        # Different frame should not be equal
        imu_diff_frame = ImuData('imu', CoordinateFrame.NED, timestamps, lin_acc, ang_vel, orientations)
        self.assertNotEqual(imu, imu_diff_frame)

        # Different lin_acc should not be equal
        imu_diff_lin_acc = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc + 1.0, ang_vel, orientations)
        self.assertNotEqual(imu, imu_diff_lin_acc)

        # Different ang_vel should not be equal
        imu_diff_ang_vel = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel + 1.0, orientations)
        self.assertNotEqual(imu, imu_diff_ang_vel)

        # None vs non-None orientations should not be equal
        imu_no_orientations = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel, None)
        self.assertNotEqual(imu, imu_no_orientations)

        # Different orientations should not be equal
        imu_diff_orientations = ImuData('imu', CoordinateFrame.FLU, timestamps, lin_acc, ang_vel,
                                         np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]))
        self.assertNotEqual(imu, imu_diff_orientations)

    def test_to_PathData_unsupported_frame_raises(self):
        """ Test error when frame is unsupported for to_PathData. """
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_from_txt_file', 'imu.txt').absolute()
        imu = ImuData.from_txt(file_path, '/robot', CoordinateFrame.FLU, nine_axis=False)
        imu.frame = CoordinateFrame.ENU
        with self.assertRaises(RuntimeError):
            imu.to_PathData(
                initial_pos=np.array([0, 0, 0], dtype=float),
                initial_vel=np.array([0, 0, 0], dtype=float),
                initial_ori=np.array([0, 0, 0, 1], dtype=float),
                use_ang_vel=True)


if __name__ == "__main__":
    unittest.main()
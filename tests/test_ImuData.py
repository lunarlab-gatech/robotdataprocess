from copy import deepcopy
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame
from robotdataprocess.data_types.ImuData import ImuData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
import unittest

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImuData(unittest.TestCase):
    
    # TODO: Write test cases for: 
    # - Orientation loading for from_ros2_bag
    # - Orientation conversion from NED to ROS frame works as well.
    # These are technically written, but don't test on any orientation other than identity.
    
    def test_from_txt_file(self):
        """
        Test that we can load IMU data from a txt file 
        and save it into a ROS2 bag.
        """

        # Load the IMU data and save it into a ROS2 bag
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_from_txt_file', 'imu.txt').absolute()
        imu_data = ImuData.from_txt_file(file_path, '/Husky1/base_link', CoordinateFrame.FLU)
        bag_path = Path(Path('.'), 'tests', 'temporary_files', 'test_ImuData', 'test_from_txt_file', 'imu_bag').absolute()
        if os.path.isdir(bag_path):
            os.remove(bag_path / 'imu_bag.db3')
            os.remove(bag_path / 'metadata.yaml')
            os.rmdir(bag_path)
        Ros2BagWrapper.write_data_to_rosbag(bag_path, [imu_data], ['/imu'], [None], None)

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
        imu_data = ImuData.from_txt_file(file_path, '/Husky2/robot', CoordinateFrame.NED, nine_axis=True)
        np.testing.assert_equal(float(imu_data.timestamps[84]), 331.79)
        np.testing.assert_array_equal(imu_data.lin_acc[84].astype(np.float128), [-8.800791, -0.004754, -9.927985])
        np.testing.assert_array_equal(imu_data.ang_vel[84].astype(np.float128), [ 0.000035,  0.001181, -0.005946])
        np.testing.assert_array_equal(imu_data.orientations[84].astype(np.float128), [0.001867, -0.000799, 0.927410, 0.374042])
        np.testing.assert_equal(imu_data.frame_id, '/Husky2/robot')

    def test_crop_data(self):
        """ Make sure data is successfully cropped. """

        # Load the IMU data
        file_path = Path(Path('.'), 'tests', 'files', 'test_ImuData', 'test_crop_data', 'imu.txt').absolute()
        imu_data = ImuData.from_txt_file(file_path, '/Husky1/base_link', CoordinateFrame.FLU)

        # Crop it and make sure it matches what we expect
        imu_data_cropped = deepcopy(imu_data)
        imu_data_cropped.crop_data(Decimal('257.745'), Decimal('258.050000'))
        np.testing.assert_array_equal(imu_data_cropped.timestamps, imu_data.timestamps[13:75])
        np.testing.assert_array_equal(imu_data_cropped.lin_acc, imu_data.lin_acc[13:75])
        np.testing.assert_array_equal(imu_data_cropped.ang_vel, imu_data.ang_vel[13:75])
        np.testing.assert_equal(imu_data_cropped.orientations, None)


    # def test_to_FLU_frame(self):
    #     """ 
    #     Makes sure that the conversion from NED to ROS functions properly.
    #     """

    #     # Load the IMU data
    #     file_path = Path(Path('.'), 'tests', 'test_outputs', 'test_from_txt_file', 'imu.txt').absolute()
    #     imu_data = ImuData.from_txt_file(file_path, '/Husky1/base_link', CoordinateFrame.NED)

    #     # Convert into a ROS frame
    #     imu_data.to_FLU_frame()

    #     # Make sure this data matches what we expect
    #     np.testing.assert_equal(float(imu_data.timestamps[87212]), 436.065000)
    #     np.testing.assert_array_equal(imu_data.lin_acc[87212].astype(np.float128), [-0.124648, 0.091863, 10.415014])
    #     np.testing.assert_array_equal(imu_data.ang_vel[87212].astype(np.float128), [0.001785, -0.004928, -0.003135])
    #     np.testing.assert_array_equal(imu_data.orientation[87212].astype(np.float128), [1, 0, 0, 0])
    #     np.testing.assert_equal(imu_data.frame_id, '/Husky1/base_link')

if __name__ == "__main__":
    unittest.main()
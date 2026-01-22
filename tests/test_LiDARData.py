from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame, LiDARData, ROSMsgLibType
import struct
import unittest

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLiDARData(unittest.TestCase):

    def test_from_npy_files(self):
        """ Ensure we load a point cloud from a folder with npy files correctly. """

        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        print(folder_path)
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)

        # Check that the values are what we expect
        np.testing.assert_array_equal(lidar_data.timestamps.astype(float), [0.1, 0.6, 1.1])
        np.testing.assert_equal(lidar_data.frame, CoordinateFrame.NED)
        np.testing.assert_equal(lidar_data.frame_id, "robot")
        np.testing.assert_array_equal(lidar_data.get_point_cloud_at_index(0)[0][3], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(lidar_data.get_point_cloud_at_index(0)[0][35], [26.67838478088379, 0.3280501961708069, -9.796014785766602])
        np.testing.assert_array_equal(lidar_data.get_point_cloud_at_index(2)[0][-1], [5.671111583709717, 9.91832280305971e-7, 2.6444828510284424])
        np.testing.assert_equal(len(lidar_data.point_clouds), 3)

    # def test_to_FLU_frame(self):
    #     """ Ensure our transformations are correct """

    #     point_cloud_simple = [[ 0,  1, 2], [-1, -3, 4]]
    #     lidar_data = LiDARData("robot", [0], point_cloud_simple, None, CoordinateFrame.NED)
    #     lidar_data.to_FLU_frame()

    #     # Assert that the transformation completed successfully
    #     expected_pc = np.array([[[ 0, -1, -2], [-1,  3, -4]]])
    #     np.testing.assert_array_equal(lidar_data.point_clouds, expected_pc)

    #     # Make sure it doesn't change when we call it again
    #     lidar_data.to_FLU_frame()
    #     expected_pc = np.array([[[ 0, -1, -2], [-1,  3, -4]]])
    #     np.testing.assert_array_equal(lidar_data.point_clouds, expected_pc)

    def test_visualize(self):
        """ Just ensure that this code doesn't crash. """

        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        print(folder_path)
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)
        lidar_data.visualize(testing=True)
    
    def test_calculate_point_channels(self):
        """ Ensure the scan line (channels) are calculated correctly for each point"""
        point_cloud_simple = [np.array([[ 0,   1,  2],
                                [-1,  -3,  4],
                                [ 4,   1,  2],
                                [ 4,  10,  2],
                                [ 4,  10,  1],
                                [ 4,  10,  0]]),
                              np.array([[ 4,  10, -2],
                                [ 4,  10, -1],
                                [ 0,   1, -2],
                                [10,  10, -1],
                                [10,  10, -3],
                                [np.nan, np.nan, np.nan]])]
        lidar_data = LiDARData("robot", [0, 1], point_cloud_simple, None, CoordinateFrame.FLU)
        lidar_data.calculate_point_channels(41, -20, 20)

        np.testing.assert_array_equal(lidar_data.channels, [[40, 40, 40, 31, 25, 20],
                                                            [ 9, 15, 0, 16, 8, 65535]])
        
    def test_make_dense(self):
        """ Ensure invalid points (infinities and NaNs are removed) """

        pc_list = [np.array([[0, 0, 0], [1, 4, 6], [0, 1, 0], [np.inf, np.inf, np.inf], [0, 1, np.inf]])]
        lidar_data = LiDARData('lidar_link', np.array([0]), pc_list, None, CoordinateFrame.NED)

        lidar_data.make_dense()

        pc_expected = np.array([[1, 4, 6], [0, 1, 0]])
        np.testing.assert_array_equal(pc_expected, lidar_data.get_point_cloud_at_index(0)[0])

    # NOTE: Only testing ROSBAGS right now
    def test_get_ros_msg_type(self):
        """ Ensure we get the correct ROS message type. """

        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        print(folder_path)
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)
        ros_msg_type = lidar_data.get_ros_msg_type(ROSMsgLibType.ROSBAGS)
        self.assertEqual(ros_msg_type, 'sensor_msgs/msg/PointCloud2')

    def test_get_ros_msg(self):
        """ Ensure we can create a ROS PointCloud2 message correctly. """

        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        print(folder_path)
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)
        lidar_data.calculate_point_channels(51, -25, 25)

        # Get a ROS message for the first point cloud (index 0)
        ros_msg = lidar_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 0)

        # Validate header
        self.assertEqual(ros_msg.header.frame_id, "robot")
        self.assertEqual(ros_msg.header.stamp.sec, 0)
        self.assertEqual(ros_msg.header.stamp.nanosec, 100000000)  # 0.1 seconds = 100000000 nanoseconds

        # Validate point cloud structure
        self.assertEqual(ros_msg.height, 1)
        self.assertEqual(ros_msg.width, lidar_data.point_clouds[0].shape[0])  # Number of points
        self.assertEqual(ros_msg.is_bigendian, False)
        self.assertEqual(ros_msg.is_dense, False)
        self.assertEqual(ros_msg.point_step, 24) 
        self.assertEqual(ros_msg.row_step, 24 * lidar_data.point_clouds[0].shape[0])

        # Validate fields (x, y, z)
        self.assertEqual(len(ros_msg.fields), 6)
        self.assertEqual(ros_msg.fields[0].name, 'x')
        self.assertEqual(ros_msg.fields[0].offset, 0)
        self.assertEqual(ros_msg.fields[0].datatype, 7)  # FLOAT32
        self.assertEqual(ros_msg.fields[0].count, 1)
        self.assertEqual(ros_msg.fields[1].name, 'y')
        self.assertEqual(ros_msg.fields[1].offset, 4)
        self.assertEqual(ros_msg.fields[2].name, 'z')
        self.assertEqual(ros_msg.fields[2].offset, 8)
        self.assertEqual(ros_msg.fields[3].name, 'ring')
        self.assertEqual(ros_msg.fields[3].offset, 12)
        self.assertEqual(ros_msg.fields[4].name, 'time')
        self.assertEqual(ros_msg.fields[4].offset, 16)
        self.assertEqual(ros_msg.fields[5].name, 'intensity')
        self.assertEqual(ros_msg.fields[5].offset, 20)

        # Validate data array length
        expected_data_length = lidar_data.point_clouds[0].shape[0] * 24  # num_points * point_step
        self.assertEqual(len(ros_msg.data), expected_data_length)

        # Validate actual point cloud values by decoding the binary data
        binary_data = bytes(ros_msg.data)
        num_points = ros_msg.width
        unpacked_points = []
        for i in range(num_points):
            offset = i * 24  # 12 bytes per point (3 floats × 4 bytes)
            x, y, z, r, t, i = struct.unpack('<fffHxxff', binary_data[offset:offset+24])  # little-endian floats
            unpacked_points.append([x, y, z, r, t, i])
        unpacked_points = np.array(unpacked_points)

        # Compare unpacked points with original point cloud data (though with zeros turned to NaNs)
        org_points = lidar_data.point_clouds[0].astype(np.float32)
        org_points_wNaNs = np.where(org_points == 0, np.nan, org_points)
        np.testing.assert_allclose(unpacked_points[:,0:3], org_points_wNaNs)

        # Verify specific known points
        np.testing.assert_array_almost_equal(unpacked_points[35], [26.67838478, 0.3280502, -9.79601479, 5, 0, 255], decimal=5)

        # Test another index to ensure timestamps are handled correctly
        ros_msg_2 = lidar_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 1)
        self.assertEqual(ros_msg_2.header.stamp.sec, 0)
        self.assertEqual(ros_msg_2.header.stamp.nanosec, 600000000)  # 0.6 seconds


if __name__ == "__main__":
    unittest.main()
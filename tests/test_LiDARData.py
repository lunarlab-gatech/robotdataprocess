from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame, LiDARData, ROSMsgLibType, Ros2BagWrapper
import shutil
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

    def test_to_npy_files(self):
        """ Ensure we can save LiDAR data to npy files and load it back identically. """

        # Load original data from fixtures
        src_folder = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        lidar_data = LiDARData.from_npy_files(src_folder, "robot", CoordinateFrame.NED)

        # Save to a temporary folder
        out_folder = Path(Path('.'), 'tests', 'temporary_files', 'test_LiDARData', 'test_to_npy_files').absolute()
        if out_folder.is_dir():
            shutil.rmtree(out_folder)
        lidar_data.to_npy_files(out_folder)

        # Reload and compare
        loaded = LiDARData.from_npy_files(out_folder, "robot", CoordinateFrame.NED)
        self.assertEqual(loaded.len(), lidar_data.len())
        np.testing.assert_array_equal(loaded.timestamps, lidar_data.timestamps)
        for i in range(lidar_data.len()):
            np.testing.assert_array_almost_equal(
                np.array(loaded.point_clouds[i]),
                np.array(lidar_data.point_clouds[i])
            )
        self.assertEqual(loaded.channels, lidar_data.channels)  # Both None

        # Test round-trip with channels (same number of points per frame required by from_npy_files)
        pc = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
              np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)]
        channels = [np.array([0, 5], dtype=np.uint16),
                    np.array([12, 3], dtype=np.uint16)]
        lidar_with_channels = LiDARData("sensor", [Decimal("1.0"), Decimal("2.0")], pc, channels, CoordinateFrame.FLU)

        out_folder_ch = Path(Path('.'), 'tests', 'temporary_files', 'test_LiDARData', 'test_to_npy_files_channels').absolute()
        if out_folder_ch.is_dir():
            shutil.rmtree(out_folder_ch)
        lidar_with_channels.to_npy_files(out_folder_ch)

        loaded_ch = LiDARData.from_npy_files(out_folder_ch, "sensor", CoordinateFrame.FLU)
        self.assertEqual(loaded_ch.len(), 2)
        for i in range(2):
            np.testing.assert_array_almost_equal(
                np.array(loaded_ch.point_clouds[i]),
                np.array(lidar_with_channels.point_clouds[i])
            )
            np.testing.assert_array_equal(
                np.array(loaded_ch.channels[i]),
                np.array(lidar_with_channels.channels[i])
            )

        # Cleanup
        shutil.rmtree(out_folder)
        shutil.rmtree(out_folder_ch)

    def test_to_FLU_frame(self):
        """ Ensure NED-to-FLU transformation flips Y and Z when getting point clouds. """
        point_cloud = [np.array([[ 0.0,  1.0,  2.0],
                                  [-1.0, -3.0,  4.0]], dtype=np.float32)]
        lidar_data = LiDARData("robot", [Decimal("1.0")], point_cloud, None, CoordinateFrame.NED)
        lidar_data.to_FLU_frame()

        self.assertEqual(lidar_data.frame, CoordinateFrame.FLU)

        pc, _ = lidar_data.get_point_cloud_at_index(0)
        expected_pc = np.array([[ 0.0, -1.0, -2.0],
                                [-1.0,  3.0, -4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(pc, expected_pc)

        # Calling again should be a no-op (already FLU)
        lidar_data.to_FLU_frame()
        self.assertEqual(len(lidar_data.transformations), 1)

    def test_to_FLU_frame_from_ENU(self):
        """ ENU-to-FLU transformation: X→-Y, Y→+X, Z stays. """
        point_cloud = [np.array([[ 1.0,  0.0,  0.0],   # pure East  → FLU forward=0, left=-1
                                  [ 0.0,  1.0,  0.0],   # pure North → FLU forward=1, left=0
                                  [ 0.0,  0.0,  1.0],   # pure Up    → unchanged
                                  [ 3.0,  4.0, -2.0]], dtype=np.float32)]
        lidar_data = LiDARData("robot", [Decimal("1.0")], point_cloud, None, CoordinateFrame.ENU)
        lidar_data.to_FLU_frame()

        self.assertEqual(lidar_data.frame, CoordinateFrame.FLU)

        pc, _ = lidar_data.get_point_cloud_at_index(0)
        expected_pc = np.array([[ 0.0, -1.0,  0.0],
                                 [ 1.0,  0.0,  0.0],
                                 [ 0.0,  0.0,  1.0],
                                 [ 4.0, -3.0, -2.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(pc, expected_pc)

        # Calling again should be a no-op (already FLU)
        lidar_data.to_FLU_frame()
        self.assertEqual(len(lidar_data.transformations), 1)

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

        # Test that channels are calculated correctly after cropping
        point_clouds_extended = [
            np.array([[10, 10, 5], [10, 10, 0]]),   # index 0, timestamp 0.5 (will be cropped)
            np.array([[ 0,  1,  2],                 # index 1, timestamp 1.0
                      [-1, -3,  4],
                      [ 4,  1,  2],
                      [ 4, 10,  2],
                      [ 4, 10,  1],
                      [ 4, 10,  0]]),
            np.array([[ 4, 10, -2],                 # index 2, timestamp 1.5
                      [ 4, 10, -1],
                      [ 0,  1, -2],
                      [10, 10, -1],
                      [10, 10, -3],
                      [np.nan, np.nan, np.nan]]),
            np.array([[10, 10, -5], [10, 10, 0]]),  # index 3, timestamp 2.0 (will be cropped)
        ]
        timestamps = [Decimal("0.5"), Decimal("1.0"), Decimal("1.5"), Decimal("2.0")]
        lidar_data_cropped = LiDARData("robot", timestamps, point_clouds_extended, None, CoordinateFrame.FLU)

        # Crop to keep only timestamps 1.0 and 1.5 (indices 1 and 2)
        lidar_data_cropped.crop_data(Decimal("1.0"), Decimal("1.5"))
        lidar_data_cropped.calculate_point_channels(41, -20, 20)

        # Verify channels are calculated correctly for the cropped data
        # Should match the original test values since we're using the same point clouds
        _, channels_0 = lidar_data_cropped.get_point_cloud_at_index(0)
        _, channels_1 = lidar_data_cropped.get_point_cloud_at_index(1)
        np.testing.assert_array_equal(channels_0, [40, 40, 40, 31, 25, 20])
        np.testing.assert_array_equal(channels_1, [ 9, 15,  0, 16,  8, 65535])
        
    def test_make_dense(self):
        """ Ensure invalid points (infinities and NaNs are removed) """

        pc_list = [np.array([[0, 0, 0], [1, 4, 6], [0, 1, 0], [np.inf, np.inf, np.inf], [0, 1, np.inf]])]
        lidar_data = LiDARData('lidar_link', np.array([0]), pc_list, None, CoordinateFrame.NED)
        lidar_data.make_dense()
        pc_expected = np.array([[1, 4, 6], [0, 1, 0]])
        np.testing.assert_array_equal(pc_expected, lidar_data.get_point_cloud_at_index(0)[0])

        pc_list = [np.array([[0, 0, 0], [1, 4, 6], [0, 1, 0], [np.inf, np.inf, np.inf], [0, 1, np.inf]])]
        lidar_data = LiDARData('lidar_link', np.array([0]), pc_list, [np.array([0, 1, 2, 3, 4], dtype=np.uint16)], CoordinateFrame.NED)
        lidar_data.make_dense()
        pc_expected = np.array([[1, 4, 6], [0, 1, 0]])
        channels_expected = np.array([1,2], dtype=np.uint16)
        np.testing.assert_array_equal(pc_expected, lidar_data.get_point_cloud_at_index(0)[0])
        np.testing.assert_array_equal(channels_expected, lidar_data.get_point_cloud_at_index(0)[1])

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

    def test_crop_data(self):
        """ Ensure crop_data correctly filters data and raises errors on out-of-bounds access. """

        # Create point clouds with distinct values so we can verify correct indexing after crop
        point_clouds = [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # index 0, timestamp 0.5
            np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),  # index 1, timestamp 1.0
            np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]),  # index 2, timestamp 1.5
            np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]]),  # index 3, timestamp 2.0
            np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]),  # index 4, timestamp 2.5
        ]
        timestamps = [Decimal("0.5"), Decimal("1.0"), Decimal("1.5"), Decimal("2.0"), Decimal("2.5")]
        lidar_data = LiDARData("robot", timestamps, point_clouds, None, CoordinateFrame.FLU)

        # Verify initial state
        self.assertEqual(lidar_data.len(), 5)

        # Crop data to keep only timestamps in [1.0, 2.0]
        lidar_data.crop_data(Decimal("1.0"), Decimal("2.0"))

        # Verify cropped length
        self.assertEqual(lidar_data.len(), 3)

        # Verify timestamps are correct after cropping
        np.testing.assert_array_equal(
            lidar_data.timestamps.astype(float),
            [1.0, 1.5, 2.0]
        )

        # Verify that cropped indices return correct point clouds
        # Index 0 after crop should be original index 1 (timestamp 1.0)
        pc_0, _ = lidar_data.get_point_cloud_at_index(0)
        np.testing.assert_array_equal(pc_0[1], [3.0, 3.0, 3.0])

        # Index 1 after crop should be original index 2 (timestamp 1.5)
        pc_1, _ = lidar_data.get_point_cloud_at_index(1)
        np.testing.assert_array_equal(pc_1[1], [5.0, 5.0, 5.0])

        # Index 2 after crop should be original index 3 (timestamp 2.0)
        pc_2, _ = lidar_data.get_point_cloud_at_index(2)
        np.testing.assert_array_equal(pc_2[1], [7.0, 7.0, 7.0])

        # Verify that out-of-bounds index raises an error
        with self.assertRaises(IndexError):
            lidar_data.get_point_cloud_at_index(3)

        # Verify that get_ros_msg also raises an error for out-of-bounds index
        lidar_data.calculate_point_channels(16, -15, 15)
        with self.assertRaises(ValueError):
            lidar_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 3)

        # Verify that get_ros_msg works for valid index after cropping
        ros_msg = lidar_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 2)
        self.assertEqual(ros_msg.header.stamp.sec, 2)
        self.assertEqual(ros_msg.header.stamp.nanosec, 0)


    def test_calculate_point_channels_already_exists(self):
        """ Test that RuntimeError is raised when channels already calculated. """
        pc = [np.array([[1.0, 2.0, 3.0]])]
        channels = [np.array([0], dtype=np.uint16)]
        lidar = LiDARData("robot", [0], pc, channels, CoordinateFrame.FLU)
        with self.assertRaises(RuntimeError):
            lidar.calculate_point_channels(16, -15, 15)

    def test_get_ros_msg_no_channels_raises(self):
        """ Test RuntimeError when channels are None (not yet calculated). """
        pc = [np.array([[1.0, 2.0, 3.0]])]
        lidar = LiDARData("robot", [0], pc, None, CoordinateFrame.FLU)
        with self.assertRaises(RuntimeError):
            lidar.get_ros_msg(ROSMsgLibType.ROSBAGS, 0)

    def test_from_ros2_bag(self):
        """ Ensure we can write LiDAR data to a ROS2 bag and read it back, loading frame_id from the bag. """

        # Create LiDAR data from npy files and compute channels (required for get_ros_msg)
        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)
        lidar_data.calculate_point_channels(51, -25, 25)

        # Write to a ROS2 bag
        bag_path = Path(Path('.'), 'tests', 'temporary_files', 'test_LiDARData', 'test_from_ros2_bag', 'lidar_bag').absolute()
        if bag_path.is_dir():
            shutil.rmtree(bag_path)
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        Ros2BagWrapper.write_data_to_rosbag(bag_path, [lidar_data], ['/lidar'], [None], None)

        # Read it back -- frame_id should be loaded from the bag header
        loaded_data = LiDARData.from_ros2_bag(bag_path, '/lidar', CoordinateFrame.NED)

        # Verify frame_id was loaded from the bag
        self.assertEqual(loaded_data.frame_id, "robot")

        # Verify basic structure
        self.assertEqual(loaded_data.len(), 3)
        self.assertEqual(loaded_data.frame, CoordinateFrame.NED)

        # Verify timestamps match (compare as floats due to precision differences in bag round-trip)
        np.testing.assert_array_almost_equal(
            loaded_data.timestamps.astype(float),
            lidar_data.timestamps.astype(float),
            decimal=5
        )

        # Verify point cloud data round-trips correctly for a known point
        pc_loaded, _ = loaded_data.get_point_cloud_at_index(0)
        np.testing.assert_array_almost_equal(
            pc_loaded[35], [26.67838478, 0.3280502, -9.79601479], decimal=4
        )

        # Cleanup
        shutil.rmtree(bag_path)


if __name__ == "__main__":
    unittest.main()
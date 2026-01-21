import inspect
from multiprocessing import Process
import numpy as np
import os
from pathlib import Path
from robotdataprocess import LiDARData, CoordinateFrame, ImuData, OdometryData
from robotdataprocess.data_types.Data import Data, ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.data_types.OdometryData import PATH_SLICE_STEP
from robotdataprocess.ModuleImporter import ModuleImporter
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
import struct
import time
from typing import Any, Callable, Optional
from numpy.typing import NDArray
import unittest

@unittest.skipIf(os.getenv("SKIP_ROS2_TESTS") == "True", "ROS2 not installed")
class TestRosPublisher(unittest.TestCase):

    def util_ROS2_test(self, data: Data, topic_class: Any, topic_name: str, 
                       msg_to_dict_fn: Callable, assert_data_dict_equal: Callable,
                       verbose: bool = False, data_msg_type:  Optional[str] = None):
        
        rclpy = ModuleImporter.get_module('rclpy')
        Node = ModuleImporter.get_module_attribute('rclpy.node', 'Node')

        # Create a ROS2 node to subscribe to the published topic
        class Listener(Node):
            def __init__(self):
                super().__init__('listener')
                self.subscription = self.create_subscription(topic_class, topic_name, self.callback, 10)
                self.msg_to_dict_fn = msg_to_dict_fn
                self.received = []
                self.get_logger().info(f"Listener initialized!")
            
            # Generic callback function to save recieved messages
            def callback(self, msg):
                self.received.append(msg_to_dict_fn(msg))
                self.get_logger().info(f"Received msg {len(self.received)-1} at time {self.received[-1]['stamp']}")

        
        # Launch the publisher we initialize rclpy (otherwise rclpy.init breaks process forking for ROS2)
        p = Process(target=publish_data_ROS_multiprocess, args=([data], [topic_name], [data_msg_type], [500], [1], 
                                                                ROSMsgLibType.RCLPY, False, verbose, 0.0, True))
        p.start()

        # Initialize ROS2 and create the listener node
        rclpy.init()
        node = Listener()

        # Start listening here and meanwhile launch the publisher 
        try:
            start_time = time.time()
            timeout_sec = 5.0
            while rclpy.ok() and (time.time() - start_time) < timeout_sec:
                rclpy.spin_once(node, timeout_sec=0.1)

            # Make sure we recieved at least one message
            self.assertTrue(len(node.received) > 0)

            # Make sure data is correct for each message we recieved
            data_ts_floats = data.timestamps.astype(float)
            for j in range(len(node.received)):
                stamp = node.received[j]['stamp']
                matches = np.where(np.isclose(data_ts_floats, stamp, atol=1e-6))[0]
                if matches.size > 0:
                    # Ensure that data matches the recieved msg dictionary
                    matched_index = int(matches[0])
                    assert_data_dict_equal(data, matched_index, node.received[j])
                else:
                    self.fail("Recieved ROS Message has a timestamp that doesn't match any of the sent messages!")

        finally:
            # ---------- Guaranteed cleanup ----------
            node.destroy_node()
            rclpy.shutdown()

        
    def test__run_ROS2_publisher_process(self):
        """ Test that we can publish to ROS2 without losing data."""

        # ================== Test ImageData ================== 
        # Create an ImageDataInMemory object
        file_path = Path(Path('.'), 'tests', 'files', 'test_RosPublisher', 
                         'test__run_ROS2_publisher_process', 'ImageData').absolute()
        image_data = ImageDataInMemory.from_image_files(file_path, '/cam0')

        # Lazily import ROS2 libraries
        Image = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Image')

        # Write msg_to_dict function
        CvBridge = ModuleImporter.get_module_attribute('cv_bridge', 'CvBridge')
        bridge = CvBridge()
        def msg_to_dict_fn(msg: Image) -> dict:
            image: NDArray = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            return {
                "image": image,
                "height": msg.height,
                "width": msg.width,
                "encoding": msg.encoding,
                "frame_id": msg.header.frame_id,
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }

        # Write assert_data_dict_equal function
        def assert_data_dict_equal(data_sent: ImageData, matched_index: int, msg_recieved: dict):
            np.testing.assert_array_equal(data_sent.images[matched_index], msg_recieved["image"])
            np.testing.assert_equal(data_sent.height, msg_recieved["height"])
            np.testing.assert_equal(data_sent.width, msg_recieved["width"])
            np.testing.assert_equal(data_sent.encoding, ImageData.ImageEncoding.from_ros_str(msg_recieved["encoding"]))
            np.testing.assert_equal(data_sent.frame_id, msg_recieved["frame_id"])
            np.testing.assert_almost_equal(float(data_sent.timestamps[matched_index]), msg_recieved["stamp"])

        # # Test that we can send data over ROS2 and get it back successfully
        p = Process(target=self.util_ROS2_test, args=(image_data, Image, 
                                '/cam0/image_raw', msg_to_dict_fn, assert_data_dict_equal))
        p.start()
        p.join()
        self.assertEqual(p.exitcode, 0, f"Child process failed with exit code {p.exitcode}")

        # ================== Test LiDARData ================== 
        # Create a LiDARData object
        file_path = Path(Path('.'), 'tests', 'files', 'test_RosPublisher', 
                         'test__run_ROS2_publisher_process', 'LiDARData').absolute()
        lidar_data = LiDARData.from_npy_files(file_path, "lidar_link", CoordinateFrame.NED)
        lidar_data.calculate_point_channels(32, -25, 25)

        # Lazily import ROS2 libraries
        PointCloud2 = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'PointCloud2')

        # Write msg_to_dict function
        CvBridge = ModuleImporter.get_module_attribute('cv_bridge', 'CvBridge')
        bridge = CvBridge()
        def msg_to_dict_fn(msg: PointCloud2) -> dict:

            # Map ROS datatype to struct format
            datatype_to_struct = {
                7: 'f',  # FLOAT32
                4: 'I',  # UINT32
                2: 'B',  # UINT8
            }

            # Build struct format for all fields in the message
            struct_endian = '>' if msg.is_bigendian else '<'
            struct_fmt = struct_endian
            for f in msg.fields:
                if f.datatype not in datatype_to_struct:
                    raise NotImplementedError(f"Datatype {f.datatype} for field '{f.name}' not supported")
                struct_fmt += datatype_to_struct[f.datatype]

            # Extract all points
            points = []
            for i in range(0, len(msg.data), msg.point_step):
                values = struct.unpack(struct_fmt, msg.data[i:i+msg.point_step])
                points.append(values)
            points_np = np.array(points, dtype=np.float32)

            # Create a dict mapping field names -> column indices
            field_indices = {f.name: idx for idx, f in enumerate(msg.fields)}

            # Now we can index by name
            xyz = points_np[:, [field_indices['x'], field_indices['y'], field_indices['z']]]
            ring = points_np[:, field_indices['ring']].astype(np.int32)
            time = points_np[:, field_indices['time']]
            intensity = points_np[:, field_indices['intensity']]

            # Return important data in a dictionary
            return {
                "point_clouds": xyz,
                "channels": ring,
                "time": time,
                "intensity": intensity,
                "frame_id": msg.header.frame_id,
                "height": msg.height,
                "width": msg.width,
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
        
        # Write assert_data_dict_equal function
        def assert_data_dict_equal(data_sent: LiDARData, matched_index: int, msg_recieved: dict):
            exp_num_points = data_sent.point_clouds[matched_index].shape[0]
            np.testing.assert_array_equal(data_sent.get_point_cloud_at_index(matched_index)[0], msg_recieved["point_clouds"])
            np.testing.assert_array_equal(data_sent.get_point_cloud_at_index(matched_index)[1], msg_recieved["channels"])
            np.testing.assert_array_equal(np.zeros(exp_num_points), msg_recieved["time"])
            np.testing.assert_array_equal(255 * np.ones(exp_num_points), msg_recieved["intensity"])
            np.testing.assert_equal(data_sent.frame_id, msg_recieved["frame_id"])
            np.testing.assert_equal(1, msg_recieved["height"])
            np.testing.assert_equal(exp_num_points, msg_recieved["width"])
            np.testing.assert_almost_equal(float(data_sent.timestamps[matched_index]), msg_recieved["stamp"])
    
        # Test that we can send data over ROS2 and get it back successfully
        p = Process(target=self.util_ROS2_test, args=(lidar_data, PointCloud2, '/points_raw', msg_to_dict_fn, assert_data_dict_equal, True))
        p.start()
        p.join()
        self.assertEqual(p.exitcode, 0, f"Child process failed with exit code {p.exitcode}")

        # ================== Test ImuData ================== 
        # Create a ImuData object
        file_path = Path(Path('.'), 'tests', 'files', os.path.basename(__file__)[:-3], 
                         inspect.currentframe().f_code.co_name, 'ImuData', 'imu.txt').absolute()
        imu_data = ImuData.from_txt_file(file_path, "base_link", CoordinateFrame.NED, True)

        # Write msg_to_dict_fn
        Imu = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Imu')
        def msg_to_dict_fn(msg: Imu) -> dict:
            return {
                "linear_acceleration": np.array([msg.linear_acceleration.x,
                                                 msg.linear_acceleration.y,
                                                 msg.linear_acceleration.z]),
                "linear_acceleration_covariance": msg.linear_acceleration_covariance,
                "angular_velocity": np.array([msg.angular_velocity.x,
                                                 msg.angular_velocity.y,
                                                 msg.angular_velocity.z]),
                "angular_velocity_covariance": msg.angular_velocity_covariance,
                "orientation": np.array([msg.orientation.x,
                                         msg.orientation.y,
                                         msg.orientation.z,
                                         msg.orientation.w]),
                "orientation_covariance": msg.orientation_covariance,
                "frame_id": msg.header.frame_id,
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
        
        # Write assert_data_dict_equal function
        def assert_data_dict_equal(data_sent: ImuData, matched_index: int, msg_recieved: dict):
            np.testing.assert_array_equal( data_sent.lin_acc[matched_index].astype(float),             msg_recieved["linear_acceleration"])
            np.testing.assert_array_equal( np.zeros(9),                                                 msg_recieved["linear_acceleration_covariance"])
            np.testing.assert_array_equal( data_sent.ang_vel[matched_index].astype(float),             msg_recieved["angular_velocity"])
            np.testing.assert_array_equal( np.zeros(9),                                                msg_recieved["angular_velocity_covariance"])
            np.testing.assert_array_equal( data_sent.orientations[matched_index].astype(float),        msg_recieved["orientation"])
            np.testing.assert_array_equal( np.zeros(9),                                  msg_recieved["orientation_covariance"])
            np.testing.assert_equal(       data_sent.frame_id,                           msg_recieved["frame_id"])
            np.testing.assert_almost_equal(float(data_sent.timestamps[matched_index]),   msg_recieved["stamp"])
        
        # Test that we can send data over ROS2 and get it back successfully
        p = Process(target=self.util_ROS2_test, args=(imu_data, Imu, '/imu0', msg_to_dict_fn, assert_data_dict_equal, False))
        p.start()
        p.join()
        self.assertEqual(p.exitcode, 0, f"Child process failed with exit code {p.exitcode}")

        # ================== Test OdometryData ================== 
        # Create an OdometryData object
        file_path = Path(Path('.'), 'tests', 'files', 'test_RosPublisher', 'test__run_ROS2_publisher_process', 'OdometryData', 'odomGT.csv').absolute()
        odom_data = OdometryData.from_csv(file_path, "odom", "base_link", CoordinateFrame.FLU, True, None)

        # Write msg_to_dict_fn
        PathMsg = ModuleImporter.get_module_attribute('nav_msgs.msg', 'Path')
        def msg_to_dict_fn(msg: PathMsg) -> dict:
            return {
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "frame_id": msg.header.frame_id,
                "position": np.array([msg.poses[-1].pose.position.x,
                                      msg.poses[-1].pose.position.y,
                                      msg.poses[-1].pose.position.z]),
                "orientation": np.array([msg.poses[-1].pose.orientation.x,
                                         msg.poses[-1].pose.orientation.y,
                                         msg.poses[-1].pose.orientation.z,
                                         msg.poses[-1].pose.orientation.w])
            }

        # Write assert_data_dict_equal function
        def assert_data_dict_equal(data_sent: OdometryData, matched_index: int, msg_recieved: dict):
            np.testing.assert_almost_equal(float(data_sent.timestamps[matched_index]), msg_recieved["stamp"])
            np.testing.assert_equal(      data_sent.frame_id,                          msg_recieved["frame_id"])
            i = (matched_index // PATH_SLICE_STEP)
            np.testing.assert_array_equal(data_sent.positions[i * PATH_SLICE_STEP].astype(float),    msg_recieved["position"])
            np.testing.assert_array_equal(data_sent.orientations[i * PATH_SLICE_STEP].astype(float), msg_recieved["orientation"])

        # Test that we can send data over ROS2 and get it back successfully
        p = Process(target=self.util_ROS2_test, args=(odom_data, PathMsg, '/odom_gt', msg_to_dict_fn, assert_data_dict_equal, False, "Path"))
        p.start()
        p.join()
        self.assertEqual(p.exitcode, 0, f"Child process failed with exit code {p.exitcode}")
 
if __name__ == "__main__":
    unittest.main()